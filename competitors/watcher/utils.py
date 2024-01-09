import datetime
from typing import List, Optional

import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from carla import log
from carla.recourse_methods.processing import reconstruct_encoding_constraints

DECISION_THRESHOLD = 0.5

def fix_one_hot_inconsistencies(x: torch.Tensor, feature_pos: dict):

    x_enc = torch.clamp(x.clone(), 0,1)

    binary_pairs = [(k, (min(v), max(v))) for k,v in feature_pos.items()]
    
    for k, pair in binary_pairs:
        
        idx_max = torch.argmax(x_enc[pair[0]:pair[1]])
        x_enc[pair[0]:pair[1]+1] = 0
        x_enc[pair[0]+idx_max] = 1

    return x_enc

def get_immutables_mask(x: torch.Tensor, feature_pos:dict, immutables: list):

    mutable_mask = torch.ones_like(x)
    immutables_mask = torch.zeros_like(x)
    binary_pairs = [(k, (min(v), max(v))) for k,v in feature_pos.items()]
    for k, pair in binary_pairs:
        if k in immutables:
            mutable_mask[pair[0]:pair[1]+1] = 0
            immutables_mask[pair[0]:pair[1]+1] = 1
    
    return mutable_mask, immutables_mask

def wachter_recourse(
    model,
    model_backend,
    x: np.ndarray,
    w,
    g,
    columns: list,
    immutables: np.ndarray,
    cat_feature_indices: List[int],
    feature_costs: Optional[List[float]],
    lr: float,
    lambda_param: float,
    y_target: List[int],
    n_iter: int,
    t_max_min: float,
    norm: int,
    clamp: bool,
    loss_type: str,
) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x

    Parameters
    ----------
    model:
        black-box-model to discover
    x:
        Factual instance to explain.
    cat_feature_indices:
        List of positions of categorical features in x.
    binary_cat_features:
        If true, the encoding of x is done by drop_if_binary.
    feature_costs:
        List with costs per feature.
    lr:
        Learning rate for gradient descent.
    lambda_param:
        Weight factor for feature_cost.
    y_target:
        Tuple of class probabilities (BCE loss) or [Float] for logit score (MSE loss).
    n_iter:
        Maximum number of iterations.
    t_max_min:
        Maximum time amount of search.
    norm:
        L-norm to calculate cost.
    clamp:
        If true, feature values will be clamped to intverval [0, 1].
    loss_type:
        String for loss function ("MSE" or "BCE").

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # returns counterfactual instance

    x = torch.from_numpy(x).float().to(device)
    y_target = torch.tensor(y_target).float().to(device)
    lamb = torch.tensor(lambda_param).float().to(device)

    mutable_mask, immutables_mask = get_immutables_mask(x, cat_feature_indices, immutables)
    
    # x_new is used for gradient search in optimizing process
    x_new = Variable(x.clone(), requires_grad=True)
    # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
    # such that categorical data is either 0 or 1
    x_new_enc = x_new.clone()

    optimizer = optim.Adam([x_new], lr, amsgrad=True)

    if loss_type == "MSE":
        if len(y_target) != 1:
            raise ValueError(f"y_target {y_target} is not a single logit score")

        # If logit is above 0.0 we want class 1, else class 0
        target_class = int(y_target[0] > 0.0)
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "BCE":
        if y_target[0] + y_target[1] != 1.0:
            raise ValueError(
                f"y_target {y_target} does not contain 2 valid class probabilities"
            )

        # [0, 1] for class 1, [1, 0] for class 0
        # target is the class probability of class 1
        # target_class is the class with the highest probability
        target_class = torch.round(y_target[1]).int()
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    # get the probablity of the target class
    f_x_new = model._model(x_new.clone())

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)
    while f_x_new <= DECISION_THRESHOLD:
        it = 0
        while f_x_new <= 0.5 and it < n_iter:

            optimizer.zero_grad()

            # Fix inconsistencies
            x_new_enc = fix_one_hot_inconsistencies(
                x_new, cat_feature_indices
            )
            
            x_new_enc = x_new_enc*mutable_mask + x*immutables_mask
            
            # use x_new_enc for prediction results to ensure constraints
            # get the probablity of the target class
            f_x_new = model._model(x_new_enc)
            f_x_new = torch.FloatTensor(f_x_new)

            if datetime.datetime.now() - t0 > t_max:
                break
            elif f_x_new >= 0.5:
                break
                #log.info("Counterfactual Explanation Found")

            if loss_type == "MSE":
                # single logit score for the target class for MSE loss
                f_x_loss = torch.log(f_x_new / (1 - f_x_new))
            elif loss_type == "BCE":
                f_x_loss = model._model(x_new_enc)
               
            else:
                raise ValueError(f"loss_type {loss_type} not supported")

            x_cf = pd.DataFrame(np.array([x_new_enc.clone().detach().numpy()]), columns=columns)
            x_or = pd.DataFrame(np.array([x.clone().detach().numpy()]), columns=columns)

            if feature_costs:
                cost = feature_costs(x_cf, x_or, w, g, model._preprocessor)
            else:
                cost = torch.dist(x_new_enc, x, norm)

            loss = loss_fn(f_x_loss, torch.ones(1)) + lamb * cost
            loss.backward()
            optimizer.step()

            # Fix inconsistencies
            x_new_enc = fix_one_hot_inconsistencies(
                x_new, cat_feature_indices
            )

            x_new_enc = x_new_enc*mutable_mask + x*immutables_mask
           
            it += 1
        lamb -= 0.05

    return x_new_enc.cpu().detach().numpy()
