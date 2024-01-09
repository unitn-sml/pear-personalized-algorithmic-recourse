"""Train the FARE model by using the given python class instead of the script."""

from recourse_fare.models.FARE import FARE
from recourse_fare.models.EFARE import EFARE
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor
from recourse_fare.utils.Mixture import MixtureModel

from blackbox.adult.adult_scm import AdultSCM
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM
from blackbox.synthetic.synthetic_scm import SyntheticSCM
from blackbox.utils_blackbox.net import Net, NetLinear

import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer

import torch

import pandas as pd
import numpy as np

import os
import random

import dill as pickle

from argparse import ArgumentParser

from blackbox.config import MIXTURE_MEAN_LIST, NN_CONFIG
from competitors.fare.config import FARE_CONFIG, FARE_CONFIG_MCTS

if __name__ == "__main__":

    # Initialize the parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="svc", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--retrain", default=False, action="store_true", help="Retrain the model if present.")
    parser.add_argument("--clean", default=False, action="store_true", help="Retrain the model starting from scratch.")
    parser.add_argument("--uniform", default=False, action="store_true", help="Train the model with uniform costs.")
    parser.add_argument("--output", default="competitors/fare/", type=str, help="Path where to save the trained model.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Train WFARE with a corrupted graph.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed used for training")
    parser.add_argument("--check", default=False, action="store_true", help="Prevent training the model again if it exists")

    args = parser.parse_args()

    # Seed for reproducibility
    np.random.seed(2023)
    torch.manual_seed(2023)
    random.seed(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_data_path = f"blackbox/{args.dataset}/train_data_{args.dataset}.csv"
    test_data_path = f"blackbox/{args.dataset}/test_data_{args.dataset}.csv"
    blackbox_model_path = f"blackbox/{args.dataset}/model_{args.model}_{args.dataset}.pth"

    # Read data and preprocess them
    X_train = pd.read_csv(train_data_path)
    X_test = pd.read_csv(test_data_path)

    # Build weights dataframes
    if args.dataset == "adult":
        tmp_scm = AdultSCM(None)
    elif args.dataset == "synthetic":
        tmp_scm = SyntheticSCM(None)
    else:
        tmp_scm = GiveMeCreditSCM(None)
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    # Simulate training this model with a misspecified causal graph.
    edges_to_remove=None
    full_edges = list(tmp_scm.scm.edges())
    if args.corrupt_graph > 0:
        n_edges_to_remove = int(args.corrupt_graph*len(full_edges))
        print(f"Removing {n_edges_to_remove}/{len(full_edges)} edges...")
        edges_to_remove = list(range(0, len(full_edges)))
        np.random.shuffle(edges_to_remove)
        edges_to_remove = edges_to_remove[:n_edges_to_remove]
        edges_to_remove = [ e for idx,e in zip(range(0, len(full_edges)), full_edges) if idx in edges_to_remove]

    print("Current Mixture dimension: ", len(keys_weights))

    # Generate random weights.
    # For training, we use the expected value (sampled) of the mixture.
    # For testing, we sample the weights from the mixture.
    if args.uniform:
        W_train = np.ones(len(keys_weights))
    else:
        mixture = MixtureModel(mixture_means=MIXTURE_MEAN_LIST.get(args.dataset))
        W_train = np.mean(mixture.sample(1000), axis=0)

    # Transform the weights into dataframes.
    W_train = pd.DataFrame([W_train], columns=keys_weights)
    W_train = W_train.to_dict("records")[0]

    # Build a preprocessing pipeline, which can be used to preprocess
    # the elements of the dataset.
    # The Fast preprocessor does min/max scaling and categorical encoding.
    # It is much faster than then scikit learn ones and it uses dictionaries
    # and sets to perform operations on the fly.
    preprocessor = FastPreprocessor()
    preprocessor.fit(X_train)

    # Fit a simple SVC model over the data
    if args.model != "nn":
        with open(blackbox_model_path, "rb") as model:
            blackbox_model = pickle.load(model)
    else:
        if args.model == "nn":
            blackbox_model = Net(len(preprocessor.transform(X_train)[0]),
                                 layers=NN_CONFIG.get(args.dataset).get("nn").get("layers"))
            blackbox_model.load_state_dict(
                torch.load(blackbox_model_path)
            )
            blackbox_model.eval()

    # Filter the training dataset by picking only the examples which are classified negatively by the model
    if args.model != "nn":
        output = blackbox_model.predict(preprocessor.transform(X_train))
    else:
        with torch.no_grad():
            output = blackbox_model(torch.FloatTensor(preprocessor.transform(X_train))).round().numpy()
    
    X_train["predicted"] = output
    X_train = X_train[X_train.predicted == 0]
    X_train.drop(columns="predicted", inplace=True)

    policy_config = FARE_CONFIG.get(args.dataset).get("policy_config")

    environment_config = {
        "class_name": FARE_CONFIG.get(args.dataset).get("environment"),
        "additional_parameters": {
            "preprocessor": preprocessor,
            "model_type": args.model,
            "weights": W_train,
            "agnostic": True,
            "remove_edges": edges_to_remove
        }
    }
    
    mcts_config = FARE_CONFIG.get(args.dataset).get("mcts_config")

    print("[*] FARE Agent Architecture")

    # Minimize the action cost coeff if we are training a corrupt graph.
    if args.corrupt_graph > 0.0:
        mcts_config["action_cost_coeff"] = FARE_CONFIG_MCTS.get(args.dataset).get(args.corrupt_graph, 1.0)
        print("Setting action MCTS expansion penalty to", mcts_config["action_cost_coeff"])

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = FARE(blackbox_model, policy_config, environment_config, mcts_config,
                  batch_size=FARE_CONFIG.get(args.dataset).get("batch_size"))
    print(model.policy)

    # Train a FARE model given the previous configurations
    fare_model_type = "fare" if not args.uniform else "fare_uniform"

    # Model name
    fare_model_name = f"{fare_model_type}_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth"
    fare_full_model_name = f"{fare_model_type}_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth"
    fare_model_path = os.path.join(args.output, fare_model_name)
    fare_full_model_path = os.path.join(args.output, fare_full_model_name)

    if args.retrain:
        
        if not args.clean and os.path.isfile(fare_model_path):
            print("[*] Loading checkpoint from file")
            model.load(fare_model_path)
        
        if os.path.isfile(fare_model_path) and args.check:
            print(f"[*] Skipping training current model since it exists {fare_model_path}")
            exit(0)

        model.fit(X_train, max_iter=FARE_CONFIG.get(args.dataset).get("training_steps"), tensorboard="./wfare")
        
        # We save the trained WFARE model to disc
        model.save(fare_model_path)
        pickle.dump(model, open(fare_full_model_path, "wb"))
    
    else:
        if os.path.isfile(fare_model_path):
            model.load(fare_model_path)
        else:
            print(f"No model available ({fare_model_path})!")
            exit()
    
    # We build the interpretable preprocessor
    cat_selector = make_column_selector(dtype_include=[object, "category"])
    preprocessor_efare = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore",sparse=False), cat_selector), remainder="passthrough"
    )

    # We fit the EFARE preprocessor
    preprocessor_efare.fit(X_train)

    # We save the preprocessor for later use
    preprocessor_fare_full_model_path = os.path.join(args.output, f"efare_preprocessor_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")
    pickle.dump(preprocessor_efare, open(preprocessor_fare_full_model_path, "wb"))

    # Train the EFARE model
    efare_model = EFARE(model, preprocessor_efare)
    efare_model.fit(X_train[0:300], verbose=True)
    efare_full_model_path = os.path.join(args.output, f"e{fare_model_type}_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")
    efare_model.save(efare_full_model_path)