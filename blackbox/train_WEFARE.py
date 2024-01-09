"""Train the FARE model by using the given python class instead of the script."""

from recourse_fare.models.WFARE import WFARE
from recourse_fare.models.WEFARE import WEFARE
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor, StandardPreprocessor
from recourse_fare.utils.Mixture import MixtureModel

from blackbox.adult.adult_scm import AdultSCM
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM
from blackbox.synthetic.synthetic_scm import SyntheticSCM
from blackbox.utils_blackbox.net import Net

import torch

import pandas as pd
import numpy as np

import os
import random

import dill as pickle

from argparse import ArgumentParser

from blackbox.config import MIXTURE_MEAN_LIST, WFARE_CONFIG, NN_CONFIG
from utils import build_wfarefiner

if __name__ == "__main__":

    # Initialize the parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="svc", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--retrain", default=False, action="store_true", help="Retrain the model if present.")
    parser.add_argument("--clean", default=False, action="store_true", help="Retrain the model starting from scratch.")
    parser.add_argument("--quantile", default=1.0, type=float, help="Quantile we want to use.")
    parser.add_argument("--num-examples", default=500, type=int, help="How many examples to use to train E-WFARE.")

    args = parser.parse_args()

    # Seed for reproducibility
    np.random.seed(2024)
    torch.manual_seed(2024)
    random.seed(2024)
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

    print("Current Mixture dimension: ", len(keys_weights))

    # Generate random weights.
    # For training, we use the expected value (sampled) of the mixture.
    # For testing, we sample the weights from the mixture.
    mixture = MixtureModel(mixture_means=MIXTURE_MEAN_LIST.get(args.dataset))
    W_expectation = np.mean(mixture.sample(1000), axis=0)
    W_train = mixture.sample(len(X_train))
    W_test = mixture.sample(len(X_test), mixture_class=True)

    # Transform the weights into dataframes.
    W_train = pd.DataFrame(W_train, columns=keys_weights)
    W_expectation = pd.DataFrame([W_expectation], columns=keys_weights)
    W_test = pd.DataFrame(W_test, columns=["mixture_class"]+keys_weights)

    # Save weights to disk
    W_test.to_csv(f"blackbox/{args.dataset}/weights_test_{args.dataset}.csv", index=None)
    W_test.drop(columns=["mixture_class"], inplace=True)

    # Build a preprocessing pipeline, which can be used to preprocess
    # the elements of the dataset.
    # The Fast preprocessor does min/max scaling and categorical encoding.
    # It is much faster than then scikit learn ones and it uses dictionaries
    # and sets to perform operations on the fly.
    preprocessor = FastPreprocessor()
    preprocessor.fit(X_train)

    # Fit a simple SVC model over the data
    if args.model != "nn" and args.model != "linear":
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
    if args.model != "nn" and args.model != "linear":
        output = blackbox_model.predict(preprocessor.transform(X_train))
    else:
        with torch.no_grad():
            output = blackbox_model(torch.FloatTensor(preprocessor.transform(X_train))).numpy()
    
    X_train["predicted"] = output
    X_train = X_train[X_train.predicted <= 0.5]
    quantile = X_train.predicted.quantile(args.quantile)
    X_train = X_train[(X_train.predicted <= quantile)]
    print("[*] Training with quantile: ", args.quantile)

    # Get hard examples
    X_hard = X_train[(X_train.predicted <= X_train.predicted.quantile(0.45))].copy()
    W_hard = W_train.iloc[X_hard.index].copy()
    X_hard.reset_index(inplace=True, drop=True)
    W_hard.reset_index(inplace=True, drop=True)
    print("[*] Training hard examples: ", len(X_hard))

    X_train.drop(columns="predicted", inplace=True)

    print("[*] Training examples: ", len(X_train))

    # Read the trained WFARE method from disk
    # The WFARE method contains the following:
    # - Blackbox classifier
    # - Custom preprocessor
    recourse_method: WFARE = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}.pth", "rb"))

    policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

    recourse_method = WFARE(recourse_method.model, policy_config, recourse_method.environment_config, recourse_method.mcts_config,
                  batch_size=WFARE_CONFIG.get(args.dataset).get("batch_size", 50),
                  training_buffer_size=WFARE_CONFIG.get(args.dataset).get("buffer_size", 200),
                  expectation=recourse_method.expectation if WFARE_CONFIG.get(args.dataset).get("expectation", False) else None)
    recourse_method.load(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}.pth")

    # Use fine-tuned version if it is requested by the experiment
    if WFARE_CONFIG.get(args.dataset).get("finetune", False):

        print("[*] Loading finetuned version.")

        fare_model = pickle.load(open(f"competitors/fare/fare_recourse_{args.model}_{args.dataset}.pth", "rb"))
        policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

        recourse_method = build_wfarefiner(
            recourse_method, fare_model, policy_config
        )

    # Train the W-EFARE model
    wefare_model = WEFARE(recourse_method)
    if os.path.isfile(f"blackbox/{args.dataset}/wefare_{args.model}_{args.dataset}.pth") and not args.retrain:
        wefare_model.load(f"blackbox/{args.dataset}/wefare_{args.model}_{args.dataset}.pth")
    else:
        wefare_model.fit(X_train[0:args.num_examples], W_train[0:args.num_examples], verbose=True)
        wefare_model.save(f"blackbox/{args.dataset}/wefare_{args.model}_{args.dataset}.pth")

    # For testing, we use the test data
    if args.model != "nn" and args.model != "linear":
        output = blackbox_model.predict(preprocessor.transform(X_train))
    else:
        with torch.no_grad():
            output = blackbox_model(torch.FloatTensor(preprocessor.transform(X_test))).round().numpy()

    X_test["predicted"] = output
    X_test = X_test[X_test.predicted == 0]
    X_test.drop(columns="predicted", inplace=True)

    # Reset the index of both test features and weights
    W_test = W_test.iloc[X_test.index]
    W_test.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)

    print("[*] Test W-EFARE with true weights.")
    df_cfs, Y_full, competitor_traces, costs_efare, _ = wefare_model.predict(X_test[0:100], W_test[0:100], full_output=True)
    print(sum(Y_full)/len(Y_full), round(np.mean(costs_efare), 2), round(np.mean([len(t) for t in competitor_traces]), 2))
    
    # We use the model to predict the test data
    print("[*] Test model with expected weights given the mixture.")
    _, Y_full, traces_full, c_full_e, _ = wefare_model.predict(X_test[0:100], W_train[0:100], full_output=True)
    print(sum(Y_full)/len(Y_full), round(np.mean(c_full_e), 2), round(np.mean([len(t) for t in traces_full]), 2))