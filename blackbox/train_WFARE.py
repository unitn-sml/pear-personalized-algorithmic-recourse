"""Train the FARE model by using the given python class instead of the script."""

from recourse_fare.models.WFARE import WFARE
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor
from recourse_fare.utils.Mixture import MixtureModel

from blackbox.adult.adult_scm import AdultSCM
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM
from blackbox.synthetic.synthetic_scm import SyntheticSCM
from blackbox.utils_blackbox.net import Net, NetLinear

import torch

import pandas as pd
import numpy as np

import os
import random

import dill as pickle

from argparse import ArgumentParser

from blackbox.config import MIXTURE_MEAN_LIST, WFARE_CONFIG, WFARE_CONFIG_MCTS, NN_CONFIG

if __name__ == "__main__":

    # Initialize the parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="svc", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--retrain", default=False, action="store_true", help="Retrain the model if present.")
    parser.add_argument("--clean", default=False, action="store_true", help="Retrain the model starting from scratch.")
    parser.add_argument("--quantile", default=1.0, type=float, help="Quantile we want to use.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Train WFARE with a corrupted graph.")
    parser.add_argument("--agnostic", default=False, action="store_true", help="Train the model without exposing the weights in the environment.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed used when training.")

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

    print("Current Mixture dimension: ", len(keys_weights))

    # Simulate training this model with a misspecified causal graph.
    edges_to_remove=None
    full_edges = list(tmp_scm.scm.edges())
    if args.corrupt_graph > 0:
        n_edges_to_remove = int(args.corrupt_graph*len(full_edges))
        print(f"Removing {n_edges_to_remove}/{len(full_edges)} edges...")
        assert n_edges_to_remove > 0
        edges_to_remove = list(range(0, len(full_edges)))
        np.random.shuffle(edges_to_remove)
        edges_to_remove = edges_to_remove[:n_edges_to_remove]
        edges_to_remove = [ e for idx,e in zip(range(0, len(full_edges)), full_edges) if idx in edges_to_remove]

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
    if args.model != "nn":
        with open(blackbox_model_path, "rb") as model:
            blackbox_model = pickle.load(model)
    else:
        blackbox_model = Net(len(preprocessor.transform(X_train)[0]),
                                layers=NN_CONFIG.get(args.dataset).get("nn").get("layers"))
        blackbox_model.load_state_dict(
                torch.load(blackbox_model_path)
            )
        blackbox_model.eval()

    # Filter the training dataset by picking only the examples which are classified negatively by the model
    if args.model != "nn":
        output = blackbox_model.predict_proba(preprocessor.transform(X_train))[:,1]
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

    policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

    environment_config = {
        "class_name": WFARE_CONFIG.get(args.dataset).get("environment"),
        "additional_parameters": {
            "preprocessor": preprocessor,
            "model_type": args.model,
            "agnostic":  WFARE_CONFIG.get(args.dataset).get("agnostic", False),
            "remove_edges": edges_to_remove
        }
    }
    
    mcts_config = WFARE_CONFIG.get(args.dataset).get("mcts_config")

    # Minimize the action cost coeff if we are training a corrupt graph.
    #if args.corrupt_graph > 0.0:
    #    mcts_config["action_cost_coeff"] = WFARE_CONFIG_MCTS.get(args.dataset).get(args.corrupt_graph, 1.0)
    #    mcts_config["number_of_simulations"] = 35
    #    print("Setting action MCTS expansion penalty to ", mcts_config["action_cost_coeff"])

    print("[*] WFARE Agent Architecture")

    model = WFARE(blackbox_model, policy_config, environment_config, mcts_config,
                  batch_size=WFARE_CONFIG.get(args.dataset).get("batch_size", 50),
                  training_buffer_size=WFARE_CONFIG.get(args.dataset).get("buffer_size", 200),
                  expectation=W_expectation if WFARE_CONFIG.get(args.dataset).get("expectation", False) else None,
                  sample_from_hard_examples=WFARE_CONFIG.get(args.dataset).get("sample_hard", 0.0))
    print(model.policy)

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Train a FARE model given the previous configurations
    if args.retrain:
        
        if not args.clean and os.path.isfile(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth"):
            print("[*] Loading checkpoint from file")
            model.load(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")
        
        model.fit(X_train, W_train, X_hard=X_hard, W_hard=W_hard,
                  max_iter=WFARE_CONFIG.get(args.dataset).get("training_steps"), 
                  tensorboard="./wfare")
        
        # We save the trained WFARE model to disc
        model.save(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")
        pickle.dump(model, open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "wb"))
    else:
        if os.path.isfile(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth"):
            model.load(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")
        else:
            print(f"No model available (blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth)!")
            exit()

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
    
    # We use the model to predict the test data
    print("[*] Test model with expected weights given the mixture.")
    _, Y_full, traces_full, c_full_e, _ = model.predict(X_test[0:100], W_train[0:100], full_output=True)
    print(sum(Y_full)/len(Y_full), round(np.mean(c_full_e), 2), round(np.mean([len(t) for t in traces_full]), 2))

    print("[*] Test model with true weights.")
    _, Y_full, traces_full, c_full, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True)
    print(sum(Y_full)/len(Y_full), round(np.mean(c_full), 2), round(np.mean([len(t) for t in traces_full]), 2))
