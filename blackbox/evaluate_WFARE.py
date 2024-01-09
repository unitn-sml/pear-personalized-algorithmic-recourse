import dill as pickle
import random

import numpy as np
import pandas as pd
import torch

from blackbox.adult.adult_scm import AdultSCM, DEFAULT_EDGES_ADULT
from blackbox.adult.adult_scm import AdultSCM, DEFAULT_EDGES_ADULT
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM, DEFAULT_EDGES_GIVEMECREDIT
from blackbox.synthetic.synthetic_scm import SyntheticSCM, DEFAULT_EDGES_SYNTHETIC

from recourse_fare.models.WFARE import WFARE
from recourse_fare.utils.functions import get_single_action_costs

import matplotlib.pyplot as plt
import seaborn as sns

from recourse_fare.utils.Mixture import MixtureModel

from blackbox.config import MIXTURE_MEAN_LIST, WFARE_CONFIG
from utils import filter_negative_classes, build_wfarefiner

from argparse import ArgumentParser

if __name__ == "__main__":

    # Initialize the parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="nn", help="Model type we want to train (svc, tree, nn, linear)")
    parser.add_argument("--iterations", type=int, default=250, help="How many example we want to run.")
    parser.add_argument("--quantile", type=float, default=1.0, help="How many example we want to run.")
    parser.add_argument("--random", default=False, action="store_true", help="Model type we want to train (svc, tree, nn, linear)")
    parser.add_argument("--seed", default=2023, type=int, help="Seed used for the evaluation.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Percentage of arcs we want to remove from the graph during the estimation phase.")

    args = parser.parse_args()

    # Set seeds for reproducibility
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build the mixture (prior for the estimation)
    mixture = MixtureModel(
        mixture_means=MIXTURE_MEAN_LIST.get(args.dataset)
    )

    # Generate correct graphs for the users.
    if args.dataset == "adult":
        tmp_scm = AdultSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_ADULT
    elif args.dataset == "givemecredit":
        tmp_scm = GiveMeCreditSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_GIVEMECREDIT
    elif args.dataset == "synthetic":
        tmp_scm = SyntheticSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_SYNTHETIC

    # Read the trained WFARE method from disk
    # The WFARE method contains the following:
    # - Blackbox classifier
    # - Custom preprocessor
    recourse_method: WFARE = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "rb"))

    policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

    print("[*] WFARE Agent Architecture")

    recourse_method = WFARE(recourse_method.model, policy_config, recourse_method.environment_config, recourse_method.mcts_config,
                  batch_size=WFARE_CONFIG.get(args.dataset).get("batch_size", 50),
                  training_buffer_size=WFARE_CONFIG.get(args.dataset).get("buffer_size", 200),
                  expectation=recourse_method.expectation if WFARE_CONFIG.get(args.dataset).get("expectation", False) else None)
    recourse_method.load(f"blackbox/{args.dataset}/wfare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")

    # Use fine-tuned version if it is requested by the experiment
    if WFARE_CONFIG.get(args.dataset).get("finetune", False):

        print("[*] Loading finetuned version.")

        fare_model = pickle.load(open(f"competitors/fare/fare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "rb"))
        policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

        recourse_method = build_wfarefiner(
            recourse_method, fare_model, policy_config
        )

    # Build the dataframes with the weights
    W_test = pd.read_csv(f"blackbox/{args.dataset}/weights_test_{args.dataset}.csv")

    # Extract the mixture population
    mixture_classes = W_test["mixture_class"]
    W_test.drop(columns=["mixture_class"], inplace=True)

    W_test.rename(
        columns={c: eval(c) for c in W_test.columns},
        inplace=True
    )

    # Read data
    X = pd.read_csv(f"blackbox/{args.dataset}/test_data_{args.dataset}.csv")
    
    if args.model != "nn":
        # Keep only the instances which are negatively classified
        X["predicted"] = recourse_method.model.predict_proba(
            recourse_method.environment_config.get("additional_parameters").get("preprocessor").transform(X)
        )[:, 1]
    else:
        with torch.no_grad():
            recourse_method.model.eval()
            output = recourse_method.model(torch.FloatTensor(
                recourse_method.environment_config.get("additional_parameters").get("preprocessor").transform(X)
            ))

            X["predicted"] = output.numpy()

    X, W_test, mixture_classes, original_model_score = filter_negative_classes(X, W_test, mixture_classes, quantile=args.quantile, num_examples=args.iterations)

    # Build the graph using the default edges (in case the model was trained with corrupted graphs)
    G = [{"edges": DEFAULT_EDGES.copy()} for i in range(len(X))]

    # Generate expected costs given the mixture
    W_test_rnd = np.mean(mixture.sample(1000), axis=0)
    W_test_rnd = [W_test_rnd for _ in range(len(X))]
    W_test_rnd = pd.DataFrame(W_test_rnd, columns=W_test.columns)

    # Generate the counterfactuals and traces given the true weights
    _, Y, traces, costs, root_nodes = recourse_method.predict(
        X, W_test, None, full_output=True, verbose=True)
    
    # We regenerate the costs given the fact that the graph might be different
    costs, Y = recourse_method.evaluate_trace_costs(traces, X, W_test, G=G)
    
    # Save the validity, cost and elicitation result to disk
    data = pd.DataFrame(list(zip(range(0, len(X)), Y, costs, np.ones(len(X)))), columns=["user_idx","recourse", "cost", "elicitation"])
    data["length"] = [len(t) for t in traces]
    data["mixture_class"] = mixture_classes
    data["model_score"] = original_model_score
    data.to_csv(f"validity_cost_elicitation-T-{args.dataset}-{args.quantile}.csv", index=None)
    
    # Generate the counterfactuals and traces given the expected weights
    _, Y_e, traces_e, costs_e, root_nodes_e = recourse_method.predict(
        X, W_test_rnd, None, full_output=True, verbose=True)

    # If we use random weights, then we regenerate from the distribution
    costs_e, Y_e = recourse_method.evaluate_trace_costs(traces_e, X, W_test, G=G)

    data_e = pd.DataFrame(list(zip(range(0, len(X)), Y_e, costs_e, np.ones(len(X)))), columns=["user_idx","recourse", "cost", "elicitation"])
    data["length"] = [len(t) for t in traces]
    data_e["mixture_class"] = mixture_classes
    data["model_score"] = original_model_score

    print("[*] Check the improvement between the expected value and true value.")

    print(f"AVG. COST: ", data[data.recourse == 1].cost.mean())
    print(f"AVG. COST (E): ", data_e[data_e.recourse == 1].cost.mean())
    print(f"LEN TRACES: ", np.mean([len(t) for t in traces]))
    print(f"LEN TRACES (E): ", np.mean([len(t) for t in traces_e]))
    
    recourse_vars = data[(data.recourse == 1) & (data_e.recourse == 1)]
    recourse_vars_e = data_e[(data.recourse == 1) & (data_e.recourse == 1)]
    print("Potential improvements: ", np.mean((recourse_vars_e.cost - recourse_vars.cost)>0))
    print("Potential errors: ", np.mean((recourse_vars_e.cost - recourse_vars.cost)<0))

    print(np.sum(Y)/len(Y), np.sum(Y_e)/len(Y_e))
    for c in sorted(mixture_classes.unique()):

        cost_users_e = data_e[data_e.mixture_class == c]
        cost_users = data[data.mixture_class == c]

        cost_c = cost_users[(cost_users.recourse == 1) & (cost_users_e.recourse == 1)].cost
        cost_c_e = cost_users_e[(cost_users.recourse == 1) & (cost_users_e.recourse == 1)].cost

        print(f"[{c}][{len(cost_c)}] AVG. COST: ",
              round(cost_c.mean(),2),
              round(cost_c_e.mean(), 2))

    # Get the costs for each node and plot
    # a KDE visualization of the costs. 
    node_costs = []
    for r in root_nodes:
        c = get_single_action_costs(r)
        for cost_single in c:
            node_costs.append([cost_single, "true"])
    for r in root_nodes_e:
        c = get_single_action_costs(r)
        for cost_single in c:
            node_costs.append([cost_single, "expectation"])

    node_costs = pd.DataFrame(node_costs, columns=["cost", "preferences"])

    sns.boxplot(data=data[data.recourse==1], x="length", y="model_score")
    plt.show()
