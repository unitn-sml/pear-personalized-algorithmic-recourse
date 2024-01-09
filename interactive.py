from mpi4py import MPI
import dill as pickle
import random
import os

from blackbox.config import MIXTURE_MEAN_LIST, WFARE_CONFIG
from blackbox.adult.adult_scm import AdultSCM, DEFAULT_EDGES_ADULT
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM, DEFAULT_EDGES_GIVEMECREDIT
from blackbox.synthetic.synthetic_scm import SyntheticSCM, DEFAULT_EDGES_SYNTHETIC

from utils import filter_negative_classes, build_wfarefiner

from recourse_fare.models.PEAR import PEAR
from recourse_fare.models.XPEAR import XPEAR
from recourse_fare.models.WEFARE import WEFARE
from recourse_fare.user.user import NoiselessUser, LogisticNoiseUser
from recourse_fare.utils.Mixture import MixtureModel

import numpy as np
import pandas as pd
import torch

from argparse import ArgumentParser

if __name__ == "__main__":

    # Add the argument parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="nn", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--questions", default=3, type=int, help="How many questions we shoudl ask.")
    parser.add_argument("--choice-set-size", default=2, type=int, help="Size of the choice set.")
    parser.add_argument("--test-set-size", default=300, type=int, help="How many users we should pick from the test set for evaluation.")
    parser.add_argument("--mcmc-steps", default=50, type=int, help="How many steps should the MCMC procedure perform.")
    parser.add_argument("--logistic-user", default=False, action="store_true", help="Use a logistic user rather than a noiseless one.")
    parser.add_argument("--random-choice-set", default=False, action="store_true", help="Use a random choice set rather than EUS.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Percentage of arcs we want to remove from the graph during the estimation phase.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Make the procedure verbose.")
    parser.add_argument("--xpear", default=False, action="store_true", help="Use XPEAR for the interactive session.")
    parser.add_argument("--batching", default=1, type=int, help="Run MCMC only after performing N questions (default: 1).")
    parser.add_argument("--quantile", type=float, default=1.0, help="How many example we want to run.")
    parser.add_argument("--output", default="", type=str, help="Path where the results will be stored as csv.")
    parser.add_argument("--mcts-steps", default=5, type=int, help="Simulations done by MCTS.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed used for the evaluation.")
    parser.add_argument("--finetune", default=False, action="store_true", help="Perform preference elicitation with finetuning (works only for PEAR).")

    # Parse the arguments
    args = parser.parse_args()

    # Launch the script in a parallel fashion
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    bcast_data = None

    # Set seeds for reproducibility
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read the trained WFARE method from disk
    # The WFARE method contains the following:
    # - Blackbox classifier
    # - Custom preprocessor
    recourse_method = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))

    # If we use xpear then we load the corresponding method
    if args.xpear:
        recourse_method = WEFARE(recourse_method)
        recourse_method.load(f"blackbox/{args.dataset}/wefare_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth")

    # Use fine-tuned version if it is requested by the experiment
    if (WFARE_CONFIG.get(args.dataset).get("finetune", False) and not args.xpear) or args.finetune:

        print("[*] Loading finetuned version.")

        fare_model = pickle.load(open(f"competitors/fare/fare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))
        policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

        # We add the FARE model as the baseline for the optimization process.
        recourse_method = build_wfarefiner(
            recourse_method, fare_model, policy_config
        )

    # Create the user model required
    if args.logistic_user:
        user = LogisticNoiseUser()
    else:
        user = NoiselessUser()

    # Get edges and nodes. We get the FULL set of nodes and edges.
    # In case of the corrupted graph, this information is used only to make it
    # easier to sample, but then the weights of the missing edges will not be
    # used.
    if args.dataset == "adult":
        tmp_scm = AdultSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_ADULT
    elif args.dataset == "givemecredit":
        tmp_scm = GiveMeCreditSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_GIVEMECREDIT
    elif args.dataset == "synthetic":
        tmp_scm = SyntheticSCM(None)
        DEFAULT_EDGES = DEFAULT_EDGES_SYNTHETIC

    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

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
            )).numpy()
            X["predicted"] = output
    
    iterations = args.test_set_size
    X, W_test, mixture_classes, original_model_score = filter_negative_classes(X, W_test, mixture_classes, quantile=args.quantile, num_examples=iterations)

    # Remove a percentage of edges from the graph to simulate a corrupted baseline.
    # This procedure is done only by the master process and then sent to the others.
    # This way we ensure consistency between the results.
    edges_to_remove=None
    if rank == 0:
        full_edges = list(tmp_scm.scm.edges())
        if args.corrupt_graph > 0:
            n_edges_to_remove = int(args.corrupt_graph*len(full_edges))
            print(f"Removing {n_edges_to_remove}/{len(full_edges)} edges...")
            edges_to_remove = list(range(0, len(full_edges)))
            np.random.shuffle(edges_to_remove)
            edges_to_remove = edges_to_remove[:n_edges_to_remove]
            edges_to_remove = [ e for idx,e in zip(range(0, len(full_edges)), full_edges) if idx in edges_to_remove]
    # We broadcast the edges we want to remove.
    edges_to_remove = comm.bcast(edges_to_remove, root=0)

    # Build the mixture (prior for the estimation)
    mixture = MixtureModel(
        mixture_means=MIXTURE_MEAN_LIST.get(args.dataset)
    )

    # Create and interactive FARE object and predict the test instances
    if args.xpear:
        interactive_class = XPEAR
    else:
        interactive_class = PEAR

    interactive = interactive_class(recourse_method, user, mixture, keys_weights,
                        questions=int(args.questions), mcmc_steps=args.mcmc_steps,
                        verbose=args.verbose, choice_set_size=args.choice_set_size,
                        random_choice_set=args.random_choice_set)
    
    # Hack into the environment parameters if we want to remove certain edges
    # from the default configurations when building the graph used for the estimation.
    if args.corrupt_graph > 0:
        interactive.environment_config["additional_parameters"]["remove_edges"] = edges_to_remove

    # Given how many users we want to analyze, get a slice of
    # the data with the corresponding users
    perrank = iterations // size
    data_slice = (0 + rank * perrank,  0 + (rank + 1) * perrank)

    # Generate correct graphs for the users. In theory, this only makes sense
    # if we have a corrupt_graph value above 0
    G = [{"edges": DEFAULT_EDGES.copy()} for i in range(len(X))]

    # Current slice
    X_test_slice, W_test_slice = X[data_slice[0]:data_slice[1]], W_test[data_slice[0]:data_slice[1]]
    G_test_slice = G[data_slice[0]:data_slice[1]]

    # Seed again the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate the counterfactuals and traces
    (counterfactuals, Y, traces, costs_e, _), W_updated, failed_users = interactive.predict(
        X_test_slice, W_test_slice, G_test_slice,
        full_output=True, batching=args.batching,
        mcts_steps=args.mcts_steps)

    # Regenerate the true costs, given the found traces
    # We drop the recourse information.
    costs, _ = interactive.recourse_model.evaluate_trace_costs(traces, X_test_slice, W_test_slice, G_test_slice)

    # Send the complete results
    complete_trace = [counterfactuals, Y, traces, costs, W_updated, failed_users, data_slice[0], data_slice[1]]
    complete_trace = comm.gather(complete_trace, root=0)

    # If we are the master process, then we print all
    if rank == 0:
        
        # Sort the traces based on their interval
        complete_trace = sorted(complete_trace, key=lambda x: x[6])

        user_idx = []
        intervention_costs = []
        failed_users_all = []
        validity = []
        all_traces = []
        length = []

        # Unwind the results and store the traces
        for counterfactuals, Y, traces, costs, W_updated, failed_users, start_slice, end_slice in complete_trace:
            
            user_range = list(range(start_slice, end_slice))

            user_idx += user_range
            validity += Y
            intervention_costs += costs
            failed_users_all += failed_users
            length += [len(t) for t in traces]
            
            all_traces += [[idx, p,a] for t, idx in zip(traces, user_range) for p,a in t]

        # Format the choice set size if we use random
        choice_set_size = args.choice_set_size if not args.random_choice_set else "R"

        # Save the validity, cost and elicitation result to disk
        data = pd.DataFrame(list(zip(user_idx, validity, intervention_costs, failed_users_all)), columns=["user_idx","recourse", "cost", "elicitation"])
        data["length"] = length
        data["mixture_class"] = mixture_classes
        data["model_score"] = original_model_score
        data["seed"] = np.ones(len(mixture_classes))*args.seed
        data.to_csv(
            os.path.join(
                args.output,
                f"validity_cost_elicitation-{args.dataset}-{args.questions}-{args.corrupt_graph}-{args.logistic_user}-{choice_set_size}-{args.test_set_size}-{args.quantile}-{args.seed}.csv"
            ), 
            index=None)

        # Save the traces to disk
        data = pd.DataFrame(all_traces, columns=["user_idx", "action", "argument"])
        data.to_csv(
            os.path.join(
                args.output,
                f"traces-{args.dataset}-{args.questions}-{args.corrupt_graph}-{args.logistic_user}-{choice_set_size}-{args.test_set_size}-{args.quantile}-{args.seed}.csv"
            ), 
            index=None)
        # Save estimated weights to disk
        weights = pd.concat([x[4] for x in complete_trace])
        weights.to_csv(
            os.path.join(
                args.output,
                f"estimated_weights-{args.dataset}-{args.questions}-{args.corrupt_graph}-{args.logistic_user}-{choice_set_size}-{args.test_set_size}-{args.quantile}-{args.seed}.csv"
            ), 
            index=None)