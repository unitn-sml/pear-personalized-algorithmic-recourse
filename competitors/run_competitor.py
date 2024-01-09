import warnings
warnings.filterwarnings("ignore")

from competitors.cscf.model import lauch_CSCF

from blackbox.adult.adult_scm import AdultSCM, DEFAULT_EDGES_ADULT
from blackbox.adult.adult_scm import AdultSCM, DEFAULT_EDGES_ADULT
from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM, DEFAULT_EDGES_GIVEMECREDIT
from blackbox.synthetic.synthetic_scm import SyntheticSCM, DEFAULT_EDGES_SYNTHETIC

from blackbox.config import MIXTURE_MEAN_LIST, DATA_CONFIG

from competitors.fare.config import FARE_CONFIG
from blackbox.config import WFARE_CONFIG
from utils import filter_negative_classes, build_wfarefiner

from recourse_fare.models.FARE import FARE
from recourse_fare.models.EFARE import EFARE
from recourse_fare.models.WFARE import WFARE
from recourse_fare.models.WEFARE import WEFARE
from recourse_fare.models.WFAREFiner import WFAREFiner
from recourse_fare.user.user import NoiselessUser
from recourse_fare.utils.Mixture import MixtureModel

from recourse_fare.utils.functions import import_dyn_class

from recourse_fare.environment_w import EnvironmentWeights

import torch

from itertools import permutations

import pandas as pd
import numpy as np

import random
from argparse import ArgumentParser

import dill as pickle

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from tqdm import tqdm

import os

def find_closest(env, program_name, arguments):

    args_type = env.programs_library.get(program_name).get("args")
    potential_args = env.arguments.get(args_type).copy()
    potential_args = sorted(potential_args)

    for idx in range(len(potential_args)):
        if potential_args[idx] == arguments:
            return potential_args[idx]
        elif arguments < potential_args[idx] and not isinstance(potential_args[idx], str):
            return potential_args[idx]
    
    return potential_args[-1]

def intervention_from_feature_changes(recourse_model, feature_changes: list, verbose=False) -> list:
    intervention = []
    env: EnvironmentWeights = import_dyn_class(recourse_model.environment_config.get("class_name"))(
        None,
        None,
        recourse_model,
        **recourse_model.environment_config.get("additional_parameters"))
    
    condition = True
    for a, v in feature_changes:
        found=False
        for k, feature in env.program_feature_mapping.items():
            if a == feature:

                program_args_type = env.programs_library.get(k).get("args")
                current_vals = env.arguments.get(program_args_type)
                max_val = max(current_vals)
                min_val = min(current_vals)

                if isinstance(v, str):
                    condition &= v in current_vals
                else:
                    condition &= (v <= max_val and v >= min_val)

                intervention.append(
                    (k,v)
                )
                found=True
                break
        if not found and verbose:
            print(f"Unable to convert {a}")

    # We add the stop action, since CF methods do not do it.
    intervention.append(("STOP", 0))
    
    return intervention, condition

def find_differences(original:dict, cf: dict, exclude=[]):
    differences = []
    for c in original:
        if c not in exclude:
            if original[c].values != cf[c].values:
                if is_numeric_dtype(original[c]):
                    differences.append((c, (cf[c]-original[c]).values[0]))
                elif is_string_dtype(original[c]):
                    differences.append((c, cf[c].values[0]))
                else:
                    print(f"Skipping {c}. It is not string nor numeric.")
    return differences

def compute_cost(x, y, w, g, preprocessor, return_interventions=False, custom_intervention=None):
    costs = []
    interventions = []
    valid_interventions = []

    x = preprocessor.inverse_transform(x, type="df")
    y = preprocessor.inverse_transform(y, type="df")

    prototypes = find_differences(
        x, y
    )

    intervention, is_a_valid_intervention = intervention_from_feature_changes(recourse_method, prototypes, preprocessor) if not custom_intervention else custom_intervention

    cost, _ = recourse_method.evaluate_trace_costs([intervention], x, w, g, bin_argument=False)
    costs.append(cost[0])
    interventions.append(intervention)
    valid_interventions.append(is_a_valid_intervention)

    if return_interventions:
        return np.array(costs), interventions, valid_interventions
    else:
        return np.array(costs)


AVAILABLE_COMPETITORS = [
    "cscf",
    "cscf_uniform",
    "face_l2",
    "face_cost",
    "face_cost_uniform",
    "watcher_l2",
    "watcher_cost",
    "fare",
    "rl",
    "mcts",
    "efare",
    "fare_uniform",
    "rl_uniform",
    "mcts_uniform",
    "wfare",
    "wfare_finetuned",
    "xpear_gt"
]


if __name__ == "__main__":

    # Add the argument parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="nn", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--method", type=str, default="face", help="Competitor we want to use.")
    parser.add_argument("--test-set-size", default=100, type=int, help="How many users we should pick from the test set for evaluation.")
    parser.add_argument("--quantile", type=float, default=1.0, help="How many example we want to run.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Make the procedure verbose.")
    parser.add_argument("--output", default="", type=str, help="Path where the results will be stored as csv.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed used for the evaluation.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Percentage of arcs we want to remove from the graph during the estimation phase.")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the competitor is available
    if args.method not in AVAILABLE_COMPETITORS:
        print(f"The selected method {args.method} does not exists! Use of of {AVAILABLE_COMPETITORS}")
        exit()

    # Set seeds for reproducibility
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    recourse_method = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_0.0_2023.pth", "rb"))
    blackbox_model = recourse_method.model

    # Create the user model required
    user = NoiselessUser()

    # Get edges and nodes. We get the FULL set of nodes and edges.
    # In case of the corrupted graph, this information is used only to make it
    # easier to sample, but then the weights of the missing edges will not be
    # used.
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    # Remove edges if we are dealing with a corrupted causal graph.
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
    full_edges = list(tmp_scm.scm.edges())

    # Build the mixture (prior for the estimation)
    mixture = MixtureModel(
        mixture_means=MIXTURE_MEAN_LIST.get(args.dataset)
    )

    # Build the dataframes with the weights
    W_test = pd.read_csv(f"blackbox/{args.dataset}/weights_test_{args.dataset}.csv")

    # Get the mixture classes out of the way
    mixture_classes = W_test["mixture_class"]
    W_test.drop(columns="mixture_class", inplace=True)

    W_test.rename(
        columns={c: eval(c) for c in W_test.columns},
        inplace=True
    )

    # Get the mixture and generated the expected weights.
    mixture = MixtureModel(mixture_means=MIXTURE_MEAN_LIST.get(args.dataset))

    train_data_path = f"blackbox/{args.dataset}/train_data_{args.dataset}.csv"
    test_data_path = f"blackbox/{args.dataset}/test_data_{args.dataset}.csv"
    immutables = DATA_CONFIG.get(args.dataset).get("immutables")

    preprocessor_blackbox = recourse_method.environment_config.get("additional_parameters").get("preprocessor")

    # Read the test data and the test weights
    X_test = pd.read_csv(test_data_path)

    # Filter the training dataset by picking only the examples which are classified negatively by the model
    if args.model != "nn":
        output = blackbox_model.predict_proba(preprocessor_blackbox.transform(X_test))[:, 1]
    else:
        with torch.no_grad():
            blackbox_model.eval()
            output = blackbox_model(torch.FloatTensor(preprocessor_blackbox.transform(X_test))).numpy()
    
    X_test["predicted"] = output
    X_test, W_test, mixture_classes, original_model_score = filter_negative_classes(X_test, W_test, mixture_classes,
                                                                                    quantile=args.quantile,
                                                                                    num_examples=args.test_set_size)

    single_weights = np.mean(mixture.sample(1000), axis=0)
    W_expected = [single_weights for _ in range(len(W_test))]
    W_uniform = [np.ones(len(single_weights)) for _ in range(len(W_test))]
    W_expected = pd.DataFrame(W_expected, columns=W_test.columns)
    W_uniform = pd.DataFrame(W_uniform, columns=W_test.columns)

    # Build the graph
    G = [{"edges": DEFAULT_EDGES.copy()} for i in range(len(X_test))]

    # Build the corrupted graph
    G_corrupted = None
    if edges_to_remove:
        G_corrupted = [{"edges": [e for e in DEFAULT_EDGES.copy() if e not in edges_to_remove]} for i in range(len(X_test))]
        print("[*] Using a corrupted graph.")

    competitor_traces = None
    valid_intervention = np.ones(len(X_test)).tolist()

    # Set again the seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run the recourse method
    if args.method in ["fare", "rl", "mcts", "fare_uniform", "rl_uniform", "mcts_uniform"]:

        fare_recourse_name = "fare_uniform" if "uniform" in args.method else "fare"
        model: FARE = pickle.load(open(f"competitors/fare/{fare_recourse_name}_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "rb"))

        df_cfs, Y, competitor_traces, costs_cscf, _ = model.predict(X_test,
                                           full_output=True,
                                           agent_only=args.method in ["rl", "rl_uniform"],
                                           mcts_only=args.method in ["mcts", "mcts_uniform"])

        # Compute the correct costs again. Here, we recompute the cost by
        # assuming to have the CORRECT causal graph.
        costs_correct, Y = recourse_method.evaluate_trace_costs(
            competitor_traces,
            X_test,
            W_test, G)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]
    
    elif args.method in ["efare"]:

        preprocessor_efare = pickle.load(open(f"competitors/fare/efare_preprocessor_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "rb"))

        fare_recourse_name = "fare_uniform" if "uniform" in args.method else "fare"
        fare_model: FARE = pickle.load(open(f"competitors/fare/{fare_recourse_name}_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth", "rb"))
        
        model = EFARE(fare_model, preprocessor_efare)
        model.load(f"competitors/fare/e{fare_recourse_name}_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")

        df_cfs, Y, competitor_traces, costs_efare, _ = model.predict(X_test, full_output=True)

        # Compute the correct costs again. Here, we recompute the cost by
        # assuming to have the CORRECT causal graph.
        costs_correct, Y = recourse_method.evaluate_trace_costs(
            competitor_traces,
            X_test,
            W_test, G)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]

    elif args.method in ["xpear_gt"]:

        xpear_model: WFARE = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))
        
        model = WEFARE(xpear_model)
        model.load(f"blackbox/{args.dataset}/wefare_{args.model}_{args.dataset}_{args.corrupt_graph}_{args.seed}.pth")

        df_cfs, Y, competitor_traces, costs_correct, _ = model.predict(X_test, W_test, full_output=True)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]
    
    elif args.method in ["wfare"]:

        model: WFARE = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))

        df_cfs, Y, competitor_traces, costs_correct, _ = model.predict(X_test,
                                                                    W = W_test,
                                                                    full_output=True)
    
        # Compute the correct costs again. Here, we recompute the cost by
        # assuming to have the CORRECT causal graph.
        costs_correct, Y = recourse_method.evaluate_trace_costs(
            competitor_traces,
            X_test,
            W_test, G)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]

    elif args.method in ["wfare_finetuned"]:

        wfare_model: WFARE = pickle.load(open(f"blackbox/{args.dataset}/wfare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))

        fare_model: FARE = pickle.load(open(f"competitors/fare/fare_recourse_{args.model}_{args.dataset}_{args.corrupt_graph}_2023.pth", "rb"))
        policy_config = WFARE_CONFIG.get(args.dataset).get("policy_config")

        assert wfare_model.environment_config["additional_parameters"].get("remove_edges", None) == fare_model.environment_config["additional_parameters"].get("remove_edges", None)

        model = build_wfarefiner(
            wfare_model, fare_model, policy_config
        )

        df_cfs, Y, competitor_traces, costs_correct, _ = model.predict(X_test,
                                                                    W = W_expected,
                                                                    full_output=True)

        # Compute the correct costs again. Here, we recompute the cost by
        # assuming to have the CORRECT causal graph.
        costs_correct, Y = recourse_method.evaluate_trace_costs(
            competitor_traces,
            X_test,
            W_test, G)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]

    elif args.method in ["cscf", "cscf_uniform"]:

        # Hack into the environment parameters if we want to remove certain edges
        # from the default configurations when building the graph used for the estimation.
        env_config_cscf = recourse_method.environment_config.copy()
        if args.corrupt_graph > 0:
            env_config_cscf["additional_parameters"]["remove_edges"] = edges_to_remove       

        df_cfs, Y, competitor_traces, costs_cscf = lauch_CSCF(
            X_test,
            W_expected if args.method == "cscf" else W_uniform,
            env_config_cscf,
            recourse_method.model,
            verbose=True,
            population = 50,
            generations = 25,
            seed = args.seed
        )

        # Compute the correct costs again. Here, we recompute the cost by
        # assuming to have the CORRECT causal graph.
        costs_correct, Y = recourse_method.evaluate_trace_costs(
            competitor_traces,
            X_test,
            W_test, G)
        
        all_traces = [[idx, p,a] for idx, t in enumerate(competitor_traces) for p,a in t]

    else:

        from competitors.data_wrapper import DataWrapper
        from competitors.blackbox_wrapper import BlackBoxWrapper
        from competitors.watcher.watcher import Wachter
        from competitors.face.face import Face

        # Build the data wrapper
        data = DataWrapper(
            train_data_path,
            test_data_path,
            preprocessor_blackbox,
            immutables = immutables
        )

        # Build the black-box wrapper
        ml_model = BlackBoxWrapper(
            recourse_method.model,
            preprocessor_blackbox,
            data
        )


        if args.method in ["face_l2", "face_cost", "face_cost_uniform"]:

            hyperparams = { "mode": "knn",
                            "n_neighbours": 50,
                            "fraction": 0.1,
                            "epsilon_constraints": 1.0
                        }
            if args.method in ["face_cost", "face_cost_uniform"]:
                 hyperparams["weight_function"] = compute_cost

            competitor = Face(ml_model, hyperparams)
            df_cfs, Y = competitor.get_counterfactuals(
                preprocessor_blackbox.transform(X_test, type="df"),
                W_expected if args.method == "face_cost" else W_uniform,
                G=G_corrupted
            )

        elif args.method in ["watcher_l2", "watcher_cost_uniform", "watcher_cost"]:

            hyperparams = {
                "lr": 1.00,
                "n_iter": 100,
                "t_max_min": 0.06
            }
            if args.method == "watcher_cost":
                hyperparams["feature_cost"] = compute_cost

            competitor = Wachter(ml_model, hyperparams)
            df_cfs, Y = competitor.get_counterfactuals(
                preprocessor_blackbox.transform(X_test, type="df"),
                W_expected if args.method == "watcher_cost" else W_uniform,
            )

        X_test = preprocessor_blackbox.transform(X_test, type="df")

        costs_correct = []
        competitor_traces = []
        valid_intervention = []
        for idx in range(len(df_cfs)):
            c, inter, is_it_valid = compute_cost(
                X_test.iloc[[idx]],
                df_cfs.iloc[[idx]],
                W_test.iloc[[idx]],
                [G[idx]],
                preprocessor_blackbox,
                return_interventions=True,
                custom_intervention=None
            )
            costs_correct.append(c[0])
            valid_intervention.append(is_it_valid[0])
            competitor_traces += inter
    
        all_traces = []
        for k, t in enumerate(competitor_traces):
            for p,a in t:
                all_traces.append([k,p,a])
    
    # Save the validity and cost result to disk
    data = pd.DataFrame(list(zip(range(len(Y)), Y, costs_correct, valid_intervention)), columns=["user_idx", "recourse", "cost", "is_it_valid"])
    data["length"] = [len(t) for t in competitor_traces]
    data["mixture_class"] = mixture_classes
    data["model_score"] = original_model_score
    data["seed"] = np.ones(len(mixture_classes))*args.seed
    data.to_csv(
        os.path.join(
            args.output,
            f"validity_cost-{args.dataset}-{args.method}-{args.quantile}-{args.test_set_size}-{args.corrupt_graph}-{args.seed}.csv"
        ), 
        index=None)

    # Save the traces to disk
    df_traces = pd.DataFrame(all_traces, columns=["user_idx", "action", "argument"])
    df_traces.to_csv(
        os.path.join(
            args.output,
            f"traces-{args.dataset}-{args.method}-{args.quantile}-{args.test_set_size}-{args.corrupt_graph}-{args.seed}.csv"
        ), 
        index=None)


    # Print average cost and average validity
    print("AVG. COST: ", round(data["cost"].mean(),3), round(data["cost"].std(),3))
    print("AVG. COST REC: ", round(data[data.recourse == 1]["cost"].mean(), 3), round(data[data.recourse == 1]["cost"].std(), 3))
    print("VALIDITY: ", data["recourse"].mean())