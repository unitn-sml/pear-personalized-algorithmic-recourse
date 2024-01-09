from tqdm import tqdm

from recourse_fare.utils.functions import import_dyn_class

from competitors.cscf.problem_factory import ProblemFactory, pool
from competitors.cscf.CSCF import CSCF

from pymoo.util.display import MultiObjectiveDisplay

import pandas as pd

import numpy as np

def optimization_cfg(total_pop):
    # For the EA
    n_elites_frac = 0.2
    n_elites = 1  # doesn't make a difference since we use NDS and not crowding distance
    offsprings_frac = 0.8
    n_offspring = int(offsprings_frac * total_pop)
    mutants_frac = 0.2
    n_mutants = int(mutants_frac * total_pop)
    bias = 0.7
    eliminate_duplicates = True

    return n_elites, n_offspring, n_mutants, bias, eliminate_duplicates


def setup_optimizer(n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, seed):
    algorithm = CSCF(
        n_elites=n_elites,
        n_offsprings=n_offspring,
        n_mutants=n_mutants,
        bias=bias,
        eliminate_duplicates=eliminate_duplicates,
    )
    return algorithm


def lauch_CSCF(X, W, environment_config, blackbox_model, verbose=False, population=200, generations=50, seed=2023):

    X = X.to_dict(orient='records')
    W = W.to_dict(orient='records')

    counterfactuals=[]
    Y=[]
    traces=[]
    final_costs=[]

    for i in tqdm(range(len(X)),  desc="Eval CSCF", disable=not verbose):

        env = import_dyn_class(environment_config.get("class_name"))(
            X[i].copy(),
            W[i].copy(),
            blackbox_model,
            **environment_config.get("additional_parameters"))
        
        env.start_task()

        problem = ProblemFactory(env)

        n_elites, n_offspring, n_mutants, bias, eliminate_duplicates = optimization_cfg(total_pop=population)
        n_generations = generations

        algorithm = setup_optimizer(n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, 2023)

        termination = ("n_gen", n_generations)

        # perform a copy of the algorithm to ensure reproducibility
        import copy
        obj = copy.deepcopy(algorithm)

        # let the algorithm know what problem we are intending to solve and provide other attributes
        obj.setup(problem, termination=termination,
                  seed=seed,
                  display=MultiObjectiveDisplay(),
                  save_history=False,
                  verbose=False,
                  return_least_infeasible=True)

        # until the termination criterion has not been met
        evals = 0
        while obj.has_next():
            # perform an iteration of the algorithm
            obj.next()
            evals += obj.evaluator.n_eval

        res = obj.result()

        if res.X is not None:

            res_s = np.array([problem.decoder.decode(instance) for instance in res.X], dtype=np.int64)

            sequences = [
                problem.create_sequence(sol) for i, sol in enumerate(res_s)
            ]

            costs = []
            rewards = []
            for C, F, s in zip(res.G, res.F, sequences):
                rewards.append(C[0])
                costs.append(F[0])

            # Invert the cost and rewards to make it similar to the other experiments
            rewards = [1 if r == -1 else np.inf for r in rewards]

            # Multiply the reward with the cost
            costs = np.multiply(costs, rewards)

            # For each sequence, pick the least expensive one
            c_r_seq = list(zip(costs, rewards, sequences))
            c_r_seq.sort(key=lambda x: x[0])

            # Get the first sequence and save its costs
            cost_b, reward_b, sequence_b = c_r_seq[0]

            # Apply the sequence to the environment to get the counterfactual
            # And compute again the cost
            computed_cost = 0
            for (a,v) in sequence_b:
                pindex = env.programs_library[a].get("index")
                aindex = env.inverse_complete_arguments.get(v).get(a)
                assert aindex is not None
                computed_cost += env.get_cost(pindex, aindex)
                env.act(a,v)
            
            assert (computed_cost == cost_b) or (not np.isfinite(cost_b)), (computed_cost, cost_b)

            counterfactuals.append(env.features.copy())
            Y.append(1 if reward_b == 1 and np.isfinite(reward_b) else 0)
            traces.append(sequence_b)
            final_costs.append(computed_cost)

        else:
            assert False

        env.end_task()
    
    # Close multiprocessing pool
    pool.close()
    
    return pd.DataFrame(counterfactuals), Y, traces, final_costs
