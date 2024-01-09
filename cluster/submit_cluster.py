import argparse
import subprocess
import os

def check_if_run_result_exists(args, questions):
    path = os.path.join(
                args.output,
                f"validity_cost_elicitation-{args.dataset}-{questions}-{args.corrupt_graph}-{args.logistic_user}-{args.choice_set_size}-{args.test_set_size}-{args.quantile}.csv"
            )
    return os.path.isfile(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="nn", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--queue", type=str, default="short_cpuQ", help="Queue where to launch the experiments.")
    parser.add_argument("--min-questions", type=int, default=0, help="Minimum number of questions we can ask.")
    parser.add_argument("--questions", default=10, type=int, help="How many questions we should ask.")
    parser.add_argument("--questions-step", default=1, type=int, help="Questions step.")
    parser.add_argument("--test-set-size", default=100, type=int, help="How many users we should pick from the test set for evaluation.")
    parser.add_argument("--choice-set-size", default=2, type=int, help="Size of the choice set.")
    parser.add_argument("--logistic-user", default=False, action="store_true", help="Use a logistic user rather than a noiseless one.")
    parser.add_argument("--random-choice-set", default=False, action="store_true", help="Use a random choice set rather than EUS.")
    parser.add_argument("--corrupt-graph", default=0.0, type=float, help="Specify how much to corrupt a graph")
    parser.add_argument("--quantile", default=1.0, type=float, help="Data quantile to be used (use more difficult users.).")
    parser.add_argument("--test", default=False, action="store_true", help="Test the script locally.")
    parser.add_argument("--xpear", default=False, action="store_true", help="Use XPEAR.")
    parser.add_argument("--output", default="", type=str, help="Path where to save the file.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed which need to be used for evaluation.")

    args = parser.parse_args()

    for c in range(args.min_questions, args.questions+1, args.questions_step):

        if check_if_run_result_exists(args, c):
            print("Skipping since it exists ", f"{args.dataset}-{c}-{args.corrupt_graph}-{args.logistic_user}-{args.choice_set_size}-{args.test_set_size}-{args.quantile}")
            continue

        experiment_name=f"{args.dataset[0]}_{args.quantile}_{args.corrupt_graph}_{args.model}_{c}_{args.choice_set_size}_{str(args.logistic_user)[0]}"
        
        if args.test:
            command = "./launch_adult.sh"
            my_env = os.environ.copy()
            my_env["Q"] = str(c)
            my_env["L"] = str(args.logistic_user)
            my_env["C"] = str(args.choice_set_size)
            my_env["D"] = str(args.dataset)
            my_env["M"] = str(args.model)
            my_env["RC"] = str(args.random_choice_set)
        else:
            command = f'qsub -V -N {experiment_name} -q {args.queue} -v XPEAR="{args.xpear}",SEED="{args.seed}",Q="{c}",QA="{args.quantile}",L="{args.logistic_user}",C="{args.choice_set_size}",D="{args.dataset}",M="{args.model}",S="{args.test_set_size}",CG="{args.corrupt_graph}",O="{args.output}",RC="{args.random_choice_set}" launch.sh'

        result = subprocess.run(
            command.split(" "), capture_output=True
        )
        print(result.stdout.decode())
        print(result.stderr.decode())