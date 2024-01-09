import argparse
import subprocess
import os

def check_if_run_result_exists(args):
    path = os.path.join(
                args.output,
                f"validity_cost-{args.dataset}-{args.method}-{args.quantile}-{args.test_set_size}-{args.seed}.csv"
            )
    return os.path.isfile(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="nn", help="Model type we want to train (svc, tree, nn)")
    parser.add_argument("--method", type=str, default="face", help="Competitor we want to use.")
    parser.add_argument("--quantile", type=float, default=1.0, help="How many example we want to run.")
    parser.add_argument("--queue", type=str, default="short_cpuQ", help="Queue where to launch the experiments.")
    parser.add_argument("--test-set-size", default=300, type=int, help="How many users we should pick from the test set for evaluation.")
    parser.add_argument("--output", default="", type=str, help="Path where to save the file.")
    parser.add_argument("--seed", default=2023, type=int, help="Seed which need to be used for evaluation.")

    args = parser.parse_args()

    if check_if_run_result_exists(args):
        print("Skipping since it exists ", f"validity_cost-{args.dataset}-{args.method}-{args.quantile}-{args.test_set_size}-{args.seed}.csv")
        exit

    experiment_name=f"{args.dataset[0]}_{args.model[0]}_{args.method[0:2]}_{args.quantile}"
    
    command = f'qsub -V -N {experiment_name} -q {args.queue} -v SEED="{args.seed}",L="{args.quantile}",D="{args.dataset}",M="{args.model}",MM="{args.method}",S="{args.test_set_size}",O="{args.output}" competitors/cluster/launch.sh'

    result = subprocess.run(
        command.split(" "), capture_output=True
    )
    print(result.stdout.decode())
    print(result.stderr.decode())