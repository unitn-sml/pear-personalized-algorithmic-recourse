import pandas as pd
import numpy as np

import os

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=".", help="Path where to look for the files")
    args = parser.parse_args()

    complete_data = []

    all_files = []
    dir_path = args.path
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            if "validity_cost" in path:
                all_files.append(
                    path
                )
    
    all_files = sorted(all_files)

    index_set_dict = {}
    complete_data_dict = {}

    average_costs = []

    all_dataset = []
    for f in all_files:

        if "-T-" in f:
            continue

        _, dataset, questions, wrong_graph, logistic, choice_set, test_set, quantile, seed = f.replace(".csv", "").split("-")
        
        data = pd.read_csv(os.path.join(dir_path,f))[['user_idx', 'recourse', 'cost', 'length', 'mixture_class', 'model_score']]
        data["corruption"] = np.ones(len(data))*float(wrong_graph)
        data["seed"] = np.ones(len(data))*int(seed)
        data["quantile"] = np.ones(len(data))*float(quantile)
        data["dataset"] = [dataset for _ in range(len(data))]

        if quantile == "0.5":
            continue

        all_dataset.append(data)
    
    all_dataset = pd.concat(all_dataset, axis=0)

    list_of_dfs = []
    columns = []
    methods = None
    print("\t".join(["Quantile", "Corruption", "Validity", "Cost", "Length"]))
    for q in sorted(all_dataset["quantile"].unique(), reverse=True):
        for c in all_dataset["corruption"].unique():
            line = []
            tmp = all_dataset[(all_dataset["corruption"] == c) & (all_dataset["quantile"] == q)][["user_idx", "recourse", "cost", "length"]].copy()
            tmp.dropna(inplace=True)

            condition = (tmp.recourse==1) | (tmp.recourse==True)

            validity_all = tmp.groupby('user_idx')["recourse"].agg([np.mean, np.var])
            cost_length = tmp[condition].groupby('user_idx')[["cost", "length"]].agg([np.mean, np.var])

            validity = validity_all["mean"].mean()
            validity_std = np.sqrt(validity_all["var"].fillna(0).mean())

            cost = cost_length[("cost", "mean")].mean()
            cost_std = np.sqrt(cost_length[("cost", "var")].fillna(0).mean()) 

            length = cost_length[("length", "mean")].mean()
            length_std = np.sqrt(cost_length[("length", "var")].fillna(0).mean())


            if len(tmp) > 0:
                validity_std = "\small{"+f"${round(validity,2):.2f} \pm {round(validity_std,2):.2f}$"+"}"
                cost_std = "\small{"+f"${round(cost,2):.2f} \pm {round(cost_std,2):.2f}$"+"}"
                length_std = "\small{"+f"${round(length,2):.2f} \pm {round(length_std,2):.2f}$"+"}"
                line += [validity_std, cost_std, length_std]
            else:
                line += ["0.0", "-", "-"]
            
            print("\t".join([str(q), str(c)])+"\t"+"\t".join(line))