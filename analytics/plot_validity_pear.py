import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(rc={'figure.figsize':(8,5)})
sns.set(font_scale=1.5)
sns.set_style("ticks", {'axes.grid' : False})
sns.set_palette(sns.color_palette("colorblind"))
plt.rcParams['lines.markersize'] = 12

plt.rcParams.update({
    "text.usetex": True
})

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", type=str, default=".", help="Path where to look for the files")
    args = parser.parse_args()

    complete_data = []

    all_files = []
    dir_path = args.path
    for dir_path in args.path:
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                if "validity_cost_elicitation" in path:
                    all_files.append(
                        (dir_path, path)
                    )
    
    all_files = sorted(all_files)

    index_set_dict = {}
    complete_data_dict = {}

    average_costs = []

    all_indexes = set()
    for dir_path, f in all_files:
        data = pd.read_csv(os.path.join(dir_path,f))
        if not "-T-" in f:
            _, dataset, questions, wrong_graph, logistic, choice_set, test_set, quantile, seed = f.replace(".csv", "").split("-")
            if questions == "1":
                data.loc[(data.elicitation == 1) & (data.recourse == 0), "elicitation"] = 0
        data = data[data.recourse == 1]
        if len(all_indexes) == 0:
            all_indexes.update(data.user_idx.values)
        else:
            all_indexes = all_indexes.intersection(set(data.user_idx.values))

    for dir_path, f in all_files:
        
        if "-T-" in f:
            continue

        _, dataset, questions, wrong_graph, logistic, choice_set, test_set, quantile, seed = f.replace(".csv", "").split("-")
        
        key = f"{wrong_graph}_{logistic}_{choice_set}_{test_set}"

        if wrong_graph == "0.1":
            continue
        
        if seed != "2023":
            continue

        if quantile != "1.0":
            continue

        if choice_set != "2":
            continue


        data = pd.read_csv(os.path.join(dir_path,f))

        validity = round(data.recourse.sum()/len(data),2)
        
        average_costs.append([validity, round(data["cost"].mean(),2), round(data.length.mean(),2), int(questions), wrong_graph, choice_set, logistic])

    complete_df = pd.DataFrame(
        average_costs, columns=["Validity", "Cost", "Length", "Questions", "Wrong Graph", "Choice Set", "Logistic"]
    )

    df = complete_df.groupby(["Choice Set", "Logistic"])
    for key, item in df:
        print(key)
        for k, r in df.get_group(key).sort_values(by=['Questions'])[["Validity", "Cost", "Length"]].iterrows():
            print(" ".join([str(v) for v in r.values.tolist()]))
        print("\n\n")