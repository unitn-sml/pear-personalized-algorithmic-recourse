import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(font_scale=1.5)
sns.set_style("ticks", {'axes.grid' : False})
sns.set_palette(sns.color_palette("colorblind"))

plt.rcParams.update({
    "text.usetex": True
})

import argparse

METHODS = {
    "cscf": "$\mathtt{CSCF}$",
    "cscf_uniform": "$\mathtt{CSCF}_{U}$",
    "face_l2": "$\mathtt{FACE}_{\ell_2}$",
    "face_cost": "$\mathtt{FACE}$",
    "face_cost_uniform": "$\mathtt{FACE}_{U}$",
    "fare": "$\mathtt{FARE}$",
    "efare": "$\mathtt{EFARE}$",
    "rl": "$\mathtt{RL}$",
    "mcts": "$\mathtt{MCTS}$",
    "fare_uniform": "$\mathtt{FARE}_{U}$",
    "rl_uniform": "$\mathtt{RL}_{U}$",
    "mcts_uniform": "$\mathtt{MCTS}_{U}$",
    "xpear_nl": "$\mathtt{XPEAR}_{NL}$",
    "xpear_l": "$\mathtt{XPEAR}_{L}$",
    "pear_nl": "$\mathtt{PEAR}_{NL}$",
    "pear_l": "$\mathtt{PEAR}_{L}$"
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=".", help="Path where to look for the files")
    parser.add_argument("--corrupt-graph", type=float, default=0.0)
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

        if len(f.replace(".csv", "").split("-")) > 6:
            _, dataset, method, quantile, test_set, corruption, seed = f.replace(".csv", "").split("-")

            if float(corruption) != args.corrupt_graph:
                continue

        else:
            _, dataset, method, quantile, test_set, seed = f.replace(".csv", "").split("-")
        
        data = pd.read_csv(os.path.join(dir_path,f))[['user_idx', 'recourse', 'cost', 'length', 'mixture_class', 'model_score']]
        data["seed"] = np.ones(len(data))*int(seed)
        data["quantile"] = np.ones(len(data))*float(quantile)
        data["method"] = [method for _ in range(len(data))]
        data["dataset"] = [dataset for _ in range(len(data))]

        if "uniform" in method:
            continue

        if quantile == "0.5":
            continue
            
        all_dataset.append(data)
    
    all_dataset = pd.concat(all_dataset, axis=0)

    list_of_dfs = []
    columns = []
    methods = None
    for q in all_dataset["quantile"].unique():
        for m in all_dataset.method.unique():
            line = [q, m]
            for d in ["adult", "givemecredit"]:
                tmp = all_dataset[(all_dataset.dataset == d) & (all_dataset.method == m) & (all_dataset["quantile"] == q)][["user_idx", "recourse", "cost", "length"]].copy()
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
            
            list_of_dfs.append(line)

    list_of_dfs = sorted(list_of_dfs, key= lambda x: (x[0], x[2], x[3]), reverse=True)
    
    df = pd.DataFrame(list_of_dfs, columns=[("quantile", "metric"),("method", "metric"), ("adult", "validity"), ("adult", "cost"), ("adult", "length"), ("givemecredit", "validity"), ("givemecredit", "cost"), ("givemecredit", "length"),], index=None)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Method", ""])


    for q in df[("quantile", "metric")].unique():
        print(q)
        selection =df[df[("quantile", "metric")] == q]
        selection[("method", "metric")] = selection[("method", "metric")].apply(lambda x: METHODS.get(x))
        selection.drop(columns=[("quantile", "metric")], inplace=True)

        #print(selection.to_list())
        for index, row in selection.iterrows():
            print("\t".join(row.to_list()))

        #df_tex = selection.to_latex(index=False,
        #                    formatters={
        #                        ("method", "metric"): METHODS.get
        #                    },
        #                    escape=False,
        #                    na_rep="-")
        #print(df_tex)