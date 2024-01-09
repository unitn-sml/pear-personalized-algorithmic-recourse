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
    "pear_nl": "$\mathtt{PEAR}_{NL}$",
    "pear_l": "$\mathtt{PEAR}_{L}$",
    "fare": "$\mathtt{FARE}$",
    "efare": "$\mathtt{EFARE}$",
    "cscf": "$\mathtt{CSCF}$",
    "face_cost": "$\mathtt{FACE}$",
    "rl": "$\mathtt{RL}$",
    "mcts": "$\mathtt{MCTS}$"
}
ORDER = list([v for k,v in METHODS.items()])


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
            
            if "xpear" in m or "face_l2" in m:
                continue

            for d in ["adult", "givemecredit"]:
                tmp = all_dataset[(all_dataset.dataset == d) & (all_dataset.method == m) & (all_dataset["quantile"] == q)][["user_idx", "recourse", "cost", "length"]].copy()
                tmp.dropna(inplace=True)

                condition = (tmp.recourse==1) | (tmp.recourse==True)

                validity_all = tmp.groupby('user_idx')["recourse"].agg([np.mean, np.var])
                cost_length = tmp[condition].groupby('user_idx')[["cost", "length"]].agg([np.mean, np.var])

                if len(tmp) > 0:
                    for v, c in zip(validity_all["mean"].values, cost_length[("cost", "mean")].values):
                        list_of_dfs.append(
                            [q, 
                             METHODS.get(m),
                             "$\mathtt{Adult}$" if d == "adult" else "$\mathtt{GiveMeSomeCredit}$",
                             v, np.sqrt(validity_all["var"].fillna(0).mean()), c, np.sqrt(cost_length[("cost", "var")].fillna(0).mean())]
                        )


                validity = validity_all["mean"].mean()
                validity_std = np.sqrt(validity_all["var"].fillna(0).mean())

                cost = cost_length[("cost", "mean")].mean()
                cost_std = np.sqrt(cost_length[("cost", "var")].fillna(0).mean()) 

                length = cost_length[("length", "mean")].mean()
                length_std = np.sqrt(cost_length[("length", "var")].fillna(0).mean()) 

    #list_of_dfs = sorted(list_of_dfs, key= lambda x: (x[0], x[3], x[5]), reverse=True)
    
    df = pd.DataFrame(list_of_dfs, columns=["quantile", "Method", "Dataset", "Validity", "validity_std", "Cost", "cost_std"], index=None)

    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.catplot(
            data=df,
            x="Dataset", y="Cost", hue="Method",
            col="quantile",
            kind="bar",
            legend="brief",
            aspect=1.8,
            palette=sns.color_palette("colorblind")[0:2]+sns.color_palette("pastel")[2:],
            hue_order=ORDER,
            col_order=[1.0, 0.25],
            capsize=.05,

        )

        # Rename Axes
        axes = g.axes.flatten()
        for ax in axes:
            ax.set_title("Users = $\mathtt{Hard}$" if "0.25" in ax.get_title() else "Users = $\mathtt{All}$", fontsize=25)
            ax.set_ylabel(ax.get_ylabel(), fontsize=25)
            ax.set_xlabel("", fontsize=25)
            ax.tick_params(labelsize=25)
    
        plt.savefig(f"competitors_bars_cost.png", dpi=600)

        g = sns.catplot(
            data=df,
            x="Dataset", y="Validity", hue="Method",
            col="quantile",
            kind="bar",
            legend="brief",
            aspect=1.8,
            palette=sns.color_palette("colorblind")[0:2]+sns.color_palette("pastel")[2:],
            hue_order=ORDER,
            col_order=[1.0, 0.25],
            capsize=.05,

        )

        # Rename Axes
        axes = g.axes.flatten()
        for ax in axes:
            ax.set_title("Users = $\mathtt{Hard}$" if "0.25" in ax.get_title() else "Users = $\mathtt{All}$", fontsize=25)
            ax.set_ylabel(ax.get_ylabel(), fontsize=25)
            ax.set_xlabel("", fontsize=25)
            ax.tick_params(labelsize=25)
    
        plt.savefig(f"competitors_bars_validity.png", dpi=600)
        plt.show()