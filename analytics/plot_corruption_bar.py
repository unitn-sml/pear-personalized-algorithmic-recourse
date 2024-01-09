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
    "face_l2": "$\mathtt{FACE}_{\ell_2}$",
    "face_cost": "$\mathtt{FACE}$",
    "rl": "$\mathtt{RL}$",
    "mcts": "$\mathtt{MCTS}$"
}
ORDER = list([v for k,v in METHODS.items()])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", type=str, default=".", help="Path where to look for the files")
    args = parser.parse_args()

    complete_data = []

    all_files = []
    dir_path = args.path
    for pth in args.path:
        for path in os.listdir(pth):
            if os.path.isfile(os.path.join(pth, path)):
                if "validity_cost" in path:
                    all_files.append(
                        (pth, path)
                    )
    
    all_files = sorted(all_files)

    index_set_dict = {}
    complete_data_dict = {}

    average_costs = []

    all_dataset = []
    for dir_path, f in all_files:

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
    for q in all_dataset["quantile"].unique():
        for c in all_dataset["corruption"].unique():
            for d in ["adult", "givemecredit"]:
            
                tmp = all_dataset[(all_dataset["corruption"] == c) & (all_dataset["quantile"] == q) & (all_dataset["dataset"] == d)][["user_idx", "recourse", "cost", "length"]].copy()
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
                    for v, cst in zip(validity_all["mean"].values, cost_length[("cost", "mean")].values):
                        list_of_dfs.append(
                            ["$\mathtt{All}$" if q == 1.0 else "$\mathtt{Hard}$",
                             c,
                             d,
                            v, np.sqrt(validity_all["var"].fillna(0).mean()), cst, np.sqrt(cost_length[("cost", "var")].fillna(0).mean())]
                        )
            

    df = pd.DataFrame(list_of_dfs, columns=["Users", "\% Corruption", "Dataset", "Validity", "validity_std", "Cost", "cost_std"], index=None)

    print(df)

    palette_vals = sns.color_palette("Set1")[0:2]
    palette_vals.reverse()

    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.relplot(
            data=df,
            x="\% Corruption", y="Cost", hue="Users",
            col="Dataset",
            kind="line",
            legend="brief",
            aspect=1.5,
            markers=['*', '^'],
            linewidth=4,
            palette=palette_vals,
            hue_order=["$\mathtt{All}$", "$\mathtt{Hard}$"]
        )

        # Rename Axes
        axes = g.axes.flatten()
        for ax in axes:
            ax.set_title("$\mathtt{Adult}$" if "adult" in axes[0].get_title() else "$\mathtt{GiveMeSomeCredit}$", fontsize=25)
            ax.set_ylabel(ax.get_ylabel(), fontsize=25)
            ax.set_xlabel(ax.get_xlabel(), fontsize=25)
            ax.tick_params(labelsize=25)

        #plt.setp(g._legend.get_texts(), fontsize='25')
        #plt.setp(g._legend.get_title(), fontsize='25')

        plt.savefig(f"corruption_bars_cost.png", dpi=600)
        plt.show()