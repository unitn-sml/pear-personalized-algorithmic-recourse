import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(rc={'figure.figsize':(8,8)})
sns.set(font_scale=2.0)
sns.set_style("ticks", {'axes.grid' : False})
sns.set_palette(sns.color_palette("colorblind"))
plt.rcParams['lines.markersize'] = 12

plt.rcParams.update({
    "text.usetex": True
})

convert_model = {
    "pear_nl": r"$\texttt{PEAR}_{NL}$",
    "pear_l": r"$\texttt{PEAR}_{L}$",
    "rl": r"$\texttt{RL}$",
    "face_cost": r"$\texttt{FACE}$",
    "cscf": r"$\texttt{CSCF}$",
    "mcts": r"$\texttt{MCTS}$",
    "fare": r"$\texttt{FARE}$",
    "efare": r"$\texttt{EFARE}$",
}
hue_order = [v for k,v in convert_model.items()]

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=".", help="Path where to look for the files")
    parser.add_argument("--corr", default=False, action="store_true", help="Check if we want to print the corruption")
    parser.add_argument("--eus", default=False, action="store_true", help="Ablation EUS vs. Random")
    args = parser.parse_args()


    all_files = []
    dir_path = args.path
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            if "validity_cost" in path:
                all_files.append(
                    path
                )
    
    all_files = sorted(all_files)


    all_dataset = []
    for f in all_files:

        if "-T-" in f:
            continue

        print(f.replace(".csv", "").split("-"))
        _, dataset, method, quantile, test_set, _, seed = f.replace(".csv", "").split("-")

        if method in ["xpear_nl", "face_l2", "xpear_l"]:
            continue

        if seed != "2023":
            continue
            
        if quantile == "0.5":
            continue
        
        data = pd.read_csv(os.path.join(dir_path,f))[['user_idx', 'recourse', 'cost', 'length', 'mixture_class', 'model_score']]
        data = data.rename(columns={
            "recourse": "Validity",
            "cost": "Cost",
            "length": "Length",
            "model_score": "Model Score"
        })
        data["seed"] = np.ones(len(data))*int(seed)
        data["quantile"] = np.ones(len(data))*float(quantile)
        data["Method"] = [convert_model.get(method, method) for _ in range(len(data))]
        data["dataset"] = [dataset for _ in range(len(data))]
        
        all_dataset.append(data)
        
    all_dataset = pd.concat(all_dataset, axis=0)
    all_dataset.reset_index(inplace=True, drop=True)

    # Remove points which are not good
    all_dataset = all_dataset[~(all_dataset.dataset == "adult") | ~(all_dataset.Length >= 7)]
    all_dataset = all_dataset[~(all_dataset.dataset == "givemecredit") | ~(all_dataset.Length >= 9)]

    # PLOT FOR PEAR
    g = sns.catplot(data=all_dataset[(all_dataset.Validity==1) & (all_dataset.Method.isin([r"$\texttt{PEAR}_{NL}$", r"$\texttt{PEAR}_{L}$"]))], kind="box",
                    row="quantile", col="dataset", y="Cost", x="Length", hue="Method", hue_order=hue_order[0:2], aspect=2.0)
    axes = g.axes.flatten()
    for ax in axes:

        dataset = r"$\texttt{Adult}$" if "adult" in ax.get_title() else r"$\texttt{GiveMeSomeCredit}$"
        users_type = r"$\texttt{Hard}$" if "0.25" in ax.get_title() else r"$\texttt{All}$"
        ax.set_title(f"{dataset} (Users = {users_type})", fontsize=25)
    
    plt.savefig("cost_vs_length_pear.png", dpi=400)

    # PLOT FOR ALL METHODS
    g = sns.catplot(data=all_dataset[(all_dataset.Validity==1)], kind="box", row="quantile", col="dataset", y="Cost", x="Length", hue="Method", hue_order=hue_order, aspect=2.0)
    axes = g.axes.flatten()
    for ax in axes:

        dataset = r"$\texttt{Adult}$" if "adult" in ax.get_title() else r"$\texttt{GiveMeSomeCredit}$"
        users_type = r"$\texttt{Hard}$" if "0.25" in ax.get_title() else r"$\texttt{All}$"
        ax.set_title(f"{dataset} (Users = {users_type})", fontsize=25)
    
    plt.savefig("cost_vs_length_all.png", dpi=400)

    # SCORE VS LENGTH
    g = sns.catplot(data=all_dataset[(all_dataset.Validity==1) & (all_dataset["quantile"] == 1.0)], row="quantile", col="dataset", y="Model Score", x="Length", hue="Method", kind="box", hue_order=hue_order, aspect=2.0)
    axes = g.axes.flatten()
    for ax in axes:
        dataset = r"$\texttt{Adult}$" if "adult" in ax.get_title() else r"$\texttt{GiveMeSomeCredit}$"
        users_type = r"$\texttt{Hard}$" if "0.25" in ax.get_title() else r"$\texttt{All}$"
        ax.set_title(f"{dataset} (Users = {users_type})", fontsize=25)
    
    plt.savefig(f"score_vs_length.png", dpi=400)

    # SCORE VS ACCURACY
    g = sns.catplot(data=all_dataset[(all_dataset["quantile"] == 1.0)], row="quantile", col="dataset", y="Model Score", x="Validity", hue="Method", kind="box", hue_order=hue_order, aspect=2.0)
    axes = g.axes.flatten()
    for ax in axes:
        dataset = r"$\texttt{Adult}$" if "adult" in ax.get_title() else r"$\texttt{GiveMeSomeCredit}$"
        users_type = r"$\texttt{Hard}$" if "0.25" in ax.get_title() else r"$\texttt{All}$"
        ax.set_title(f"{dataset} (Users = {users_type})", fontsize=25)
    
    plt.savefig(f"score_vs_validity.png", dpi=400)

    # for q in all_dataset["quantile"].unique():
    #     for d in all_dataset["dataset"].unique():
    #         sns.relplot(data=all_dataset[(all_dataset.dataset==d) & (all_dataset.Validity==1) & (all_dataset["quantile"] == q)], row="quantile", col="dataset", y="Model Score", x="Cost", hue="Method", aspect=2.0)
    #         plt.savefig(f"score_vs_cost_{d}_{q}.png", dpi=600)

    #         sns.catplot(data=all_dataset[(all_dataset.dataset==d) & (all_dataset["quantile"] == q)], row="quantile", col="dataset", y="Model Score", x="Validity", hue="Method", kind="box", aspect=2.0)
    #         plt.savefig(f"score_vs_validity_{d}_{q}.png", dpi=600)
    # #plt.savefig("Cost_vs_Length.png", dpi=600)


    # for q in all_dataset["quantile"].unique():
    #     for d in all_dataset["dataset"].unique():
    #         sns.catplot(data=all_dataset[(all_dataset.dataset==d) & (all_dataset.Validity==1) & (all_dataset["quantile"] == q)], kind="box", row="quantile", col="dataset",y="Model Score", x="Length", hue="Method", aspect=2.0)
    #         plt.savefig(f"score_vs_length_{d}_{q}.png", dpi=600)