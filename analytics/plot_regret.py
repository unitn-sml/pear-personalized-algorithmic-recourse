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

        print(f)

        best = pd.read_csv(
            os.path.join(
                dir_path, f"validity_cost_elicitation-T-{dataset}-{quantile}.csv")
            )
        worst = pd.read_csv(
            os.path.join(
                dir_path,
                f"validity_cost_elicitation-{dataset}-0-{wrong_graph}-{logistic}-{choice_set}-{test_set}-{quantile}-{seed}.csv"
            )
        )

        data = pd.read_csv(os.path.join(dir_path,f))

        validity = round(data.recourse.sum()/len(data),2)
        print(f"[{questions}/{logistic}/{quantile}/{wrong_graph}] COST: ", validity, round(data[(data.recourse == 1)]["cost"].mean(),2), round(data[(data.recourse == 1)].length.mean(),2))

        average_costs.append([validity, round(data[(data.recourse == 1)]["cost"].mean(),2), round(data[(data.recourse == 1)].length.mean(),2), questions, wrong_graph, choice_set, logistic, quantile])


        data= data[data.recourse == 1]
        worst = worst.loc[data.index]
        best = best.loc[data.index]        
        
        data["original_cost"] = data.cost.copy()


        cost_std = data[(worst["cost"]-best["cost"] > 0) & (data.recourse == 1)]

        data["cost"] = (data["cost"]-best["cost"])/(worst["cost"]-best["cost"])
        data["cost"] = data["cost"].apply(lambda x: 1 if x > 1 else x)
        data["cost"] = data["cost"].apply(lambda x: 0 if x < 0 else x)
        data.loc[worst["cost"]-best["cost"] < 0, "cost"] = 1

        data.dropna(inplace=True)

        wrong_graph = 0.0 if wrong_graph in ["True", "False"] else float(wrong_graph)
        noisy_user = "Logistic" if logistic == "True" else "Noiseless"

        complete_data += [[dataset, int(questions), v, wrong_graph, noisy_user, choice_set, quantile, seed] for v in data.cost.values]

    complete_data = pd.DataFrame(complete_data, columns=["Dataset", "\# of Questions", "Normalized Average Regret", "Corruption", "Resp. Model", "\# Choice Set", "quantile", "seed"])


    complete_data = complete_data[(complete_data["\# Choice Set"] != "R")]
    complete_data = complete_data[(complete_data["quantile"] != "0.25")]

    avg_regret = complete_data[complete_data["\# of Questions"] == 10].groupby(["\# Choice Set", "Resp. Model"])["Normalized Average Regret"].agg([np.mean, np.std])

    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.relplot(
            data=complete_data,
            x="\# of Questions", y="Normalized Average Regret", hue="\# Choice Set",
            style="Resp. Model",
            col="Dataset",
            kind="line",
            dashes=True,
            markers=['o', '^'],
            linewidth=4,
            markersize=15,
            ci=None,
            legend="brief",
            aspect=1.5,
        )


        axes = g.axes.flatten()
        for ax in axes:
            ax.set_title("$\mathtt{Adult}$" if "adult" in axes[0].get_title() else "$\mathtt{GiveMeSomeCredit}$", fontsize=25)

            ax.set(xticks=np.arange(0, complete_data["\# of Questions"].max()+1, 2))
            ax.set(ylim=(0, 1))
            ax.set_xlabel("\# of Questions", fontsize=25)
            ax.set_ylabel("Normalized Average Regret",fontsize=25)
            ax.tick_params(labelsize=25)
        

        plt.savefig(f"regret_plot.png", dpi=500)

        v = pd.DataFrame(average_costs, columns=["val", "cost", "length", "questions", "wrong_graph", "choice_set", "logistic", "quantile"])

        print(v.groupby(["quantile", "questions", "logistic"]).min('min'))