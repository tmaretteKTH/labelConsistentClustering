import os
import pandas as pd
import matplotlib.pyplot as plt

algos = [
    "OverCover",
    "greedyAndProject",
]  # Clustering algorithm
baselines = ["Carv", "Resilient"]
# datasets = ["Uber", "OnlineRetail", "Electricity", "Twitter"]
datasets = ["OnlineRetail", "Electricity"]
ks = ["30"]  # Number of clusters
Bs = [f"{i}0%" for i in [2, 4, 6]]
colors_list = ["#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#66c2a5"]
colors_list = ["#1b9e77", "#7570b3", "#d95f02", "black", "#66a61e"]
markers = ["+", "o", "x", 5, "1"]
if not os.path.exists("plots"):
    os.makedirs("plots")
metrics = ["nbUpdates", "score"]
for k in ks:
    for dataset in datasets:
        for metric in metrics:
            nameDataset = dataset
            for baseline in baselines:
                # Compare with non-consistent algo
                for algo in algos:
                    fontsize = 16
                    fig = plt.figure(figsize=(8, 2.2), dpi=200)
                    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.2, hspace=0.8)
                    plt.xlabel("timeStep", fontsize=fontsize)
                    if metric == "score":
                        plt.ylabel("Solution cost", fontsize=fontsize)
                    else:
                        plt.ylabel("Number of updates", fontsize=fontsize)

                    plt.xticks([0, 5, 10, 15, 20], fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    pathToFolder = f"setup2/{dataset}/{k}"
                    if not os.path.exists("plots/" + pathToFolder):
                        os.makedirs("plots/" + pathToFolder)

                    i = 0

                    for b in Bs:

                        if not os.path.isfile(
                            f"results/setup2/{dataset}/{k}-{b}/[{baseline}]{algo}.csv"
                        ):
                            print("Error, missing", dataset, k, b, baseline, algo)
                            continue
                        p = pd.read_csv(
                            f"results/setup2/{dataset}/{k}-{b}/[{baseline}]{algo}.csv",
                            sep=",",
                        )
                        scores = p[metric]
                        timesteps = p["t"]
                        plt.plot(
                            timesteps,
                            scores,
                            color=colors_list[i],
                            label=f"{algo}-{b}",
                            marker=markers[i],
                        )
                        i += 1
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-100%/[Resilient]Resilient.csv",
                        sep=",",
                    )
                    X, Y = p["t"], p[metric]
                    plt.plot(
                        X,
                        Y,
                        color=colors_list[4],
                        label="Resilient",
                        marker=markers[4],
                    )
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-100%/[Carv]Carv.csv",
                        sep=",",
                    )
                    X, Y = p["t"], p[metric]
                    plt.scatter(
                        X,
                        Y,
                        color=colors_list[3],
                        label="Carve",
                        marker=markers[3],
                        s=61,
                    )
                    plt.title(
                        f"{nameDataset} dataset, k={max(p["k"])}\n ",
                        fontsize=12,
                    )
                    if not os.path.exists(f"plots/setup2/{dataset}/{k}/"):
                        os.makedirs(f"plots/setup2/{dataset}/{k}/")
                    plt.savefig(
                        f"plots/setup2/{dataset}/{k}/[{baseline}]{algo}-{metric}.pdf",
                        bbox_inches="tight",
                    )
                    plt.legend(
                        prop={"size": 12}, loc="center left", bbox_to_anchor=(1, 0.5)
                    )
                    plt.savefig(
                        f"plots/setup2/{dataset}/{k}/[{baseline}]{algo}-{metric}-legend.pdf",
                        bbox_inches="tight",
                    )
                    plt.clf()
                    plt.close()
