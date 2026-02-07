import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algos = [
    "OverCover",
    "greedyAndProject",
]  # Clustering algorithm
baselines = ["Resilient"]
datasets = ["OnlineRetail", "Uber", "Twitter"]  # "Electricity"
# datasets = ["OnlineRetail"]
ks = ["30"]  # Number of clusters
Bs = [f"{i}0%" for i in [2, 4, 6]]
# colors_list = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
colors_list = ["#66c2a5", "#fc8d62", "#8da0cb"]
colors_list = ["#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#66c2a5"]
colors_list = ["#377eb8", "#e41a1c", "#4daf4a", "black", "#984ea3"]
# e41a1c
# 377eb8
# 4daf4a
# 984ea3

markers = [".", "+", "x", "*", "d"]

if not os.path.exists("plots"):
    os.makedirs("plots")
metrics = ["nbUpdates", "score"]
for k in ks:
    for dataset in datasets:
        for metric in metrics:
            nameDataset = dataset
            for baseline in baselines:
                # Compare with non-consistent algo

                for b in Bs:
                    fontsize = 16
                    fig = plt.figure(figsize=(8, 2.2), dpi=200)
                    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.2, hspace=0.8)
                    plt.xlabel("timeStep", fontsize=fontsize)
                    if metric == "score":
                        plt.ylabel("solution cost (log scale)", fontsize=fontsize - 4)
                    else:
                        plt.ylabel("Number of updates", fontsize=fontsize)

                    plt.xticks([0, 5, 10, 15, 20], fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    pathToFolder = f"setup2/{dataset}/{k}"
                    if not os.path.exists("plots/" + pathToFolder):
                        os.makedirs("plots/" + pathToFolder)
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-100%/[Resilient]Carv.csv",
                        sep=",",
                    )
                    X, Y = p["t"], p[metric]
                    if metric == "score":
                        Y = np.log(Y)
                    plt.plot(
                        X,
                        Y,
                        color="black",
                        label="Carve",
                        marker=markers[3],
                    )
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-100%/[Resilient]Resilient.csv",
                        sep=",",
                    )
                    X, Y = p["t"], p[metric]
                    if metric == "score":
                        Y = np.log(Y)
                    plt.plot(
                        X,
                        Y,
                        color=colors_list[4],
                        label="Resilient",
                        marker=markers[4],
                    )
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-{b}/[{baseline}]greedyAndProject.csv",
                        sep=",",
                    )
                    scores = p[metric]
                    if metric == "score":
                        scores = np.log(scores)
                    timesteps = p["t"]
                    plt.plot(
                        timesteps,
                        scores,
                        color=colors_list[1],
                        label=f"greedyAndProject-{b}",
                        marker=markers[1],
                    )
                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-{b}/[{baseline}]OverCover.csv",
                        sep=",",
                    )
                    scores = p[metric]
                    if metric == "score":
                        scores = np.log(scores)
                    timesteps = p["t"]
                    plt.plot(
                        timesteps,
                        scores,
                        color=colors_list[0],
                        label=f"OverCover-{b}",
                        marker=markers[0],
                    )

                    p = pd.read_csv(
                        f"results/setup2/{dataset}/{k}-{b}/[Resilient]CFHLNS.csv",
                        sep=",",
                    )
                    X, Y = p["t"], p[metric]
                    if metric == "score":
                        Y = np.log(Y)
                    plt.plot(
                        X,
                        Y,
                        color=colors_list[2],
                        label=f"CFHLNS-{b}",
                        marker=markers[2],
                    )

                    plt.title(
                        f"{nameDataset} dataset, k={max(p["k"])}\n ",
                        fontsize=12,
                    )
                    if not os.path.exists(f"plots/setup2/{dataset}/{k}/"):
                        os.makedirs(f"plots/setup2/{dataset}/{k}/")
                    b = b[:-1]
                    plt.savefig(
                        f"plots/setup2/{dataset}/{k}/_2_{dataset}-{baseline}-{metric}-{b}.pdf",
                        bbox_inches="tight",
                    )
                    plt.legend(
                        prop={"size": 12},
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                    )
                    plt.savefig(
                        f"plots/setup2/{dataset}/{k}/_2_{dataset}-{baseline}-{metric}-{b}-legend.pdf",
                        bbox_inches="tight",
                    )
                    plt.clf()
                    plt.close()
