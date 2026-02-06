import os
import pandas as pd
import matplotlib.pyplot as plt

historicals = ["Carv", "Resilient"]  # Historical clustering algorithm
algos = historicals + [
    "greedyAndProject",
    "OverCover",
    "Chakraborty",
]  # Clustering algorithm
historicals.append("FFT")
datasets = ["Electricity", "OnlineRetail", "Abalone", "Twitter", "Uber"]  # Datasets
# datasets = ["Electricity", "OnlineRetail", "Abalone"]  # Datasets

ks = ["10", "20", "50"]  # Number of clusters

colors_list = ["#7f3b08", "#000000", "#7fbc41", "blue"]
markers = ["*", ".", "x", "+", "d", "."]
colors_list = ["#1b9e77", "#7570b3", "#d95f02", "black", "#66a61e", "blue"]
colors_list = [
    "black",
    "#984ea3",
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#542788",
]
markers = ["*", "d", "+", ".", "x", "*"]

if not os.path.exists("plots"):
    os.makedirs("plots")
for k in ks:
    for dataset in datasets:
        nameDataset = dataset.title()
        if dataset == "Onlineretail":
            nameDataset = "OnlineRetail"
        dataset = f"setup1/{dataset}-{k}"
        if not os.path.exists("plots/" + dataset):
            os.makedirs("plots/" + dataset)
        for hist in historicals:
            i = 0
            fontsize = 8
            fig = plt.figure(figsize=(3, 2.2), dpi=200)
            fig.subplots_adjust(top=0.82, bottom=0.18, left=0.2, hspace=0.8)
            plt.xlabel("Number of updates", fontsize=fontsize)
            plt.ylabel("Solution cost", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            for algo in algos:
                p = pd.read_csv(f"results/{dataset}/[{hist}]{algo}.csv", sep=",")
                if algo == "Chakraborty":
                    algoName = "CFHLNS"
                else:
                    algoName = algo
                bs = p["nbUpdates"] if algo in historicals else p["b"]
                scores = p["score"]
                if algo not in historicals:
                    plt.scatter(
                        bs,
                        scores,
                        color=colors_list[i],
                        label=f"{algoName}",
                        marker=markers[i],
                    )
                else:
                    if algo == "Carv":
                        algoText = algo + "e"
                    else:
                        algoText = algo
                    plt.scatter(
                        bs,
                        scores,
                        color=colors_list[i],
                        label=f"{algoText}",
                        marker=markers[i],
                        s=100,
                    )
                i += 1
            plt.suptitle(
                f"{nameDataset} dataset, k={k}",
                fontsize=fontsize,
                x=(fig.subplotpars.right + fig.subplotpars.left) / 2,
            )
            if hist == "Carv":
                plt.title(
                    # rf"Historical algorithm: $\mathbf{{{hist}e}}$",
                    f"Historical algorithm: {hist}e",
                    fontsize=fontsize,
                )
            else:
                plt.title(
                    f"Historical algorithm: {hist}",
                    fontsize=fontsize,
                )
            plt.savefig(
                f"plots/{dataset}/_1_{nameDataset}-{k}-{hist}.pdf",
                bbox_inches="tight",
            )
            plt.legend(prop={"size": fontsize})
            plt.savefig(
                f"plots/{dataset}/_1_{nameDataset}-{k}-{hist}-legend.pdf",
                bbox_inches="tight",
            )

            plt.clf()
            plt.close()
