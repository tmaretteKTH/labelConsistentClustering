import os
import pandas as pd
import matplotlib.pyplot as plt

algos = [
    "OverCover",
    "greedyAndProject",
]  # Clustering algorithm
baseline = "Carv"
dataset = "Twitter"
ks = ["5", "10", "20", "50"]  # Number of clusters
Bs = ["2", "4", "6"]
colors_list = ["#7f3b08", "#542788", "#7fbc41", "#000000"]
markers = ["+", "o", "x", "v"]
colors_list = [
    "#7f3b08",
    "#542788",
    "#000000",
    "#7fbc41",
]
markers = ["+", "o", "x", "1"]


if not os.path.exists("plots/scalability"):
    os.makedirs("plots/scalability")
for b in Bs:
    # with fixed b, plot all combination of k,algo on same plot
    nameb = b
    b = f"{b}0%"
    pathToFolder = f"scalability"
    if not os.path.exists("plots/" + pathToFolder):
        os.makedirs("plots/" + pathToFolder)
    nameDataset = "Twitter"
    for algo in algos:
        fontsize = 8
        fig = plt.figure(figsize=(3, 2.2), dpi=200)
        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.2, hspace=0.8)
        plt.xlabel("Data size", fontsize=fontsize)
        plt.ylabel("time (s)", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        i = 0

        for k in ks:
            p = pd.read_csv(
                f"results/scalability/{k}-{b}/[{baseline}]{algo}.csv",
                sep=",",
            )
            time = list(p["time"])
            n = list(p["n"])
            time.pop()
            n.pop()
            plt.plot(
                n,
                time,
                color=colors_list[i],
                label=f"{algo}, k={k}",
                marker=markers[i],
            )
            i += 1
        plt.title(
            f"{nameDataset} dataset, b={b}\n Historical algorithm: {baseline}e",
            fontsize=fontsize,
        )
        plt.savefig(f"plots/{pathToFolder}/{algo}-{nameb}.pdf")
        plt.legend(prop={"size": fontsize})
        plt.savefig(f"plots/{pathToFolder}/{algo}-{nameb}-legend.pdf")

        plt.clf()
        plt.close()
