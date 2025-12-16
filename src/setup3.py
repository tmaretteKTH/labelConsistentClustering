import numpy as np
import os
from data import *
from helperFunctions import *
from algorithms.FFT import *
from algorithms.carv import *
from algorithms.resilientkclustering import *
from algorithms.greedyAndProject import *
from algorithms.overCover import *
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "historical",
    type=str,
    help="Clustering algorithm for historical clustering, can be: 'FFT', 'Carv' and 'Resilient'",
)
parser.add_argument("k", type=int, help="Number of centers opened")


def createHistorical(points, alg, k, seed):
    if alg == "Random":
        historicalCenters = randomClustering(points, k, seed)
        historicalAssign = findClosestCenter(points, historicalCenters)
    if alg == "FFT":
        historicalCenters = FFT(points, k)
        historicalAssign = findClosestCenter(points, historicalCenters)
    if alg == "Carv":
        historicalCenters = carv(points, k, seed, epsilon)
        historicalAssign = findClosestCenter(points, historicalCenters)
    if alg == "Resilient":
        historicalCenters, historicalAssign = resilientkcenter(
            points, k, 0.5, 1.1, 0.5, 0.5, seed
        )
    historicalCluster = (historicalCenters, historicalAssign)
    return historicalCluster


def findBestRstarAndClustering(points, k, Bs, rStars, hC, lC, clusterAlgo, seed):
    score = clusteringScore(points, hC, lC) * 2
    bestCenters = hC
    bestAssign = lC
    for rStar in rStars:
        for b in Bs:
            if clusterAlgo == "greedyAndProject":
                clusterCenters, clusterAssign = greedyAndProject(
                    points, k, b, hC, lC, rStar, seed
                )
            if clusterAlgo == "OverCover":
                clusterCenters, clusterAssign = overCover(
                    points, k, b, hC, lC, rStar, seed
                )
            if clusterAlgo == "Carv":
                clusterCenters = Carvrstar(points, rStar, k, seed)
                clusterAssign = findClosestCenter(points, clusterCenters)
            newScore = clusteringScore(points, clusterCenters, clusterAssign)
            if newScore < score:
                score = newScore
                bestCenters = clusterCenters
                bestAssign = clusterAssign
    clusterCenters = bestCenters
    clusterAssign = bestAssign
    return clusterCenters, clusterAssign


def scoreUpdatesCurve(points, k, histCluster, histClusterName, clusterAlgo, seed):
    hC, lh = histCluster
    ourAlgos = ["greedyAndProject", "OverCover", "Carv"]
    clusterCenters, clusterAssign = [], []
    n = len(points)
    b = int(n / 200)
    if clusterAlgo in ["FFT", "Resilient", "Carv"]:
        b = 2 * n + 2
    bestB = [0]
    f = open(f"results/setup3/{dataset}/[{histClusterName}]{clusterAlgo}.csv", "w")
    f.write("b,score,k,nbUpdates\n")
    f.close()
    bestScore = clusteringScore(points, hC, lh)
    rMin = clusteringScore(points, FFT(points, k)) / 3
    rMax = historicalScore * 2
    rStars = [rMin]  # all candidates rStar
    while rStars[-1] < rMax:
        rStars.append(rStars[-1] * (1 + epsilon))
    while b != n:
        if b > n:
            b = n

        if clusterAlgo in ourAlgos:
            clusterCenters, clusterAssign = findBestRstarAndClustering(
                points, k, bestB + [b], rStars, hC, lh, clusterAlgo, seed
            )
        if clusterAlgo == "Resilient":
            clusterCenters, clusterAssign = resilientkcenter(
                points, k, 0.5, 1.1, 0.5, 0.5, seed
            )
        if clusterAlgo == "FFT":
            clusterCenters = FFT(points, k)
            clusterAssign = findClosestCenter(points, clusterCenters)
        score = clusteringScore(points, clusterCenters, clusterAssign)
        numberUpdates = clusterDistance(hC, lh, clusterCenters, clusterAssign)
        f = open(f"results/setup3/{dataset}/[{histClusterName}]{clusterAlgo}.csv", "a")
        f.write(f"{b},{score},{len(clusterCenters)},{numberUpdates}\n")
        f.close()
        if score < bestScore:
            bestB = [b]
            bestScore = score
        if b != n:
            b *= bepsilon
            b = int(b)


################################
###     EXECUTION            ###
################################
seed = 2026
epsilon = 0.25  # rStar step
bepsilon = 1.33  # budget step
algos = [
    "Resilient",
    "Carv",
    "OverCover",
    "greedyAndProject",
]  # Algorithms to run
args = parser.parse_args()
hist = args.historical
k = args.k
dataset = "Uber"

historicalPoints, newPoints, n = getConfigTemporalEvolution(dataset, -1)
historicalClustering = createHistorical(historicalPoints, hist, k, seed)
dataset = dataset + "-" + str(k)  # Fancy name
historicalScore = clusteringScore(
    newPoints, historicalClustering[0], historicalClustering[1]
)
historicalk = len(historicalClustering[0])
folderpath = f"results/setup3/{dataset}/"
if not os.path.exists(folderpath):
    os.makedirs(folderpath)
f = open(
    f"results/setup3/{dataset}/[{hist}].csv",
    "w",
)
f.write(f"b,score,k\n0,{historicalScore},{historicalk}\n")
f.close()
for algo in algos:
    scoreUpdatesCurve(newPoints, k, historicalClustering, hist, algo, seed)
