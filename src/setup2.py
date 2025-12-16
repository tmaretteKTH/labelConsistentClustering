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
    "dataset",
    type=str,
)

parser.add_argument(
    "historical",
    type=str,
    help="Clustering algorithm for historical clustering, can be: 'FFT', 'Carv' and 'Resilient'",
)
parser.add_argument("k", type=int, help="Number of centers opened")


def createHistorical(points, alg, k):
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


def computeClustering(points, k, Bs, epsilon, hC, lC, clusterAlgo, seed):
    ourAlgos = ["greedyAndProject", "OverCover", "iterativeRemoval", "Carv"]
    clusterCenters, clusterAssign = [], []
    rMin = clusteringScore(points, FFT(points, k)) / 3
    rMax = clusteringScore(points, hC, lC) * 3
    rStars = [rMin]  # all candidates rStar
    while rStars[-1] < rMax:
        rStars.append(rStars[-1] * (1 + epsilon))
    if clusterAlgo in ourAlgos:
        clusterCenters, clusterAssign = findBestRstarAndClustering(
            points, k, Bs, rStars, hC, lC, clusterAlgo, seed
        )
    if clusterAlgo == "Resilient":
        clusterCenters, clusterAssign = resilientkcenter(
            points, k, 0.5, 1.1, 0.5, 0.5, seed
        )
    if clusterAlgo == "FFT":
        clusterCenters = FFT(points, k)
        clusterAssign = findClosestCenter(points, clusterCenters)
    return clusterCenters, clusterAssign


################################
###     EXECUTION            ###
################################
seed = 2026
epsilon = 0.25  # rStar step
args = parser.parse_args()
hist = args.historical
k = args.k
dataset = args.dataset
algos = [
    hist,
    "greedyAndProject",
    "OverCover",
]  # Algorithms to run
timesteps = 20

if dataset == f"OnlineRetail":
    dataNext = onlineRetailGetNext
    X, n = onlineRetailGetN(timesteps)
if dataset == f"Electricity":
    dataNext = electricityGetNext
    X, n = electricityGetN(timesteps)
if dataset == "Twitter":
    dataNext = twitterGetNext
    X, n = twitterGetN(timesteps)
if dataset == "Uber":
    dataNext = uberGetNext
    X, n = uberGetN(timesteps)

for algo in algos:
    if algo == "FFT" or algo == "Resilient" or algo == "Carv":
        Bs = [10]
    else:
        Bs = [2, 4, 6]
    pastBs = [0]
    X, nextDate, points = dataNext(n, X)
    historicalClustering = createHistorical(points, hist, k)
    historicalScore = clusteringScore(
        points, historicalClustering[0], historicalClustering[1]
    )
    historicalk = len(historicalClustering[0])
    for b in Bs:
        effectiveB = int(n * (b / 10))
        pastBs.append(effectiveB)
        folderpath = f"results/setup2/{dataset}/{k}-{b*10}%/"
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        f = open(
            f"results/setup2/{dataset}/{k}-{b*10}%/[{hist}]{algo}.csv",
            "w",
        )
        f.write(f"t,score,k,nbUpdates\n0,{historicalScore},{historicalk},0\n")
        f.close()
        oldCenters, oldAssign = historicalClustering
        for t in range(timesteps):
            previousStepPoints = [points[i] for i in oldCenters]
            if t == 0:
                X, nextDate, points = dataNext(n, X)  # step 0 (historical)
                X, nextDate, points = dataNext(n, X, nextDate)  # step 1
            else:
                X, nextDate, points = dataNext(n, X, nextDate)
            oldCenters = []
            points = list(points)
            for c in previousStepPoints:
                points.append(c)
                oldCenters.append(len(points) - 1)
            points = np.array(points)
            oldAssign = findClosestCenter(
                points, oldCenters
            )  # update historical assignment for new points
            clusterCenters, clusterAssign = computeClustering(
                points, k, pastBs, epsilon, oldCenters, oldAssign, algo, seed
            )
            score = clusteringScore(points, clusterCenters, clusterAssign)
            numberUpdates = clusterDistance(
                oldCenters, oldAssign, clusterCenters, clusterAssign
            )
            f = open(f"results/setup2/{dataset}/{k}-{b*10}%/[{hist}]{algo}.csv", "a")
            f.write(f"{t+1},{score},{len(clusterCenters)},{numberUpdates}\n")
            f.close()
