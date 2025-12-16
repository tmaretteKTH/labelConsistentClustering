import os
from data import twitterScalability
from helperFunctions import *
from algorithms.FFT import *
from algorithms.carv import *
from algorithms.resilientkclustering import *
from algorithms.greedyAndProject import *
from algorithms.overCover import *
import argparse
import time

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument("k", type=int, help="Number of centers opened")

parser.add_argument("b", type=int, help="budget")


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
            newScore = clusteringScore(points, clusterCenters, clusterAssign)
            if newScore < score:
                score = newScore
                bestCenters = clusterCenters
                bestAssign = clusterAssign
    clusterCenters = bestCenters
    clusterAssign = bestAssign
    return clusterCenters, clusterAssign


def computeClustering(points, k, Bs, epsilon, hC, lC, clusterAlgo, seed):
    ourAlgos = ["greedyAndProject", "OverCover", "iterativeRemoval"]
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
    if clusterAlgo == "Carv":
        clusterCenters = carv(points, k, seed, epsilon)
        clusterAssign = findClosestCenter(points, clusterCenters)
    return clusterCenters, clusterAssign


################################
###     EXECUTION            ###
################################
seed = 2026
epsilon = 0.5  # rStar step
algos = [
    "greedyAndProject",
    "OverCover",
]
args = parser.parse_args()
hist = "Carv"
k = args.k
b = args.b
minN = 100
maxN = 70000
trials = 5

X = twitterScalability(maxN)

nStep = 1.33


for algo in algos:
    folderpath = f"results/scalability/{k}-{b*10}%"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    f = open(
        f"{folderpath}/[{hist}]{algo}.csv",
        "w",
    )
    f.write(f"n,time\n")
    f.close()
    n = minN
    while n != maxN:
        if n > maxN:
            n = maxN
        effectiveB = int(n * (b / 10))
        points = X[:n]
        historicalClustering = createHistorical(points, hist, k, seed)
        historicalScore = clusteringScore(
            points, historicalClustering[0], historicalClustering[1]
        )
        historicalk = len(historicalClustering[0])
        histCenters, histAssign = historicalClustering
        total = 0
        for _ in range(trials):
            start = time.time()

            clusterCenters, clusterAssign = computeClustering(
                points, k, [b], epsilon, histCenters, histAssign, algo, seed
            )
            end = time.time()
            total += end - start
        f = open(f"{folderpath}/[{hist}]{algo}.csv", "a")
        f.write(f"{n},{total/trials}\n")
        f.close()
        if n == maxN:
            break
        n = int(n * nStep)
