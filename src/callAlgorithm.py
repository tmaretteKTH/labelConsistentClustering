from algorithms.FFT import *
from algorithms.carv import *
from algorithms.resilientkclustering import *
from algorithms.greedyAndProject import *
from algorithms.overCover import *
from algorithms.CFHLNS import *


def createHistorical(points, alg, k, epsilon, seed):
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
            if clusterAlgo == "CFHLNS":
                clusterCenters, clusterAssign = CFHLNS(
                    points, k, b, hC, lC, rStar, seed
                )
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
