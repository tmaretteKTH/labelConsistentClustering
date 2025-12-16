import numpy as np
from helperFunctions import *
from algorithms.FFT import FFT


def updateCovered(points, covered, newCenter, rStar, realIndex):
    covered[newCenter] = True
    unCoveredPoints = np.where(covered == False)[0]
    for x in unCoveredPoints:
        if dist(points, realIndex[newCenter], realIndex[x]) <= 2 * rStar:
            covered[x] = True
    return covered


def Carvrstar(points, rStar, k, seed, realIndex=[]):
    rng = np.random.RandomState(seed)
    # k is given as an upper bound, so that we stop the search if we need k+1 centers to cover our dataset
    if list(realIndex) == []:
        realIndex = np.arange(len(points))
    n = len(realIndex)
    if n == 0:
        return np.array([])
    covered = np.full(n, False, dtype=bool)
    newClusterCenter = rng.choice(np.where(covered == False)[0])
    covered[newClusterCenter] = True
    clusterCenters = [realIndex[newClusterCenter]]
    covered = updateCovered(points, covered, newClusterCenter, rStar, realIndex)
    while sum(covered) < n and len(clusterCenters) < k:
        newClusterCenter = rng.choice(np.where(covered == False)[0])
        clusterCenters.append(realIndex[newClusterCenter])
        updateCovered(points, covered, newClusterCenter, rStar, realIndex)
    if sum(covered) < n:
        # we failed at covering
        return [0]
    return np.array(clusterCenters)


def carv(points, k, seed, epsilon, realIndex=[]):
    if realIndex == []:
        realIndex = np.arange(len(points))
    TworStar = clusteringScore(points, FFT(points, k))  # this is at most 2rStar
    bestClusterCenters = [0]
    rStar = TworStar / 3
    rStop = TworStar * 2
    bestScore = clusteringScore(points, [0])
    while rStar < rStop:
        clusterCenters = Carvrstar(points, rStar, k, seed, realIndex)
        if len(clusterCenters) <= k:
            score = clusteringScore(points, clusterCenters)
            if score < bestScore:
                rStop = min(rStop, score)
                bestScore = score
                bestClusterCenters = clusterCenters
        rStar *= 1 + epsilon
    return bestClusterCenters
