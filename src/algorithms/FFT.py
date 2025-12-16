import numpy as np
from helperFunctions import *


def updateCovered(points, covered, newCenter, rStar, realIndex):
    covered[newCenter] = True
    unCoveredPoints = np.where(covered == False)[0]
    for x in unCoveredPoints:
        if dist(points, realIndex[newCenter], realIndex[x]) <= 2 * rStar:
            covered[x] = True
    return covered


def furtherstNonCoveredPoint(points, clustering, realIndex):
    furthestPoint = realIndex[0]
    maxDist = 0
    for x in realIndex:
        if x in clustering:
            continue
        currDist = min([dist(points, x, c) for c in clustering])
        if currDist >= maxDist:
            maxDist = currDist
            furthestPoint = x
    return furthestPoint


def FFT(points, k, realIndex=[]):
    if realIndex == []:
        realIndex = np.arange(len(points))
    n = len(realIndex)
    if n == 0:
        return np.array([])
    clusterCenters = [realIndex[-1]]
    while len(clusterCenters) < k:
        newClusterCenter = furtherstNonCoveredPoint(points, clusterCenters, realIndex)
        clusterCenters.append(newClusterCenter)
    return np.array(clusterCenters)


def FFTrstar(points, rStar, k, realIndex=[]):
    # k is given as an upper bound, so that we stop the search if we need k+1 centers to cover our dataset
    # FFT algorithm but we stop once all the points are covered
    if realIndex == []:
        realIndex = np.arange(len(points))
    n = len(realIndex)
    if n == 0:
        return np.array([])
    covered = np.full(n, False, dtype=bool)
    newClusterCenter = realIndex[-1]
    covered[newClusterCenter] = True
    clusterCenters = [realIndex[newClusterCenter]]
    covered = updateCovered(points, covered, newClusterCenter, rStar, realIndex)
    while sum(covered) < n and len(clusterCenters) < k:
        newClusterCenter = furtherstNonCoveredPoint(points, clusterCenters, realIndex)
        clusterCenters.append(realIndex[newClusterCenter])
        updateCovered(points, covered, newClusterCenter, rStar, realIndex)
    return np.array(clusterCenters)
