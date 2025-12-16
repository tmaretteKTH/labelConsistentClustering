import numpy as np
from helperFunctions import *
from algorithms.carv import Carvrstar


def greedyAndProject(
    points, k, b, historicalClusterCenters, historicalClusterAssign, rStar, seed
):
    S = Carvrstar(points, rStar, k, seed)  # cover points with radius 2r^*
    if len(S) > k:
        return historicalClusterCenters, historicalClusterAssign
    alpha = {}
    histAssignWithIndex = [historicalClusterCenters[i] for i in historicalClusterAssign]
    for c in historicalClusterCenters:
        alpha[c] = np.sum(np.array(histAssignWithIndex) == c)
    C0 = []
    for s in S:
        # get all hist centers at distance less than rStar
        Ns = [c for c in historicalClusterCenters if dist(points, s, c) <= rStar]
        if Ns == []:
            C0.append(s)
        else:
            Ns.sort(key=lambda x: -alpha[x])
            C0.append(Ns[0])
    C1 = [c for c in historicalClusterCenters if c not in C0]
    C1.sort(key=lambda x: -alpha[x])
    C1 = C1[
        : (k - len(C0))
    ]  # we pick the largest historical centers to complete our set of centers until we have k
    C = C0 + C1
    lC = historicalClusterAssign.copy()
    pointsIndices = np.arange(len(points))
    closestCenter = findClosestCenter(points, C)
    for x in pointsIndices:
        if historicalClusterCenters[historicalClusterAssign[x]] not in C:
            lC[x] = closestCenter[x]
        else:
            # update index
            lC[x] = C.index(historicalClusterCenters[historicalClusterAssign[x]])
    if clusterDistance(historicalClusterCenters, historicalClusterAssign, C, lC) > b:
        # We overspent
        return historicalClusterCenters, historicalClusterAssign
    distToCluster = np.array(
        list(
            map(
                lambda x: dist(points, x, C[lC[x]]),
                pointsIndices,
            )
        )
    )
    canUpgrade = True
    while (
        clusterDistance(historicalClusterCenters, historicalClusterAssign, C, lC) < b
        and canUpgrade
    ):
        # we can reassign more points
        x = np.argmax(distToCluster)
        closestDistance = dist(points, x, C[closestCenter[x]])
        if closestDistance < distToCluster[x]:
            lC[x] = closestCenter[x]
            distToCluster[x] = closestDistance
        else:
            # furthest point cannot be reassigned, useless to continue
            canUpgrade = False
    return C, lC
