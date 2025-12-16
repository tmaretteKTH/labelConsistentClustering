import numpy as np

from helperFunctions import *
from algorithms.carv import Carvrstar


def overCover(
    points,
    k,
    b,
    historicalClusterCenters,
    historicalClusterAssign,
    rStar,
    seed,
):
    pointsIndices = np.arange(len(points))
    # Now we can compute F, the set of points far from their historical center
    F = [
        pointIndex
        for pointIndex in pointsIndices
        if dist(
            points,
            pointIndex,
            historicalClusterCenters[historicalClusterAssign[pointIndex]],
        )
        > rStar
    ]
    if len(F) > b:
        return (
            historicalClusterCenters,
            historicalClusterAssign,
        )  # If F is too large, we cant reassign them all and we know rStar is to small
    alpha = {}
    alphaAssign = [historicalClusterAssign[i] for i in pointsIndices if i not in F]
    # we look at assignments of points that are not far from their assigned historical center
    histAssignWithIndex = [historicalClusterCenters[i] for i in alphaAssign]
    for i in range(len(historicalClusterCenters)):
        c = historicalClusterCenters[i]
        alpha[c] = np.sum(np.array(histAssignWithIndex) == c)
    beta = b - len(F)
    while beta > 0:
        # remove centers
        smallestCenter = min(alpha, key=alpha.get)
        if beta - alpha[smallestCenter] >= 0:
            beta -= alpha[smallestCenter]
            alpha.pop(smallestCenter)
        else:
            break
        if alpha == {}:
            break
    C = list(alpha.keys())
    F = []
    O = []
    for i in pointsIndices:
        ch = historicalClusterCenters[historicalClusterAssign[i]]
        if dist(points, i, ch) > rStar or ch not in C:
            for c in C:
                if dist(points, i, c) <= rStar:
                    O.append(i)
                    # i already covered by some preserved hist center
                    continue
            F.append(i)
    G = Carvrstar(points, rStar, k - len(C), seed, F)
    # Equivalent to Carv(F, 2*rStar)
    C += list(G)
    # Compute the new point labels
    closestCenters = findClosestCenter(points, C)
    lC = historicalClusterAssign.copy()
    for x in pointsIndices:
        if x in F or x in O:
            lC[x] = closestCenters[x]
        else:
            lC[x] = C.index(
                historicalClusterCenters[historicalClusterAssign[x]]
            )  # getNewCenterindex
            # Reassign points one by one until we spent all the budget
    canUpgrade = True
    distToCluster = np.array(
        list(
            map(
                lambda x: dist(points, x, C[lC[x]]),
                pointsIndices,
            )
        )
    )
    while (
        clusterDistance(historicalClusterCenters, historicalClusterAssign, C, lC) < b
        and canUpgrade
    ):
        # we can reassign more points
        x = np.argmax(distToCluster)
        closestDistance = dist(points, x, C[closestCenters[x]])
        if closestDistance < distToCluster[x]:
            lC[x] = closestCenters[x]
            distToCluster[x] = closestDistance
        else:
            # furthest point cannot be reassigned
            canUpgrade = False

    return C, lC
