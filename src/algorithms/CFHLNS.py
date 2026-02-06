import numpy as np
from helperFunctions import *


def isInBall(points, x, C, R):
    for c in C:
        if dist(points, c, x) <= R:
            return True
    return False


def CFHLNS(
    points, k, b, historicalClusterCenters, historicalClusterAssign, rStar, seed
):
    # Phase 1
    # U is all points of P2 at distance more than 2rStar of a point P1
    pointsIndices = np.arange(len(points))
    U = []
    for x in pointsIndices:
        if isInBall(points, x, historicalClusterCenters, 2 * rStar):
            continue
        U.append(x)
    C2 = []
    while U != []:
        x = U.pop()
        if isInBall(points, x, C2, 2 * rStar):
            continue
        C2.append(x)
    # print(C2)
    if len(C2) > k:
        return (
            historicalClusterCenters,
            historicalClusterAssign,
        )
    # Phase 2
    # compute Fol1(c,2r*)
    # print("Phase 2")
    Fol1 = [0 for _ in range(k)]
    for i in range(len(historicalClusterAssign)):
        assignedCenter = historicalClusterAssign[i]
        if dist(points, i, assignedCenter) <= 2 * rStar:
            Fol1[assignedCenter] += 1

    sortedHistorical = sorted(
        historicalClusterCenters,
        key=lambda x: -Fol1[list(historicalClusterCenters).index(x)],
    )

    T = []
    for i in range(len(historicalClusterCenters)):
        c = sortedHistorical[i]
        if isInBall(points, c, T, 2 * rStar):
            continue
        T.append(c)
    # print("Phase 3")
    for c in T:
        for cprime in sortedHistorical:
            if dist(points, c, cprime) <= rStar:
                C2.append(cprime)
                break
    if len(C2) > k:
        return (
            historicalClusterCenters,
            historicalClusterAssign,
        )
    while len(C2) < k:
        allHistInC2 = True
        for cprime in sortedHistorical:
            if cprime not in C2:
                C2.append(cprime)
                allHistInC2 = False
                break
        if allHistInC2:
            break
    newAssign = []
    # print("Assignement")
    closestCenter = findClosestCenter(points, C2)
    for x in pointsIndices:
        lhx = historicalClusterCenters[historicalClusterAssign[x]]
        if lhx in C2 and dist(points, x, lhx) <= 2 * rStar:
            newAssign.append(C2.index(lhx))
        else:
            newAssign.append(closestCenter[x])
    # Check sanity
    budgetSpent = clusterDistance(
        historicalClusterCenters, historicalClusterAssign, C2, newAssign
    )
    if budgetSpent > b:
        return (
            historicalClusterCenters,
            historicalClusterAssign,
        )
    distToCluster = np.array(
        list(
            map(
                lambda x: dist(points, x, C2[newAssign[x]]),
                pointsIndices,
            )
        )
    )
    canUpgrade = True
    while (
        clusterDistance(
            historicalClusterCenters, historicalClusterAssign, C2, newAssign
        )
        < b
        and canUpgrade
    ):
        # we can reassign more points
        x = np.argmax(distToCluster)
        closestDistance = dist(points, x, C2[closestCenter[x]])
        if closestDistance < distToCluster[x]:
            newAssign[x] = closestCenter[x]
            distToCluster[x] = closestDistance
        else:
            # furthest  point cannot be reassigned, useless to continue
            canUpgrade = False
    return C2, newAssign
