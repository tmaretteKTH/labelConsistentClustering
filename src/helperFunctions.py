import numpy as np


def dist(points, x, y):
    return np.linalg.norm(points[x] - points[y])


def randomClustering(points, k, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(np.arange(len(points)), k)


def findClosestCenter(points, clusterCenters):
    n = len(points)
    closestClusters = np.zeros(n, dtype=int)
    for x in range(n):
        distances = np.array(list(map(lambda c: dist(points, x, c), clusterCenters)))
        closestClusters[x] = np.argmin(distances)
    return closestClusters


def clusteringScore(points, clusterCenters, clusterAssign=[]):
    if len(clusterAssign) == 0:
        clusterAssign = findClosestCenter(points, clusterCenters)
    return max(
        [dist(points, x, clusterCenters[clusterAssign[x]]) for x in range(len(points))]
    )


def clusterDistance(
    historicalClusterCenters,
    historicalClusterAssign,
    algClusterCenters,
    algClusterAssign,
):
    points = np.arange(len(historicalClusterAssign))
    historicalSet = {
        (x, historicalClusterCenters[historicalClusterAssign[x]]) for x in points
    }
    algSet = {(x, algClusterCenters[algClusterAssign[x]]) for x in points}
    # If a point change assignment, it will be counted twice
    return len(historicalSet ^ algSet) / 2
