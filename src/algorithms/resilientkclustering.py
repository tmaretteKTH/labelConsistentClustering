import numpy as np
import networkx as nx
from helperFunctions import dist
from algorithms.FFT import FFT
from data import *


def log(b, x):
    return np.log2(x) / np.log2(b)


def resilientMST(G, l, rng):
    Gprime = G.copy()
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        alpha = rng.random()
        if w == 0:
            Gprime.add_edge(u, v, weight=w)
        else:
            i = np.floor(alpha + log(l, w))
            Gprime.add_edge(u, v, weight=l ** (i - alpha))
    T = nx.minimum_spanning_tree(Gprime)
    return T


def resilientkcenter(points, k, epsilon, l, alpha, beta, seed):
    rng = np.random.RandomState(seed)
    # will open (alpha + beta) k clusters
    n = len(points)
    P = np.arange(n)
    C = P[rng.choice(n, int(alpha * k), replace=False)]
    H = nx.Graph()
    H.add_nodes_from(P)
    for p in P:
        if p in C:
            for q in P:
                if q == p:
                    continue
                if q in C:
                    H.add_edge(p, q, weight=0)
                else:
                    H.add_edge(p, q, weight=dist(points, p, q))
    T = resilientMST(H, l, rng)
    sigma = np.zeros(len(P.copy()), dtype=int)
    for x, y in T.edges:
        if x in C and y in C:
            (assigny,) = np.where(C == y)[0]
            (assignx,) = np.where(C == x)[0]
            sigma[x] = assignx
            sigma[y] = assigny
        if y in C and x not in C:
            (assign,) = np.where(C == y)[0]
            sigma[x] = assign
        else:
            assign = np.where(C == x)[0]
            sigma[y] = assign
    E = sorted(T.edges(data=True), key=lambda e: e[2]["weight"])[
        int(1 - np.ceil(epsilon * n)) :
    ]  # indices of points with high clustering cost
    L = set()
    for p, q, d in E:
        L.add(p)
        L.add(q)
    L = list(L)
    Cprime = FFT(points, int(beta * k))
    for v in L:
        distances = np.array(list(map(lambda c: dist(points, v, c), Cprime)))
        sigma[v] = int(len(C) + np.argmin(distances))
    return (list(C) + list(Cprime)), sigma
