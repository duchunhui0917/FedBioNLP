import numpy as np
import networkx as nx


def get_fix_consensus_matrix(V):
    n = len(V)
    alpha = 1 / n
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i][j] = V[i][j] * alpha
            else:
                L[i][j] = 1 - np.sum(V[i][:]) * alpha
    return L


def get_fix_spectral_gap(V):
    L = get_fix_consensus_matrix(V)
    spectral_values = np.linalg.eig(L)[0]
    spectral_values.sort()
    return float(1 - spectral_values[-2])


def get_degree_consensus_matrix(V):
    n = len(V)
    for i in range(n):
        V[i][i] = 1
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                alpha = 1 / max(np.sum(V[i][:]), np.sum(V[:][j]))
                L[i][j] = V[i][j] * alpha
    for i in range(n):
        L[i][i] = 1 - np.sum(L[i][:])
    return L


def get_degree_spectral_gap(V):
    L = get_degree_consensus_matrix(V)
    spectral_values = np.linalg.eig(L)[0]
    spectral_values.sort()
    return float(1 - spectral_values[-2])


def get_laplacian_spectral_gap(V):
    G = nx.Graph(V)
    spectral_values = np.sort(nx.linalg.spectrum.laplacian_spectrum(G))[::-1]
    spectral_gap = spectral_values[-2] / spectral_values[0]
    return float(spectral_gap)


def init_mst(digraph):
    graph = nx.Graph(digraph)
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['cap'] = min(digraph[edge[0]][edge[1]]['cap'], digraph[edge[1]][edge[0]]['cap'])
    mst = nx.maximum_spanning_edges(graph, weight='cap', data=False)
    pairs = list(mst)
    for i, j in pairs:
        if (j, i) not in pairs:
            pairs.append((j, i))
    return pairs


def init_overlay(G):
    pairs = list(nx.edges(G))
    for i, j in pairs:
        if (j, i) not in pairs:
            pairs.append((j, i))

    return pairs


def pairs2matrix(pairs, n):
    V = np.zeros((n, n))
    for (i, j) in pairs:
        V[i][j] = 1
        V[j][i] = 1
    return V


def pairs2graph(pairs):
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    return graph


def get_value(t, spectral_gap):
    return t / spectral_gap
