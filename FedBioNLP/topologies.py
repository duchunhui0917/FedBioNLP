import networkx as nx
import random
import numpy as np
from FedBioNLP import pairs2matrix, get_degree_spectral_gap
from FedBioNLP import plot_bandwidth

random.seed(66)

pairs = [(0, 1), (0, 2), (0, 3)]
V = pairs2matrix(pairs, 4)
print(get_degree_spectral_gap(V))


def simple_graph():
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(6)))
    graph.add_edges_from([
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
        (4, 5),
        (5, 4)
    ])
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['cap'] = 1
    return graph


def bad_graph(k=3):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(k * (2 * k + 3) + k + 1)))
    for kk in range(k):
        for i in range(2 * k + 2):
            graph.add_edge(kk * (2 * k + 3) + i, kk * (2 * k + 3) + i + 1)
    for kk in range(k):
        graph.add_edge(kk * (2 * k + 3) + 1, (kk + 1) * (2 * k + 3))
    graph.add_edge(k * (2 * k + 3) + 1, 0)
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['cap'] = random.randint(2, 3)
    return graph


def regular_pairs(n, d, idxes):
    pairs = []
    for i in range(n):
        for j in range(k // 2):
            pairs.append((i, (i + j + 1) % n))
            pairs.append((i, (i + j - 1) % n))
    return pairs


def random_regular_graph(n, d):
    graph = nx.random_regular_graph(d, n)
    return graph


def cluster_regular_graph(n, d, idxes):
    edges = []
    for k in range(d):
        flags = [False for _ in range(n)]
        for i in range(n):
            if not flags[i]:
                j = (i + 1 + d) % n
                edges.append((idxes[i], idxes[j]))
                flags[i], flags[j] = True, True
    graph = nx.Graph()
    graph.add_nodes_from(list(range(n)))
    graph.add_edges_from(edges)
    return graph


def random_ring_graph(n):
    idxes = list(range(n))
    random.shuffle(idxes)
    pairs = [(idxes[i], idxes[(i + 1) % n]) for i in range(n)]

    graph = nx.Graph()
    graph.add_nodes_from(list(range(n)))
    graph.add_edges_from(pairs)
    return graph


def cluster_ring_graph(n, cluster_labels):
    idxes = [np.where(cluster_labels == i)[0].tolist() for i in range(n)]
    idx_for_cluster = [c[i] for i in range(n) for c in idxes if i < len(c)]
    pairs = [(idx_for_cluster[i], idx_for_cluster[(i + 1) % n]) for i in range(n)]

    graph = nx.Graph()
    graph.add_nodes_from(list(range(n)))
    graph.add_edges_from(pairs)
    return graph


def star_pairs(n):
    pairs = [(0, i) for i in range(1, n)]
    return pairs


def line_pairs(n):
    pairs = [(i, i + 1) for i in range(n - 1)]
    return pairs


def fractional_graph():
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(20)))
    graph.add_edges_from([
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (0, 4),
        (2, 6),
        (3, 10),
        (5, 12),
        (7, 14),
        (9, 16),
        (13, 17),
        (15, 19)
    ])
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['cap'] = random.randint(10, 100) + random.random()

    return graph


def nsf_graph():
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(24)))
    graph.add_edges_from([
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 5),
        (5, 3),
        (3, 9),
        (9, 3),
        (3, 10),
        (10, 3),
        (4, 5),
        (5, 4),
        (4, 6),
        (6, 4),
        (4, 7),
        (7, 4),
        (5, 6),
        (6, 5),
        (5, 8),
        (8, 5),
        (5, 10),
        (10, 5),
        (6, 7),
        (7, 6),
        (7, 8),
        (8, 7),
        (8, 11),
        (11, 8),
        (9, 10),
        (10, 9),
        (9, 13),
        (13, 9),
        (9, 15),
        (15, 9),
        (9, 17),
        (17, 9),
        (10, 11),
        (11, 10),
        (10, 13),
        (13, 10),
        (11, 12),
        (12, 11),
        (11, 14),
        (14, 11),
        (12, 14),
        (14, 12),
        (12, 22),
        (22, 12),
        (13, 14),
        (14, 13),
        (13, 18),
        (18, 13),
        (14, 20),
        (20, 14),
        (15, 16),
        (16, 15),
        (16, 17),
        (17, 16),
        (16, 0),
        (0, 16),
        (17, 18),
        (18, 17),
        (18, 19),
        (19, 18),
        (18, 0),
        (0, 18),
        (19, 20),
        (20, 19),
        (19, 21),
        (21, 19),
        (20, 21),
        (21, 20),
        (20, 22),
        (22, 20),
        (21, 23),
        (23, 21),
        (22, 23),
        (23, 22)
    ])
    for edge in graph.edges:
        graph[edge[0]][edge[1]]['cap'] = random.randint(10, 100) + random.random()
    return graph


def customize_graph(pairs):
    graph = nx.Graph()
    graph.add_edges_from(pairs)
    return graph


def real_graph():
    node_names = ['Virginia',
                  'California',
                  'Oregon',
                  'Ireland',
                  'Frankfurt',
                  'Tokyo',
                  'Seoul',
                  'Singapore',
                  'Sydney',
                  'Mumbai',
                  'San Paulo']
    nodes = [(i, {'name': name}) for i, name in enumerate(node_names)]
    edges = [
        (0, 1, {'bandwidth': 260}),
        (0, 2, {'bandwidth': 280}),
        (0, 3, {'bandwidth': 180}),
        (0, 4, {'bandwidth': 80}),
        (0, 5, {'bandwidth': 60}),
        (0, 6, {'bandwidth': 50}),
        (0, 7, {'bandwidth': 30}),
        (0, 8, {'bandwidth': 40}),
        (0, 9, {'bandwidth': 40}),
        (0, 10, {'bandwidth': 140}),
        (1, 2, {'bandwidth': 320}),
        (1, 3, {'bandwidth': 15}),
        (1, 4, {'bandwidth': 40}),
        (1, 5, {'bandwidth': 40}),
        (1, 6, {'bandwidth': 55}),
        (1, 7, {'bandwidth': 35}),
        (1, 8, {'bandwidth': 62}),
        (1, 9, {'bandwidth': 32}),
        (1, 10, {'bandwidth': 83}),
        (2, 3, {'bandwidth': 42}),
        (2, 4, {'bandwidth': 56}),
        (2, 5, {'bandwidth': 40}),
        (2, 6, {'bandwidth': 88}),
        (2, 7, {'bandwidth': 70}),
        (2, 8, {'bandwidth': 62}),
        (2, 9, {'bandwidth': 42}),
        (2, 10, {'bandwidth': 73}),
        (3, 4, {'bandwidth': 260}),
        (3, 5, {'bandwidth': 16}),
        (3, 6, {'bandwidth': 46}),
        (3, 7, {'bandwidth': 18}),
        (3, 8, {'bandwidth': 60}),
        (3, 9, {'bandwidth': 72}),
        (3, 10, {'bandwidth': 42}),
        (4, 5, {'bandwidth': 32}),
        (4, 6, {'bandwidth': 32}),
        (4, 7, {'bandwidth': 60}),
        (4, 8, {'bandwidth': 11}),
        (4, 9, {'bandwidth': 52}),
        (4, 10, {'bandwidth': 42}),
        (5, 6, {'bandwidth': 360}),
        (5, 7, {'bandwidth': 260}),
        (5, 8, {'bandwidth': 130}),
        (5, 9, {'bandwidth': 130}),
        (5, 10, {'bandwidth': 36}),
        (6, 7, {'bandwidth': 225}),
        (6, 8, {'bandwidth': 125}),
        (6, 9, {'bandwidth': 163}),
        (6, 10, {'bandwidth': 38}),
        (7, 8, {'bandwidth': 138}),
        (7, 9, {'bandwidth': 189}),
        (7, 10, {'bandwidth': 20}),
        (8, 9, {'bandwidth': 170}),
        (8, 10, {'bandwidth': 25}),
        (9, 10, {'bandwidth': 52}),
    ]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def real_example_graph():
    node_names = ['Virginia',
                  'Frankfurt',
                  'Singapore',
                  'San Paulo']
    nodes = [(i, {'name': name}) for i, name in enumerate(node_names)]
    edges = [
        (0, 1, {'bandwidth': 80}),
        (0, 2, {'bandwidth': 30}),
        (0, 3, {'bandwidth': 140}),
        (1, 2, {'bandwidth': 60}),
        (1, 3, {'bandwidth': 42}),
        (2, 3, {'bandwidth': 20}),
    ]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


graph1 = real_graph()
plot_bandwidth(graph1)
