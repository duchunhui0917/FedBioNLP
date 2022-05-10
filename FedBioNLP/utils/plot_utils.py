import os
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
import networkx as nx

base_dir = os.path.expanduser('~/FedBioNLP')


def plot_heatmap(name_list):
    length_dict = {}
    for name in name_list:
        print(name)
        h5_path = os.path.join(base_dir, f'data/{name}_data.h5')
        length_list = []
        with h5py.File(h5_path, 'r+') as hf:
            attributes = json.loads(hf["attributes"][()])
            l = len(attributes['index_list'])
            for i in range(l):
                x = hf[f'/X/{i}'][()]
                x = x.strip().split()
                length_list.append(len(x))
        length_dict[name] = length_list
    x = list(length_dict.keys())
    y = list(length_dict.values())
    plt.boxplot(y)
    plt.xticks(range(1, len(x) + 1), x)
    plt.ylabel('Text Length')
    plt.show()


def plot_class_samples(mtx):
    n, c = mtx.shape[0], mtx.shape[1]
    distributions_tran = mtx.transpose()
    s = 0
    for j in range(c):
        plt.barh(range(n), distributions_tran[j], left=s)
        s += distributions_tran[j]
    plt.xlabel('number of class samples')
    plt.show()


def plot_bandwidth(graph):
    n = graph.number_of_nodes()
    node_names = [graph.nodes[node]['name'] for node in graph.nodes]
    fig, ax = plt.subplots(figsize=(7, 6))
    bandwidths = np.zeros((n, n), dtype=int)

    for (i, j) in graph.edges:
        bandwidths[i][j] = graph[i][j]['bandwidth']
        bandwidths[j][i] = graph[i][j]['bandwidth']
    mask = np.zeros_like(bandwidths)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(bandwidths,
                    xticklabels=node_names,
                    yticklabels=node_names,
                    mask=mask,
                    fmt='d',
                    annot=True,
                    cmap="YlGnBu",
                    cbar=False,
                    # annot_kws={'size': 18}
                    )

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=45)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45)
    plt.savefig('../results/statistics/bandwidths.eps')
    plt.show()


def plot_graph(graph):
    pylab.figure(1, figsize=(7, 6))
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=False, node_size=400, node_color='#2ca02c', alpha=0.8)
    edge_labels = {}
    for edge in graph.edges:
        edge_labels[edge] = graph[edge[0]][edge[1]]['bandwidth']
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    labels = {node: graph.nodes[node]['name'] for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, font_size=10, labels={n: lab for n, lab in labels.items() if n in pos})
    pylab.savefig('../results/statistics/graph.eps')
    pylab.show()

# name_list = ['GAD', 'EU-ADR', 'PGR_Q1', 'CoMAGC']
# name_list = ['i2b2_BIDMC', 'i2b2_Partners']
# name_list = ['AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL']
#
# sta_ht(name_list)
