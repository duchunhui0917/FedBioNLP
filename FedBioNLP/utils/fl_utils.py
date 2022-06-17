import os.path
from matplotlib import pyplot as plt
import math
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import h5py
import pickle
import logging

logger = logging.getLogger(os.path.basename(__file__))


def get_embedding_Kmeans(text, n_clients, path, bsz=16):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info('embedding has been loaded')

    else:
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda:0')
        embeddings = embedder.encode(text, show_progress_bar=True, batch_size=bsz)

        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info('embedding has been dumped')

    print("start kmeans embedding")
    clustering_model = KMeans(n_clusters=n_clients)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment


def cmp_param_inner_prod(param1, param2):
    n_params = len(param1)
    prod = 0
    for i in range(n_params):
        prod12 = float(torch.dot(torch.reshape(param1[i], (-1,)), torch.reshape(param2[i], (-1,))))
        prod1 = float(torch.dot(torch.reshape(param1[i], (-1,)), torch.reshape(param1[i], (-1,))))
        prod2 = float(torch.dot(torch.reshape(param2[i], (-1,)), torch.reshape(param2[i], (-1,))))
        cur_prod = prod12 / (math.sqrt(prod1) * math.sqrt(prod2))
        prod = (prod * i + cur_prod) / (i + 1)
    return prod


def cmp_distribution_inner_prod(distribution1, distribution2):
    prod12 = np.dot(distribution1, distribution2)
    prod1 = np.dot(distribution1, distribution1)
    prod2 = np.dot(distribution2, distribution2)
    prod = prod12 / (np.sqrt(prod1) * np.sqrt(prod2))
    return float(prod)


def sta_param_inner_prod(client_models_list):
    n_iterations = len(client_models_list)
    n_clients = len(client_models_list[0])
    prods = np.zeros((n_iterations, n_clients, n_clients))
    for ite in range(n_iterations):
        client_models = client_models_list[ite]
        params = [list(model.parameters()) for model in client_models]
        for i in range(n_clients):
            for j in range(n_clients):
                prod = cmp_param_inner_prod(params[i], params[j])
                prods[ite][i][j] = prod
    return prods


def sta_param_norm(client_models_list):
    iterations = len(client_models_list)
    n_clients = len(client_models_list[0])
    norms = []
    for ite in range(iterations):
        client_models = client_models_list[ite]
        n_params = len(list(client_models[0].parameters()))
        params = [list(model.parameters()) for model in client_models]
        norm = 0
        count = 0
        for p in range(n_params):
            for i in range(n_clients):
                cur_norm = math.sqrt(
                    float(torch.dot(torch.reshape(params[i][p], (-1,)), torch.reshape(params[i][p], (-1,))))
                )
                norm = (norm * count + cur_norm) / (count + 1)
                count += 1
        norms.append(norm)
    plt.plot(norms)
    plt.xlabel('Iterations')
    plt.ylabel('Norm')
    plt.show()
    return norms


def flat_model(model):
    param = np.array([])
    for p in model.parameters():
        param = np.concatenate([param, p.detach().numpy()], axis=None)
    return param


def generate_idxes_dirichlet(targets, n_clients, n_classes, beta=0.5, seed=None):
    idxes = [[] for _ in range(n_clients)]
    targets = np.array(targets)
    for k in range(n_classes):
        idx_k = np.where(targets == k)[0]
        if seed is not None:
            np.random.seed(seed + k)
        distribution = np.random.dirichlet(np.repeat(beta, n_clients))
        distribution = (np.cumsum(distribution) * idx_k.size).astype(int)[:-1]
        idx_k_split = np.split(idx_k, distribution)
        for i in range(n_clients):
            idxes[i].extend(idx_k_split[i].tolist())
    return idxes


def generate_idxes_group(targets, n_clients, n_classes, beta, n_groups, seed):
    idxes = []
    targets_per_cluster = len(targets) // n_groups
    for i in range(n_groups):
        cur_targets = targets[i * targets_per_cluster:(i + 1) * targets_per_cluster]
        cur_idxes = generate_idxes_dirichlet(cur_targets, n_clients // n_groups, n_classes, beta, seed)
        cur_idxes = [[idx + i * targets_per_cluster for idx in cur_idx] for cur_idx in cur_idxes]
        idxes.extend(cur_idxes)
    cur_targets = targets[(n_groups - 1) * targets_per_cluster:len(targets)]
    cur_idxes = generate_idxes_dirichlet(cur_targets, n_clients % n_groups, n_classes, beta)
    cur_idxes = [[idx + (n_groups - 1) * targets_per_cluster for idx in cur_idx] for cur_idx in cur_idxes]
    idxes.extend(cur_idxes)
    return idxes


class BatchIterator:
    def __init__(self, n, x):
        self.idx = 0
        self.n = n
        self.x = x
        self.ite_x = iter(x)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.n:
            try:
                res = next(self.ite_x)
            except StopIteration:
                self.ite_x = iter(self.x)
                res = next(self.ite_x)
            self.idx += 1
            return res
        else:
            raise StopIteration

    def __len__(self):
        return self.n
