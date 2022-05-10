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


def get_embedding_Kmeans(embedding_exist, corpus, n_clients, bsz=16):
    embedding_data = {}
    corpus_embeddings = []
    if embedding_exist == False:
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda:0')  # server only
        corpus_embeddings = embedder.encode(corpus, show_progress_bar=True,
                                            batch_size=bsz)  # smaller batch size for gpu

        embedding_data['data'] = corpus_embeddings
    else:
        corpus_embeddings = corpus
    ### KMEANS clustering
    print("start Kmeans")
    num_clusters = n_clients
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    print("end Kmeans")
    # TODO: read the center points

    return cluster_assignment, embedding_data


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


def generate_idxes_kmeans(data_file, partition_file, embedding_file, task_type, n_clients, bsz=16):
    with h5py.File(partition_file, "a") as partition:
        if f'/kmeans_clusters={n_clients}' not in partition:
            print("start reading data")
            with h5py.File(data_file, "r") as f:
                attributes = json.loads(f["attributes"][()])
                print(attributes.keys())
                total_index_list = attributes['index_list']
                if "train_index_list" in attributes:
                    train_index_list = attributes['train_index_list']
                    test_index_list = attributes['test_index_list']
                else:
                    split_point = int(len(total_index_list) * 0.8)
                    random.shuffle(total_index_list)
                    train_index_list, test_index_list = total_index_list[:split_point], total_index_list[split_point:]
                    attributes['train_index_list'] = train_index_list
                    attributes['test_index_list'] = test_index_list

                print(len(total_index_list), len(train_index_list), len(test_index_list))

                corpus = []
                if task_type == 'name_entity_recognition':  # specifically wnut and wikiner datesets
                    for i in f['X'].keys():
                        sentence = f['X'][i][()]
                        sentence = [i.decode('UTF-8') for i in sentence]
                        corpus.append(" ".join(sentence))

                elif task_type == 'reading_comprehension':  # specifically Squad1.1 dataset
                    for i in f['context_X'].keys():
                        question_components = []
                        # context = f['context_X'][i][()].decode('UTF-8')
                        question = f['question_X'][i][()].decode('UTF-8')
                        answer_start = f['Y'][i][()][0]
                        answer_end = f['Y'][i][()][1]
                        answer = f['context_X'][i][()].decode('UTF-8')[answer_start: answer_end]

                        question_components.append(question)
                        question_components.append(answer)
                        corpus.append(" ".join(question_components))

                elif task_type == 'sequence_to_sequence':
                    for i in f['Y'].keys():
                        sentence = f['Y'][i][()].decode('UTF-8')
                        corpus.append(sentence)
                else:
                    for i in f['X'].keys():
                        sentence = f['X'][i][()].decode('UTF-8')
                        corpus.append(sentence)

            print("start process embedding data and kmeans partition")
            if os.path.exists(embedding_file) == False:
                cluster_assignment, corpus_embedding = get_embedding_Kmeans(False, corpus, n_clients, bsz)
                embedding_data = {}
                embedding_data['data'] = corpus_embedding
                with open(embedding_file, 'wb') as f:
                    pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(embedding_file, 'rb') as f:
                    embedding_data = pickle.load(f)
                    embedding_data = embedding_data['data']
                    if isinstance(embedding_data, dict):
                        embedding_data = embedding_data['data']
                    cluster_assignment, corpus_embedding = get_embedding_Kmeans(True, embedding_data, n_clients, bsz)

            print("start insert data")
            partition_pkl_train = {}
            partition_pkl_test = {}

            for cluster_id in range(n_clients):
                partition_pkl_train[cluster_id] = []
                partition_pkl_test[cluster_id] = []

            for index in train_index_list:
                idx = cluster_assignment[index]
                if idx in partition_pkl_train:
                    partition_pkl_train[idx].append(index)
                else:
                    partition_pkl_train[idx] = [index]

            for index in test_index_list:
                idx = cluster_assignment[index]
                if idx in partition_pkl_test:
                    partition_pkl_test[idx].append(index)
                else:
                    partition_pkl_test[idx] = [index]

            print("Store kmeans partition to file.")
            partition[f'/kmeans_clusters={n_clients}/n_clients'] = n_clients
            partition[f'/kmeans_clusters={n_clients}/client_assignment'] = cluster_assignment

            for i in sorted(partition_pkl_train.keys()):
                train_path = f'/kmeans_clusters={n_clients}/partition_index/{i}/train/'
                test_path = f'/kmeans_clusters={n_clients}/partition_index/{i}/test/'
                train = partition_pkl_train[i]
                test = partition_pkl_test[i]
                partition[train_path] = train
                partition[test_path] = test

        # partition exists in the file
        train_idxes = []
        test_idxes = []
        for i in range(n_clients):
            train_path = f'/kmeans_clusters={n_clients}/partition_index/{i}/train/'
            test_path = f'/kmeans_clusters={n_clients}/partition_index/{i}/test/'
            train_idxes.append(partition[train_path][()])
            test_idxes.append(partition[test_path][()])

        return train_idxes, test_idxes


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
