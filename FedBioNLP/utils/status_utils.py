import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from .MMD import mmd
from .CKA import linear_CKA
import torch
import os

logger = logging.getLogger(os.path.basename(__file__))


def process_feature(feature, mode):
    if len(feature.shape) == 3:
        if mode == 'average':
            x = np.mean(feature, axis=1)
        elif mode == 'first':
            x = feature[:, 0, :]
        elif mode == 'last':
            x = feature[:, -1, :]
        elif mode == 'first_last':
            x = np.concatenate([feature[:, 0, :], feature[:, -1, :]], axis=1)
        elif mode == 'squeeze':
            x = feature.reshape(feature.shape[0], -1)
        else:
            raise NotImplementedError
    else:
        x = feature
    return x


def cmp_cosine_euclidean(features, pca=False, mode='average'):
    n = len(features)
    for i in range(n):
        f1 = features[i]
        for j in range(n):
            if i < j:
                logger.info(f'client {i} and {j}')
                f2 = features[j]
                for layer, (feature1, feature2) in enumerate(zip(f1, f2)):
                    x1, x2 = process_feature(feature1, mode), process_feature(feature2, mode)
                    if pca:
                        pca = PCA(n_components=2)
                        x1 = pca.fit_transform(x1)
                        print(pca.explained_variance_ratio_)

                        pca = PCA(n_components=2)
                        x2 = pca.fit_transform(x2)
                        print(pca.explained_variance_ratio_)

                    MMD = mmd(x1, x2)
                    x1_mean = x1.mean(axis=0, keepdims=True)
                    x2_mean = x2.mean(axis=0, keepdims=True)
                    cosine_sim_mean = cosine_similarity(x1_mean, x2_mean)[0][0]
                    euclidean_dist_mean = pairwise_distances(x1_mean, x2_mean)[0][0]

                    cosine_sim = cosine_similarity(x1, x2).mean()
                    euclidean_dist = pairwise_distances(x1, x2).mean()

                    logger.info(
                        f'layer: {layer}, '
                        f'cosine_sim_mean: {cosine_sim_mean:.4f}, euclidean_dist_mean: {euclidean_dist_mean:.4f}, '
                        f'cosine_sim: {cosine_sim:.4f}, euclidean_dist: {euclidean_dist:.4f}, '
                        f'MMD: {MMD:.4f}'
                    )


def cmp_CKA_sim(features1, features2, mode=None):
    for layer, (feature1, feature2) in enumerate(zip(features1, features2)):
        x1, x2 = process_feature(feature1, mode), process_feature(feature2, mode)
        metric = linear_CKA(x1, x2)
        logging.info(f'layer{layer}, shape1: {feature1.shape}, shape2: {feature2.shape}, metric: {metric:.4f}')


def cmp_kmeans_sim(features1, features2, pca=False, mode=None):
    for layer, (feature1, feature2) in enumerate(zip(features1, features2)):
        x = np.concatenate([feature1, feature2])
        x = process_feature(x, mode)
        if pca:
            pca = PCA(n_components=10)
            x = pca.fit_transform(x)

        y_true = np.concatenate([np.zeros(feature1.shape[0]), np.ones(feature2.shape[0])])

        # kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans = GaussianMixture(n_components=2)
        kmeans.fit(x)
        y_pred = kmeans.predict(x)
        metric = rand_score(y_true, y_pred)
        logging.info(f'layer{layer}, shape1: {feature1.shape}, shape2: {feature2.shape}, metric: {metric:.4f}')


def visualize_features(features, labels, style=None, mode=None):
    for layer, feature in enumerate(features):
        x = process_feature(feature, mode)
        X_features = TSNE(n_components=2, random_state=33).fit_transform(x)

        palette = sns.color_palette("bright", n_colors=np.unique(labels).shape[0])
        if style is not None:
            sns.scatterplot(X_features[:, 0], X_features[:, 1], hue=labels, style=style, palette=palette, s=8)
        else:
            sns.scatterplot(X_features[:, 0], X_features[:, 1], hue=labels, palette=palette, s=8)
        plt.title(f'layer {layer}')
        plt.show()


def param_l2_norm(model1, model2):
    l2_norms = []
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        l2_norm = torch.norm(param1 - param2, 2).item()
        l2_norms.append(l2_norm)
    l2_norm = sum(l2_norms) / len(l2_norms)
    return l2_norm


def tensor_cos_sim(tensor1, tensor2):
    tensor1, tensor2 = torch.flatten(tensor1), torch.flatten(tensor2)
    # cosine_sim = F.cosine_similarity(tensor1, tensor2).item()
    sim11 = torch.dot(tensor1, tensor1)
    sim22 = torch.dot(tensor2, tensor2)
    sim12 = torch.dot(tensor1, tensor2)
    if sim11 != 0 and sim22 != 0:
        # cosine_sim = sim12
        # cosine_sim = sim12 / torch.sqrt(sim11 * sim22)
        # cosine_sim = torch.norm(tensor1, 2)
        cosine_sim = torch.norm(tensor1 - tensor2, 2)
    else:
        cosine_sim = 0
    return cosine_sim


def distribution_cosine(dist1, dist2):
    pass


def status_mtx(datasets, n_classes):
    np.set_printoptions(precision=2)
    n = len(datasets)
    mtx = np.zeros((n, n_classes), dtype=np.int)
    mtx_ = np.zeros((n, n_classes), dtype=np.float)
    for i in range(n):
        for data, label in datasets[i]:
            mtx[i][label[0]] += 1
        mtx_[i][:] = mtx[i][:] / sum(mtx[i][:])
    return mtx, mtx_
