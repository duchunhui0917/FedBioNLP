import numpy as np


def dependency_to_matrix(dependency, sequence_length):
    ls = eval(dependency)
    dep_matrix = np.zeros((sequence_length, sequence_length), dtype=np.int64)
    for (_, governor, dependent) in ls:
        governor -= 1
        dependent -= 1
        dep_matrix[governor][dependent] = 1
        dep_matrix[dependent][governor] = 1
    return dep_matrix


def get_self_loop_dep_matrix(dep_matrix, n):
    res = np.zeros_like(dep_matrix)
    for i in range(n):
        res[i][i] = 1
    return res


def get_local_dep_matrix(dep_matrix, pos1, pos2):
    res = np.zeros_like(dep_matrix)
    res[pos1, :] = dep_matrix[pos1, :]
    res[:, pos1] = dep_matrix[:, pos1]

    res[pos2, :] = dep_matrix[pos2, :]
    res[:, pos2] = dep_matrix[:, pos2]
    return res


def get_sdp(dep_matrix, pos1, pos2):
    n = len(dep_matrix)

    dist = [float('inf') for _ in range(n)]
    dist[pos1] = 0
    pre = [-1 for _ in range(n)]
    mark = [False for _ in range(n)]
    mark[pos1] = True
    node = pos1
    while True:
        for i in range(n):
            if dep_matrix[node][i] > 0 and dist[node] + dep_matrix[node][i] < dist[i]:
                dist[i] = dist[node] + dep_matrix[node][i]
                pre[i] = node
        min_dist = float('inf')
        node = -1
        for i in range(n):
            if dist[i] < min_dist and not mark[i]:
                min_dist = dist[i]
                node = i
        if node == -1:
            break
        mark[node] = True

    v = pos2

    sdp = [v]
    while True:
        u = pre[v]
        sdp.insert(0, u)
        if u == -1:
            return None
        v = u
        if u == pos1:
            return sdp


def get_subtree_dep_matrix(dep_matrix, sdp, K_LCA):
    res = np.zeros_like(dep_matrix)
    n = len(dep_matrix)

    if K_LCA < 0:
        return res, set()

    for i in range(len(sdp) - 1):
        u, v = sdp[i], sdp[i + 1]
        res[u, v] = 1
        res[v, u] = 1
    for k in range(K_LCA):
        cur_res = res.copy()
        for i in range(n):
            for j in range(i, n):
                if cur_res[i][j] == 1:
                    res[i, :] = dep_matrix[i, :]
                    res[:, i] = dep_matrix[:, i]

                    res[j, :] = dep_matrix[j, :]
                    res[:, j] = dep_matrix[:, j]
    subtree = np.where(res == 1)
    subtree = np.array(subtree).flatten()
    subtree = set(subtree)
    return res, subtree
