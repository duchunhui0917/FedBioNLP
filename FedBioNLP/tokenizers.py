import logging
import random
import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os

logger = logging.getLogger(os.path.basename(__file__))


def re_tokenizer(args, model_name, max_seq_length):
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dep_e_texts = args['dep_e_text']
    args.update({
        "input_ids": [],
        "input_mask": [],
        "e1_mask": [],
        "e2_mask": []
    })

    t = tqdm(dep_e_texts)
    for dep_e_text in t:
        dep_e_ls = dep_e_text.split(' ')

        # add [CLS] token
        tokens = ["[CLS]"]
        e1_mask = [0]
        e2_mask = [0]
        e1_mask_val = 0
        e2_mask_val = 0
        for i, word in enumerate(dep_e_ls):
            if len(tokens) >= max_seq_length - 1:
                break
            if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                if word in ["<e1>"]:
                    e1_mask_val = 1
                elif word in ["</e1>"]:
                    e1_mask_val = 0
                if word in ["<e2>"]:
                    e2_mask_val = 1
                elif word in ["</e2>"]:
                    e2_mask_val = 0
                continue

            token = tokenizer.tokenize(word)
            if len(tokens) + len(token) > max_seq_length - 1:
                break
            tokens.extend(token)
            e1_mask.extend([e1_mask_val] * len(token))
            e2_mask.extend([e2_mask_val] * len(token))

        # add [SEP] token
        tokens.append("[SEP]")
        e1_mask.append(0)
        e2_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        e1_mask += padding
        e2_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(e1_mask) == max_seq_length
        assert len(e2_mask) == max_seq_length

        args["input_ids"].append(input_ids)
        args["input_mask"].append(input_mask)
        args["e1_mask"].append(e1_mask)
        args["e2_mask"].append(e2_mask)

    logger.info('tokenizing finished')

    return args


def dependency_to_matrix(dependency, max_seq_length):
    ls = eval(dependency)
    dep_matrix = np.zeros((max_seq_length, max_seq_length), dtype=np.int64)
    for (_, governor, dependent) in ls:
        governor -= 1
        dependent -= 1
        dep_matrix[governor][dependent] = 1
        dep_matrix[dependent][governor] = 1
    return dep_matrix


def add_self_loop_dep_matrix(dep_matrix, n):
    for i in range(n):
        dep_matrix[i][i] = 1
    return dep_matrix


def get_local_dep_matrix(dep_matrix, pos1, pos2):
    res = np.zeros_like(dep_matrix)
    res[pos1, :] = dep_matrix[pos1, :]
    res[:, pos1] = dep_matrix[:, pos1]

    res[pos2, :] = dep_matrix[pos2, :]
    res[:, pos2] = dep_matrix[:, pos2]
    return res


def get_sdp_dep_matrix(dep_matrix, pos1, pos2):
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

    res = np.zeros_like(dep_matrix)
    v = pos2

    len_sdp = 0
    while True:
        u = pre[v]
        if u == -1:
            return dep_matrix, None
        res[u][v] = 1
        res[v][u] = 1
        v = u
        len_sdp += 1
        if u == pos1:
            return res, len_sdp


def re_dep_tokenizer(args, model_name, max_seq_length, pruned_dep=False):
    """
    org_text: 'A mechanic tightens the bolt with a spanner .'
    deps: [('ROOT', 0, 3), ('det', 2, 1), ('nsubj', 3, 2), ('det', 5, 4), ('obj', 3, 5), ('case', 8, 6), ('det', 8, 7),
     ('obl', 3, 8), ('punct', 3, 9)]
    text: 'A <e1> mechanic </e1> tightens the bolt with a <e2> spanner </e2> .'
    tokens: ['CLS', 'A', 'mechanic', 'tighten', '##s', 'the', 'bolt', 'with', 'a', 'spanner', '.', 'SEP', 'PAD']
    input_mask:
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    valid_mask:
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    e1_mask:
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    e2_mask:
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


    Args:
        pruned_dep:
        args:
        model_name:
        max_seq_length:

    Returns:

    """
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dep_e_texts = args['dep_e_text']
    dependencies = args['dependency']
    args.update({
        "input_ids": [],
        "input_mask": [],
        "mlm_input_ids": [],
        "valid_ids": [],
        "e1_mask": [],
        "e2_mask": [],
        "dep_matrix": [],
        "len_sdp": []
    })

    ls = [(x, y) for x, y in zip(dep_e_texts, dependencies)]
    t = tqdm(ls)
    for (dep_e_text, dependency) in t:
        dep_e_ls = dep_e_text.split(' ')
        dep_matrix = dependency_to_matrix(dependency, max_seq_length)

        # add [CLS] token
        tokens = ["[CLS]"]
        mlm_tokens = ["[CLS]"]
        valid = [0]
        e1_mask = []
        e2_mask = []
        e1_mask_val = 0
        e2_mask_val = 0
        e1_start = 0
        e2_start = 0
        for i, word in enumerate(dep_e_ls):
            if len(tokens) >= max_seq_length - 1:
                break
            if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
                if word in ["<e1>"]:
                    e1_mask_val = 1
                    e1_start = len(e1_mask)
                elif word in ["</e1>"]:
                    e1_mask_val = 0
                if word in ["<e2>"]:
                    e2_mask_val = 1
                    e2_start = len(e2_mask)
                elif word in ["</e2>"]:
                    e2_mask_val = 0
                continue

            token = tokenizer.tokenize(word)
            mlm_token = token[:]
            for t in range(len(mlm_token)):
                if random.random() < 0.15:
                    mlm_token[t] = '[MASK]'

            if len(tokens) + len(token) > max_seq_length - 1:
                break
            tokens.extend(token)
            mlm_tokens.extend(mlm_token)
            e1_mask.append(e1_mask_val)
            e2_mask.append(e2_mask_val)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                else:
                    valid.append(0)

        # add [SEP] token
        tokens.append("[SEP]")
        mlm_tokens.append("[SEP]")
        valid.append(0)
        e1_mask.append(0)
        e2_mask.append(0)

        # convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        mlm_input_ids = tokenizer.convert_tokens_to_ids(mlm_tokens)
        input_mask = [1] * len(input_ids)

        # zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        mlm_input_ids += padding
        input_mask += padding
        valid += padding
        e1_mask += [0] * (max_seq_length - len(e1_mask))
        e2_mask += [0] * (max_seq_length - len(e2_mask))

        mlm_input_ids = [-100 if x != tokenizer.mask_token_id else x for x in mlm_input_ids]
        # process pruned dependency matrix
        local_dep_matrix = get_local_dep_matrix(dep_matrix, e1_start, e2_start)
        sdp_dep_matrix, len_sdp = get_sdp_dep_matrix(dep_matrix, e1_start, e2_start)
        if pruned_dep:
            dep_matrix = local_dep_matrix | sdp_dep_matrix
        dep_matrix = add_self_loop_dep_matrix(dep_matrix, sum(valid))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(mlm_input_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(e1_mask) == max_seq_length
        assert len(e2_mask) == max_seq_length

        args["input_ids"].append(input_ids)
        args["input_mask"].append(input_mask)
        args["mlm_input_ids"].append(mlm_input_ids)
        args["valid_ids"].append(valid)
        args["e1_mask"].append(e1_mask)
        args["e2_mask"].append(e2_mask)
        args["dep_matrix"].append(dep_matrix)
        args["len_sdp"].append(len_sdp)

    logger.info('tokenizing finished')

    return args
