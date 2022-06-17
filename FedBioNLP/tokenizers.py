import logging
import random
import string
import time

import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
from .utils.dependency_parsing import *
import json
import copy

logger = logging.getLogger(os.path.basename(__file__))


def add_space_for_punctuation(s):
    punctuations = string.punctuation
    ns = copy.deepcopy(s)
    j = 0
    for i, x in enumerate(s):
        if x in punctuations:
            ns = ns[:j] + ' ' + ns[j] + ' ' + ns[j + 1:]
            j += 2
        j += 1
    return ns


def re_tokenizer(args, model_name='distilbert-base-cased'):
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts = args['text']
    args.update({
        "input_ids": [],
        "attention_mask": [],
        "e1_mask": [],
        "e2_mask": []
    })

    t = tqdm(texts)
    for text in t:
        text_ls = text.split(' ')

        # add [CLS] token
        tokens = ["[CLS]"]
        e1_mask = [0]
        e2_mask = [0]
        e1_mask_val = 0
        e2_mask_val = 0
        for i, word in enumerate(text_ls):
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
            mlm_token = token[:]
            for t in range(len(mlm_token)):
                if random.random() < 0.15:
                    mlm_token[t] = '[MASK]'

            tokens.extend(token)
            e1_mask.extend([e1_mask_val] * len(token))
            e2_mask.extend([e2_mask_val] * len(token))

        # add [SEP] token
        tokens.append("[SEP]")
        e1_mask.append(0)
        e2_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        args["input_ids"].append(input_ids)
        args["attention_mask"].append(attention_mask)
        args["e1_mask"].append(e1_mask)
        args["e2_mask"].append(e2_mask)

    max_length = max([len(token) for token in args["input_ids"]])
    logger.info(f'max sequence length: {max_length}')
    ls = zip(args["input_ids"], args["attention_mask"], args["e1_mask"], args["e2_mask"])

    for i, (input_ids, attention_mask, e1_mask, e2_mask) in enumerate(ls):
        # zero-pad up to the sequence length
        padding = [0] * (max_length - len(input_ids))
        args['input_ids'][i] = input_ids + padding
        args['attention_mask'][i] = attention_mask + padding
        args['e1_mask'][i] = e1_mask + padding
        args['e2_mask'][i] = e2_mask + padding

        assert len(args['input_ids'][i]) == max_length
        assert len(args['attention_mask'][i]) == max_length
        assert len(args['e1_mask'][i]) == max_length
        assert len(args['e2_mask'][i]) == max_length

    logger.info('tokenizing finished')

    return args


def re_dep_tokenizer(args, model_name='distilbert-base-cased', mlm_method='None', mlm_prob=0.15, K_LCA=1):
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
        K_LCA:
        mlm_prob:
        mlm_method:
        args:
        model_name:

    Returns:

    """
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dep_e_texts = args['dep_e_text']
    dependencies = args['dependency']
    args.update({
        'tokens': [],
        'input_ids': [],
        'attention_mask': [],
        'mlm_input_ids': [],
        'mlm_labels': [],
        'valid_ids': [],
        'e1_mask': [],
        'e2_mask': [],
        'dep_matrix': []
    })
    model_max_length = tokenizer.model_max_length
    cls_id, mask_id, sep_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[MASK]', '[SEP]'])

    ratio_subtree = []
    ratio_mask = []
    ls = [(x, y) for x, y in zip(dep_e_texts, dependencies)]
    t = tqdm(ls)
    for (dep_e_text, dependency) in t:
        dep_e_ls = dep_e_text.split(' ')

        # add [CLS] token
        tokens = ['[CLS]']
        valid_ids = [0]
        e1_mask = []
        e2_mask = []
        e1_mask_val = 0
        e2_mask_val = 0
        e1_start = 0
        e2_start = 0
        for i, word in enumerate(dep_e_ls):
            if word in ['<e1>', '</e1>', '<e2>', '</e2>']:
                if word in ['<e1>']:
                    e1_mask_val = 1
                    e1_start = len(e1_mask)
                elif word in ['</e1>']:
                    e1_mask_val = 0
                if word in ['<e2>']:
                    e2_mask_val = 1
                    e2_start = len(e2_mask)
                elif word in ['</e2>']:
                    e2_mask_val = 0
                continue

            token = tokenizer.tokenize(word)

            if len(tokens) + len(token) >= model_max_length:
                break
            tokens.extend(token)
            e1_mask.append(e1_mask_val)
            e2_mask.append(e2_mask_val)
            for m in range(len(token)):
                if m == 0:
                    valid_ids.append(1)
                else:
                    valid_ids.append(0)

        # add [SEP] token
        tokens.append('[SEP]')
        valid_ids.append(0)
        e1_mask.append(0)
        e2_mask.append(0)

        # convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        args['tokens'].append(tokens)
        args['input_ids'].append(input_ids)
        args['attention_mask'].append(attention_mask)
        args['valid_ids'].append(valid_ids)
        args['e1_mask'].append(e1_mask)
        args['e2_mask'].append(e2_mask)

        # prun dependency matrix
        valid_seq_length = sum(valid_ids)
        assert e1_start < valid_seq_length
        assert e2_start < valid_seq_length
        dep_matrix = dependency_to_matrix(dependency, valid_seq_length)
        sdp = get_sdp(dep_matrix, e1_start, e2_start)
        self_loop_dep_matrix = get_self_loop_dep_matrix(dep_matrix, valid_seq_length)
        subtree_dep_matrix, subtree = get_subtree_dep_matrix(dep_matrix, sdp, K_LCA=K_LCA)
        dep_matrix = self_loop_dep_matrix | subtree_dep_matrix
        args['dep_matrix'].append(dep_matrix)

        mlm_input_ids, mlm_labels = input_ids[:], input_ids[:]
        num_mask = 0
        if mlm_method == 'subtree':
            node = 0
            mask = False
            for i, x in enumerate(mlm_input_ids):
                if valid_ids[i] == 1:
                    if node in subtree:
                        mask = True
                    else:
                        mask = False
                    node += 1
                if mask and random.random() < mlm_prob and x != cls_id and x != sep_id:
                    mlm_input_ids[i] = mask_id
                    num_mask += 1
                else:
                    mlm_labels[i] = -100
        elif mlm_method == 'sentence':
            mlm_input_ids = input_ids[:]
            excluded_nodes = random.sample(list(range(valid_seq_length)), k=len(subtree))
            for i, x in enumerate(mlm_input_ids):
                if i not in excluded_nodes and random.random() < mlm_prob and x != cls_id and x != sep_id:
                    mlm_input_ids[i] = mask_id
                    num_mask += 1
                else:
                    mlm_labels[i] = -100
        else:
            mlm_labels = [-100] * len(input_ids)
        args['mlm_input_ids'].append(mlm_input_ids)
        args['mlm_labels'].append(mlm_labels)

        ratio_subtree.append(len(subtree) / valid_seq_length)
        ratio_mask.append(num_mask / len(input_ids))
    max_length = max([len(token) for token in args["input_ids"]])
    logger.info(f'max sequence length: {max_length}')
    ratio_subtree = sum(ratio_subtree) / len(ratio_subtree)
    logger.info(f'ratio of subtree nodes: {ratio_subtree:.4f}')
    ratio_mask = sum(ratio_mask) / len(ratio_mask)
    logger.info(f'ratio of mask nodes: {ratio_mask:.4f}')

    ls = zip(args['tokens'], args['input_ids'], args['attention_mask'], args['mlm_input_ids'], args['mlm_labels'],
             args['valid_ids'], args['e1_mask'], args['e2_mask'], args['dep_matrix'])
    for i, (
            tokens, input_ids, attention_mask, mlm_input_ids, mlm_labels, valid_ids, e1_mask, e2_mask,
            dep_matrix) in enumerate(ls):
        # zero-pad up to the sequence length
        padding = [0] * (max_length - len(input_ids))
        args['tokens'][i] = tokens + ['[PAD]'] * (max_length - len(input_ids))
        args['input_ids'][i] = input_ids + padding
        args['attention_mask'][i] = attention_mask + padding
        args['valid_ids'][i] = valid_ids + padding
        args['mlm_input_ids'][i] = mlm_input_ids + padding
        args['mlm_labels'][i] = mlm_labels + [-100] * (max_length - len(input_ids))

        args['e1_mask'][i] = e1_mask + [0] * (max_length - len(e1_mask))
        args['e2_mask'][i] = e2_mask + [0] * (max_length - len(e2_mask))

        args['dep_matrix'][i] = np.pad(
            dep_matrix, ((0, (max_length - dep_matrix.shape[0])), (0, (max_length - dep_matrix.shape[1])))
        )

        assert len(args['tokens'][i]) == max_length
        assert len(args['input_ids'][i]) == max_length
        assert len(args['attention_mask'][i]) == max_length
        assert len(args['mlm_input_ids'][i]) == max_length
        assert len(args['valid_ids'][i]) == max_length
        assert len(args['e1_mask'][i]) == max_length
        assert len(args['e2_mask'][i]) == max_length
        assert len(args['dep_matrix'][i]) == max_length

    logger.info('tokenizing finished')

    return args


def nlp_tokenizer(args, parser_args, model_name):
    logger.info('start tokenizing text')
    text = args['text']

    if model_name == 'LSTM':
        with open(parser_args.vocab_file, 'r') as f:
            vocab = json.load(f)
        input_ids = []
        t = tqdm(text)

        for i, s in enumerate(t):
            input_id = []
            s = add_space_for_punctuation(s)
            ls = s.split()
            for token in ls:
                token = token.lower()
                if token in vocab:
                    input_id.append(vocab[token])
                else:
                    input_id.append(vocab['[OOV]'])
            input_ids.append(input_id)

        lengths = [len(input_id) for input_id in input_ids]
        lengths = sorted(lengths)
        max_length = lengths[-1]
        logger.info(f'max sequence length: {max_length}')
        median_length = lengths[len(lengths) // 2]
        logger.info(f'median sequence length: {median_length}')
        max_length = min(max_length, median_length * 2)

        for i, input_id in enumerate(input_ids):
            if len(input_id) > max_length:
                input_ids[i] = input_id[:max_length]
            else:
                input_ids[i] += [vocab['[PAD]']] * (max_length - len(input_id))
                args.update({'input_ids': input_ids})

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(text, padding=True, truncation=True)
        args.update({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
        max_length = len(inputs['input_ids'][0])
        logger.info(f'max sequence length: {max_length}')

    return args


def token_classification_tokenizer(args, special_args, parser_args, model_name):
    logger.info('start tokenizing text')
    text = args['text']
    labels = args['label']
    label_vocab = special_args['label vocab']

    if model_name == 'LSTM':
        with open(parser_args.vocab_file, 'r') as f:
            vocab = json.load(f)
            logger.info(f'{parser_args.vocab_file} has been loaded')
        input_ids = []
        t = tqdm(text)
        for i, s in enumerate(t):
            input_id = []
            s = add_space_for_punctuation(s)

            ls = s.split()
            for token in ls:
                token = token.lower()
                if token in vocab:
                    input_id.append(vocab[token])
                else:
                    input_id.append(vocab['[OOV]'])
            input_ids.append(input_id)

        lengths = [len(input_id) for input_id in input_ids]
        max_length = max(lengths)

        for i, input_id in enumerate(input_ids):
            if len(input_id) > max_length:
                input_ids[i] = input_id[:max_length]
            else:
                input_ids[i] += [vocab['[PAD]']] * (max_length - len(input_id))
        args.update({'input_ids': input_ids})

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(text, padding=True, truncation=True)
        max_length = len(inputs['input_ids'][0])
        args.update({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
    
    logger.info(f'max sequence length: {max_length}')
    labels = [x + [label_vocab['O']] * (max_length - len(x)) for x in labels]
    args.update({
        'label': labels
    })

    return args
