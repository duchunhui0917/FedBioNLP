import logging
import random
import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
from .utils.dependency_parsing import *

logger = logging.getLogger(os.path.basename(__file__))


def re_tokenizer(args, model_name):
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts = args['text']
    args.update({
        "input_ids": [],
        "attention_mask": [],
        "mlm_input_ids": [],
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
    ls = zip(args["input_ids"], args["attention_mask"], args["mlm_input_ids"], args["e1_mask"], args["e2_mask"])
    for i, (input_ids, attention_mask, mlm_input_ids, e1_mask, e2_mask) in enumerate(ls):
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


def re_dep_tokenizer(args, model_name, mlm_method='None', mlm_prob=0.15, K_LCA=1):
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
        'input_ids': [],
        'attention_mask': [],
        'mlm_input_ids': [],
        'valid_ids': [],
        'e1_mask': [],
        'e2_mask': [],
        'dep_matrix': []
    })
    model_max_length = tokenizer.model_max_length
    cls_id, mask_id, sep_id = tokenizer.convert_tokens_to_ids(['CLS', '[MASK]', '[SEP]'])

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
        args['input_ids'].append(input_ids)
        args['attention_mask'].append(attention_mask)
        args['valid_ids'].append(valid_ids)
        args['e1_mask'].append(e1_mask)
        args['e2_mask'].append(e2_mask)

        # prun dependency matrix
        seq_length = sum(valid_ids)
        assert e1_start < seq_length
        assert e2_start < seq_length
        dep_matrix = dependency_to_matrix(dependency, seq_length)
        sdp = get_sdp(dep_matrix, e1_start, e2_start)
        self_loop_dep_matrix = get_self_loop_dep_matrix(dep_matrix, seq_length)
        subtree_dep_matrix, subtree = get_subtree_dep_matrix(dep_matrix, sdp, K_LCA=K_LCA)
        dep_matrix = self_loop_dep_matrix | subtree_dep_matrix
        args['dep_matrix'].append(dep_matrix)

        if mlm_method == 'subtree':
            assert max(subtree) <= sum(valid_ids)
            mlm_input_ids = input_ids[:]
            node = 0
            mask = False
            for i, x in enumerate(mlm_input_ids):
                if valid_ids[i] == 1:
                    if node not in subtree:
                        mask = True
                    else:
                        mask = False
                    node += 1
                if mask and random.random() < mlm_prob and x != cls_id and x != sep_id:
                    mlm_input_ids[i] = mask_id
            mlm_input_ids = [-100 if x != mask_id else x for x in mlm_input_ids]
        elif mlm_method == 'all':
            mlm_input_ids = input_ids[:]
            for i, x in enumerate(mlm_input_ids):
                if random.random() < mlm_prob and x != cls_id and x != sep_id:
                    mlm_input_ids[i] = mask_id
            mlm_input_ids = [-100 if x != mask_id else x for x in mlm_input_ids]
        else:
            mlm_input_ids = [0] * len(input_ids)
        args['mlm_input_ids'].append(mlm_input_ids)

    max_length = max([len(token) for token in args["input_ids"]])
    logger.info(f'max sequence length: {max_length}')

    ls = zip(args['input_ids'], args['attention_mask'], args['mlm_input_ids'],
             args['valid_ids'], args['e1_mask'], args['e2_mask'], args['dep_matrix'])
    for i, (input_ids, attention_mask, mlm_input_ids, valid_ids, e1_mask, e2_mask, dep_matrix) in enumerate(ls):
        # zero-pad up to the sequence length
        padding = [0] * (max_length - len(input_ids))
        args['input_ids'][i] = input_ids + padding
        args['attention_mask'][i] = attention_mask + padding
        args['mlm_input_ids'][i] = mlm_input_ids + padding
        args['valid_ids'][i] = valid_ids + padding

        args['e1_mask'][i] = e1_mask + [0] * (max_length - len(e1_mask))
        args['e2_mask'][i] = e2_mask + [0] * (max_length - len(e2_mask))

        args['dep_matrix'][i] = np.pad(
            dep_matrix, ((0, (max_length - dep_matrix.shape[0])), (0, (max_length - dep_matrix.shape[1])))
        )

        assert len(args['input_ids'][i]) == max_length
        assert len(args['attention_mask'][i]) == max_length
        assert len(args['mlm_input_ids'][i]) == max_length
        assert len(args['valid_ids'][i]) == max_length
        assert len(args['e1_mask'][i]) == max_length
        assert len(args['e2_mask'][i]) == max_length
        assert len(args['dep_matrix'][i]) == max_length

    logger.info('tokenizing finished')

    return args


def sc_tokenizer(args, model_name, max_seq_length):
    logger.info('start tokenizing text')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = args['text']
    # max_seq_length = max([len(t.split(' ')) for t in text])
    # logger.info(max_seq_length)
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_seq_length)
    args.update({
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask']
    })
    return args
