import copy
import json
import os
import random
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

base_dir = os.path.expanduser('~/FedBioNLP')


def get_attributes_text(file_name):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')
    print(h5_path)
    rf = h5py.File(h5_path, 'r')
    attributes = json.loads(rf["attributes"][()])
    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']
    index_list = attributes['index_list']

    try:
        atts = eval(attributes['atts'])
    except:
        atts = ['X', 'Y']

    print(f'num of train/test: {len(train_index_list)}/{len(test_index_list)}')

    train_data = [[rf[x][str(idx)][()].decode('UTF-8') for idx in train_index_list] for x in atts]
    test_data = [[rf[x][str(idx)][()].decode('UTF-8') for idx in test_index_list] for x in atts]
    data = [[rf[x][str(idx)][()].decode('UTF-8') for idx in index_list] for x in atts]

    rf.close()
    return attributes, atts, train_data, test_data, data


def update_re_h5(file_name):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')

    with h5py.File(h5_path, 'r+') as hf:
        attributes = json.loads(hf["attributes"][()])
        index_list = attributes['index_list']
        for idx in index_list:
            sentence = hf['X'][str(idx)][()].decode('UTF-8')
            pos = sentence.index('<e1>')
            sentence = sentence[:pos] + ' <e1> ' + sentence[pos + 4:]
            pos = sentence.index('</e1>')
            sentence = sentence[:pos] + '  </e1>' + sentence[pos + 5:]
            pos = sentence.index('<e2>')
            sentence = sentence[:pos] + ' <e2> ' + sentence[pos + 4:]
            pos = sentence.index('</e2>')
            sentence = sentence[:pos] + ' </e2> ' + sentence[pos + 5:]

            hf['X'][str(idx)][()] = sentence

        hf["attributes"][()] = json.dumps(attributes)


def update_h5(file_name):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')

    with h5py.File(h5_path, 'r+') as hf:
        attributes = json.loads(hf["attributes"][()])
        train_source_list = attributes['train_source_list']
        test_source_list = attributes['test_source_list']
        train_index_list = attributes['train_index_list']
        test_index_list = attributes['test_index_list']

        source2doc = {'onbn': 0, 'onnw': 1, 'onmz': 2, 'connl03': 3, 'onbc': 4, 'onwb': 5, 'wnut16': 6}
        doc_idx = {}
        for idx in train_index_list:
            doc_idx.update({idx: source2doc[train_source_list[idx][0]]})
        for idx in test_index_list:
            doc_idx.update({idx: source2doc[test_source_list[idx - 14000][0]]})
        attributes['doc_index'] = doc_idx
        hf["attributes"][()] = json.dumps(attributes)


# update_h5('ploner')


def sample_false_true(data, num_false, num_true):
    y = data[-1]
    # false_idx = [i for i in range(len(y)) if y[i] == '0']
    # true_idx = [i for i in range(len(y)) if y[i] == '1']

    false_idx = [i for i in range(len(y)) if y[i] == 'negative']
    true_idx = [i for i in range(len(y)) if y[i] == 'positive']

    sampled_false_idx = random.sample(false_idx, num_false)
    sampled_true_idx = random.sample(true_idx, num_true)
    sampled_idx = sampled_false_idx + sampled_true_idx
    random.shuffle(sampled_idx)

    unsampled_false_idx = [idx for idx in false_idx if idx not in sampled_false_idx]
    unsampled_true_idx = [idx for idx in true_idx if idx not in sampled_true_idx]
    unsampled_idx = unsampled_false_idx + unsampled_true_idx
    random.shuffle(unsampled_idx)

    sampled_data = [[x[i] for i in sampled_idx] for x in data]
    unsampled_data = [[x[i] for i in unsampled_idx] for x in data]
    return sampled_data, unsampled_data


def split_h5(file_name, nums, sampled_name):
    sampled_wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_0_data.h5"), 'w')
    unsampled_wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_1_data.h5"), 'w')

    num0, num1, num2, num3 = nums

    attributes, atts, train_data, test_data, data = get_attributes_text(file_name)
    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']

    train_index0, train_index1 = train_index_list[:num0], train_index_list[num0:num0 + num1]
    test_index0, test_index1 = test_index_list[:num2], test_index_list[num2:num2 + num3]

    sampled_train_data = [[x[i] for i in train_index0] for x in data]
    unsampled_train_data = [[x[i] for i in train_index1] for x in data]
    sampled_test_data = [[x[i] for i in test_index0] for x in data]
    unsampled_test_data = [[x[i] for i in test_index1] for x in data]

    ls = [[sampled_wf, sampled_train_data, sampled_test_data],
          [unsampled_wf, unsampled_train_data, unsampled_test_data]]
    for wf, train_data, test_data in ls:
        idx = 0
        index_list = []
        train_index_list = []
        test_index_list = []

        train_num = len(train_data[0])
        test_num = len(test_data[0])

        t = tqdm(range(train_num))
        for i in t:
            for a, x in enumerate(train_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            index_list.append(idx)
            train_index_list.append(idx)
            idx += 1

        t = tqdm(range(test_num))
        for i in t:
            for a, x in enumerate(test_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            index_list.append(idx)
            test_index_list.append(idx)
            idx += 1

        assert len(index_list) == len(train_index_list) + len(test_index_list)
        attributes['index_list'] = index_list
        attributes['train_index_list'] = train_index_list
        attributes['test_index_list'] = test_index_list
        wf['/attributes'] = json.dumps(attributes)

        wf.close()


def undersampling_h5(file_name, nums, sampled_name):
    sampled_wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_0_data.h5"), 'w')
    unsampled_wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_1_data.h5"), 'w')

    attributes, atts, train_data, test_data, data = get_attributes_text(file_name)
    train_num_false, train_num_true, test_num_false, test_num_true = nums
    sampled_train_data, unsampled_train_data = sample_false_true(train_data, train_num_false, train_num_true)
    sampled_test_data, unsampled_test_data = sample_false_true(test_data, test_num_false, test_num_true)

    ls = [[sampled_wf, sampled_train_data, sampled_test_data],
          [unsampled_wf, unsampled_train_data, unsampled_test_data]]
    for wf, train_data, test_data in ls:
        idx = 0
        index_list = []
        train_index_list = []
        test_index_list = []

        train_num = len(train_data[0])
        test_num = len(test_data[0])

        t = tqdm(range(train_num))
        for i in t:
            for a, x in enumerate(train_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            index_list.append(idx)
            train_index_list.append(idx)
            idx += 1

        t = tqdm(range(test_num))
        for i in t:
            for a, x in enumerate(test_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            index_list.append(idx)
            test_index_list.append(idx)
            idx += 1

        assert len(index_list) == len(train_index_list) + len(test_index_list)
        attributes['index_list'] = index_list
        attributes['train_index_list'] = train_index_list
        attributes['test_index_list'] = test_index_list
        wf['/attributes'] = json.dumps(attributes)

        wf.close()


def under_sampling(file_name, nums, sampled_name):
    """

    Args:
        file_name:
        nums: [train_num, test_num]
        sampled_name:

    Returns:

    """
    wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_data.h5"), 'w')

    attributes, atts, train_data, test_data, data = get_attributes_text(file_name)
    train_num, test_num = nums
    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']

    sampled_train_index = random.sample(train_index_list, train_num)
    sampled_test_index = random.sample(test_index_list, test_num)

    sampled_train_data = [[x[i] for i in sampled_train_index] for x in data]
    sampled_test_data = [[x[i] for i in sampled_test_index] for x in data]

    idx = 0
    index_list = []
    train_index_list = []
    test_index_list = []

    t = tqdm(range(train_num))
    for i in t:
        for a, x in enumerate(sampled_train_data):
            wf[f'/{atts[a]}/{idx}'] = x[i]

        index_list.append(idx)
        train_index_list.append(idx)
        idx += 1

    t = tqdm(range(test_num))
    for i in t:
        for a, x in enumerate(sampled_test_data):
            wf[f'/{atts[a]}/{idx}'] = x[i]

        index_list.append(idx)
        test_index_list.append(idx)
        idx += 1

    assert len(index_list) == len(train_index_list) + len(test_index_list)
    attributes['index_list'] = index_list
    attributes['train_index_list'] = train_index_list
    attributes['test_index_list'] = test_index_list
    wf['/attributes'] = json.dumps(attributes)

    wf.close()


def sample_by_class(data, nums, label_vocab):
    y = data[-1]
    y = [label_vocab[_] for _ in y]

    sampled_data = [[] for _ in range(len(data))]
    for c, num in enumerate(nums):
        idx = [i for i in range(len(y)) if y[i] == c]
        sampled_idx = random.sample(idx, num)
        for i, x in enumerate(data):
            sampled_data[i] += [x[i] for i in sampled_idx]
    return sampled_data


def under_sampling_by_class(file_name, nums, sampled_name):
    wf = h5py.File(os.path.join(base_dir, f"data/{file_name}_{sampled_name}_data.h5"), 'w')

    attributes, atts, train_data, test_data, data = get_attributes_text(file_name)
    label_vocab = attributes['label_vocab']
    sampled_train_data = sample_by_class(train_data, nums[:len(nums) // 2], label_vocab)
    sampled_test_data = sample_by_class(test_data, nums[len(nums) // 2:], label_vocab)

    idx = 0
    index_list = []
    train_index_list = []
    test_index_list = []

    train_num = len(sampled_train_data[0])
    test_num = len(sampled_test_data[0])

    t = tqdm(range(train_num))
    for i in t:
        for a, x in enumerate(sampled_train_data):
            wf[f'/{atts[a]}/{idx}'] = x[i]

        index_list.append(idx)
        train_index_list.append(idx)
        idx += 1

    t = tqdm(range(test_num))
    for i in t:
        for a, x in enumerate(sampled_test_data):
            wf[f'/{atts[a]}/{idx}'] = x[i]

        index_list.append(idx)
        test_index_list.append(idx)
        idx += 1

    assert len(index_list) == len(train_index_list) + len(test_index_list)
    attributes['index_list'] = index_list
    attributes['train_index_list'] = train_index_list
    attributes['test_index_list'] = test_index_list
    wf['/attributes'] = json.dumps(attributes)

    wf.close()


def merge_h5(name_list):
    h5_name = '*'.join(name_list)
    wf = h5py.File(os.path.join(base_dir, f'data/{h5_name}_data.h5'), 'w')
    attributes = {}

    doc_index = {}
    index_list = []
    train_index_list = []
    test_index_list = []
    idx = 0

    for d, file_name in enumerate(name_list):
        attributes, atts, train_data, test_data, data = get_attributes_text(file_name)
        train_num = len(train_data[0])
        test_num = len(test_data[0])

        t = tqdm(range(train_num))
        for i in t:
            for a, x in enumerate(train_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            doc_index[idx] = d
            index_list.append(idx)
            train_index_list.append(idx)
            idx += 1

        t = tqdm(range(test_num))
        for i in t:
            for a, x in enumerate(test_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            doc_index[idx] = d
            index_list.append(idx)
            test_index_list.append(idx)
            idx += 1

    assert len(doc_index) == len(index_list) == len(train_index_list) + len(test_index_list)
    attributes['doc_index'] = doc_index
    attributes['index_list'] = index_list
    attributes['train_index_list'] = train_index_list
    attributes['test_index_list'] = test_index_list
    wf['/attributes'] = json.dumps(attributes)

    wf.close()


def over_sampling_h5(file_name, times=1., aug=False, only_false=True):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')
    if aug:
        if only_false:
            h5_times_path = os.path.join(base_dir, f'data/{file_name}_{times}_aug_data.h5')
        else:
            h5_times_path = os.path.join(base_dir, f'data/{file_name}_{times}_aug_size_data.h5')
    else:
        if only_false:
            h5_times_path = os.path.join(base_dir, f'data/{file_name}_{times}_data.h5')
        else:
            h5_times_path = os.path.join(base_dir, f'data/{file_name}_{times}_size_data.h5')

    rf = h5py.File(h5_path, 'r')
    wf = h5py.File(h5_times_path, 'w')

    attributes = json.loads(rf["attributes"][()])

    index_list = attributes['index_list']
    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']

    print(f'num of train/test: {len(train_index_list)}/{len(test_index_list)}')

    # over sampling train
    train_x = [rf['X'][str(idx)][()].decode('UTF-8') for idx in train_index_list]
    train_y = [rf['Y'][str(idx)][()].decode('UTF-8') for idx in train_index_list]

    if only_false:
        train_x_true = [train_x[idx] for idx in range(len(train_x)) if train_y[idx] == '1']
        train_x_false = [train_x[idx] for idx in range(len(train_x)) if train_y[idx] == '0']
        sample_idx = list(range(len(train_x_false)))
        k = int(len(train_x_true) * times) - len(train_x_false)
        sample_idx = random.choices(sample_idx, k=k)
        train_x_sample = [train_x_false[idx] for idx in sample_idx]
        if aug:
            train_x_sample = aug_data(train_x_sample, file_name)

        train_x += train_x_sample
        train_y += ['0' for _ in range(len(train_x_sample))]
    else:
        sample_idx = list(range(len(train_x)))
        k = int(len(train_x) * times) - len(train_x)
        sample_idx = random.choices(sample_idx, k=k)
        train_x_sample = [train_x[idx] for idx in sample_idx]
        train_y_sample = [train_y[idx] for idx in sample_idx]
        if aug:
            train_x_sample = aug_data(train_x_sample, file_name)
        train_x += train_x_sample
        train_y += train_y_sample

    # retain test
    test_x = [rf['X'][str(idx)][()].decode('UTF-8') for idx in test_index_list]
    test_y = [rf['Y'][str(idx)][()].decode('UTF-8') for idx in test_index_list]

    # update attributes, x, y
    doc_index = {}
    index_list = []
    train_index_list = []
    test_index_list = []

    idx = 0
    for x, y in zip(train_x, train_y):
        wf[f'/X/{idx}'] = x
        wf[f'/Y/{idx}'] = y
        doc_index[idx] = 0
        index_list.append(idx)
        train_index_list.append(idx)
        idx += 1

    for x, y in zip(test_x, test_y):
        wf[f'/X/{idx}'] = x
        wf[f'/Y/{idx}'] = y
        doc_index[idx] = 0
        index_list.append(idx)
        test_index_list.append(idx)
        idx += 1

    print(f'num of train/test: {len(train_index_list)}/{len(test_index_list)}')
    attributes['doc_index'] = doc_index
    attributes['index_list'] = index_list
    attributes['train_index_list'] = train_index_list
    attributes['test_index_list'] = test_index_list
    wf['/attributes'] = json.dumps(attributes)

    rf.close()
    wf.close()


# txt2h5(['GAD', 'EU-ADR', 'PGR_Q1'])
# update_h5('semeval_2010_task8')
# txt2h5(['PGR_Q1', 'PGR_Q2'])
# sta_ht()

# csv2tsv(['HPRD50'])
# undersampling_h5('AIMed_2|2', nums=[390, 390, 90, 90], sampled_name='balance')
# undersampling_h5('AIMed_1|2', nums=[390, 390, 90, 90], sampled_name='balance')
# undersampling_h5('AIMed_2|2', nums=[390, 100, 80, 20], sampled_name='8:2')
# undersampling_h5('AIMed_cur', nums=[400, 100, 80, 20], sampled_name='8:2')
# undersampling_h5('AIMed_2|2', nums=[98, 390, 22, 90], sampled_name='ratio_reverse')
# undersampling_h5('PGR_2|2', nums=[709, 709, 177, 177], sampled_name='5:5')
# undersampling_h5('PGR_1|2', nums=[426, 992, 106, 248], sampled_name='3:7')

# merge_h5(['AIMed_1|2', 'AIMed_2|2'])
# merge_h5(['LLL'])
# merge_h5(['PGR_Q1', 'PGR_Q2'])
# merge_h5(['AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL'])
# merge_h5(['AIMed', 'AIMed_797|797|189|189'])
# merge_h5(['AIMed', 'AIMed_label_reverse'])

# merge_h5(['AIMed_1|2', 'AIMed_2|2'])
# merge_h5(['AIMed_1|2', 'PGR_2797'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2', 'PGR_2797'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2_label_reverse'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2', 'AIMed_2|2_label_reverse'])

# merge_h5(['AIMed_1|2', 'AIMed_2|2_balance'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2_back_translate'])
# merge_h5(['AIMed_2:8', 'AIMed_2:8'])
# merge_h5(['AIMed_2:8', 'AIMed_5:5'])
# merge_h5(['AIMed_2:8', 'AIMed_8:2'])
# merge_h5(['AIMed_2:8', 'AIMed_5:5', 'AIMed_8:2'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2_balance', 'AIMed_2|2_ratio_reverse'])
# merge_h5(['AIMed_1|2', 'AIMed_2|2_2:8'])
# merge_h5(['AIMed', 'BioInfer'])
# merge_h5(['PGR_1|2_3:7', 'PGR_2|2_3:7'])
# split_h5('PGR_Q1', [1718, 1718, 430, 430], 'equal')
# undersampling_h5('PGR_Q2_1|2', [500, 500, 125, 125], '5:5')
# merge_h5(['PGR_Q2_1|2_1:9', 'PGR_Q2_2|2_5:5'])
# merge_h5(['PGR_Q2_1|2_9:1', 'PGR_Q2_2|2_5:5'])
# merge_h5(['PGR_Q2_1|2_5:5', 'PGR_Q2_2|2_1:9'])
# merge_h5(['PGR_Q2_1|2_5:5', 'PGR_Q2_2|2_9:1'])
# merge_h5(['PGR_Q1_1|2', 'PGR_Q1_2|2'])
# undersampling_h5('sst_2', [100, 100, 900, 900], '200')
# undersampling_h5('sst_2', [3000, 300, 900, 900], '3000:300')

# merge_h5(['PGR_Q1_100', 'BioInfer'])
# update_h5('semeval_2010_task8')

# undersampling('20news', [1000, 7532], sampled_name='1000')
# undersampling_by_class('sst_2', [300, 3000, 912, 909], sampled_name='300:3000')
under_sampling_by_class('20news',
                        [6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 4,
                         399, 397, 364, 396, 393, 385, 398, 376, 319, 394, 396, 396, 398, 392, 390, 310, 395, 394, 389,
                         251],
                        sampled_name='115')
