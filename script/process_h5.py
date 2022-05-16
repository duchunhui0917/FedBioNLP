import copy
import json
import os
import random
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

base_dir = os.path.expanduser('~/FedBioNLP/data')


def get_attributes_text(file_name):
    h5_path = os.path.join(base_dir, f'{file_name}_data.h5')
    print(h5_path)
    rf = h5py.File(h5_path, 'r')
    attributes = json.loads(rf["attributes"][()])
    tmp_train_index_list = attributes['train_index_list']
    tmp_test_index_list = attributes['test_index_list']
    atts = eval(attributes['atts'])

    print(f'num of train/test: {len(tmp_train_index_list)}/{len(tmp_test_index_list)}')

    train_data = [[rf[x][str(idx)][()].decode('UTF-8') for idx in tmp_train_index_list] for x in atts]
    test_data = [[rf[x][str(idx)][()].decode('UTF-8') for idx in tmp_test_index_list] for x in atts]
    rf.close()
    return attributes, atts, train_data, test_data


def update_h5(file_name):
    h5_path = os.path.join(base_dir, f'data_files/{file_name}_data.h5')

    with h5py.File(h5_path, 'r+') as hf:
        attributes = json.loads(hf["attributes"][()])

        hf["attributes"][()] = json.dumps(attributes)


def sample_false_true(data, num_false, num_true):
    y = data[-1]
    false_idx = [i for i in range(len(y)) if y[i] == '0']
    true_idx = [i for i in range(len(y)) if y[i] == '1']
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


def undersampling_h5(file_name, nums, sampled_name):
    sampled_wf = h5py.File(os.path.join(base_dir, f"{file_name}_{sampled_name}_0_data.h5"), 'w')
    unsampled_wf = h5py.File(os.path.join(base_dir, f"{file_name}_{sampled_name}_1_data.h5"), 'w')

    attributes, atts, train_data, test_data = get_attributes_text(file_name)
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


def merge_h5(name_list):
    h5_name = '*'.join(name_list)
    wf = h5py.File(os.path.join(base_dir, f'{h5_name}_data.h5'), 'w')
    attributes = {}

    doc_index = {}
    index_list = []
    train_index_list = []
    test_index_list = []
    idx = 0

    for d, file_name in enumerate(name_list):
        attributes, atts, train_data, test_data = get_attributes_text(file_name)
        train_num = len(train_data[0])
        test_num = len(test_data[0])

        false_num = 0
        true_num = 0
        t = tqdm(range(train_num))
        for i in t:
            for a, x in enumerate(train_data):
                wf[f'/{atts[a]}/{idx}'] = x[i]

            if train_data[-1][i] == '0':
                false_num += 1
            else:
                true_num += 1

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
    base_dir = os.path.expanduser('~/FedBioNLP/data')
    h5_path = os.path.join(base_dir, f'{file_name}_data.h5')
    if aug:
        if only_false:
            h5_times_path = os.path.join(base_dir, f'{file_name}_{times}_aug_data.h5')
        else:
            h5_times_path = os.path.join(base_dir, f'{file_name}_{times}_aug_size_data.h5')
    else:
        if only_false:
            h5_times_path = os.path.join(base_dir, f'{file_name}_{times}_data.h5')
        else:
            h5_times_path = os.path.join(base_dir, f'{file_name}_{times}_size_data.h5')

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

# undersampling_h5('AIMed_cur', nums=[400, 100, 80, 20], sampled_name='8:2')
# undersampling_h5('AIMed_2|2', nums=[98, 390, 22, 90], sampled_name='ratio_reverse')

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
merge_h5(['AIMed_1|2', 'AIMed_2|2', 'PGR_2797'])
