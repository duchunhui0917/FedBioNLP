import copy

from sklearn.datasets import load_svmlight_files
import os
import scipy.io as scio
import os
import numpy as np

names = ['books', 'dvd', 'elec', 'kitchen']
base_dir = os.path.expanduser('~/FedBioNLP')

train_file = os.path.join(base_dir, f'data/Amazon_review/train_400.mat')
test_file = os.path.join(base_dir, f'data/Amazon_review/test_400.mat')
train_data = {'fts': np.array([]), 'labels': np.array([])}
test_data = {'fts': np.array([]), 'labels': np.array([])}
for i, name in enumerate(names):
    train_name_file = os.path.join(base_dir, f'data/Amazon_review/{name}_400_train.mat')
    test_name_file = os.path.join(base_dir, f'data/Amazon_review/{name}_400_test.mat')

    train_name_data = scio.loadmat(train_name_file)
    test_name_data = scio.loadmat(test_name_file)

    if i != 0:
        train_data['fts'] = np.concatenate([train_data['fts'], train_name_data['fts']], axis=0)
        train_data['labels'] = np.concatenate([train_data['labels'], train_name_data['labels']])

        test_data['fts'] = np.concatenate([test_data['fts'], test_name_data['fts']])
        test_data['labels'] = np.concatenate([test_data['labels'], test_name_data['labels']])
    else:
        train_data['fts'] = train_name_data['fts']
        train_data['labels'] = train_name_data['labels']

        test_data['fts'] = test_name_data['fts']
        test_data['labels'] = test_name_data['labels']

scio.savemat(train_file, train_data)
scio.savemat(test_file, test_data)
