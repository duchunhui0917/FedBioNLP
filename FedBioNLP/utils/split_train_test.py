import os.path

import pandas as pd
import numpy as np

base_dir = os.path.expanduser('~/cross_silo_FL')
total_path = 'data/data_files/gene_disease/PGR_Q2/total.tsv'
train_path = 'data/data_files/gene_disease/PGR_Q2/train.tsv'
test_path = 'data/data_files/gene_disease/PGR_Q2/test.tsv'
df = pd.read_csv(os.path.join(base_dir, total_path), sep='\t')
rel = df['RELATION'].values

rel_true = np.where(rel == 1)[0]
rel_true_train = np.random.choice(rel_true, int(len(rel_true) * 0.8), replace=False)
rel_true_test = np.array([x for x in rel_true if x not in rel_true_train])

rel_false = np.where(rel == 0)[0]
rel_false_train = np.random.choice(rel_false, int(len(rel_false) * 0.8), replace=False)
rel_false_test = np.array([x for x in rel_false if x not in rel_false_train])

train_idx = np.concatenate([rel_true_train, rel_false_train])
train_idx = np.sort(train_idx)
test_idx = np.concatenate([rel_true_test, rel_false_test])
test_idx = np.sort(test_idx)

df_train = df.loc[train_idx]
df_train.to_csv(os.path.join(base_dir, train_path), sep='\t', encoding='utf8', index=False)
df_test = df.loc[test_idx]
df_test.to_csv(os.path.join(base_dir, test_path), sep='\t', encoding='utf8', index=False)
