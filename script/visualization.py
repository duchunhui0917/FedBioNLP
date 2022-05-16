import os.path
import numpy as np
from FedBioNLP.processors import process_dataset
from FedBioNLP.systems import Centralized
import argparse
import pickle
import logging
import warnings
import datetime
from FedBioNLP import process_dataset, plot_class_samples, status_mtx, init_log, set_seed, visualize_features

base_dir = os.path.expanduser('~/FedBioNLP')

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load_model', default=True)
parser.add_argument('--load_rep', default=False)
parser.add_argument('--vis_docs', default=False)
parser.add_argument('--vis_labels', default=True)
parser.add_argument('--mode', default='average')

# common FL hyperparameters
parser.add_argument('--dataset_name', type=str,
                    default='AIMed',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--alg', type=str, default='SCL')
parser.add_argument('--split_type', default='idx_split',
                    choices=['centralized', 'idx_split', 'label_shift', 'feature_shift'])
parser.add_argument('--beta', type=int, default=0.5)
parser.add_argument('--n_clients', type=int, default=1)
parser.add_argument('--sm', default=0)
parser.add_argument('--cm', default=0)
parser.add_argument('--centralized', default=False)
parser.add_argument('--personalized', default=True)
parser.add_argument('--aggregate_method', default='equal',
                    choices=['equal', 'sample', 'attention'])
parser.add_argument('--layers',
                    default='.*embedding*layer.0*layer.1*layer.2*layer.3*layer.4*layer.5*patch*gcn*classifier')
# parser.add_argument('--layers', default='.')

# for FedProx, MOON, pFedMe
parser.add_argument('--mu', type=float, default=1)
# for MOON
parser.add_argument('--temperature', type=float, default=0.5)
# for pFedMe
parser.add_argument('--n_inner_loops', type=int, default=4)
# for PartialFL, FedMatch
parser.add_argument('--pk', type=str, default='classifier')
# for ICFA
parser.add_argument('--n_clusters', type=int, default=0)
# for SCL
parser.add_argument('--SCL', default=True)

# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                    choices=['CNN',
                             'distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=50)
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--n_batches', default=0)
parser.add_argument('--opt', default='Adam',
                    choices=['SGD', 'Adam', 'WPOptim'])
parser.add_argument('--batch_size', default=128)
parser.add_argument('--max_seq_length', default=256)

args = parser.parse_args()

model_name = args.model_name.replace('/', '_')
ckpt = f'ckpt/{args.alg}/{args.dataset_name}_{model_name}.pth'
args.ckpt = os.path.join(base_dir, ckpt)
log_file = f'log/{args.alg}/{args.dataset_name}_{model_name}_{datetime.datetime.now():%y-%m-%d %H:%M}.log'
log_file = os.path.join(base_dir, log_file)
init_log(log_file)

_ = process_dataset(args.dataset_name, args.model_name, args.split_type, args.n_clients, args.max_seq_length, args)
clients, train_datasets, test_datasets, train_dataset, test_dataset, model, res = _
doc_index = test_dataset.doc_index

cs = Centralized(train_datasets, test_datasets, train_dataset, test_dataset, model, args)
if args.load_model:
    cs.load(args.ckpt)
    print(args.ckpt)
    logging.info('model has been loaded')

p = f'rep/{args.alg}/{args.dataset_name}_{model_name}.pkl'
rep = os.path.join(base_dir, p)
if os.path.exists(rep) and args.load_rep:
    with open(rep, 'rb') as f:
        ls = pickle.load(f)
    print(rep)
    logging.info('pickle has been loaded')
else:
    ls = []
    for i, test_loader in enumerate(cs.test_loaders):
        metric, inputs, labels, features, logits = cs.test_model(data_loader=test_loader)
        ls.append({'metric': metric, 'labels': labels, 'features': features})
    with open(rep, 'wb') as f:
        pickle.dump(ls, f)
    print(rep)
    logging.info('pickle has been dumped')


def vis(nodes, all_layers=True):
    if all_layers:
        features_n = [x['features'] for x in ls]
        labels_n = [x['labels'][-1] for x in ls]
    else:
        features_n = [[x['features'][-1]] for x in ls]
        labels_n = [x['labels'][-1] for x in ls]

    vis_features = [np.concatenate([features_n[n][l] for n in nodes]) for l in range(len(features_n[0]))]
    # vis_labels = np.concatenate(
    #     [labels_n[n] for n in nodes]
    # )
    vis_docs = np.concatenate(
        [np.ones(len(labels_n[n]), dtype=np.int64) * n for n in nodes])

    vis_labels = np.concatenate([
        labels_n[n] + np.ones(len(labels_n[n]), dtype=np.int64) * 2 * i for i, n in enumerate(nodes)
    ])
    if args.vis_docs:
        visualize_features(vis_features, vis_docs, mode=args.mode)
    if args.vis_labels:
        visualize_features(vis_features, vis_labels, mode=args.mode)


for x in ls:
    print(x['metric'])

vis([0])
