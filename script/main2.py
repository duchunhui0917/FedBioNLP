import json
import sys

sys.path.append('..')

import argparse
import datetime
from FedBioNLP.systems import *
from FedBioNLP import process_dataset, plot_class_samples, status_mtx, init_log, set_seed
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load', default=False)
parser.add_argument('--train', default=True)

# common FL hyperparameters
parser.add_argument('--dataset_name', type=str,
                    default='LLL',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--alg', type=str, default='PartialFL')
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
# parser.add_argument('--pk', type=list, default=['classifier'])
parser.add_argument('--pk', type=list, default=['embeddings', 'transformer.layer.0', 'transformer.layer.1'])
# for ICFA
parser.add_argument('--n_clusters', type=int, default=0)
# for SCL
parser.add_argument('--SCL', default=False)
# for
parser.add_argument('--MaskedLM', default=False)

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
parser.add_argument('--batch_size', default=8)
parser.add_argument('--max_seq_length', default=384)


def run():
    base_dir = os.path.expanduser('~/FedBioNLP')
    set_seed(2)
    model_name = args.model_name.replace('/', '_')
    ckpt = f'ckpt/{args.alg}/{args.dataset_name}_{model_name}'
    args.ckpt = os.path.join(base_dir, ckpt)
    log_file = f'log/{args.alg}/{args.dataset_name}_{model_name}_{datetime.datetime.now():%y-%m-%d %H:%M}.log'
    log_file = os.path.join(base_dir, log_file)
    init_log(log_file)

    tmp = process_dataset(args.dataset_name, args.model_name, args.split_type, args.n_clients, args.max_seq_length,
                          args)
    client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, res = tmp
    n_classes = train_dataset.n_classes

    # mtx, mtx_ = status_mtx(train_datasets, n_classes)
    # plot_class_samples(mtx)
    # mtx = json.dumps(mtx.tolist())
    # logger.info(f'clients train class samples\n{mtx}')
    #
    # mtx, mtx_ = status_mtx([train_dataset, test_dataset], n_classes)
    # plot_class_samples(mtx)
    # mtx = json.dumps(mtx.tolist())
    # logger.info(f'train test class samples\n{mtx}')

    if args.alg in ['centralized', 'SCL', 'MaskedLM']:
        system = Centralized
    elif args.alg == 'FedAvg':
        system = FedAvg
    elif args.alg == 'FedProx':
        system = FedProx
    elif args.alg == 'MOON':
        system = MOON
    elif args.alg == 'FedGS':
        system = FedGS
    elif args.alg == 'FedGP':
        system = FedGP
    elif args.alg == 'HarmoFL':
        system = HarmoFL
    elif args.alg == 'PartialFL':
        system = PartialFL
    elif args.alg == 'FedMatch':
        system = PartialFL
    elif args.alg == 'pFedMe':
        system = pFedMe
    elif args.alg == 'ICFA':
        args.cluster_models = res['cluster_models']
        system = ICFA
    else:
        raise NotImplementedError

    s = system(train_datasets, test_datasets, train_dataset, test_dataset, model, args)

    if args.load:
        print(ckpt)
        s.load(ckpt)

    if args.train:
        s.run()

    s.test_model()
    for test_loader in s.test_loaders:
        s.test_model(data_loader=test_loader)


args = parser.parse_args()
run()
