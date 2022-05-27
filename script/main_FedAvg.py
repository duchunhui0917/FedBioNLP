import os
import sys

sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
                    default='HPRD50*LLL',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--alg', type=str, default='FedAvg')
parser.add_argument('--split_type', default='idx_split',
                    choices=['centralized', 'idx_split', 'label_shift', 'feature_shift'])
parser.add_argument('--beta', type=int, default=0.5)
parser.add_argument('--n_clients', type=int, default=1)
parser.add_argument('--sm', default=0)
parser.add_argument('--cm', default=0)
parser.add_argument('--save_central', default=False)
parser.add_argument('--save_global', default=True)
parser.add_argument('--save_personal', default=True)
parser.add_argument('--aggregate_method', default='sample',
                    choices=['equal', 'sample', 'attention'])
parser.add_argument('--layers',
                    default='.*embedding*layer.0*layer.1*layer.2*layer.3*layer.4*layer.5*patch*gcn*mlm*classifier')
# parser.add_argument('--layers', default='.')


# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                    choices=['CNN',
                             'distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--n_batches', default=0)
parser.add_argument('--opt', default='Adam',
                    choices=['SGD', 'Adam', 'WPOptim'])
parser.add_argument('--batch_size', default=16)
parser.add_argument('--weight_sampler', default=False)


def run():
    if args.alg == 'centralized':
        system = Centralized
    if args.alg == 'SCL':
        system = SCL
    elif args.alg == 'FedAvg':
        system = FedAvg
        args.mlm_method = 'None'
        args.mlm_prob = 0.15
        args.num_gcn_layers = 0
        args.K_LCA = 1
    elif args.alg == 'FedProx':
        system = FedProx
        args.mu = 1
    elif args.alg == 'MOON':
        system = MOON
        args.mu = 1
        args.temperature = 0.5
    elif args.alg == 'FedGS':
        system = FedGS
    elif args.alg == 'FedGP':
        system = FedGP
    elif args.alg == 'HarmoFL':
        system = HarmoFL
    elif args.alg == 'PartialFL':
        system = PartialFL
        args.pk = 'classifier'
    elif args.alg == 'GSN':
        system = PartialFL
        args.mlm_method = 'subtree'
        args.mlm_prob = 0.15
        args.num_gcn_layers = 0
        args.K_LCA = 1
        args.pk = 'mlm'
    elif args.alg == 'FedMatch':
        system = PartialFL
    elif args.alg == 'pFedMe':
        system = pFedMe
        args.n_inner_loops = 4
        args.mu = 1
    elif args.alg == 'ICFA':
        args.n_clusters = 0
        system = ICFA
    else:
        raise NotImplementedError

    base_dir = os.path.expanduser('~/FedBioNLP')
    set_seed(2)
    model_name = args.model_name.replace('/', '_')
    ckpt = f'ckpt/{args.alg}/{args.dataset_name}'

    if not os.path.exists(os.path.join(base_dir, ckpt)):
        os.makedirs(os.path.join(base_dir, ckpt), exist_ok=True)
    args.ckpt = os.path.join(base_dir, ckpt, f'{model_name}')

    log_file = f'log/{args.alg}/{args.dataset_name}'
    if not os.path.exists(os.path.join(base_dir, log_file)):
        os.makedirs(os.path.join(base_dir, log_file), exist_ok=True)
    log_file = os.path.join(base_dir, log_file, f'{model_name}_{datetime.datetime.now():%y-%m-%d %H:%M}.log')
    init_log(log_file)

    tmp = process_dataset(args.dataset_name, args.model_name, args.split_type, args.n_clients, args)
    client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, res = tmp
    n_classes = train_dataset.n_classes

    mtx, mtx_ = status_mtx(train_datasets, n_classes)
    plot_class_samples(mtx)
    mtx = json.dumps(mtx.tolist())
    logger.info(f'clients train class samples\n{mtx}')

    mtx, mtx_ = status_mtx(test_datasets, n_classes)
    plot_class_samples(mtx)
    mtx = json.dumps(mtx.tolist())
    logger.info(f'clients test class samples\n{mtx}')

    mtx, mtx_ = status_mtx([train_dataset, test_dataset], n_classes)
    plot_class_samples(mtx)
    mtx = json.dumps(mtx.tolist())
    logger.info(f'train test class samples\n{mtx}')

    s = system(train_datasets, test_datasets, train_dataset, test_dataset, model, args)

    load_ckpt = os.path.join(base_dir, ckpt, f'{model_name}_global')
    if args.load:
        print(load_ckpt)
        s.load(load_ckpt)

    if args.train:
        s.run()

    s.test_model()
    for test_loader in s.test_loaders:
        s.test_model(data_loader=test_loader)


args = parser.parse_args()
run()
