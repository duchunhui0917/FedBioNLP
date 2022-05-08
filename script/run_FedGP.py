from FedBioNLP.processors import process_dataset
from FedBioNLP.systems import FedGPSystem
import argparse
from FedBioNLP.utils import set_seed, init_log
from FedBioNLP.utils.plt_utils import plot_dirichlet
from FedBioNLP.utils.fl_utils import sta_dis
import os
import logging
import torch
import datetime

set_seed(23333)
parser = argparse.ArgumentParser()
logger = logging.getLogger(os.path.basename(__file__))

# control hyperparameters
parser.add_argument('--load', default=False)
parser.add_argument('--train', default=True)

# FL hyperparameters
parser.add_argument('--dataset_name', type=str, default='AIMed_1|2*PGR_2797',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--split_type', default='idx_split',
                    choices=['centralized', 'idx_split', 'label_shift', 'feature_shift'])
parser.add_argument('--beta', type=int, default=0.5)
parser.add_argument('--n_clients', type=int, default=1)
parser.add_argument('--mu', type=float, default=1)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--avg', default=False)
parser.add_argument('--sm', default=0)
parser.add_argument('--cm', default=0)
parser.add_argument('--centralized', default=False)
parser.add_argument('--personalized', default=True)
parser.add_argument('--layers', default='embedding*layer.0*layer.1*layer.2*layer.3*layer.4*layer.5*classifier')
# parser.add_argument('--layers', default='.')
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
parser.add_argument('--aggregate_method', default='equal',
                    choices=['equal', 'sample', 'attention'])
parser.add_argument('--batch_size', default=32)
parser.add_argument('--max_seq_length', default=384)

args = parser.parse_args()
base_dir = os.path.expanduser('~/FedBioNLP')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = args.model_name.replace('/', '_')
ckpt = f'ckpt/FedGP/{args.dataset_name}_{model_name}'
ckpt = os.path.join(base_dir, ckpt)
log_file = f'log/FedGP/{args.dataset_name}_{model_name}_{datetime.datetime.now():%y-%m-%d %H:%M}.log'
log_file = os.path.join(base_dir, log_file)
init_log(log_file)

res = process_dataset(args.dataset_name, split_type=args.split_type, max_seq_length=args.max_seq_length,
                      model_name=args.model_name)
client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model = res
n_classes = train_dataset.n_classes
doc_index = test_dataset.doc_index

logger.info('client train distribution')
distributions = sta_dis(train_datasets, n_classes)
plot_dirichlet(distributions, args.dataset_name)

logger.info('train test distribution')
distributions = sta_dis([train_dataset, test_dataset], n_classes)
plot_dirichlet(distributions, args.dataset_name)

cs = FedGPSystem(client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, ckpt,
                 n_iterations=args.n_iterations, n_epochs=args.n_epochs, n_batches=args.n_batches,
                 lr=args.lr, opt=args.opt,
                 aggregate_method=args.aggregate_method, batch_size=args.batch_size,
                 sm=args.sm, cm=args.cm, centralized=args.centralized, personalized=args.personalized,
                 layers=args.layers)

if args.load:
    print(ckpt)
    cs.load(ckpt)

if args.train:
    metrics = cs.run()

for i, data_loader in enumerate(cs.test_loaders):
    cs.test_model(data_loader=data_loader)
