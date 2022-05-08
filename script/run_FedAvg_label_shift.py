from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP import set_seed
from FedBioNLP import plot_dirichlet
from FedBioNLP import sta_dis
import os

set_seed(23333)
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load', default=False)
parser.add_argument('--train', default=True)

# FL hyperparameters
parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--alg', default='FedAvg',
                    choices=['centralized', 'SOLO', 'server', 'FedAvg', 'FedProx', 'MOON', 'PersonalizedFL'])
parser.add_argument('--split_type', default='label_shift',
                    choices=['centralized', 'idx_split', 'label_shift', 'feature_shift'])
parser.add_argument('--beta', type=int, default=10)
parser.add_argument('--n_clients', type=int, default=5)
parser.add_argument('--avg', default=False)
parser.add_argument('--sm', default=0)
parser.add_argument('--cm', default=0)

# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                    choices=['CNN',
                             'distilbert-base-cased', 'bert-base-cased', 'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--opt', default='Adam',
                    choices=['SGD', 'Adam', 'WPOptim'])
parser.add_argument('--aggregate_method', default='sample',
                    choices=['equal', 'sample', 'attention'])
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()

base_dir = os.path.expanduser('~/FedBioNLP')

res = process_dataset(args.dataset_name, args.n_clients, beta=args.beta, split_type=args.split_type,
                      model_name=args.model_name)
client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model = res
n_classes = train_dataset.n_classes
doc_index = test_dataset.doc_index

logging.info('client train distribution')
distributions = sta_dis(train_datasets, n_classes)
plot_dirichlet(distributions, args.dataset_name)

logging.info('train test distribution')
distributions = sta_dis([train_dataset, test_dataset], n_classes)
plot_dirichlet(distributions, args.dataset_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = args.model_name.replace('/', '_')

p = f'ckpt/FedAvg_label_shift/{args.dataset_name}_beta={args.beta}_n={args.n_clients}_{model_name}'
ckpt = os.path.join(base_dir, p)
cs = FedAvgSystem(client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, ckpt,
                  n_iterations=args.n_iterations, lr=args.lr, epochs=args.n_epochs, opt=args.opt,
                  aggregate_method=args.aggregate_method, batch_size=args.batch_size,
                  sm=args.sm, cm=args.cm)

if args.load:
    print(ckpt)
    cs.load(ckpt)

if args.train:
    metrics = cs.run(avg=args.avg)
