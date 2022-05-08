from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP import set_seed
import os

os.path.expanduser('~/FedBioNLP')

set_seed(233)
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load', default=False)
parser.add_argument('--train', default=True)
parser.add_argument('--visual', default=True)

# FL hyperparameters
parser.add_argument('--dataset_name', type=str, default='AIMed*PGR_Q1',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])
parser.add_argument('--alg', default='FedMatch',
                    choices=['centralized', 'SOLO', 'server', 'FedAvg', 'FedProx', 'MOON', 'PersonalizedFL'])
parser.add_argument('--split_type', default='idx_split',
                    choices=['centralized', 'idx_split', 'label_shift', 'feature_shift'])
parser.add_argument('--beta', type=int, default=0.5)
parser.add_argument('--n_clients', type=int, default=1)
parser.add_argument('--centralized', default=False)
parser.add_argument('--personalized', default=True)
parser.add_argument('--num_layers', default='encoder*gcn*classifier')

# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                    choices=['CNN',
                             'distilbert-base-cased', 'bert-base-cased', 'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=50)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--n_batches', default=0)
parser.add_argument('--opt', default='Adam')
parser.add_argument('--aggregate_method', default='equal')
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()

base_dir = os.path.expanduser('~/FedBioNLP')
if not os.path.exists(os.path.join(base_dir, 'ckpt')):
    os.mkdir(os.path.join(base_dir, 'ckpt'))

res = process_dataset(args.dataset_name,
                      args.n_clients,
                      beta=args.beta,
                      split_type=args.split_type,
                      model_name=args.model_name,
                      horizon=True)
client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model = res
n_classes = train_dataset.n_classes
doc_index = test_dataset.doc_index

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

p = f'ckpt/{args.alg}/{args.dataset_name}'
ckpt = os.path.join(base_dir, p)
cs = PersonalizedFLSystem(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                          model, device, ckpt,
                          n_iterations=args.n_iterations, n_epochs=args.n_epochs, n_batches=args.n_batches,
                          lr=args.lr, opt=args.opt,
                          aggregate_method=args.aggregate_method, batch_size=args.batch_size, pk='patch',
                          centralized=args.centralized, personalized=args.personalized, layers=args.layers)

if args.load:
    print(ckpt)
    cs.load(ckpt)

if args.train:
    metrics = cs.run()
