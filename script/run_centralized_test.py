from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP.utils import set_seed
from FedBioNLP.utils.fl_utils import sta_dis
from FedBioNLP.utils.plt_utils import plot_dirichlet
import os

set_seed(23659)
parser = argparse.ArgumentParser()
base_dir = os.path.expanduser('~/FedBioNLP')

# control hyperparameters
parser.add_argument('--load', default=True)
parser.add_argument('--train', default=False)

# FL hyperparameters
parser.add_argument('--dataset_name', type=str, default='AIMed_1|2*AIMed_2|2_back_translate',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL',
                             'i2b2', 'i2b2_BIDMC', 'i2b2_Partners',
                             'semeval_2010_task8',
                             'wikiner', 'ploner',
                             'squad_1.1',
                             'cnn_dailymail', 'cornell_movie_dialogue'
                             ])

# training hyperparameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                    choices=['CNN',
                             'distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=50)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--n_batches', default=0)
parser.add_argument('--opt', default='Adam')
parser.add_argument('--aggregate_method', default='sample')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--max_seq_length', default=384)

args = parser.parse_args()
res = process_dataset(args.dataset_name, split_type='idx_split', model_name=args.model_name,
                      max_seq_length=args.max_seq_length)
client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model = res
n_classes = train_dataset.n_classes
doc_index = test_dataset.doc_index

distributions = sta_dis([train_dataset, test_dataset], n_classes)
plot_dirichlet(distributions, args.dataset_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = args.model_name.replace('/', '_')
ckpt_p = f'ckpt/centralized/{args.dataset_name}_{model_name}.pth'
cos_p = f'grad_cos/centralized/{args.dataset_name}_{model_name}.json'
ckpt = os.path.join(base_dir, ckpt_p)
grad_cos = os.path.join(base_dir, cos_p)

cs = CentralizedSystem(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                       model, device, ckpt, grad_cos,
                       n_iterations=args.n_iterations, n_epochs=args.n_epochs, n_batches=args.n_batches,
                       lr=args.lr, opt=args.opt,
                       aggregate_method=args.aggregate_method, batch_size=args.batch_size)

if args.load:
    print(ckpt)
    cs.load(ckpt)

if args.train:
    metrics = cs.run()

cs.test_model()
for test_loader in cs.test_loaders:
    cs.test_model(data_loader=test_loader)
