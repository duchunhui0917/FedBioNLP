from FedBioNLP.datasets import process_dataset
import argparse
from FedBioNLP import set_seed, plot_dirichlet
from FedBioNLP import sta_dis
from FedBioNLP import visualize_features
import os

os.path.expanduser('~/cross_silo_FL')

set_seed(233)
parser = argparse.ArgumentParser()

# control hyperparameters
parser.add_argument('--load', default=False)
parser.add_argument('--train', default=True)
parser.add_argument('--visual', default=True)

# FL hyperparameters
parser.add_argument('--dataset_name', type=str, default='PGR_Q1*PGR_Q2',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100',
                             '20news', 'agnews', 'sst_2', 'sentiment140',
                             'GAD', 'EU-ADR', 'PGR_Q1', 'PGR_Q2', 'CoMAGC', 'PolySearch',
                             'AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL', 'merged',
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
                             'distilbert-base-cased', 'bert-base-cased', 'dmis-lab/biobert-v1.1'])
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--opt', default='Adam')
parser.add_argument('--aggregate_method', default='sample')
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()

base_dir = os.path.expanduser('~/cross_silo_FL')
if not os.path.exists(os.path.join(base_dir, 'ckpt')):
    os.mkdir(os.path.join(base_dir, 'ckpt'))

train_dataset, test_dataset, model = process_dataset(args.dataset_name,
                                                     model_name=args.model_name,
                                                     discriminator=True)
n_classes = train_dataset.n_classes
doc_index = test_dataset.doc_index

distributions = sta_dis([train_dataset, test_dataset], n_classes)
plot_dirichlet(distributions, args.dataset_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

p = f'ckpt/discriminator/{args.dataset_name}'
ckpt = os.path.join(base_dir, p)
cs = CentralizedSystem(train_dataset, test_dataset, model, device, ckpt,
                       n_iterations=args.n_iterations, lr=args.lr, epochs=args.n_epochs, opt=args.opt,
                       aggregate_method=args.aggregate_method, batch_size=args.batch_size)

if args.load:
    print(ckpt)
    cs.load(ckpt)

if args.train:
    metrics = cs.run()

metric1, inputs1, features1, logits1, labels1 = cs.test_model(visual=True)

ckpt = os.path.join(base_dir, f'ckpt/centralized/AIMed*BioInfer*PGR_Q1*PGR_Q2')
print(ckpt)
cs.load(ckpt)
metric2, inputs2, features2, logits2, labels2 = cs.test_model(visual=True)

if args.visual:
    visualize_features(inputs1, doc_index)
    for feature in features1:
        visualize_features(feature, doc_index)
    visualize_features(logits1, doc_index)
