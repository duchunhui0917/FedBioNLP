import os.path
import numpy as np
from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP import visualize_features, cmp_cosine_euclidean
import pickle

parser = argparse.ArgumentParser()
base_dir = os.path.expanduser('~/FedBioNLP')
parser.add_argument('--load_model', default=True)
parser.add_argument('--load_rep', default=True)
parser.add_argument('--alg', default='GRL')
parser.add_argument('--dataset_name', default='AIMed_1|2*PGR_2797')
parser.add_argument('--model_name', default='distilbert-base-cased',
                    choices=['distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--mode', default='average', choices=['average', 'first', 'first_last', 'squeeze'])
parser.add_argument('--batch_size', default=8)
parser.add_argument('--GRL', default=False)
parser.add_argument('--MaskedLM', default=False)
parser.add_argument('--vis_docs', default=True)
parser.add_argument('--vis_labels', default=False)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

res = process_dataset(args.dataset_name,
                      split_type='idx_split',
                      model_name=args.model_name,
                      GRL=args.GRL,
                      MaskedLM=args.MaskedLM)
clients, train_datasets, test_datasets, train_dataset, test_dataset, model = res
doc_index = test_dataset.doc_index

model_name = args.model_name.replace('/', '_')
ckpt_p = f'ckpt/centralized/{args.dataset_name}_{model_name}.pth'
cos_p = f'grad_cos/centralized/{args.dataset_name}_{model_name}.json'
ckpt = os.path.join(base_dir, ckpt_p)
grad_cos = os.path.join(base_dir, cos_p)

cs = CentralizedSystem(clients, train_datasets, test_datasets, train_dataset, test_dataset, model, device,
                       ckpt, grad_cos,
                       batch_size=args.batch_size)
if args.load_model:
    cs.load(ckpt)
    print(ckpt)
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


def cmp(pairs, all_layers=True):
    i, j = pairs
    if all_layers:
        features_n = [x['features'] for x in ls]
    else:
        features_n = [x['features'][-1] for x in ls]
    cmp_cosine_euclidean(features_n[i], features_n[j], mode=args.mode)


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

vis([0, 1])
