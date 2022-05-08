import copy
import os.path
import numpy as np
from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP.utils.sta_utils import param_cosine

parser = argparse.ArgumentParser()
base_dir = os.path.expanduser('~/FedBioNLP')
parser.add_argument('--load_model', default=True)
parser.add_argument('--load_rep', default=False)
parser.add_argument('--alg', default='centralized')
parser.add_argument('--dataset_name', default='LLL')
parser.add_argument('--model_name', default='distilbert-base-cased',
                    choices=['distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--mode', default='average', choices=['average', 'first', 'first_last', 'squeeze'])
parser.add_argument('--batch_size', default=32)
parser.add_argument('--GRL', default=False)
parser.add_argument('--MaskedLM', default=False)
args = parser.parse_args()

layers = ['encoder', 'classifier']
layers = ['embedding', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'classifier']


def get_grad_cosine(state_dicts):
    np.set_printoptions(precision=4)
    n_params = len(state_dicts)
    cos_sims = {layer: np.zeros((n_params, n_params)) for layer in layers}

    for i in range(n_params):
        for j in range(n_params):
            sim = param_cosine(state_dicts[i], state_dicts[j], layers)
            for k, v in cos_sims.items():
                cos_sims[k][i][j] = sim[k]

    for k, v in cos_sims.items():
        print(f'layer {k} cosine similarity matrix')
        print(v)
    return cos_sims


def compute_grad_state_dicts(model, state_dicts):
    grad_state_dicts = copy.deepcopy(state_dicts)
    global_state_dict = model.state_dict()
    for key, val in global_state_dict.items():
        for sd in grad_state_dicts:
            sd[key] = val - sd[key]
    return grad_state_dicts


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

res = process_dataset(args.dataset_name,
                      split_type='idx_split',
                      model_name=args.model_name,
                      GRL=args.GRL,
                      MaskedLM=args.MaskedLM)
clients, train_datasets, test_datasets, train_dataset, test_dataset, model = res
doc_index = test_dataset.doc_index

model_name = args.model_name.replace('/', '_')
p = f'ckpt/{args.alg}/{args.dataset_name}_{model_name}'
ckpt = os.path.join(base_dir, p)
cs = CentralizedSystem(clients, train_datasets, test_datasets, train_dataset, test_dataset, model, device, ckpt,
                       batch_size=args.batch_size)
if args.load_model:
    model = copy.deepcopy(cs.model)

    p = f'ckpt/{args.alg}/AIMed_{model_name}'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_AIMed = copy.deepcopy(cs.model)

    p = f'ckpt/{args.alg}/AIMed_797|797_{model_name}'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_AIMed_797_797 = copy.deepcopy(cs.model)

    p = f'ckpt/FedAvg/AIMed*AIMed_797|797*LLL_distilbert-base-cased'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_FedAvg = copy.deepcopy(cs.model)

    p = f'ckpt/FedAvg/AIMed*AIMed_797|797*LLL_distilbert-base-cased_0'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_AIMed_FedAvg = copy.deepcopy(cs.model)

    p = f'ckpt/FedAvg/AIMed*AIMed_797|797*LLL_distilbert-base-cased_1'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_AIMed_1280_FedAvg = copy.deepcopy(cs.model)

    p = f'ckpt/FedAvg/AIMed*AIMed_797|797*LLL_distilbert-base-cased_2'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_AIMed_797_FedAvg = copy.deepcopy(cs.model)

    # p = f'ckpt/{args.alg}/BioInfer_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # print(ckpt)
    # cs.load(ckpt)
    # model_BioInfer = copy.deepcopy(cs.model)

    # p = f'ckpt/{args.alg}/HPRD50_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # print(ckpt)
    # cs.load(ckpt)
    # model_HPRD50 = copy.deepcopy(cs.model)

    # p = f'ckpt/{args.alg}/IEPA_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # print(ckpt)
    # cs.load(ckpt)
    # model_IEPA = copy.deepcopy(cs.model)

    p = f'ckpt/{args.alg}/LLL_{model_name}'
    ckpt = os.path.join(base_dir, p)
    print(ckpt)
    cs.load(ckpt)
    model_LLL = copy.deepcopy(cs.model)

    param_state_dicts = [model_AIMed_FedAvg.state_dict(), model_AIMed_1280_FedAvg.state_dict(),
                         model_AIMed_797_FedAvg.state_dict()]
    grad_state_dicts = compute_grad_state_dicts(model_FedAvg, param_state_dicts)
    get_grad_cosine(grad_state_dicts)
