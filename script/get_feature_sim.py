import copy
import os.path
from FedBioNLP.processors import process_dataset
import argparse
from FedBioNLP import cmp_CKA_sim, cmp_l2_norm

parser = argparse.ArgumentParser()
base_dir = os.path.expanduser('~/FedBioNLP')
parser.add_argument('--load_model', default=True)
parser.add_argument('--load_rep', default=False)
parser.add_argument('--alg', default='centralized')
parser.add_argument('--dataset_name', default='AIMed_1|2*PGR_2797')
parser.add_argument('--model_name', default='distilbert-base-cased',
                    choices=['distilbert-base-cased',
                             'bert-base-cased',
                             'dmis-lab/biobert-v1.1'])
parser.add_argument('--mode', default='average', choices=['average', 'first', 'first_last', 'squeeze'])
parser.add_argument('--batch_size', default=32)
parser.add_argument('--GRL', default=False)
parser.add_argument('--MaskedLM', default=False)
args = parser.parse_args()


def print_false():
    for i, test_loader in enumerate(cs.test_loaders):
        metric, inputs, labels, features, logits = cs.test_model(data_loader=test_loader)
        fn = metric['false_negative']
        fp = metric['false_positive']
        print(fn)
        print(fp)
        text = test_loader.dataset.text
        fn_data = [text[idx] for idx in fn]
        fp_data = [text[idx] for idx in fp]
        print('false negative data')
        # for x in fn_data:
        #     print(x)
        #     print(' ')
        print('false positive data')
        # for x in fp_data:
        #     print(x)
        #     print(' ')


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
                       ckpt, grad_cos=grad_cos,
                       batch_size=args.batch_size)
if args.load_model:
    cs.load(ckpt)
    model = copy.deepcopy(cs.model)
    print(ckpt)
    logging.info('model has been loaded')
    metric, inputs, labels, features, logits = cs.test_model()

    # p = f'ckpt/{args.alg}/AIMed_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # cs.load(ckpt)
    # print(ckpt)
    # print_false()
    #
    # p = f'ckpt/{args.alg}/BioInfer_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # cs.load(ckpt)
    # print(ckpt)
    # print_false()
    #
    # p = f'ckpt/{args.alg}/HPRD50_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # cs.load(ckpt)
    # print(ckpt)
    # print_false()
    #
    # p = f'ckpt/{args.alg}/IEPA_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # cs.load(ckpt)
    # print(ckpt)
    # print_false()
    #
    p = f'ckpt/{args.alg}/AIMed_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_AIMed = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_AIMed, logits = cs.test_model()

    p = f'ckpt/{args.alg}/BioInfer_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_BioInfer = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_BioInfer, logits = cs.test_model()

    p = f'ckpt/{args.alg}/HPRD50_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_HPRD50 = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_HPRD50, logits = cs.test_model()

    p = f'ckpt/{args.alg}/IEPA_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_IEPA = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_IEPA, logits = cs.test_model()

    p = f'ckpt/{args.alg}/IEPA_6.7_size_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_IEPA_size = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_IEPA_size, logits = cs.test_model()

    p = f'ckpt/{args.alg}/LLL_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_LLL = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_LLL, logits = cs.test_model()

    p = f'ckpt/{args.alg}/AIMed*BioInfer*HPRD50*IEPA*LLL_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_centralized = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_centralized, logits = cs.test_model()

    p = f'ckpt/{args.alg}/PGR_Q1_{model_name}'
    ckpt = os.path.join(base_dir, p)
    cs.load(ckpt)
    model_PGR_Q1 = copy.deepcopy(cs.model)
    print(ckpt)
    metric, inputs, labels, features_PGR_Q1, logits = cs.test_model()

    # p = f'ckpt/FedAvg/AIMed*BioInfer*HPRD50*IEPA*LLL_{model_name}'
    # ckpt = os.path.join(base_dir, p)
    # cs.load(ckpt)
    # print(ckpt)
    # metric, inputs, labels, features_FedAvg, logits = cs.test_model()

    print('AIMed')
    cmp_CKA_sim(features_AIMed, features, mode=args.mode)
    cmp_l2_norm(model_AIMed, model)
    print('BioInfer')
    cmp_CKA_sim(features_BioInfer, features, mode=args.mode)
    cmp_l2_norm(model_BioInfer, model)
    print('HPRD50')
    cmp_CKA_sim(features_HPRD50, features, mode=args.mode)
    cmp_l2_norm(model_HPRD50, model)
    print('IEPA')
    cmp_CKA_sim(features_IEPA, features, mode=args.mode)
    cmp_l2_norm(model_IEPA, model)
    print('IEPA_6.7_size')
    cmp_CKA_sim(features_IEPA_size, features, mode=args.mode)
    cmp_l2_norm(model_IEPA_size, model)
    print('LLL')
    cmp_CKA_sim(features_LLL, features, mode=args.mode)
    cmp_l2_norm(model_LLL, model)
    print('centralized')
    cmp_CKA_sim(features_centralized, features, mode=args.mode)
    cmp_l2_norm(model_centralized, model)
    print('PGR_Q1')
    cmp_CKA_sim(features_PGR_Q1, features, mode=args.mode)
    cmp_l2_norm(model_PGR_Q1, model)
