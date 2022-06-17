import os

from torchvision import datasets as tv_datasets
from torchvision import transforms
import h5py
import json
import numpy as np
from .tokenizers import *
from .datasets import *
from .models import *
from .utils.fl_utils import generate_idxes_dirichlet, generate_idxes_group, get_embedding_Kmeans

logger = logging.getLogger(os.path.basename(__file__))
base_dir = os.path.expanduser('~/FedBioNLP')


def process_dataset(dataset_name, model_name, split_type, n_clients, parser_args):
    res = {}
    clients = list(range(n_clients))
    train_datasets = {}
    test_datasets = {}
    n_classes = None
    transform_aug = None
    transform_normal = None
    task_type = None
    doc_index = None
    train_doc_index = None
    test_doc_index = None

    # load dataset
    if dataset_name == 'MNIST':
        n_classes = 10
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = tv_datasets.FashionMNIST(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_dataset = tv_datasets.FashionMNIST(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
    elif dataset_name == 'CIFAR10':
        n_classes = 10
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = tv_datasets.CIFAR10(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_dataset = tv_datasets.CIFAR10(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
        model = ImageConvNet()
    elif dataset_name == 'MNIST*USPS':
        n_classes = 10

        my_train_dataset = ImageCCSADataset
        my_test_dataset = ImageDataset
        model = ImageConvCCSANet()

        transform_aug = transform_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_MNIST_dataset = tv_datasets.MNIST(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_MNIST_dataset = tv_datasets.MNIST(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_MNIST_data = train_MNIST_dataset.data
        train_MNIST_data = np.expand_dims(train_MNIST_data, -1).repeat(3, -1)
        test_MNIST_data = test_MNIST_dataset.data
        test_MNIST_data = np.expand_dims(test_MNIST_data, -1).repeat(3, -1)

        train_USPS_dataset = tv_datasets.USPS(root=os.path.join(base_dir, 'data'), train=True, download=True)
        test_USPS_dataset = tv_datasets.USPS(root=os.path.join(base_dir, 'data'), train=False, download=True)
        train_USPS_data = train_USPS_dataset.data
        train_USPS_data = np.expand_dims(train_USPS_data, -1).repeat(3, -1)
        test_USPS_data = test_USPS_dataset.data
        test_USPS_data = np.expand_dims(test_USPS_data, -1).repeat(3, -1)

        train_args = [train_USPS_data, train_MNIST_data, train_USPS_dataset.targets, train_MNIST_dataset.targets]
        test_args = [test_MNIST_data, test_MNIST_dataset.targets]


    elif dataset_name == 'CIFAR100':
        n_classes = 100
        my_train_dataset = my_test_dataset = ImageDataset

        transform_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = tv_datasets.CIFAR100(root=os.path.join(base_dir, 'data'),
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
        test_dataset = tv_datasets.CIFAR100(root=os.path.join(base_dir, 'data'),
                                            train=False,
                                            transform=transforms.ToTensor())
        train_args = [train_dataset.data, np.array(train_dataset.targets)]
        test_args = [test_dataset.data, np.array(test_dataset.targets)]
    else:
        data_file = os.path.join(base_dir, f'data/{dataset_name}_data.h5')

        with h5py.File(data_file, 'r+') as df:
            attributes = json.loads(df["attributes"][()])

            label_vocab = attributes['label_vocab']
            index_list = attributes['index_list']
            train_idx = attributes['train_index_list']
            test_idx = attributes['test_index_list']
            task_type = attributes['task_type']
            if 'doc_index' in attributes:
                doc_index = attributes['doc_index']
            else:
                doc_index = {str(i): 0 for i in index_list}
            logger.info(label_vocab)

            # ls["attributes"][()] = json.dumps(attributes)

            data = []
            targets = []
            unique_docs = set()
            # specifically for wikiner, ploner
            if task_type == 'seq_tagging':
                args = {'text': [],
                        'label': [],
                        'doc': []}
                special_args = {'label vocab': label_vocab}
                n_classes = attributes['num_labels']
                if parser_args.model_name == 'LSTM':
                    model = TokenClassificationLSTM(n_classes)
                else:
                    model = TokenClassificationBERT(model_name, n_classes)
                tokenizer = model.tokenizer
                my_train_dataset = my_test_dataset = model.dataset

                t = tqdm(index_list)
                for idx in t:
                    sentence = df['X'][str(idx)][()]
                    sentence = [i.decode('UTF-8') for i in sentence]
                    args['text'].append(' '.join(sentence))

                    label = df['Y'][str(idx)][()]
                    label = [i.decode('UTF-8') for i in label]
                    args['label'].append([label_vocab[y] for y in label])

                    doc = doc_index[str(idx)]
                    args['doc'].append(doc)
                    unique_docs.add(doc)

                args = tokenizer(args, special_args, parser_args, model_name)
            # specifically for squad_1.1
            elif task_type == 'reading_comprehension':
                tokenizer = nlp_tokenizer
                args = {'text': [],
                        'label': [],
                        'doc': []}
                special_args = {}
                for i in df['context_X'].keys():
                    question_components = []
                    question = df['question_X'][i][()].decode('UTF-8')
                    answer_start = df['Y'][i][()][0]
                    answer_end = df['Y'][i][()][1]
                    answer = df['context_X'][i][()].decode('UTF-8')[answer_start: answer_end]

                    question_components.append(question)
                    question_components.append(answer)
                    data.append(" ".join(question_components))
            # specifically for cnn_dailymail, cornell_movie_dialogue
            elif task_type == 'sequence_to_sequence':
                tokenizer = nlp_tokenizer
                args = {}
                special_args = {}
                for i in df['Y'].keys():
                    sentence = df['Y'][i][()].decode('UTF-8')
                    data.append(sentence)
            # specifically for semeval_2010_task8...

            elif task_type == 'relation_extraction':
                n_classes = attributes['num_labels']

                model = REModel(model_name, n_classes, parser_args.num_gcn_layers)
                special_args = {}
                # model = RELatentGCNModel(model_name, n_classes)
                # if parser_args.n_clusters:
                #     cluster_models = [REModel(model_name, n_classes, parser_args.num_gcn_layers) for _ in
                #                       range(parser_args.n_clusters)]
                #     res.update({'cluster_models': cluster_models})
                # if parser_args.horizon:
                #     model = REHorizonModel(model_name, n_classes)
                # if parser_args.CCSA:
                #     model = RECCSAModel(model_name, n_classes)
                if parser_args.alg == 'DomainGRL':
                    model = REDomainGRLModel(model_name, n_classes, parser_args.num_gcn_layers, parser_args.lambda_)
                if parser_args.alg == 'DomainMMD':
                    model = REDomainMMDModel(model_name, n_classes, parser_args.num_gcn_layers, parser_args.lambda_)
                my_train_dataset = my_test_dataset = model.dataset
                n_classes = attributes['num_labels']

                try:
                    tokenizer = re_dep_tokenizer
                    args = {'text': [],
                            'e_text': [],
                            'dep_text': [],
                            'dep_e_text': [],
                            'dependency': [],
                            'doc': [],
                            'label': []}
                    for idx in index_list:
                        text = df['text'][str(idx)][()].decode('UTF-8')
                        e_text = df['e_text'][str(idx)][()].decode('UTF-8')
                        dep_text = df['dep_text'][str(idx)][()].decode('UTF-8')
                        dep_e_text = df['dep_e_text'][str(idx)][()].decode('UTF-8')
                        dependency = df['dependency'][str(idx)][()].decode('UTF-8')
                        label = df['label'][str(idx)][()].decode('UTF-8')
                        doc = doc_index[str(idx)]

                        unique_docs.add(doc)

                        args['text'].append(text)
                        args['e_text'].append(e_text)
                        args['dep_text'].append(dep_text)
                        args['dep_e_text'].append(dep_e_text)
                        args['dependency'].append(dependency)
                        args['label'].append(label_vocab[label])
                        args['doc'].append(doc)
                    args = tokenizer(args, model_name, parser_args.mlm_method, parser_args.mlm_prob, parser_args.K_LCA)
                except:
                    tokenizer = re_tokenizer
                    args = {'text': [],
                            'label': [],
                            'doc': []}

                    for idx in index_list:
                        sentence = df['X'][str(idx)][()].decode('UTF-8')
                        label = df['Y'][str(idx)][()].decode('UTF-8')
                        doc = doc_index[str(idx)]

                        args['text'].append(sentence)
                        args['label'].append(label_vocab[label])
                        args['doc'].append(doc)
                        unique_docs.add(doc)

                    args = tokenizer(args, model_name)
                n_docs = len(unique_docs)
                if parser_args.alg == 'MTL':
                    model = REMTLModel(model_name, n_classes, parser_args.num_gcn_layers, n_docs)



            # specifically for 20news, agnews, sst_2, sentiment140, semeval_2010_task8
            elif task_type == 'text_classification':
                args = {'text': [],
                        'label': [],
                        'doc': []}
                special_args = {}

                n_classes = attributes['num_labels']
                if parser_args.model_name == 'LSTM':
                    model = SequenceClassificationLSTM(n_classes)
                else:
                    model = SequenceClassificationBERT(model_name, n_classes, parser_args.load_pretrain)
                tokenizer = model.tokenizer
                my_train_dataset = my_test_dataset = model.dataset

                for idx in index_list:
                    sentence = df['X'][str(idx)][()].decode('UTF-8')
                    args['text'].append(sentence)
                    label = df['Y'][str(idx)][()].decode('UTF-8')
                    args['label'].append(label_vocab[label])
                    doc = doc_index[str(idx)]
                    args['doc'].append(doc)
                    unique_docs.add(doc)

                args = tokenizer(args, parser_args, model_name)
            else:
                raise NotImplementedError

    train_args = {k: [v[i] for i in train_idx] for k, v in args.items()}
    train_args = dict(train_args, **special_args)
    test_args = {k: [v[i] for i in test_idx] for k, v in args.items()}
    test_args = dict(test_args, **special_args)
    num_train, num_test = len(train_idx), len(test_idx)
    logger.info('creating centralized train/test dataset')

    centralized_train_dataset = my_train_dataset(train_args, num_train, n_classes, transform_aug, train_doc_index)
    centralized_test_dataset = my_test_dataset(test_args, num_test, n_classes, transform_normal, test_doc_index)
    logger.info(f'number of classes: {n_classes}')
    logger.info(f'number of samples of train/test dataset: {num_train}/{num_test}')

    if split_type == 'centralized':
        clients = [0]
        train_datasets = {0: centralized_train_dataset}
        test_datasets = {0: centralized_test_dataset}
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'doc_split':
        n_docs = len(unique_docs)
        clients = list(range(n_docs))

        logger.info(f'unique docs: {unique_docs}')

        for d in unique_docs:
            cur_train_idx = [idx for idx in train_idx if doc_index[str(idx)] == d]
            cur_test_idx = [idx for idx in test_idx if doc_index[str(idx)] == d]
            train_args = {k: [v[idx] for idx in cur_train_idx] for k, v in args.items()}
            train_args = dict(train_args, **special_args)
            test_args = {k: [v[idx] for idx in cur_test_idx] for k, v in args.items()}
            test_args = dict(test_args, **special_args)

            num_train, num_test = len(cur_train_idx), len(cur_test_idx)
            logger.info('creating doc train/test datasets')

            train_dataset = my_train_dataset(train_args, num_train, n_classes, transform_aug, train_doc_index)
            train_datasets.update({d: train_dataset})
            test_dataset = my_test_dataset(test_args, num_test, n_classes, transform_normal, test_doc_index)
            test_datasets.update({d: test_dataset})

            logger.info(f'doc: {d}, number of samples of train/test dataset: {num_train}/{num_test}')
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'label_split':
        logger.info('start splitting data according to label shift.')
        if parser_args.n_clusters == 0:
            train_idxes = generate_idxes_dirichlet(train_args['label'], n_clients, n_classes, parser_args.beta)
            test_idxes = generate_idxes_dirichlet(test_args['label'], n_clients, n_classes, parser_args.beta)
        else:
            train_idxes = generate_idxes_group(train_args['label'], n_clients, n_classes, parser_args.beta,
                                               n_clients // parser_args.n_clusters, parser_args.seed)
            test_idxes = generate_idxes_group(test_args['label'], n_clients, n_classes, parser_args.beta,
                                              n_clients // parser_args.n_clusters, parser_args.seed)
        for i in range(n_clients):
            train_idx = train_idxes[i]
            num_train = len(train_idx)
            client_train_args = {k: [v[idx] for idx in train_idx] for k, v in train_args.items()}
            client_train_args = dict(client_train_args, **special_args)
            train_dataset = my_train_dataset(client_train_args, num_train, n_classes, transform_aug, train_doc_index)
            train_datasets.update({i: train_dataset})

            test_idx = test_idxes[i]
            num_test = len(test_idx)
            client_test_args = {k: [v[idx] for idx in test_idx] for k, v in test_args.items()}
            client_test_args = dict(client_test_args, **special_args)
            test_dataset = my_test_dataset(client_test_args, num_test, n_classes, transform_normal, test_doc_index)
            test_datasets.update({i: test_dataset})
            logger.info(f'client: {i}, number of samples of train/test dataset: {num_train}/{num_test}')

        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    elif split_type == 'feature_split':
        logger.info('start splitting data according to feature shift.')
        train_path = os.path.join(base_dir, f'embedding/{parser_args.dataset_name}_train.pkl')
        test_path = os.path.join(base_dir, f'embedding/{parser_args.dataset_name}_test.pkl')
        train_classes = get_embedding_Kmeans(train_args['text'], parser_args.n_clients, train_path)
        test_classes = get_embedding_Kmeans(test_args['text'], parser_args.n_clients, test_path)

        train_idxes = [[] for _ in range(n_clients)]
        for i, c in enumerate(train_classes):
            train_idxes[c].append(i)
        test_idxes = [[] for _ in range(n_clients)]
        for i, c in enumerate(test_classes):
            test_idxes[c].append(i)

        for i in range(n_clients):
            train_idx = train_idxes[i]
            num_train = len(train_idx)
            client_train_args = {k: [v[idx] for idx in train_idx] for k, v in train_args.items()}
            client_train_args = dict(client_train_args, **special_args)
            train_dataset = my_train_dataset(client_train_args, num_train, n_classes, transform_aug, train_doc_index)
            train_datasets.update({i: train_dataset})

            test_idx = test_idxes[i]
            num_test = len(test_idx)
            client_test_args = {k: [v[idx] for idx in test_idx] for k, v in test_args.items()}
            client_test_args = dict(client_test_args, **special_args)
            test_dataset = my_test_dataset(client_test_args, num_test, n_classes, transform_normal, test_doc_index)
            test_datasets.update({i: test_dataset})
            logger.info(f'client: {i}, number of samples of train/test dataset: {num_train}/{num_test}')
        return clients, train_datasets, test_datasets, centralized_train_dataset, centralized_test_dataset, model, res
    else:
        raise Exception("Invalid split type.")
