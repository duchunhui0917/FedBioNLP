import json

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import logging
import os
from .tokenizers import *

logger = logging.getLogger(os.path.basename(__file__))


class BaseDataset(Dataset):
    def __init__(self, args, n_samples, n_classes, transform, doc_index):
        self.args = args
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.transform = transform
        self.doc_index = doc_index

    def __len__(self):
        return self.n_samples

    @staticmethod
    def metric(inputs, labels, logits, test_mode=True, case_study=False):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        labels = labels[0]
        logits = logits[0]
        metric = {}
        if test_mode:
            pred_labels = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels, average='macro')
            recall = recall_score(labels, pred_labels, average='macro')
            f1 = f1_score(labels, pred_labels, average='macro')
            cf = confusion_matrix(labels, pred_labels)
            cf = json.dumps(cf.tolist())
            logger.info(f'confusion matrix\n{cf}')
            metric.update({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
            for key, val in metric.items():
                logger.info(f'{key}: {val:.4f}')
        else:
            score, pred_labels = logits.max(-1)
            acc = float((pred_labels == labels).long().sum()) / labels.size(0)
            metric.update({'acc': acc})
        return metric


class NLPDataset(BaseDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(NLPDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.text = args['text']
        self.doc = args['doc']

        if isinstance(args['input_ids'][0][0], int):
            self.input_ids = torch.LongTensor(args['input_ids'])
        else:
            self.input_ids = torch.FloatTensor(args['input_ids'])
        if 'attention_mask' in args:
            self.attention_mask = torch.LongTensor(args['attention_mask'])
        else:
            self.attention_mask = None
        self.label = torch.LongTensor(args['label'])

    def __getitem__(self, item):
        if self.attention_mask is not None:
            data = [self.input_ids[item], self.attention_mask[item]]
        else:
            data = [self.input_ids[item]]
        label = [self.label[item]]
        return data, label


class NERDataset(NLPDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(NERDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.label_vocab = args['label vocab']

    def metric(self, inputs, labels, logits, test_mode=True, case_study=False):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        labels = labels[0]  # (B, L)
        logits = logits[0]  # (B, L, C)
        metric = {}
        if test_mode:
            pred_labels = np.argmax(logits, axis=-1)  # (B, L)
            labels, pred_labels = labels.flatten(), pred_labels.flatten()
            acc = accuracy_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels, average='macro')
            recall = recall_score(labels, pred_labels, average='macro')
            f1 = f1_score(labels, pred_labels, average='macro')
            cf = confusion_matrix(labels, pred_labels)
            cf = json.dumps(cf.tolist())
            logger.info(f'confusion matrix\n{cf}')
            metric.update({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
            for key, val in metric.items():
                logger.info(f'{key}: {val:.4f}')
            true_mask = labels != self.label_vocab['O']
            true_acc = float(((pred_labels == labels) & true_mask).sum() / true_mask.sum())
            logger.info(f'true acc: {true_acc:.4f}')
        else:
            _, pred_labels = logits.max(-1)
            labels, pred_labels = labels.flatten(), pred_labels.flatten()
            acc = float((pred_labels == labels).long().sum()) / labels.size(0)
            metric.update({'acc': acc})
            true_mask = labels != self.label_vocab['O']
            true_acc = float(((pred_labels == labels) & true_mask).sum() / true_mask.sum())
            metric.update({'true acc': true_acc})
        return metric


class ImageDataset(BaseDataset):
    def __init__(self, args, n_classes=None, transform=None, doc_index=None):
        super(ImageDataset, self).__init__(args, n_classes, transform, doc_index)
        self.data = args[0]
        self.targets = torch.LongTensor(args[1])

    def __getitem__(self, item):
        data = [self.transform(self.data[item])]
        target = [self.targets[item]]
        return data, target


class ImageCCSADataset(BaseDataset):
    def __init__(self, args, n_classes=None, transform=None, doc_index=None):
        super(ImageCCSADataset, self).__init__(args, n_classes, transform, doc_index)
        num_sample = 5
        self.src_x = args[0]
        self.tgt_x = args[1]
        self.src_y = torch.LongTensor(args[2])
        self.tgt_y = torch.LongTensor(args[3])

        tgt_idx = [np.where(self.tgt_y == i)[0] for i in range(n_classes)]
        tgt_idx = [np.random.choice(idx, num_sample) for idx in tgt_idx]
        tgt_idx = np.concatenate(tgt_idx)
        self.tgt_x = self.tgt_x[tgt_idx]
        self.tgt_y = self.tgt_y[tgt_idx]

        positive, negative = [], []

        for trs in range(len(self.src_y)):
            for trt in range(len(self.tgt_y)):
                if self.src_y[trs] == self.tgt_y[trt]:
                    positive.append([trs, trt])
                else:
                    negative.append([trs, trt])
        logger.info(f"num of positive/negative pairs: {len(positive)}/{len(negative)}")
        np.random.shuffle(negative)
        self.pairs = positive + negative[:3 * len(positive)]
        random.shuffle(self.pairs)
        self.n_samples = len(self.pairs)

    def __getitem__(self, item):
        src_idx, tgt_idx = self.pairs[item]
        data = [self.transform(self.src_x[src_idx]), self.transform(self.tgt_x[tgt_idx])]
        targets = [self.src_y[src_idx], self.tgt_y[tgt_idx]]
        return data, targets

    @staticmethod
    def metric(inputs, labels, logits, test_mode=True):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        src_labels, tgt_labels = labels
        src_logits, tgt_logits = logits
        metric = {}
        src_score, src_pred_labels = src_logits.max(-1)
        src_acc = float((src_pred_labels == src_labels).long().sum()) / src_labels.size(0)
        tgt_score, tgt_pred_labels = tgt_logits.max(-1)
        tgt_acc = float((tgt_pred_labels == tgt_labels).long().sum()) / tgt_labels.size(0)

        metric.update({'src_acc': src_acc, 'tgt_acc': tgt_acc})
        return metric


class REDataset(BaseDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(REDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.text = args['text']
        self.tokens = args['tokens']
        self.doc = args['doc']

        self.input_ids = torch.LongTensor(args['input_ids'])
        self.attention_mask = torch.LongTensor(args['attention_mask'])
        self.e1_mask = torch.LongTensor(args['e1_mask'])
        self.e2_mask = torch.LongTensor(args['e2_mask'])
        self.label = torch.LongTensor(args['label'])

        zero = torch.zeros_like(self.input_ids)
        self.mlm_input_ids = torch.LongTensor(args['mlm_input_ids']) if 'mlm_input_ids' in args else zero
        self.mlm_labels = torch.LongTensor(args['mlm_labels']) if 'mlm_labels' in args else zero
        self.valid_ids = torch.LongTensor(args['valid_ids']) if 'valid_ids' in args else zero
        # self.dep_matrix = torch.LongTensor(args['dep_matrix']) if 'dep_matrix' in args else zero
        if 'dep_text' in args:
            self.dep_text = args['dep_text']
        self.dep_matrix = zero

    def __getitem__(self, item):
        data = [self.input_ids[item], self.attention_mask[item], self.e1_mask[item], self.e2_mask[item],
                self.valid_ids[item], self.mlm_input_ids[item], self.dep_matrix[item]]
        label = [self.label[item], self.mlm_labels[item]]
        return data, label

    def metric(self, inputs, labels, logits, test_mode=True, case_study=False):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            case_study:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        # input_ids, attention_mask, e1_mask, e2_mask, valid_ids, mlm_input_ids, dep_matrix = inputs
        re_labels, mlm_labels = labels
        re_logits = logits[0]

        metric = {}
        if test_mode:
            re_pred_labels = np.argmax(re_logits, axis=-1)
            acc = accuracy_score(re_labels, re_pred_labels)
            precision = precision_score(re_labels, re_pred_labels, average='macro')
            recall = recall_score(re_labels, re_pred_labels, average='macro')
            f1 = f1_score(re_labels, re_pred_labels, average='macro')
            cf = confusion_matrix(re_labels, re_pred_labels)
            cf = json.dumps(cf.tolist())
            logger.info(f'confusion matrix\n{cf}')
            metric.update({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
            for key, val in metric.items():
                logger.info(f'{key}: {val:.4f}')
            # if np.max(mlm_labels) > 0:
            #     _, mlm_logits, unmask_logits = logits
            #     n1, n2 = len(mlm_logits), len(mlm_logits[0])
            #     score = np.zeros((n1, n2))
            #     for i in range(n1):
            #         for j, (logits, label, input_id) in enumerate(zip(mlm_logits[i], mlm_labels[i], mlm_input_ids[i])):
            #             logits = np.exp(logits)
            #             logits /= np.sum(logits, axis=-1)
            #             if label != -100:
            #                 score[i][j] = logits[label]
            #             else:
            #                 score[i][j] = logits[input_id]
            #
            #     unmask_labels = input_ids.copy()
            #     unmask_labels[(mlm_input_ids == 103) | (input_ids == 0)] = -100
            #
            #     mlm_pred_labels = np.argmax(mlm_logits, axis=-1)
            #     mlm_mask = (mlm_labels != -100)
            #     mlm_acc = ((mlm_pred_labels == mlm_labels) & mlm_mask).sum() / mlm_mask.sum()
            #     metric.update({'mlm acc': mlm_acc})
            #     logger.info(f'mlm acc: {mlm_acc:.4f}')
            #
            #     unmask_pred_labels = np.argmax(unmask_logits, axis=-1)
            #     unmask_mask = (unmask_labels != -100)
            #     unmask_acc = ((unmask_pred_labels == unmask_labels) & unmask_mask).sum() / unmask_mask.sum()
            #     metric.update({'unmask acc': unmask_acc})
            #     logger.info(f'unmask acc: {unmask_acc:.4f}')
            if case_study:
                self.case_study(re_pred_labels, re_labels, re_logits, score, mlm_labels)
        else:
            score, re_pred_labels = re_logits.max(-1)
            acc = float((re_pred_labels == re_labels).long().sum()) / re_labels.size(0)
            metric.update({'acc': acc})

            # if torch.max(mlm_labels) > 0:
            #     _, mlm_logits, unmask_logits = logits
            #     unmask_labels = input_ids.masked_fill((mlm_input_ids == 103) | (input_ids == 0), -100)
            #
            #     score, mlm_pred_labels = mlm_logits.max(-1)
            #     mlm_mask = (mlm_labels != -100)
            #     mlm_acc = float(((mlm_pred_labels == mlm_labels) & mlm_mask).sum() / mlm_mask.sum())
            #     metric.update({'mlm acc': mlm_acc})
            #
            #     score, unmask_pred_labels = unmask_logits.max(-1)
            #     unmask_mask = (unmask_labels != -100)
            #     unmask_acc = float(((unmask_pred_labels == unmask_labels) & unmask_mask).sum() / unmask_mask.sum())
            #     metric.update({'unmask acc': unmask_acc})

        return metric

    def case_study(self, pre_labels, labels, logits, score, mlm_labels):
        logits = np.exp(logits)
        logits /= np.sum(logits, axis=1, keepdims=True)
        for tc in range(self.n_classes):
            for fc in range(self.n_classes):
                if tc != fc:
                    loc = np.where((labels == tc) & (pre_labels == fc))[0]
                    for l in loc:
                        print(self.text[l] + f' {tc} {fc} {logits[l]}')
                        print(self.tokens[l])
                        print(score[l])
                        print(mlm_labels[l])

        # for tc in range(self.n_classes):
        #     loc = np.where(labels == tc)[0]
        #     bad_case += [self.text[l] + f'|{tc}' for l in loc]
        # for case in bad_case:
        #     print(case)


class REMDomainAdaptationDataset(REDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(REMDomainAdaptationDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.doc = torch.LongTensor(args['doc'])

    def __getitem__(self, item):
        data = [self.input_ids[item], self.attention_mask[item], self.e1_mask[item], self.e2_mask[item],
                self.valid_ids[item], self.dep_matrix[item]]
        label = [self.label[item], self.doc[item]]
        return data, label

    def metric(self, inputs, labels, logits, test_mode=True, case_study=False):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        re_labels, doc = labels
        re_logits, domain_logits = logits

        metric = {}
        if test_mode:
            re_pred_labels = np.argmax(re_logits, axis=-1)
            acc = accuracy_score(re_labels, re_pred_labels)
            precision = precision_score(re_labels, re_pred_labels, average='macro')
            recall = recall_score(re_labels, re_pred_labels, average='macro')
            f1 = f1_score(re_labels, re_pred_labels, average='macro')
            cf = confusion_matrix(re_labels, re_pred_labels)
            cf = json.dumps(cf.tolist())
            logger.info(f'confusion matrix\n{cf}')
            metric.update({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})
            for key, val in metric.items():
                logger.info(f'{key}: {val:.4f}')

            domain_pred_labels = np.argmax(domain_logits, axis=-1)
            domain_acc = accuracy_score(doc, domain_pred_labels)
            metric.update({'domain acc': domain_acc})
            logger.info(f'domain acc: {domain_acc:.4f}')
            if case_study:
                self.case_study(re_pred_labels, re_labels)
        else:
            score, re_pred_labels = re_logits.max(-1)
            acc = float((re_pred_labels == re_labels).long().sum()) / re_labels.size(0)
            metric.update({'acc': acc})

            score, domain_pred_labels = domain_logits.max(-1)
            domain_acc = float((domain_pred_labels == doc).long().sum() / doc.size(0))
            metric.update({'domain acc': domain_acc})
        return metric


class RelationExtractionMMDTrainDataset(BaseDataset):
    def __init__(self, args, n_classes=None, transform=None, doc_index=None):
        super(RelationExtractionMMDTrainDataset, self).__init__(args, n_classes, transform, doc_index)
        self.data = args[0]
        self.pos1 = args[1]
        self.pos2 = args[2]
        self.docs = args[3]
        self.targets = torch.LongTensor(args[4])

        src_idx = np.where(self.docs == 0)[0]
        tgt_idx = np.where(self.docs == 1)[0]
        src_num = len(src_idx)
        tgt_num = len(tgt_idx)

        if src_num > tgt_num:
            ratio = src_num // tgt_num
            tmp = np.repeat(tgt_idx, ratio)
            tgt_idx = np.concatenate([tgt_idx, np.random.choice(tmp, src_num - tgt_num, replace=False)])
            np.random.shuffle(tgt_idx)
        else:
            ratio = tgt_num // src_num
            tmp = np.repeat(src_idx, ratio)
            src_idx = np.concatenate([src_idx, np.random.choice(tmp, tgt_num - src_num, replace=False)])
            np.random.shuffle(src_idx)

        self.n_samples = max(src_num, tgt_num)

        ls = [self.data, self.pos1, self.pos2, self.targets]
        self.src_data, self.src_pos1, self.src_pos2, self.src_targets = [x[src_idx] for x in ls]
        self.tgt_data, self.tgt_pos1, self.tgt_pos2, self.tgt_targets = [x[tgt_idx] for x in ls]

    def __getitem__(self, item):
        src = [self.src_data[item], self.src_pos1[item], self.src_pos2[item]]
        tgt = [self.tgt_data[item], self.tgt_pos1[item], self.tgt_pos2[item]]
        data = src + tgt
        targets = [self.src_targets[item], self.tgt_targets[item]]
        return data, targets

    @staticmethod
    def metric(inputs, labels, logits, test_mode=True):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        src_labels, tgt_labels = labels
        src_logits, tgt_logits = logits
        metric = {}
        src_score, src_pred_labels = src_logits.max(-1)
        src_acc = float((src_pred_labels == src_labels).long().sum()) / src_labels.size(0)
        tgt_score, tgt_pred_labels = tgt_logits.max(-1)
        tgt_acc = float((tgt_pred_labels == tgt_labels).long().sum()) / tgt_labels.size(0)

        metric.update({'src_acc': src_acc, 'tgt_acc': tgt_acc})
        return metric


class RelationExtractionCCSADataset(BaseDataset):
    def __init__(self, args, n_classes=None, transform=None, doc_index=None):
        super(RelationExtractionCCSADataset, self).__init__(args, n_classes, transform, doc_index)
        self.data = args[0]
        self.pos1 = args[1]
        self.pos2 = args[2]
        self.docs = args[3]
        self.targets = torch.LongTensor(args[4])

        src_idx = np.where(self.docs == 0)[0]
        tgt_idx = np.where(self.docs == 1)[0]
        src_num = len(src_idx)
        tgt_num = len(tgt_idx)

        positive, negative = [], []

        self.src_y = self.targets[src_idx]
        self.tgt_y = self.targets[tgt_idx]
        for trs in range(len(self.src_y)):
            for trt in range(len(self.src_y)):
                if self.src_y[trs] == self.tgt_y[trt]:
                    positive.append([trs, trt])
                else:
                    negative.append([trs, trt])
        logger.info(f"num of positive/negative pairs: {len([positive])}/{len(negative)}")
        self.pairs = positive + negative
        random.shuffle(self.pairs)

    def __getitem__(self, item):
        src_idx, tgt_idx = self.pairs[item]
        data = [self.data[src_idx], self.pos1[src_idx], self.pos2[src_idx],
                self.data[tgt_idx], self.pos1[tgt_idx], self.pos2[tgt_idx]]
        targets = [self.targets[src_idx], self.targets[tgt_idx]]
        return data, targets

    @staticmethod
    def metric(inputs, labels, logits, test_mode=True):
        """
        logits and labels are numpy if test_mode is True else torch
        Args:
            inputs:
            labels:
            logits:
            test_mode:

        Returns:

        """
        src_labels, tgt_labels = labels
        src_logits, tgt_logits = logits
        metric = {}
        src_score, src_pred_labels = src_logits.max(-1)
        src_acc = float((src_pred_labels == src_labels).long().sum()) / src_labels.size(0)
        tgt_score, tgt_pred_labels = tgt_logits.max(-1)
        tgt_acc = float((tgt_pred_labels == tgt_labels).long().sum()) / tgt_labels.size(0)

        metric.update({'src_acc': src_acc, 'tgt_acc': tgt_acc})
        return metric
