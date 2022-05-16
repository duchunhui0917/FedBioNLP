import json

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
        labels = labels[0]
        metric = {}
        if test_mode:
            pred_labels = np.argmax(logits, axis=-1)
            false_negative = np.where(pred_labels - labels == -1)[0]
            false_positive = np.where(pred_labels - labels == 1)[0]
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
            metric.update({'false_negative': false_negative, 'false_positive': false_positive})

        else:
            score, pred_labels = logits.max(-1)
            acc = float((pred_labels == labels).long().sum()) / labels.size(0)
            metric.update({'acc': acc})
        return metric


class NLPDataset(BaseDataset):
    def __init__(self, args, n_classes=None, transform=None, doc_index=None):
        super(NLPDataset, self).__init__(args, n_classes, transform, doc_index)
        self.data = args[0]
        self.targets = torch.LongTensor(args[1])

    def __getitem__(self, item):
        data = [self.data[item]]
        target = self.targets[item]
        return data, target


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


class MaskedLMDataset(BaseDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(MaskedLMDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.input_ids = torch.LongTensor(args['input_ids'])
        self.input_mask = torch.LongTensor(args['input_mask'])
        self.mlm_input_ids = torch.LongTensor(args['mlm_input_ids'])
        self.mlm_mask = torch.LongTensor(args['mlm_mask'])

    def __getitem__(self, item):
        data = [self.input_ids[item], self.input_mask[item], self.mlm_mask[item]]
        label = [self.mlm_input_ids[item]]
        return data, label

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
        input_ids, input_mask, mlm_mask = inputs
        labels = labels[0]
        metric = {}
        if test_mode:
            pred_labels = np.argmax(logits, axis=-1)
            acc = float((pred_labels == labels).sum() / mlm_mask.size)
            mask_acc = float(((pred_labels == labels) * mlm_mask).sum() / mlm_mask.sum())
            metric.update({'acc': acc, 'mask_acc': mask_acc})
        else:
            score, pred_labels = logits.max(-1)
            acc = float((pred_labels == labels).long().sum() / (mlm_mask.size(0) * mlm_mask.size(1)))
            mask_acc = float(((pred_labels == labels).long() * mlm_mask).sum() / mlm_mask.sum())
            metric.update({'acc': acc, 'mask_acc': mask_acc})
        return metric


class REGCNDataset(BaseDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(REGCNDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.input_ids = torch.LongTensor(args['input_ids'])
        self.input_mask = torch.LongTensor(args['input_mask'])
        self.mlm_input_ids = torch.LongTensor(args['mlm_input_ids'])
        self.valid_ids = torch.LongTensor(args['valid_ids'])
        self.e1_mask = torch.LongTensor(args['e1_mask'])
        self.e2_mask = torch.LongTensor(args['e2_mask'])
        self.dep_matrix = torch.LongTensor(args['dep_matrix'])
        self.label = torch.LongTensor(args['label'])

    def __getitem__(self, item):
        data = [self.input_ids[item], self.input_mask[item], self.valid_ids[item],
                self.e1_mask[item], self.e2_mask[item], self.dep_matrix[item],
                self.mlm_input_ids[item]]
        label = [self.label[item]]
        return data, label


class REDataset(BaseDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(REDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.input_ids = torch.LongTensor(args['input_ids'])
        self.input_mask = torch.LongTensor(args['input_mask'])
        self.e1_mask = torch.LongTensor(args['e1_mask'])
        self.e2_mask = torch.LongTensor(args['e2_mask'])
        self.label = torch.LongTensor(args['label'])

    def __getitem__(self, item):
        data = [self.input_ids[item], self.input_mask[item], self.e1_mask[item], self.e2_mask[item]]
        label = [self.label[item]]
        return data, label


class REGRLDataset(REDataset):
    def __init__(self, args, n_samples, n_classes=None, transform=None, doc_index=None):
        super(REGRLDataset, self).__init__(args, n_samples, n_classes, transform, doc_index)
        self.doc = torch.LongTensor(args['doc'])

    def __getitem__(self, item):
        data = [self.input_ids[item], self.input_mask[item], self.e1_mask[item], self.e2_mask[item], self.doc[item]]
        label = [self.label[item]]
        return data, label


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
