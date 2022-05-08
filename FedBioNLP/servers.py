import copy
import torch
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch import optim
from torch import nn
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))


class Server(object):
    def __init__(self, model, n_clients, selected_client_idx, agg_weights, server_dataset=None):
        self.model = copy.deepcopy(model)
        self.n_clients = n_clients
        self.selected_client_idx = selected_client_idx
        self.agg_weights = agg_weights
        self.client_models = None
        self.model_state_dicts = []

        if server_dataset is not None:
            self.dataloader = DataLoader(server_dataset)

    def aggregate(self, cur_ite):
        server_dict = self.model.state_dict()
        for key in server_dict:
            server_dict[key] = 0
            for state_dict in self.model_state_dicts:
                server_dict[key] += state_dict[key]
            server_dict[key] /= self.n_clients
        self.model.load_state_dict(server_dict)

    def cmp_cur_prob(self, cur_ite):
        probs = [self.agg_weights[idx] for idx in self.selected_client_idx[cur_ite]]
        probs = [p / sum(probs) for p in probs]
        print(f'aggregation probs:{probs}')
        return probs

    def test(self, model):
        test_labels, test_predict_labels = (np.array([]) for _ in range(2))
        for i, data in enumerate(self.dataloader):
            inputs, labels = data
            outputs = model(inputs)

            predict_labels = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            test_labels = np.concatenate([test_labels, labels.numpy()])
            test_predict_labels = np.concatenate([test_predict_labels, predict_labels])

        test_acc = accuracy_score(test_labels, test_predict_labels)
        conf_mtx = confusion_matrix(test_labels, test_predict_labels, labels=list(range(10)))
        return test_acc, conf_mtx

    def send_global_model(self):
        state_dict = self.model.state_dict()
        return state_dict
