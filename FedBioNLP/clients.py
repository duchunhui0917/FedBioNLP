import torch
import copy
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from FedBioNLP.utils.common_utils import AverageMeter
from collections import OrderedDict
from FedBioNLP.optimers import WPOptim
from FedBioNLP.utils.fl_utils import BatchIterator
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))


class BaseClient(object):
    def __init__(self, client_id, train_loader, test_loader,
                 model, device, batch_size, lr, opt, n_epochs,
                 n_batches, momentum=0):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_loader.dataset
        self.test_dataset = test_loader.dataset

        self.model = copy.deepcopy(model)
        self.device = device
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.n_batches = n_batches

        self.momentum = momentum

        if self.opt == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                       weight_decay=1e-5, momentum=0.9)
        else:
            raise Exception("Invalid optimizer. Must be 'SGD' or 'Adam'.")

    def train_model(self):
        model = nn.parallel.DataParallel(self.model)
        model.to(self.device)
        model.train()

        for epoch in range(self.n_epochs):
            avg_losses = None
            avg_metrics = None

            if self.n_batches == 0:
                t = tqdm(self.train_loader, ncols=0)
            else:
                ite_train_loader = BatchIterator(self.n_batches, self.train_loader)
                t = tqdm(ite_train_loader, ncols=0)

            for i, data in enumerate(t):
                d = OrderedDict(id=self.client_id)

                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.to(self.device) for x in inputs], [x.to(self.device) for x in labels]

                features, logits, losses = model(inputs, labels)
                losses[0].mean().backward()
                self.optimizer.step()

                metrics = self.train_dataset.metric(inputs, labels, logits, test_mode=False)

                if avg_metrics is None:
                    avg_metrics = {key: AverageMeter() for key in metrics.keys()}
                for key, val in metrics.items():
                    avg_metric = avg_metrics[key]
                    avg_metric.update(val, 1)
                    d.update({key: avg_metric.avg})

                if avg_losses is None:
                    avg_losses = [AverageMeter() for _ in range(len(losses))]

                for idx, loss in enumerate(losses):
                    avg_loss = avg_losses[idx]
                    avg_loss.update(loss.mean().item(), 1)
                    d.update({f'loss{idx}': avg_loss.avg})

                t.set_postfix(d)
        self.model.to('cpu')
        return self.model.state_dict()

    def receive_global_model(self, global_state_dict):
        local_state_dict = self.model.state_dict()
        for key, val in global_state_dict.items():
            local_state_dict[key] = self.momentum * local_state_dict[key] + (1 - self.momentum) * val
        self.model.load_state_dict(local_state_dict)


class HarmoFLClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader,
                 model, device, batch_size, lr, opt, n_epochs,
                 n_batches, momentum, perturbation=0.05):
        super(HarmoFLClient, self).__init__(client_id, train_loader, test_loader,
                                            model, device, batch_size, lr, opt, n_epochs,
                                            n_batches, momentum=0)
        self.perturbation = perturbation
        self.optimizer = WPOptim(params=self.model.parameters(), base_optimizer=optim.Adam, lr=self.lr,
                                 alpha=self.perturbation, weight_decay=1e-4)

    def train_model(self):
        model = nn.parallel.DataParallel(self.model)
        model.cuda()
        model.train()

        for epoch in range(self.n_epochs):
            avg_losses = None
            avg_metrics = None

            if self.n_batches == 0:
                t = tqdm(self.train_loader, ncols=0)
            else:
                ite_train_loader = BatchIterator(self.n_batches, self.train_loader)
                t = tqdm(ite_train_loader, ncols=0)

            for i, data in enumerate(t):
                d = OrderedDict(id=self.client_id)

                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                features, logits, losses = model(inputs, labels)
                # compute the gradient
                losses[0].mean().backward()
                # normalize the gradient and add it to the parameters
                self.optimizer.generate_delta(zero_grad=True)

                features, logits, losses = model(inputs, labels)
                # compute the gradient of the parameters with perturbation
                losses[0].mean().backward()
                # gradient descent
                self.optimizer.step(zero_grad=True)

                metrics = self.train_dataset.metric(inputs, labels, logits, test_mode=False)

                if avg_metrics is None:
                    avg_metrics = {key: AverageMeter() for key in metrics.keys()}
                for key, val in metrics.items():
                    avg_metric = avg_metrics[key]
                    avg_metric.update(val, 1)
                    d.update({key: avg_metric.avg})

                if avg_losses is None:
                    avg_losses = [AverageMeter() for _ in range(len(losses))]

                for idx, loss in enumerate(losses):
                    avg_loss = avg_losses[idx]
                    avg_loss.update(loss.mean().item(), 1)
                    d.update({f'loss{idx}': avg_loss.avg})

                t.set_postfix(d)
        self.model.to('cpu')
        return self.model.state_dict()


class FedProxClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, device, alpha, batch_size, lr, opt,
                 n_epochs, n_batches):
        super(FedProxClient, self).__init__(client_id, train_dataset, test_dataset, model, device, batch_size, lr,
                                            opt, n_epochs, n_batches)
        self.alpha = alpha

    def train_model(self):
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.to(self.device)
        model = nn.DataParallel(self.model)
        model.to(self.device)

        for epoch in range(self.n_epochs):
            d = OrderedDict(id=self.client_id)
            avg_loss = AverageMeter()
            avg_cel = AverageMeter()
            avg_nl = AverageMeter()
            avg_metrics = None

            if self.n_batches == 0:
                t = tqdm(self.train_loader, ncols=0)
            else:
                ite_train_loader = BatchIterator(self.n_batches, self.train_loader)
                t = tqdm(ite_train_loader, ncols=0)

            for i, data in enumerate(t):
                self.optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = [x.to(self.device) for x in inputs], [x.to(self.device) for x in labels]
                nl = 0
                count = 0

                for (name1, param1), (name2, param2) in zip(model.named_parameters(),
                                                            global_model.named_parameters()):
                    if 'backbone' in name1:
                        nl = (nl * count + torch.norm(param1 - param2, 2)) / (count + 1)
                    # norm_loss = (norm_loss * count + torch.norm(param1, 1)) / (count + 1)
                    count += 1
                features, logits, losses = model(inputs, labels)
                cel = losses[0].mean()
                loss = cel + self.alpha * nl
                loss.backward()
                self.optimizer.step()

                avg_loss.update(loss.item(), 1)
                avg_cel.update(cel.item(), 1)
                avg_nl.update(nl.item(), 1)
                d.update({'loss': avg_loss.avg, 'cel': avg_cel.avg, 'nl': avg_nl.avg})

                metrics = self.train_dataset.metric(inputs, labels, logits, test_mode=False)
                if avg_metrics is None:
                    avg_metrics = {key: AverageMeter() for key in metrics.keys()}
                for key, val in metrics.items():
                    avg_metric = avg_metrics[key]
                    avg_metric.update(val, 1)
                    d.update({key: avg_metric.avg})

                t.set_postfix(d)
        self.model.to('cpu')
        return self.model.state_dict()


class MOONClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, device, alpha, temperature, batch_size, lr,
                 opt,
                 epochs):
        super(MOONClient, self).__init__(client_id, train_dataset, test_dataset, model, device, batch_size,
                                         lr, opt, epochs)
        self.pre_model = copy.deepcopy(self.model)
        self.alpha = alpha
        self.temperature = temperature

    def train_model(self):
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.to(self.device)

        pre_model = copy.deepcopy(self.pre_model)
        pre_model = nn.DataParallel(pre_model)
        pre_model.to(self.device)

        model = self.model.to(self.device)
        model = nn.DataParallel(model)
        model.to(self.device)

        epoch = 0
        while epoch < self.epochs:
            d = OrderedDict(id=self.client_id)
            avg_loss = AverageMeter()
            avg_cel = AverageMeter()
            avg_scl = AverageMeter()
            avg_metrics = None

            # data_loader = self.data_loader
            t = tqdm(self.train_loader, ncols=0)

            for i, data in enumerate(t):
                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.to(self.device) for x in inputs], [x.to(self.device) for x in labels]

                features, logits, losses = model(inputs, labels)
                global_features, global_logits, global_losses = global_model(inputs, labels)
                pre_features, pre_logits, pre_losses = pre_model(inputs, labels)

                pos_sim = F.cosine_similarity(features[-1], global_features[-1], dim=-1)
                pos_sim = torch.exp(pos_sim / self.temperature)
                neg_sim = F.cosine_similarity(features[-1], pre_features[-1], dim=-1)
                neg_sim = torch.exp(neg_sim / self.temperature)

                scl = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim))
                # contrast_loss = -1.0 * torch.log(pos_sim)

                contrast_loss = scl.mean()
                cel = losses[0].mean()
                loss = cel + self.alpha * scl
                loss.backward()
                self.optimizer.step()

                avg_loss.update(loss.item(), 1)
                avg_cel.update(cel.item(), 1)
                avg_scl.update(scl.item(), 1)
                metrics = self.train_dataset.metric(inputs, labels, logits, test_mode=False)
                if avg_metrics is None:
                    avg_metrics = {key: AverageMeter() for key in metrics.keys()}
                for key, val in metrics.items():
                    avg_metric = avg_metrics[key]
                    avg_metric.update(val, 1)
                    d.update({key: avg_metric.avg})

                t.set_postfix(d)
            epoch += 1
        self.model.to('cpu')
        return self.model.state_dict()


class PersonalizedClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, device, batch_size, lr, opt, epochs, batches, pk):
        super(PersonalizedClient, self).__init__(client_id, train_loader, test_loader,
                                                 model, device, batch_size, lr, opt, epochs, batches)
        self.personalized_key = pk

    def receive_global_model(self, global_state_dict):
        local_state_dict = self.model.state_dict()
        for key, val in global_state_dict.items():
            if self.personalized_key in key:
                print(key)
                continue
            local_state_dict[key] = self.momentum * local_state_dict[key] + (1 - self.momentum) * val

        self.model.load_state_dict(local_state_dict)


class DFLClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, alpha, *args, **kwargs):
        super(DFLClient, self).__init__(client_id, train_dataset, test_dataset, model, *args, **kwargs)
        self.alpha = alpha
        self.neighbor_models = []

    def aggregate(self):
        params = [[param for param in model.parameters()] for model in self.neighbor_models]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.zeros(param.data.shape)
            for client_idx in range(len(params)):
                param.data += params[client_idx][i] * self.alpha[client_idx]
