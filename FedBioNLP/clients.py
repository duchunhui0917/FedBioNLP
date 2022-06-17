import numpy as np
import torch
import copy
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict
from .optimers import WPOptim
from .utils.fl_utils import BatchIterator
from .utils.common_utils import AverageMeter
import os
import logging
import gc
from .models import MMDLoss
from .modules import InfoNCE

logger = logging.getLogger(os.path.basename(__file__))


class BaseClient(object):
    def __init__(self, client_id, train_loader, test_loader, model, n_epochs, args):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_dataset = train_loader.dataset
        self.test_dataset = test_loader.dataset
        self.model = copy.deepcopy(model)
        self.n_epochs = n_epochs

        self.lr = args.lr
        self.opt = args.opt
        self.n_batches = args.n_batches
        self.momentum = args.cm

        self.optimizer_sd = None

        if self.opt == 'AdamW':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                       weight_decay=1e-5, momentum=0.9)
        else:
            raise Exception("Invalid optimizer. Must be 'SGD' or 'Adam'.")

    def train_model(self, ite=None):
        # model = nn.parallel.DataParallel(self.model)
        model = self.model.cuda()
        if self.optimizer_sd is not None:
            if self.opt == 'AdamW':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            elif self.opt == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.9)
            else:
                raise Exception("Invalid optimizer. Must be 'SGD' or 'Adam'.")
            state = self.optimizer_sd['state']
            for s in state.keys():
                for k, v in state[s].items():
                    if isinstance(v, torch.Tensor):
                        state[s][k] = v.cuda()
            self.optimizer.load_state_dict(self.optimizer_sd)
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

                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                if ite is not None:
                    features, logits, losses = model(inputs, labels, ite)
                else:
                    features, logits, losses = model(inputs, labels)
                losses[0].backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

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

            for key, val in d.items():
                logger.info(f'{key}: {val:.4f}')
        model.cpu()
        self.optimizer_sd = self.optimizer.state_dict()
        state = self.optimizer_sd['state']
        for s in state.keys():
            for k, v in state[s].items():
                if isinstance(v, torch.Tensor):
                    state[s][k] = v.cpu()
        del self.optimizer
        torch.cuda.empty_cache()
        sd = copy.deepcopy(model.state_dict())

        return sd

    def receive_global_model(self, global_model_dict):
        local_model_dict = self.model.state_dict()
        for key, val in global_model_dict.items():
            local_model_dict[key] = self.momentum * local_model_dict[key] + (1 - self.momentum) * val
        self.model.load_state_dict(local_model_dict)


class MTLClient(BaseClient):
    def __init__(self, client_id, train_loaders, test_loaders, model, n_epochs, args):
        super(MTLClient, self).__init__(client_id, train_loaders[0], test_loaders[0], model, n_epochs, args)
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.train_dataset = train_loaders[0].dataset
        self.test_dataset = test_loaders[0].dataset

    def train_model(self, ite=None):
        model = nn.parallel.DataParallel(self.model)
        model.cuda()
        model.train()

        for epoch in range(self.n_epochs):
            avg = [[None, None] for _ in range(len(self.train_loaders))]
            ds = [None, None]
            train_loaders = [[data for data in loader] for loader in self.train_loaders]
            ls = list(zip(*train_loaders))
            for loaders in ls:
                for i, data in enumerate(loaders):
                    avg_metrics, avg_losses = avg[i]
                    ds[i] = OrderedDict(id=i)
                    d = ds[i]

                    self.optimizer.zero_grad()

                    inputs, labels = data
                    inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]
                    inputs += [i]
                    if ite is not None:
                        features, logits, losses = model(inputs, labels, ite)
                    else:
                        features, logits, losses = model(inputs, labels)
                    losses[0].backward()

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

            for d in ds:
                logger.info(d)
        self.model.to('cpu')
        return self.model.state_dict()


class ICFAClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, args):
        super(ICFAClient, self).__init__(client_id, train_loader, test_loader, model, args)
        self.n_clusters = args.n_clusters
        self.cluster_models = args.cluster_models
        self.cluster_losses = np.zeros((self.n_clusters,))
        self.optimizers = [optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
                           for model in self.cluster_models]

    def train_model(self):
        for c in range(self.n_clusters):
            model = nn.parallel.DataParallel(self.cluster_models[c])
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

                    self.optimizers[c].zero_grad()

                    inputs, labels = data
                    inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                    features, logits, losses = model(inputs, labels)
                    losses[0].mean().backward()
                    self.optimizers[c].step()

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
            self.cluster_models[c].to('cpu')
            self.cluster_losses[c] = avg_losses[0].avg
        best_c = np.argmin(self.cluster_losses)
        return self.cluster_models[best_c].state_dict(), best_c

    def receive_cluster_models(self, cluster_model_dicts):
        for c in range(self.n_clusters):
            cluster_model_dict = self.cluster_models[c].state_dict()
            for key, val in cluster_model_dicts[c].items():
                cluster_model_dict[key] = self.momentum * cluster_model_dict[key] + (1 - self.momentum) * val
            self.cluster_models[c].load_state_dict(cluster_model_dict)


class HarmoFLClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, args):
        super(HarmoFLClient, self).__init__(client_id, train_loader, test_loader, model, args)
        self.perturbation = args.perturbation
        self.optimizer = WPOptim(params=self.model.parameters(), base_optimizer=optim.Adam, lr=self.lr,
                                 alpha=self.perturbation, weight_decay=1e-4)

    def train_model(self, ite=None):
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
    def __init__(self, client_id, train_dataset, test_dataset, model, args):
        super(FedProxClient, self).__init__(client_id, train_dataset, test_dataset, model, args)
        self.mu = args.mu

    def train_model(self, ite=None):
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.cuda()
        model = nn.DataParallel(self.model)
        model.cuda()

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
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]
                nl = 0
                count = 0

                for (name1, param1), (name2, param2) in zip(model.named_parameters(),
                                                            global_model.named_parameters()):
                    nl = (nl * count + torch.norm(param1 - param2, 2)) / (count + 1)
                    count += 1
                features, logits, losses = model(inputs, labels)
                cel = losses[0].mean()
                loss = cel + self.mu * nl
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


class pFedMeClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, args):
        super(pFedMeClient, self).__init__(client_id, train_dataset, test_dataset, model, args)
        self.mu = args.mu
        self.n_inner_loops = args.n_inner_loops

    def train_model(self, ite=None):
        # for regularization
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.cuda()
        # for obtaining gradients
        local_model = copy.deepcopy(self.model)
        if self.opt == 'Adam':
            local_optimizer = optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=self.lr)
        elif self.opt == 'SGD':
            local_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                        weight_decay=1e-5, momentum=0.9)
        else:
            raise Exception("Invalid optimizer. Must be 'SGD' or 'Adam'.")
        local_model = nn.DataParallel(local_model)
        local_model.cuda()

        model = nn.DataParallel(self.model)
        model.cuda()

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
                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                for j in range(self.n_inner_loops):
                    local_optimizer.zero_grad()
                    nls = []
                    for (name1, param1), (name2, param2) in zip(local_model.named_parameters(),
                                                                global_model.named_parameters()):
                        nls.append(torch.norm(param1 - param2, 2))
                    nl = sum(nls) / len(nls)
                    features, logits, losses = local_model(inputs, labels)
                    cel = losses[0].mean()
                    loss = cel + self.mu * nl
                    loss.backward()
                    local_optimizer.step()

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

                # update model by the gradient obtained by local model
                for param, local_param in zip(model.parameters(), local_model.parameters()):
                    param.data -= self.lr * (param.data - local_param.data)
                    local_param.data = param.data
        self.model.to('cpu')
        return self.model.state_dict()


class MOONClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, n_epochs, args):
        super(MOONClient, self).__init__(client_id, train_dataset, test_dataset, model, n_epochs, args)
        self.pre_model = copy.deepcopy(self.model)
        self.mu = args.mu
        self.temperature = args.temperature

    def train_model(self, ite=None):
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.cuda()

        pre_model = copy.deepcopy(self.pre_model)
        pre_model = nn.DataParallel(pre_model)
        pre_model.cuda()

        model = self.model
        model = nn.DataParallel(model)
        model.cuda()

        epoch = 0
        while epoch < self.n_epochs:
            d = OrderedDict(id=self.client_id)
            avg_scl = AverageMeter()
            avg_metrics = None
            avg_losses = None

            # data_loader = self.data_loader
            t = tqdm(self.train_loader, ncols=0)

            for i, data in enumerate(t):
                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                features, logits, losses = model(inputs, labels)
                with torch.no_grad():
                    global_features, global_logits, global_losses = global_model(inputs, labels)
                    pre_features, pre_logits, pre_losses = pre_model(inputs, labels)

                pos_sim = F.cosine_similarity(features[-2], global_features[-2], dim=-1)
                pos_sim = torch.exp(pos_sim / self.temperature)
                neg_sim = F.cosine_similarity(features[-2], pre_features[-2], dim=-1)
                neg_sim = torch.exp(neg_sim / self.temperature)

                scl = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim))
                scl = scl.mean()

                cel = losses[0].mean()
                loss = cel + self.mu * scl
                loss.backward()
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
                avg_scl.update(scl.item(), 1)
                d.update({'scl': avg_scl.avg})

                t.set_postfix(d)
            epoch += 1
        self.model.to('cpu')
        self.pre_model = copy.deepcopy(self.model)
        return self.model.state_dict()


class GSNClient(BaseClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, n_epochs, args):
        super(GSNClient, self).__init__(client_id, train_dataset, test_dataset, model, n_epochs, args)
        self.mu = args.mu
        self.temperature = args.temperature
        self.nce = InfoNCE(self.temperature)
        self.mmd = MMDLoss()

    def train_model(self, ite=None):
        global_model = copy.deepcopy(self.model)
        global_model = nn.DataParallel(global_model)
        global_model.cuda()

        model = self.model
        model = nn.DataParallel(model)
        model.cuda()

        epoch = 0
        while epoch < self.n_epochs:
            d = OrderedDict(id=self.client_id)
            avg_scl = AverageMeter()
            avg_mmd = AverageMeter()
            avg_pos_sim = AverageMeter()
            avg_neg_sim = AverageMeter()
            avg_metrics = None
            avg_losses = None

            # data_loader = self.data_loader
            t = tqdm(self.train_loader, ncols=0)

            for i, data in enumerate(t):
                self.optimizer.zero_grad()

                inputs, labels = data
                inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]

                features, logits, losses = model(inputs, labels)
                with torch.no_grad():
                    global_features, global_logits, global_losses = global_model(inputs, labels)

                re_rep, mlm_rep = features[-4], features[-3]
                global_re_rep, global_mlm_rep = global_features[-4], global_features[-3]

                pos_sim = self.nce(re_rep, global_re_rep).mean()
                # neg_sim = (self.nce(re_rep, mlm_rep) + self.nce(mlm_rep, global_mlm_rep)).mean()
                neg_sim = self.nce(mlm_rep, global_mlm_rep).mean()

                scl = - torch.log(pos_sim / (pos_sim + neg_sim))
                scl = self.nce(re_rep, mlm_rep).mean()

                mmd_loss = self.mmd(re_rep, global_re_rep)
                cel = losses[0].mean()
                loss = cel + self.mu * scl + mmd_loss
                loss.backward()
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
                avg_scl.update(scl.item(), 1)
                d.update({'scl': avg_scl.avg})
                avg_mmd.update(mmd_loss.item(), 1)
                d.update({'mmd': avg_mmd.avg})

                # avg_pos_sim.update(pos_sim.item(), 1)
                # d.update({'pos': avg_pos_sim.avg})
                # avg_neg_sim.update(neg_sim.item(), 1)
                # d.update({'neg': avg_neg_sim.avg})

                t.set_postfix(d)
            epoch += 1
        self.model.to('cpu')
        self.pre_model = copy.deepcopy(self.model)
        return self.model.state_dict()


class PartialFLClient(BaseClient):
    def __init__(self, client_id, train_loader, test_loader, model, n_epoch, args):
        super(PartialFLClient, self).__init__(client_id, train_loader, test_loader, model, n_epoch, args)
        self.personalized_keys = args.personal_keys

    def receive_global_model(self, global_model_dict):
        local_state_dict = self.model.state_dict()
        for key, val in global_model_dict.items():
            skip = False
            for personalized_key in self.personalized_keys:
                if personalized_key in key:
                    print(key)
                    skip = True
                    break
            if not skip:
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
