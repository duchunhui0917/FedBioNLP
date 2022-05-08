from FedBioNLP.clients import *
from FedBioNLP.utils.common_utils import set_seed
import numpy as np
from torch.utils.data import DataLoader
from FedBioNLP.utils.sta_utils import tensor_cos_sim

set_seed(2333)

logger = logging.getLogger(os.path.basename(__file__))


class BaseSystem(object):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, ckpt,
                 n_iterations=100, n_epochs=1, n_batches=0,
                 lr=0.01, batch_size=64, default_batch_size=32, opt='SGD', aggregate_method='sample'):
        self.n_samples = [len(val) for val in train_datasets.values()]
        self.n_clients = len(client_ids)
        self.dataset = test_dataset
        self.client_ids = client_ids
        self.model = copy.deepcopy(model)
        self.n_iterations = n_iterations

        self.clients = []
        self.server = None
        self.device = device
        self.n_classes = test_dataset.n_classes
        self.batch_size = batch_size if isinstance(batch_size, list) else [batch_size for _ in range(self.n_clients)]
        self.lr = lr
        self.n_epochs = n_epochs if isinstance(n_epochs, list) else [n_epochs for _ in range(self.n_clients)]
        self.n_batches = n_batches
        self.opt = opt
        self.aggregate_method = aggregate_method
        self.g_best_metric = 0

        self.ckpt = f'{ckpt}.pth'

        self.train_loaders = [DataLoader(v, batch_size=self.batch_size[k], shuffle=True, drop_last=True)
                              for k, v in train_datasets.items()]
        self.test_loaders = [DataLoader(v, batch_size=default_batch_size)
                             for k, v in test_datasets.items()]
        self.train_loader = DataLoader(train_dataset, batch_size=default_batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=default_batch_size)
        self.ite = 0

    def run(self):
        pass

    def test_model(self, model=None, data_loader=None):
        if model is None:
            model = copy.deepcopy(self.model)
        if data_loader is None:
            data_loader = self.test_loader

        model = nn.DataParallel(model)
        model = model.cuda()
        model.eval()
        res_inputs, res_labels, res_features, res_logits, res_losses = ([] for _ in range(5))

        data_loader = tqdm(data_loader)

        for i, data in enumerate(data_loader):
            inputs, labels = data

            if len(res_inputs) == 0:
                res_inputs = [[] for _ in range(len(inputs))]
            for ls, d in zip(res_inputs, inputs):
                ls.append(d.numpy())
            if len(res_labels) == 0:
                res_labels = [[] for _ in range(len(labels))]
            for ls, d in zip(res_labels, labels):
                ls.append(d.numpy())

            inputs, labels = [x.cuda() for x in inputs], [x.cuda() for x in labels]
            with torch.no_grad():
                features, logits, losses = model(inputs, labels)

            res_logits.append(logits.cpu().detach().numpy())
            res_losses.append(losses[0].mean().item())

            if len(res_features) == 0:
                res_features = [[] for _ in range(len(features))]
            for ls, feature in zip(res_features, features):
                ls.append(feature.cpu().detach().numpy())

        res_inputs = [np.concatenate(ls) for ls in res_inputs]
        res_labels = [np.concatenate(ls) for ls in res_labels]
        res_features = [np.concatenate(ls) for ls in res_features]
        res_logits = np.concatenate(res_logits)
        loss = sum(res_losses) / len(res_losses)

        metrics = self.dataset.metric(res_inputs, res_labels, res_logits)
        metrics.update({'loss': loss})
        logger.info(f'loss: {loss:.4f}')

        return metrics, res_inputs, res_labels, res_features, res_logits

    def load(self, ckpt, exclude_key=' '):
        sd = self.model.state_dict()
        for key, val in torch.load(ckpt).items():
            if exclude_key not in key:
                # logger.info(key)
                sd.update({key: val})
        self.model.load_state_dict(sd)

    def update_test_loader(self, test_dataset):
        self.test_loader = DataLoader(test_dataset, batch_size=self.default_batch_size)

    def update_model(self, model):
        self.model = copy.deepcopy(model)


class CentralizedSystem(BaseSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, path, **kwargs):
        super(CentralizedSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                                model, device, path, **kwargs)
        self.client = BaseClient(0, self.train_loader, self.test_loader, model, device,
                                 self.batch_size, self.lr, self.opt, self.n_epochs[0], self.n_batches)

    def run(self, m='f1'):
        logger.info('******running centralized system******')

        for ite in range(self.n_iterations):
            logger.info(f'iteration: {ite}')

            # train
            model_state_dict = self.client.train_model()
            self.model.load_state_dict(model_state_dict)

            # test
            metrics = self.test_model()[0]
            test_metric = metrics[m]

            if test_metric > self.g_best_metric:
                torch.save(self.model.state_dict(), self.ckpt_path)
                self.g_best_metric = test_metric
                logger.info('new model saved')
            logger.info(f'best {m}: {self.g_best_metric:.4f}')


class FedAvgSystem(BaseSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, ckpt, cm=0, sm=0, centralized=True, personalized=False, layers='.', **kwargs):
        super(FedAvgSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                           model, device, ckpt, **kwargs)

        self.clients = [
            BaseClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id], model, device,
                       self.batch_size, self.lr, self.opt, self.n_epochs[client_id], self.n_batches, momentum=cm)
            for client_id in self.client_ids
        ]
        self.weights = self.cmp_weights()
        self.momentum = sm
        self.centralized = centralized
        self.personalized = personalized
        self.p_best_metric = 0
        self.ckpts = [f'{self.ckpt}_{i}.pth' for i in range(self.n_clients)]
        self.layers = layers.split('*')

        self.personal_model_dicts = [self.model.state_dict() for _ in range(self.n_clients)]
        self.personal_grad_dicts = [[{} for _ in range(self.n_clients)] for _ in range(self.n_clients)]
        self.cos_sims = {}

    def run(self, m='f1'):
        logger.info('******running FedAvg system******')

        for self.ite in range(self.n_iterations):
            logger.info(f'iteration: {self.ite}')

            # 1. distribute server model to all clients
            global_param_dict = self.model.state_dict()
            for client in self.clients:
                client.receive_global_model(global_param_dict)

            # 2. train client models
            model_dicts = []
            for i, client in enumerate(self.clients):
                model_state_dict = client.train_model()
                model_dicts.append(model_state_dict)

            # update gradient and compute gradient cosine similarity
            self.get_personal_grad_dicts(model_dicts)
            logger.info('compute gradient cosine similarity')
            self.compute_cos_sims()

            # 3. aggregate into server model
            model_dict = self.get_global_model_dict(model_dicts)
            self.model.load_state_dict(model_dict)

            # test and save models
            self.test_save_models(m)

    def get_personal_grad_dicts(self, model_dicts):
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                for key, val in self.model.state_dict().items():
                    grad = self.personal_model_dicts[j][key] - model_dicts[j][key]
                    self.personal_grad_dicts[i][j].update({key: grad})

    def get_global_model_dict(self, model_dicts):
        logger.info(f'aggregation weights all layers: {self.weights}')
        model_dict = self.model.state_dict()
        for l, key in enumerate(model_dict.keys()):
            model_dict[key] = self.momentum * model_dict[key]
            for i, sd in enumerate(model_dicts):
                val = sd[key] * self.weights[i]
                model_dict[key] += (1 - self.momentum) * val
        return model_dict

    def test_save_models(self, m):
        g_test_metrics = []
        p_test_metrics = []

        for i, data_loader in enumerate(self.test_loaders):
            if self.centralized or self.ite == 0:
                model = copy.deepcopy(self.model)
                metrics = self.test_model(model, data_loader=data_loader)[0]
                g_test_metrics.append(metrics[m])

            if self.personalized:
                model = copy.deepcopy(self.clients[i].model)
                metrics = self.test_model(model, data_loader=data_loader)[0]
                p_test_metrics.append(metrics[m])

        g_test_metric = sum(g_test_metrics) / len(g_test_metrics) if len(g_test_metrics) != 0 else 0
        if g_test_metric > self.g_best_metric:
            self.g_best_metric = g_test_metric
            logger.info('global new model saved')

            torch.save(self.model.state_dict(), self.ckpt)
        logger.info(f'global best {m}: {self.g_best_metric:.4f}')

        p_test_metric = sum(p_test_metrics) / len(p_test_metrics) if len(p_test_metrics) != 0 else 0
        if p_test_metric > self.p_best_metric:
            self.p_best_metric = p_test_metric
            logger.info('personal new model saved')

            for i in range(self.n_clients):
                torch.save(self.clients[i].model.state_dict(), self.ckpts[i])
        logger.info(f'personal best {m}: {self.p_best_metric:.4f}')

    def compute_cos_sims(self):
        np.set_printoptions(precision=4)
        for key in self.model.state_dict().keys():
            self.cos_sims.update({key: np.zeros((self.n_clients, self.n_clients))})
            for i in range(self.n_clients):
                for j in range(self.n_clients):
                    sim = tensor_cos_sim(self.personal_grad_dicts[i][i][key], self.personal_grad_dicts[i][j][key])
                    self.cos_sims[key][i][j] = sim

        for layer in self.layers:
            cos_sim = np.zeros((self.n_clients, self.n_clients))
            count = 0
            for key, val in self.cos_sims.items():
                if layer in key:
                    cos_sim = (cos_sim * count + val) / (count + 1)
                    count += 1
            logger.info(f'layer {layer} cosine similarity matrix\n{cos_sim}')

    def cmp_weights(self):
        if self.aggregate_method == 'sample':
            weights = self.n_samples
            weights = np.array([w / sum(weights) for w in weights])
        else:
            weights = np.array([1 / self.n_clients for _ in range(self.n_clients)])
        return weights


class FedProxSystem(FedAvgSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, mu,
                 ckpt, **kwargs):
        super(FedProxSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                            model, device, ckpt, **kwargs)

        self.clients = [
            FedProxClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id], model, device, mu,
                          self.batch_size, self.lr, self.opt, self.n_epochs[client_id], self.n_batches)
            for client_id in self.client_ids
        ]


class HarmoFLSystem(FedAvgSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, ckpt, perturbation, **kwargs):
        super(HarmoFLSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                            model, device, ckpt, **kwargs)
        self.perturbation = perturbation
        self.clients = [
            HarmoFLClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id], model, device,
                          self.batch_size, self.lr, self.opt, self.n_epochs[client_id], self.n_batches, self.momentum,
                          perturbation=perturbation)
            for client_id in self.client_ids
        ]


class MOONSystem(FedAvgSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, mu,
                 temperature, ckpt, **kwargs):
        super(MOONSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                         model, device, ckpt, **kwargs)

        self.clients = [
            MOONClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id],
                       model, device, mu, temperature, self.batch_size, self.lr, self.opt, self.epochs[client_id])
            for client_id in self.client_ids
        ]


class PersonalizedFLSystem(FedAvgSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, ckpt,
                 pk='classifier', **kwargs):
        super(PersonalizedFLSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset,
                                                   test_dataset, model, device, ckpt, **kwargs)

        self.clients = [
            PersonalizedClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id], model, device,
                               self.batch_size, self.lr, self.opt, self.n_epochs[client_id], self.n_batches, pk)
            for client_id in self.client_ids
        ]


class FedGSSystem(FedAvgSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, ckpt, cm=0, sm=0, **kwargs):
        super(FedGSSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                          model, device, ckpt, **kwargs)
        self.test_metrics = [0 for _ in range(self.n_clients)]

        self.clients = [
            BaseClient(client_id, self.train_loaders[client_id], self.test_loaders[client_id], model, device,
                       self.batch_size, self.lr, self.opt, self.n_epochs[client_id], self.n_batches,
                       momentum=cm)
            for client_id in self.client_ids
        ]
        self.momentum = sm

    def run(self, m='f1'):
        logger.info('******running FedGS system******')

        for ite in range(self.n_iterations):
            logger.info(f'iteration: {ite}')

            # 1. distribute personal model to all clients
            for i, client in enumerate(self.clients):
                client.receive_global_model(self.personal_model_dicts[i])

            # 2. train client models
            param_dicts = []
            for i, client in enumerate(self.clients):
                param_dict = client.train_model()
                param_dicts.append(param_dict)

            # 3. get gradient and compute cosine similarity
            self.get_personal_grad_dicts(param_dicts)
            logger.info('compute gradient cosine similarity')
            self.compute_cos_sims()

            # 4. update gradient and compute cosine similarity
            self.update_personal_grad_dicts()
            logger.info('compute updated gradient cosine similarity')
            self.compute_cos_sims()

            # 5. compute personal models
            self.get_personal_model_dicts()

            # 6. aggregate into global model
            mode_dict = self.get_global_model_dict(param_dicts)
            self.model.load_state_dict(mode_dict)

            # test
            self.test_save_models(m)

    def update_personal_grad_dicts(self):
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                for key, val in self.cos_sims.items():
                    sign = 1 if val[i][j] > 0 else 0
                    # sign = 1 if i == j else 0
                    self.personal_grad_dicts[i][j][key] *= sign

    def get_personal_model_dicts(self):
        weights = self.weights / self.weights.sum()
        for i in range(self.n_clients):
            for j in range(self.n_clients):
                for key, val in self.model.state_dict().items():
                    self.personal_model_dicts[i][key] -= self.personal_grad_dicts[i][j][key] * weights[j]


def reshape_to_matrix(x):
    x = x.numpy()
    shape = x.shape
    if len(shape) != 2:
        x = x.reshape((shape[0], -1))
    return x, shape


class FedGPSystem(FedGSSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                 model, device, ckpt, **kwargs):
        super(FedGPSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                          model, device, ckpt, **kwargs)

    def update_personal_grad_dicts(self):
        for i in range(self.n_clients):
            for key, value in self.model.state_dict().items():
                x1 = self.personal_grad_dicts[i][i][key]
                x1_, shape1 = reshape_to_matrix(x1)
                U, D1, V = np.linalg.svd(x1_)
                # threshold = 0.9
                # sval_total = (D1 ** 2).sum()
                # sval_ratio = (D1 ** 2) / sval_total
                # r = np.sum(np.cumsum(sval_ratio) <= threshold)
                # U[:, r:] = 0
                # V[r:, :] = 0
                U_inv, V_inv = U.T, V.T

                for j in range(self.n_clients):
                    x2 = self.personal_grad_dicts[i][j][key]
                    x2_, shape2 = reshape_to_matrix(x2)
                    D2 = np.matmul(np.matmul(U_inv, x2_), V_inv)
                    sign = np.sign(D2)
                    D2 = D2 * sign
                    x2_ = np.matmul(np.matmul(U, D2), V)
                    x2 = x2_.reshape(shape2)
                    self.personal_grad_dicts[i][j][key] = torch.FloatTensor(x2)


class SoloSystem(BaseSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, train_dataset, test_dataset, model, device, **kwargs):
        super(SoloSystem, self).__init__(client_ids, train_datasets, test_datasets, train_dataset, test_dataset,
                                         model, device, **kwargs)
        self.clients = [
            BaseClient(client_id, train_datasets[client_id], test_datasets[client_id], model, device,
                       self.batch_size, self.lr, self.opt, self.epochs[client_id], **kwargs)
            for client_id in self.client_ids
        ]

    def run(self):
        logger.info('******running centralized federated learning system******')
        metrics = []

        for ite in range(self.n_iterations):
            logger.info(f'iteration: {ite}')
            lr = self.lr
            # self.cur_lr = self.lr * math.sqrt(1 / (self.cur_ite + 1))

            # test
            metric = self.test_model()
            if metric['test acc'] > self.best_test_acc:
                torch.save(self.model.state_dict(), self.ckpt)
                self.best_test_acc = metric['test acc']
                logger.info('new model saved')
            metrics.append(metric)

            # train client models
            model_state_dicts = []
            for client in self.clients:
                model_state_dict = client.train_model()
                model_state_dicts.append(model_state_dict)
            # aggregate models
            state_dict = self.compute_global_model(model_state_dicts)
            self.model.load_state_dict(state_dict)

        return metrics


class DFLSystem(BaseSystem):
    def __init__(self, client_ids, train_datasets, test_datasets, model, iterations, lr,
                 pairs=None, *args, **kwargs):
        super(DFLSystem, self).__init__(client_ids, train_datasets[0].n_classes, model, iterations, lr)
        if pairs is None:
            self.pairs = [(i, j) for i in range(self.n_clients) for j in range(i)]
        else:
            self.pairs = pairs
        self.graph = pairs2graph(self.pairs)
        V = pairs2matrix(self.pairs, self.n_clients)
        self.consensus_matrix = get_degree_consensus_matrix(V)
        np.set_logger.infooptions(precision=2)
        logger.info(self.consensus_matrix)
        self.spectral_gap = get_degree_spectral_gap(V)

        self.clients = [
            DFLClient(client_id, train_datasets[client_id], test_datasets[client_id], model,
                      self.consensus_matrix[client_id], *args, **kwargs)
            for client_id in self.client_ids
        ]
        self.client_models = [copy.deepcopy(model) for _ in range(self.n_clients)]

    def run(self):
        logger.info('******running decentralized federated learning system******')
        logger.info(f'spectral gap: {self.spectral_gap}')
        loss, train_acc, test_acc, test_auc = (np.zeros((self.iterations,)) for _ in range(4))
        ite = 0
        while ite < self.iterations:
            # lr = self.lr * math.sqrt(self.spectral_gap / (ite + 1))
            # lr = self.lr * math.sqrt(self.spectral_gap)
            # lr = self.lr * math.sqrt(1 / (ite + 1))
            lr = self.lr
            # 1. train client model
            for i, client in enumerate(self.clients):
                client_model = client.train(lr)
                self.client_models[i] = client_model

            # 2. transmit client models to neighbors
            for i, client in enumerate(self.clients):
                client.neighbor_models = self.client_models

            # 3. aggregate with neighbor client models
            for client in self.clients:
                client.aggregate()

            self.update_global_model()
            loss[ite], train_acc[ite], test_acc[ite], test_auc[ite], conf_mtx = self.test()
            logger.info(f'ite:{ite}, lr:{lr:.6f}, loss:{loss[ite]:.4f}, train acc:{train_acc[ite]:.4f},'
                        f'test acc:{test_acc[ite]:.4f}, test auc:{test_auc[ite]: .4f}')
            logger.info(conf_mtx)

            ite += 1
        return loss.tolist(), train_acc.tolist(), test_acc.tolist(), test_auc.tolist()

    def update_global_model(self):
        params = [[param for param in self.client_models[i].parameters()]
                  for i in range(self.n_clients)]
        for idx, param in enumerate(self.model.parameters()):
            param.data = torch.zeros(param.shape)
            for c in range(self.n_clients):
                param.data += params[c][idx] / self.n_clients


class ServerSystem(BaseSystem):
    def __init__(self, client_ids, train_datasets, test_dataset, server_dataset,
                 model, iterations, lr, selected_idxes=None, alg='FedAvg',
                 *args, **kwargs):
        super(ServerSystem, self).__init__(test_dataset, client_ids, model, device, **kwargs)

        if selected_idxes is None:
            self.selected_idxes = [list(range(self.n_clients)) for _ in range(self.iterations)]
        else:
            self.selected_idxes = selected_idxes
        self.alg = alg

        agg_weights = [1 for _ in range(self.n_clients)]
        self.server = Server(model, self.n_clients, self.selected_idxes, agg_weights, server_dataset)

        self.clients = [
            BaseClient(client_id, train_datasets[client_id], test_datasets[client_id], model,
                       *args, **kwargs)
            for client_id in self.client_ids
        ]

    def run(self):
        logger.info('******running centralized federated learning system******')
        test_acc = np.zeros((self.iterations,))
        client_model_list = []

        for ite in range(self.iterations):
            lr = self.lr
            # self.cur_lr = self.lr * math.sqrt(1 / (self.cur_ite + 1))

            # test
            test_acc[ite], conf_mtx = self.server.test(self.model)
            logger.info(f'ite:{ite}, lr:{lr:.6f}, test acc:{test_acc[ite]:.4f}')
            logger.info(conf_mtx)

            # 1. distribute server model to all clients
            for client in self.clients:
                client.model = copy.deepcopy(self.server.model)

            # 2. train client models
            client_models = []
            weights = []
            for client in self.clients:
                metric, model_state_dict = client.train()
                client_models.append(client_model)
                acc, _ = self.server.test(client_model)
                weights.append(acc)
            s = sum(weights)
            if self.alg == 'ABAVG':
                self.server.agg_weights = [w / s for w in weights]

            # 3. upload client models to server
            self.server.client_models = client_models

            # 4. aggregate into server model
            self.server.aggregate(ite)
            client_model_list.append(client_models)
            self.model = self.server.model

        return test_acc.tolist()
