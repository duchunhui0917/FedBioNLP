import copy

import numpy as np
import torch
from torch import nn
from FedBioNLP import param_cosine

x1 = np.random.normal(2, 4, (100, 10))
x2 = np.random.normal(0, 4, (100, 10))
x3 = np.random.normal(10, 4, (100, 10))

w1 = np.ones((100, 10)) * 0.2
b1 = 4
w3 = np.ones((100, 10)) * 1
b3 = 10

y1 = (np.sign(np.sum(x1 * w1, axis=1) - b1) + 1) / 2
y2 = (np.sign(np.sum(x2 * w1, axis=1) - b1) + 1) / 2
y3 = (np.sign(np.sum(x3 * w3, axis=1) - b3) + 1) / 2


def compute_grad_state_dicts(model, state_dicts):
    grad_state_dicts = copy.deepcopy(state_dicts)
    global_state_dict = model.state_dict()
    for key, val in global_state_dict.items():
        for sd in grad_state_dicts:
            sd[key] = val - sd[key]
    return grad_state_dicts


layers = ['1', '2', '3']


def sta_grad_cosine(grad_state_dicts):
    np.set_printoptions(precision=4)
    n_params = len(grad_state_dicts)
    cos_sims = {layer: np.zeros((n_params, n_params)) for layer in layers}

    for i in range(n_params):
        for j in range(n_params):
            sim = param_cosine(grad_state_dicts[i], grad_state_dicts[j], layers)
            for k, v in cos_sims.items():
                cos_sims[k][i][j] = sim[k]

    for k, v in cos_sims.items():
        print(f'layer {k} cosine similarity matrix')
        print(v)
    return cos_sims


def compute_global_model(model, state_dicts):
    weights = [1 / len(state_dicts) for _ in range(len(state_dicts))]
    model_state_dict = model.state_dict()
    for l, key in enumerate(model_state_dict.keys()):
        model_state_dict[key] = 0
        for i, sd in enumerate(state_dicts):
            val = sd[key] * weights[i]
            model_state_dict[key] += val
    return model_state_dict


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


x1, x2, x3 = torch.FloatTensor(x1), torch.FloatTensor(x2), torch.FloatTensor(x3)
y1, y2, y3 = torch.LongTensor(y1), torch.LongTensor(y2), torch.LongTensor(y3)

model = MyModel()
models = [copy.deepcopy(model) for _ in range(3)]

lr = 0.1
optimizers = [torch.optim.SGD(model.parameters(), lr=lr) for model in models]
X = [x1, x2, x3]
Y = [y1, y2, y3]

criterion = nn.CrossEntropyLoss()
global_model = model
for i in range(100):
    state_dicts = []
    for x, y, model, optimizer in zip(X, Y, models, optimizers):
        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        print(loss.item())
        loss.backward()
        optimizer.step()
        state_dicts.append(model.state_dict())

    model_state_dict = compute_global_model(global_model, state_dicts)
    global_model.load_state_dict(model_state_dict)
    grad_state_dicts = compute_grad_state_dicts(global_model, state_dicts)
    sta_grad_cosine(grad_state_dicts)
