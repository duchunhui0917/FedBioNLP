import copy
import os
import scipy.io as scio
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_svmlight_files
import numpy as np


def get_X_y(train_file, test_file):
    train_X, train_y, test_X, test_y = load_svmlight_files([train_file, test_file])
    train_X, test_X = (np.array(x.todense()) for x in (train_X, test_X))
    train_y, test_y = (np.array((y + 1) / 2, dtype=int) for y in (train_y, test_y))

    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    test_X = torch.FloatTensor(test_X)
    test_y = torch.LongTensor(test_y)
    return train_X, train_y, test_X, test_y


base_dir = os.path.expanduser('~/FedBioNLP')
domains = ['books', 'dvd', 'electronics', 'kitchen']
train_files = [os.path.join(base_dir, f'data/review/{domain}_train.svmlight') for domain in domains]
test_files = [os.path.join(base_dir, f'data/review/{domain}_test.svmlight') for domain in domains]
Xys = [get_X_y(f1, f2) for f1, f2 in zip(train_files, test_files)]
all_train_X = torch.cat([Xy[0] for i, Xy in enumerate(Xys) if i <= 2], dim=0)
all_train_y = torch.cat([Xy[1] for i, Xy in enumerate(Xys) if i <= 2], dim=0)
all_test_X = torch.cat([Xy[2] for Xy in Xys], dim=0)
all_test_y = torch.cat([Xy[3] for Xy in Xys], dim=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f1 = nn.Linear(5000, 500)
        self.f2 = nn.Linear(500, 128)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.classifier(x)
        return x


def test(test_loader):
    ls_acc = []
    for X, y in test_loader:
        X, y = X.cuda(), y.cuda()
        logits = net(X)
        _, pred_y = logits.max(-1)
        acc = float((pred_y == y).sum() / len(y))
        ls_acc.append(acc)
    print(f'test acc: {sum(ls_acc) / len(ls_acc):.4f}')


net = Net()
global_net = copy.deepcopy(net)
nets = [copy.deepcopy(net) for i in range(4)]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
global_optimizer = optim.Adam(global_net.parameters(), lr=0.01)
optimizers = [optim.Adam(net.parameters(), lr=0.01) for net in nets]

train_dataset = TensorDataset(all_train_X, all_train_y)
test_dataset = TensorDataset(all_test_X, all_test_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

train_loaders = [DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True) for
                 train_X, train_y, test_X, test_y in Xys]
train_loaders = train_loaders[:3]
test_loaders = [DataLoader(TensorDataset(test_X, test_y), batch_size=32, shuffle=True) for
                train_X, train_y, test_X, test_y in Xys]

net = net.cuda()
for i in range(100):
    ls_acc = []
    ls_loss = []
    for X, y in train_loader:
        X, y = X.cuda(), y.cuda()
        logits = net(X)
        _, pred_y = logits.max(-1)
        acc = float((pred_y == y).sum() / len(y))
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ls_acc.append(acc)
        ls_loss.append(loss)
    print(f'epoch {i}, loss: {sum(ls_loss) / len(ls_loss):.4f}, acc: {sum(ls_acc) / len(ls_acc):.4f}')
    test(test_loaders[3])

for net in nets:
    net.cuda()

for iteration in range(100):
    for i, train_loader in enumerate(train_loaders):
        ls_acc = []
        ls_loss = []
        for epoch in range(2):
            for X, y in train_loader:
                X, y = X.cuda(), y.cuda()
                logits = nets[i](X)
                _, pred_y = logits.max(-1)
                acc = float((pred_y == y).sum() / len(y))
                loss = criterion(logits, y)
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                ls_acc.append(acc)
                ls_loss.append(loss)
            print(f'iteration {iteration}, epoch: {epoch}, client: {i}, '
                  f'loss: {sum(ls_loss) / len(ls_loss):.4f}, acc: {sum(ls_acc) / len(ls_acc):.4f}')

    params = zip(global_net.parameters(), nets[0].parameters(), nets[1].parameters(), nets[2].parameters(),
                 nets[3].parameters())
    for (p0, p1, p2, p3, p4) in params:
        p0.data = (p1.data + p2.data + p3.data + p4.data) / 4
        p1.data = p2.data = p3.data = p4.data = p0.data

    test(test_loaders[3])
