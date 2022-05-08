## Parameters

| Parameter      | Description                                                                                            |
|----------------|--------------------------------------------------------------------------------------------------------|
| `model`        | The model architecture. Options: `CNN`, `LSTM`, `ResNet18`,`bert-base-cased`, `distilbert-base-cased`. |
| `dataset`      | Dataset to use. Options: `CIFAR10`, `CIFAR100`, `MNIST`, `FashionMNIST`, `wikiner`, `ploner`.          |
| `n_clients`    | Number of clients.                                                                                     |
| `alg`          | The training algorithm. Options: `FedAvg`, `FedProx`, `DFL`, `SOLO`.                                   |
| `lr`           | Learning rate.                                                                                         |
| `bs`           | Batch size.                                                                                            |
| `n_epochs`     | Number of local epochs.                                                                                |
| `n_iterations` | Number of iterations.                                                                                  |
| `beta`         | The concentration parameter of the Dirichlet distribution.                                             |