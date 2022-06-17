import os
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~/FedBioNLP')


def process(file):
    with open(os.path.join(base_dir, file), 'r') as f:
        lines = f.readlines()
    ls_f1 = []
    for line in lines:
        x = line.strip().split()
        if x[-2] == 'f1:' and len(x) == 6:
            ls_f1.append(float(x[-1]))
    return ls_f1


file_centralized = 'log/centralized/PGR_Q1/distilbert-base-cased_22-06-15 20:43.log'
file_10 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_10_22-06-15 18:13.log'
file_50 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_50_22-06-16 14:50.log'
file_100 = 'log/FedAvg/PGR_Q1/distilbert-base-cased_100_22-06-16 19:45.log'
files = [file_centralized, file_10, file_50, file_100]
ite = 26
labels = ['centralized', 'n=10', 'n=50', 'n=100']
for file, label in zip(files, labels):
    ls_f1 = process(file)
    if label == 'centralized':
        m = max(ls_f1)
        ls_f1 = [m for _ in range(ite)]
        plt.plot(ls_f1[:ite], label=label, linestyle='--')
    else:
        plt.plot(ls_f1[:ite], label=label)
plt.legend()
plt.title('PGR/DistilBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
