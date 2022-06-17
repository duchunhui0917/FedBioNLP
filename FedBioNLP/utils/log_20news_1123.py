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


file_centralized = 'log/centralized/20news_1123/distilbert-base-cased_22-06-15 10:16.log'
file_5_10 = 'log/FedAvg/20news_1123/distilbert-base-cased_5_10_22-06-15 12:40.log'
file_1_10 = 'log/FedAvg/20news_1123/distilbert-base-cased_1_10_22-06-15 10:04.log'
file_dot5_10 = 'log/FedAvg/20news_1123/distilbert-base-cased_0.5_10_22-06-15 15:16.log'
files = [file_centralized, file_5_10, file_1_10, file_dot5_10]
labels = ['centralized', r'$\alpha$=5', r'$\alpha$=1', r'$\alpha$=0.5']
ite = 81
for file, label in zip(files, labels):
    ls_f1 = process(file)
    if label == 'centralized':
        m = max(ls_f1)
        ls_f1 = [m for _ in range(ite)]
        plt.plot(ls_f1[:ite], label=label, linestyle='--')
    else:
        plt.plot(ls_f1[:ite], label=label)
plt.legend()
plt.title('20news_1123/DistilBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
