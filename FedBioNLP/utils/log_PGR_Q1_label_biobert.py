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


file_centralized = 'log/centralized/PGR_Q1/dmis-lab_biobert-v1.1_22-06-16 09:32.log'
file_5_10 = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_5_10_22-06-16 10:18.log'
file_1_10 = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_1_10_22-06-16 11:47.log'
file_dot5_10 = 'log/FedAvg/PGR_Q1/dmis-lab_biobert-v1.1_0.5_10_22-06-16 10:21.log'
files = [file_centralized, file_5_10, file_1_10, file_dot5_10]
ite = 21
labels = ['centralized', r'$\alpha$=5', r'$\alpha$=1', r'$\alpha$=0.5']
for file, label in zip(files, labels):
    ls_f1 = process(file)
    if label == 'centralized':
        m = max(ls_f1)
        ls_f1 = [m for _ in range(ite)]
        plt.plot(ls_f1[:ite], label=label, linestyle='--')
    else:
        plt.plot(ls_f1[:ite], label=label)
plt.legend()
plt.xticks(list(range(0, 21, 5)))
plt.title('PGR/BioBERT')
plt.xlabel('communication rounds')
plt.ylabel('F1')
plt.show()
