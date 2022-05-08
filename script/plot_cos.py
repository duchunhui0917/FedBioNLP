import os.path
import json
import matplotlib.pyplot as plt

import pandas
import seaborn as sns
import pandas as pd

alg_name = 'FedAvg'
dataset_name = 'AIMed_1|2*AIMed_2|2_back_translate'
model_name = 'distilbert-base-cased'
base_dir = os.path.expanduser('~/FedBioNLP')
grad_cos = os.path.join(base_dir, f'grad_cos/{alg_name}/{dataset_name}_{model_name}.json')
pos0, pos1 = 0, 1
with open(grad_cos, 'r') as f:
    ls = json.load(f)

df = pandas.DataFrame()
keys = ls[0].keys()

for key in keys:
    y = [x[key][pos0][pos1] for x in ls]
    y = y[:30]
    plt.plot(y, label=key)
plt.axhline(y=0, linestyle='--', color='k')

plt.title(dataset_name)
plt.xlabel('communication round')
plt.ylabel('cosine similarity')
plt.xticks(range(0, 31, 5))
plt.legend()
plt.show()
