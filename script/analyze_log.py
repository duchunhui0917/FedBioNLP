import os

n_clients = 2
layers = '.*embedding*layer.0*layer.1*layer.2*layer.3*layer.4*layer.5*classifier'
layers = layers.split('*')
dataset = 'AIMed_1|2*AIMed_2|2_back_translate'
model = 'distilbert-base-cased'
t = '22-05-10 10:42'
base_dir = os.path.expanduser('~/FedBioNLP')
log_file = os.path.join(base_dir, f'log/{dataset}_{model}_{t}.log')
org_cos_sims = {layer: [] for layer in layers}
upd_cos_sims = {layer: [] for layer in layers}
clients_global_f1 = {i: [] for i in range(n_clients)}
clients_personal_f1 = {i: [] for i in range(n_clients)}
with open(log_file, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    try:
        x1, x2, x3, x4 = line.split('\t')
        if 'compute gradient cosine similarity' in x4:
            cos = org_cos_sims
        elif 'compute updated gradient cosine similarity' in x4:
            cos = upd_cos_sims
        elif 'cosine similarity matrix' in x4:
            layer = x4.split()[1]
            sim = eval(line[i + 1])
            cos[layer].append(sim)
        elif 'personal model' in x4:
            c = x4.split()[2]
            f1 = clients_personal_f1
        elif 'global model' in x4:
            c = x4.split()[2]
            f1 = clients_global_f1
        elif 'f1' in x4:
            f1[c].append(x4.split())
    except:
        pass
    i += 1
