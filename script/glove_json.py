import os
from tqdm import tqdm
import json
from collections import OrderedDict

base_dir = os.path.expanduser('~/FedBioNLP')

names = ['50d', '100d', '200d', '300d']


def process_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    dim = len(lines[0].split(' ')) - 1
    d = OrderedDict()

    oov_emb = [0.] * dim
    t = tqdm(lines)
    for i, line in enumerate(t):
        line = line.strip().split(' ')
        word = line[0]
        embedding = [float(e) for e in line[1:]]
        d.update({word: embedding})
        oov_emb = [(x1 * i + x2) / (i + 1) for x1, x2 in zip(oov_emb, embedding)]
    pad_emb = [0.] * dim
    d.update({'[PAD]': pad_emb})
    d.update({'[OOV]': oov_emb})
    return d


for name in names:
    txt_file = os.path.join(base_dir, f'data/glove.6B/glove.6B.{name}.txt')
    json_file = os.path.join(base_dir, f'data/glove.6B/glove.6B.{name}.json')
    d = process_file(txt_file)
    with open(json_file, 'w') as f:
        json.dump(d, f)
