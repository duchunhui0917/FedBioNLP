import os
import string

from tqdm import tqdm
import json

base_dir = os.path.expanduser('~/FedBioNLP')

names = ['Books', 'Electronics', 'Home_&_Kitchen', 'Movies_&_TV']
d = {}


def process_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    t = tqdm(lines)
    for line in t:
        for s in line:
            if s in string.punctuation:
                line = line.replace(s, f' {s} ')

        words = line.strip().split()
        for word in words:
            word = word.lower()
            if word in d:
                d[word] += 1
            else:
                d.update({word: 0})


vocab_file = os.path.join(base_dir, 'data/amazon_review/vocab.json')
for name in names:
    train_file = os.path.join(base_dir, f'data/amazon_review/{name}_train.tsv')
    process_file(train_file)
    test_file = os.path.join(base_dir, f'data/amazon_review/{name}_test.tsv')
    process_file(test_file)

ls = sorted(d.items(), key=lambda x: x[1], reverse=True)
ls = ls[:10000]

sd = {}
special_tokens = ['[PAD]', '[OOV]']
idx = 0
for x in special_tokens:
    sd[x] = idx
    idx += 1
for x in ls:
    word, freq = x
    sd[word] = idx
    idx += 1

with open(vocab_file, 'w') as f:
    json.dump(sd, f)
