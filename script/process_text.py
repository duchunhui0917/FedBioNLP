import copy
import json
import os
import h5py
import pandas as pd
from FedBioNLP import aug_back_translate, aug_tfidf, aug_label_reverse, aug_label_random, aug_sent_len
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

base_dir = os.path.expanduser('~/FedBioNLP')
nlp = StanfordCoreNLP(os.path.join(base_dir, 'stanford-corenlp-4.4.0'))


def process_i2b2_lines(lines, rel):
    e1, label, e2 = rel.split('||')
    tmp = e1.split()
    pos00, pos01 = tmp[-2], tmp[-1]
    line_num, pos00 = pos00.split(':')
    _, pos01 = pos01.split(':')

    tmp = e2.split()
    pos10, pos11 = tmp[-2], tmp[-1]
    _, pos10 = pos10.split(':')
    _, pos11 = pos11.split(':')

    pos00, pos01, pos10, pos11 = int(pos00), int(pos01), int(pos10), int(pos11)

    line = lines[int(line_num) - 1]
    tokens = line.strip().split()
    tokens[pos00] = '<e1>' + tokens[pos00]
    tokens[pos01] = tokens[pos01] + '</e1>'
    tokens[pos10] = '<e2>' + tokens[pos10]
    tokens[pos11] = tokens[pos11] + '</e2>'
    x = ' '.join(tokens)
    y = label.split('\"')[1]
    return x, y


def process_re_lines(line, name):
    if name in ['GAD', 'EU-ADR']:
        x, y = line.strip().split('\t')
        pos00, pos10 = x.index('@GENE$'), x.index('@DISEASE$')
        pos01, pos11 = pos00 + 6, pos10 + 9
    elif name in ['PGR_Q1', 'PGR_Q2']:
        _, x, _, _, _, _, pos00, pos01, pos10, pos11, y = line.strip().split('\t')
        pos00, pos01, pos10, pos11 = int(pos00), int(pos01), int(pos10), int(pos11)
        y = '0' if y == 'False' else '1'
    elif name in ['CoMAGC']:
        x, _, pos00, pos01, _, pos10, pos11, y = line.strip().split('\t')
        pos00, pos01, pos10, pos11 = int(pos00), int(pos01) + 1, int(pos10), int(pos11) + 1
        y = '0' if y == 'Negative_regulation' else '1'
    elif name in ['AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL', 'merged']:
        _, _, y, x, _ = line.strip().split('\t')
        pos00, pos10 = x.index('PROTEIN1'), x.index('PROTEIN2')
        pos01, pos11 = pos00 + 8, pos10 + 8
        y = '0' if y == 'False' else '1'
    else:
        raise NotImplementedError

    text = x
    label = y
    ent1 = '<e1> ' + x[pos00:pos01] + ' </e1>'
    ent2 = '<e2> ' + x[pos10:pos11] + ' </e2>'

    if pos00 < pos10:
        e_text = x[:pos00] + ent1 + x[pos01:pos10] + ent2 + x[pos11:]
    else:
        e_text = x[:pos10] + ent2 + x[pos11:pos00] + ent1 + x[pos01:]
    return text, e_text, label


def assemble(e_ls):
    dep_ls = copy.deepcopy(e_ls)
    dep_ls.remove('<e1>')
    dep_ls.remove('</e1>')
    dep_ls.remove('<e2>')
    dep_ls.remove('</e2>')
    dep_text = ' '.join(dep_ls)
    return dep_text


def h52txt(file_name):
    h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')
    train_path = os.path.join(base_dir, f'data/{file_name}-train.tsv')
    test_path = os.path.join(base_dir, f'data/{file_name}-test.tsv')
    hf = h5py.File(h5_path, 'r')
    attributes = json.loads(hf["attributes"][()])

    train_index_list = attributes['train_index_list']
    test_index_list = attributes['test_index_list']
    for path, index_list in zip([train_path, test_path], [train_index_list, test_index_list]):
        texts = [hf['text'][str(idx)][()].decode('UTF-8') for idx in index_list]
        e_texts = [hf['e_text'][str(idx)][()].decode('UTF-8') for idx in index_list]
        labels = [hf['label'][str(idx)][()].decode('UTF-8') for idx in index_list]
        lines = []
        for text, e_text, label in zip(texts, e_texts, labels):
            line = text + '\t' + e_text + '\t' + label + '\n'
            lines.append(line)
        with open(path, 'w') as f:
            f.writelines(lines)

    hf.close()


def bio_RE_txt2h5(file_name, aug_method=None, process=True):
    dir_name = file_name.split('_')[0]

    if aug_method:
        train_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-train.tsv')
        test_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-test.tsv')
        h5_path = os.path.join(base_dir, f'data/{file_name}_{aug_method}_data.h5')
    else:
        train_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-train.tsv')
        test_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-test.tsv')
        h5_path = os.path.join(base_dir, f'data/{file_name}_data.h5')

    print(h5_path)

    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    test_index = []
    idx = 0

    for cur_index, path in zip([train_index, test_index], [train_path, test_path]):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            if not process:
                text, e_text, label = line.strip().split('\t')
            else:
                text, e_text, label = process_re_lines(line, file_name)
            dep_e_ls = nlp.word_tokenize(e_text)
            dep_e_text = ' '.join(dep_e_ls)
            dep_text = assemble(dep_e_ls)
            dep_ls = nlp.word_tokenize(dep_text)
            dependency = nlp.dependency_parse(dep_text)

            l1 = len(dep_e_ls) - 4
            l2 = max([max(x[1], x[2]) for x in dependency])
            if l1 != l2:
                print(l1, l2)
                print(dep_e_ls)
                print(dep_ls)
                print(dependency)

            hf[f'/text/{idx}'] = text
            hf[f'/e_text/{idx}'] = e_text
            hf[f'/dep_text/{idx}'] = dep_text
            hf[f'/dep_e_text/{idx}'] = dep_e_text
            hf[f'/dependency/{idx}'] = json.dumps(dependency)
            hf[f'/label/{idx}'] = label

            doc_index[idx] = 0
            index_list.append(idx)
            cur_index.append(idx)
            idx += 1
    atts = ['text', 'e_text', 'dep_text', 'dep_e_text', 'dependency', 'label']
    attributes = {
        'doc_index': doc_index,
        'label_vocab': {'0': 0, '1': 1},
        'num_labels': 2,
        'index_list': index_list,
        'train_index_list': train_index,
        'test_index_list': test_index,
        'task_type': 'relation_extraction',
        'atts': json.dumps(atts)
    }
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


def aug_text(file_name, aug_method='aug', mode='train', process=True):
    dir_name = file_name.split('_')[0]
    path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}-{mode}.tsv')
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    e_texts = []
    labels = []
    for line in lines:
        if not process:
            text, e_text, label = line.strip().split('\t')
        else:
            text, e_text, label = process_re_lines(line, file_name)
        e_texts.append(e_text)
        labels.append(label)

    aug_texts = e_texts
    aug_labels = labels
    if aug_method == 'back_translate':
        aug_texts = aug_back_translate(dir_name, file_name, mode)
    elif aug_method == 'tfidf':
        aug_texts = aug_tfidf(e_texts, dir_name)
    elif aug_method == 'label_reverse':
        aug_labels = aug_label_reverse(labels)
    elif aug_method == 'label_random':
        aug_labels = aug_label_random(labels)
    elif aug_method == 'sent_len':
        aug_labels = aug_sent_len(e_texts, 30)

    liens = []
    for aug_text, aug_label in zip(aug_texts, aug_labels):
        text = aug_text
        for x in ['<e1> ', ' </e1>', '<e2> ', ' </e2>']:
            if x not in text:
                print(text)
            text = text.replace(x, '')
        liens.append(text + '\t' + aug_text + '\t' + aug_label + '\n')
    aug_path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_{aug_method}-{mode}.tsv')
    print(aug_path)
    with open(aug_path, 'w', encoding='utf8') as f:
        f.writelines(liens)


def i2b2_txt2h5(name_list):
    h5_name = '*'.join(name_list)
    h5_path = os.path.join(base_dir, f'data/{h5_name}_data.h5')
    hf = h5py.File(h5_path, 'w')
    doc_index = {}
    index_list = []
    train_index = []
    test_index = []
    idx = 0
    for d, name in enumerate(name_list):
        dir_path = os.path.join(base_dir, f'data/i2b2/{name}/train_data')
        file_names = os.listdir(dir_path)
        for file_name in file_names:
            train_data = os.path.join(base_dir, f'data/i2b2/{name}/train_data/{file_name}')
            train_target = os.path.join(base_dir, f'data/i2b2/{name}/train_target/{file_name}')
            with open(train_data, 'r', encoding='utf8') as f:
                lines = f.readlines()
            with open(train_target, 'r', encoding='utf8') as f:
                rels = f.readlines()

            for rel in rels:
                x, y = process_i2b2_lines(lines, rel)
                hf[f'/X/{idx}'] = x
                hf[f'/Y/{idx}'] = y
                index_list.append(idx)
                train_index.append(idx)
                doc_index[idx] = d
                idx += 1

        dir_path = os.path.join(base_dir, f'data/i2b2/{name}/test_data')
        file_names = os.listdir(dir_path)
        for file_name in file_names:
            test_data = os.path.join(base_dir, f'data/i2b2/{name}/test_data/{file_name}')
            test_target = os.path.join(base_dir, f'data/i2b2/{name}/test_target/{file_name}')
            with open(test_data, 'r', encoding='utf8') as f:
                lines = f.readlines()
            with open(test_target, 'r', encoding='utf8') as f:
                rels = f.readlines()

            for rel in rels:
                x, y = process_i2b2_lines(lines, rel)
                hf[f'/X/{idx}'] = x
                hf[f'/Y/{idx}'] = y
                index_list.append(idx)
                test_index.append(idx)
                doc_index[idx] = d
                idx += 1
    attributes = {'doc_index': doc_index,
                  'label_vocab': {
                      'TrIP': 0,
                      'TrWP': 1,
                      'TrCP': 2,
                      'TrAP': 3,
                      'TrNAP': 4,
                      'TeRP': 5,
                      'TeCP': 6,
                      'PIP': 7
                  },
                  'num_labels': 8,
                  'index_list': index_list,
                  'train_index_list': train_index,
                  'test_index_list': test_index,
                  'task_type': 'relation_extraction'}
    hf['/attributes'] = json.dumps(attributes)
    hf.close()


def csv2tsv(name_list):
    for name in name_list:
        for i in ['train', 'test']:
            file_path = os.path.join(base_dir, f'data/bio_RE/{name}/{name}-{i}.csv')

            df = pd.read_csv(file_path, encoding='utf8')
            df = df[(df['passage'].str.contains('PROTEIN1')) & (df['passage'].str.contains('PROTEIN2'))]
            file_path = file_path.split('.')[0] + '.tsv'
            df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')


# bio_RE_txt2h5('GAD', process=True)
# bio_RE_txt2h5('EU-ADR', process=True)
# bio_RE_txt2h5('CoMAGC', process=True)
# bio_RE_txt2h5('PGR_Q1', process=True)
# bio_RE_txt2h5('PGR_Q2', process=True)

# bio_RE_txt2h5('AIMed', process=True)
# bio_RE_txt2h5('BioInfer', process=True)
# bio_RE_txt2h5('HPRD50', process=True)
# bio_RE_txt2h5('IEPA', process=True)
# bio_RE_txt2h5('LLL', process=True)

# bio_RE_txt2h5('AIMed_2|2', aug_method='back_translate', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='label_reverse', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='tfidf', process=False)
# bio_RE_txt2h5('AIMed_2|2', aug_method='label_random', process=False)
bio_RE_txt2h5('AIMed_2|2', aug_method='sent_len', process=False)

# aug_text('AIMed_2|2', aug_method='back_translate', mode='test', process=False)
# aug_text('AIMed_2|2', aug_method='label_random', mode='train', process=False)
# aug_text('AIMed_1|2_balance', aug_method='label_reverse', mode='train', process=False)
# aug_text('AIMed_2|2', aug_method='sent_len', mode='test', process=False)
# h52txt('AIMed_2|2')
# h52txt('AIMed_1|2_balance')
