import re
import os
import json
import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw
from tqdm import tqdm
from multiprocessing import Pool
from deep_translator import MyMemoryTranslator as Translator
from tqdm import tqdm
import time

base_dir = os.path.expanduser('~/FedBioNLP')
model_path = ''


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


# --- train tf-idf model
def train_tfidf(train_data):
    print('tfidf model not exists, start training ...')
    train_x_tokens = [_tokenizer(x) for x in train_data]
    tfidf_model = nmw.TfIdf()
    tfidf_model.train(train_x_tokens)
    print(model_path)
    tfidf_model.save(model_path)
    print('tfidf model training done!')


def sub_task(train_data):
    """

    Args:
        train_data:

    Returns:

    """
    aug = naw.TfIdfAug(model_path=model_path, tokenizer=_tokenizer)
    aug_train_data = []
    for text in tqdm(train_data):
        e1_l = text.index('<e1>')
        e1_r = text.index('</e1>') + 5
        e2_l = text.index('<e2>')
        e2_r = text.index('</e2>') + 5

        e1 = text[e1_l:e1_r]
        e2 = text[e2_l:e2_r]

        if e1_r <= e2_l:
            left = text[:e1_l]
            middle = text[e1_r:e2_l]
            right = text[e2_r:]
            aug_left = '{} '.format(aug.augment(left))
            aug_middle = ' {} '.format(aug.augment(middle))
            aug_right = ' {}'.format(aug.augment(right))

            aug_text = aug_left + e1 + aug_middle + e2 + aug_right

            aug_e1_l = len(aug_left)
            aug_e1_r = len(aug_left) + len(e1)

            aug_e2_l = len(aug_left) + len(e1) + len(aug_middle)
            aug_e2_r = len(aug_left) + len(e1) + len(aug_middle) + len(e2)
            assert aug_text[aug_e1_l:aug_e1_r] == e1
            assert aug_text[aug_e2_l:aug_e2_r] == e2
        else:
            left = text[:e2_l]
            middle = text[e2_r:e1_l]
            right = text[e1_r:]
            aug_left = '{} '.format(aug.augment(left))
            aug_middle = ' {} '.format(aug.augment(middle))
            aug_right = ' {}'.format(aug.augment(right))
            aug_text = aug_left + e2 + aug_middle + e1 + aug_right

            aug_e2_l = len(aug_left)
            aug_e2_r = len(aug_left) + len(e2)

            aug_e1_l = len(aug_left) + len(e2) + len(aug_middle)
            aug_e1_r = len(aug_left) + len(e2) + len(aug_middle) + len(e1)
            assert aug_text[aug_e1_l:aug_e1_r] == e1
            assert aug_text[aug_e2_l:aug_e2_r] == e2

        aug_train_data.append(aug_text)

    return aug_train_data


# --- multi-processing augmentation ---
def aug_tfidf(train_data, file_name):
    global model_path
    model_path = os.path.join(base_dir, f'ckpt/tfidf/{file_name}')
    train_tfidf(train_data)

    process_num = 48
    print('augment training data with {} threads ...'.format(process_num))
    pool = Pool(processes=process_num)

    thread_idx = [int(len(train_data) / (process_num / i))
                  if i else 0 for i in range(0, process_num + 1)]
    parts = [train_data[thread_idx[i]:thread_idx[i + 1]] for i in range(process_num)]
    results = pool.map(sub_task, parts)

    pool.close()
    pool.join()
    print('augment done!')

    aug_train_data = []
    for x in results:
        aug_train_data.extend(x)
    return aug_train_data


def aug_back_translate(dir_name, file_name, mode):
    path = os.path.join(base_dir, f'data/bio_RE/{dir_name}/{file_name}_en-{mode}.tsv')
    with open(path, 'r') as f:
        lines = f.readlines()
    aug_texts = []
    for line in lines:
        e_text, e_en_text = line.strip().split('\t')
        try:
            pos00 = e_en_text.index('<e1>')
            if e_en_text[pos00 + 4] != ' ':
                e_en_text = e_en_text[:pos00 + 4] + ' ' + e_en_text[pos00 + 4:]
            pos01 = e_en_text.index('</e1>')
            if e_en_text[pos01 - 1] != ' ':
                e_en_text = e_en_text[:pos01] + ' ' + e_en_text[pos01:]
            pos10 = e_en_text.index('<e2>')
            if e_en_text[pos10 + 4] != ' ':
                e_en_text = e_en_text[:pos10 + 4] + ' ' + e_en_text[pos10 + 4:]
            pos11 = e_en_text.index('</e2>')
            if e_en_text[pos11 - 1] != ' ':
                e_en_text = e_en_text[:pos11] + ' ' + e_en_text[pos11:]
        except:
            print(e_en_text)
            e_en_text = e_text
        aug_texts.append(e_en_text)

    return aug_texts


def aug_label_reverse(labels):
    aug_labels = []
    for label in labels:
        if label == '0':
            aug_labels.append('1')
        elif label == '1':
            aug_labels.append('0')
    return aug_labels
