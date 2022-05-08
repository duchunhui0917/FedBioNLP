import os.path
import spacy
import h5py
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_doc(file_name):
    data_file = os.path.join(base_dir, f'data/{file_name}_data.h5')
    with h5py.File(data_file, 'r+') as df:
        attributes = json.loads(df["attributes"][()])

        index_list = attributes['index_list']
        text = []
        for key in index_list:
            sentence = df['X'][str(key)][()].decode('UTF-8')
            text.append(sentence)
    return text


base_dir = os.path.expanduser('~/FedBioNLP')
nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
nlp.max_length = 5000000

# doc1 = nlp("I like salty fries and hamburgers.\n I am eating.")
# doc2 = nlp("Fast food tastes very good.\n I am very happy.")
text1 = load_doc('HPRD50')
text2 = load_doc('AIMed')

model_name = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding1 = tokenizer(text1, padding=True, return_tensors="pt")
encoding2 = tokenizer(text2, padding=True, return_tensors="pt")

model = AutoModel.from_pretrained(model_name)

outputs1 = model(**encoding1, output_hidden_states=True)
embedding1 = outputs1.hidden_states[0].detach().numpy()
outputs2 = model(**encoding2, output_hidden_states=True)
embedding2 = outputs2.hidden_states[0].detach().numpy()

embedding1 = embedding1.mean(0).mean(0, keepdims=True)
embedding2 = embedding2.mean(0).mean(0, keepdims=True)
sim = cosine_similarity(embedding1, embedding2)[0][0]

print('BERT similarity', sim)

# Similarity of two documents
text1 = '\n'.join(text1)
text2 = '\n'.join(text2)

doc1 = nlp(text1)
doc2 = nlp(text2)
print('spacy similarity', doc1.similarity(doc2))
