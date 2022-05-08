import h5py
import json
import matplotlib.pyplot as plt

base_dir = os.path.expanduser('~/FedBioNLP')


def process(dataset_name, model_name, max_seq_length):
    data_file = os.path.join(base_dir, f'data/{dataset_name}_data.h5')

    with h5py.File(data_file, 'r+') as df:
        attributes = json.loads(df["attributes"][()])

        label_vocab = attributes['label_vocab']
        index_list = attributes['index_list']
        train_idx = attributes['train_index_list']
        test_idx = attributes['test_index_list']
        task_type = attributes['task_type']
        if 'doc_index' in attributes:
            doc_index = attributes['doc_index']

        # ls["attributes"][()] = json.dumps(attributes)

        tokenizer = re_tokenizer
        n_classes = attributes['num_labels']
        # model = RelationExtractionBERT(model_name, n_classes)

        # model = RelationExtractionHorizonBERT(model_name, n_classes)
        args = {'org_text': [],
                'e_text': [],
                'dependency': [],
                'doc': [],
                'label': []}
        my_train_dataset = my_test_dataset = RelationExtractionDataset
        for idx in index_list:
            org_text = df['org_text'][str(idx)][()].decode('UTF-8')
            e_text = df['e_text'][str(idx)][()].decode('UTF-8')
            dependency = df['dependency'][str(idx)][()].decode('UTF-8')
            label = df['label'][str(idx)][()].decode('UTF-8')
            doc = doc_index[str(idx)]

            args['org_text'].append(org_text)
            args['e_text'].append(e_text)
            args['dependency'].append(dependency)
            args['label'].append(label_vocab[label])
            args['doc'].append(doc)

        args = tokenizer(args, model_name, max_seq_length)
        len_sdp = args["len_sdp"]

    return len_sdp


GAD_len_sdp = process('GAD', 'distilbert-base-cased', 384)
print(len(GAD_len_sdp))
GAD_len_sdp = [x for x in GAD_len_sdp if x is not None]
print(len(GAD_len_sdp))
print(sum(GAD_len_sdp) / len(GAD_len_sdp))

EU_ADR_len_sdp = process('EU-ADR', 'distilbert-base-cased', 384)
print(len(EU_ADR_len_sdp))
EU_ADR_len_sdp = [x for x in EU_ADR_len_sdp if x is not None]
print(len(EU_ADR_len_sdp))
print(sum(EU_ADR_len_sdp) / len(EU_ADR_len_sdp))

CoMAGC_len_sdp = process('CoMAGC', 'distilbert-base-cased', 384)
print(len(CoMAGC_len_sdp))
CoMAGC_len_sdp = [x for x in CoMAGC_len_sdp if x is not None]
print(len(CoMAGC_len_sdp))
print(sum(CoMAGC_len_sdp) / len(CoMAGC_len_sdp))

PGR_len_sdp = process('PGR_Q1', 'distilbert-base-cased', 384)
print(len(PGR_len_sdp))
PGR_len_sdp = [x for x in PGR_len_sdp if x is not None]
print(len(PGR_len_sdp))
print(sum(PGR_len_sdp) / len(PGR_len_sdp))

AIMed_len_sdp = process('AIMed', 'distilbert-base-cased', 384)
print(len(AIMed_len_sdp))
AIMed_len_sdp = [x for x in AIMed_len_sdp if x is not None]
print(len(AIMed_len_sdp))
print(sum(AIMed_len_sdp) / len(AIMed_len_sdp))

BioInfer_len_sdp = process('BioInfer', 'distilbert-base-cased', 384)
print(len(BioInfer_len_sdp))
BioInfer_len_sdp = [x for x in BioInfer_len_sdp if x is not None]
print(len(BioInfer_len_sdp))
print(sum(BioInfer_len_sdp) / len(BioInfer_len_sdp))

HPRD50_len_sdp = process('HPRD50', 'distilbert-base-cased', 384)
print(len(HPRD50_len_sdp))
HPRD50_len_sdp = [x for x in HPRD50_len_sdp if x is not None]
print(len(HPRD50_len_sdp))
print(sum(HPRD50_len_sdp) / len(HPRD50_len_sdp))

IEPA_len_sdp = process('IEPA', 'distilbert-base-cased', 384)
print(len(IEPA_len_sdp))
IEPA_len_sdp = [x for x in IEPA_len_sdp if x is not None]
print(len(IEPA_len_sdp))
print(sum(IEPA_len_sdp) / len(IEPA_len_sdp))

LLL_len_sdp = process('LLL', 'distilbert-base-cased', 384)
print(len(LLL_len_sdp))
LLL_len_sdp = [x for x in LLL_len_sdp if x is not None]
print(len(LLL_len_sdp))
print(sum(LLL_len_sdp) / len(LLL_len_sdp))

ls = [GAD_len_sdp, EU_ADR_len_sdp, CoMAGC_len_sdp, PGR_len_sdp,
      AIMed_len_sdp, BioInfer_len_sdp, HPRD50_len_sdp, IEPA_len_sdp, LLL_len_sdp]
plt.boxplot(ls)
plt.show()
