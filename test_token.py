from transformers import AutoTokenizer

model_name = "distilbert-base-cased"
# model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, never_split=(["<e1>", "</e1>", "<e2>", "</e2>"]))
tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])

text = 'ROTEIN1 or TPA-induced activation and tyrosine phosphorylation of p42 MAPK were completely blocked by down-regulation of PROTEIN2 betaI, epsilon, and delta, but still occurred, together with the cytosolic PLA2 mobility shift, in the absence of external Ca2+'
tokenized_text = tokenizer.tokenize(text)
tokenized_text = ['CLS'] + tokenized_text + ['SEP', 'PAD']
id_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_text)
print(id_tokens)
