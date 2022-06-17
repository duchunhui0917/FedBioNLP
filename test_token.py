from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-cased"
# model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = ['i love you', 'i love you so much']
inputs = tokenizer(text, padding=True, truncation=True)
print(inputs)

# tokenized_text = tokenizer.tokenize(text)
# tokenized_text = ['CLS'] + tokenized_text + ['SEP', 'PAD']
# id_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# print(tokenized_text)
# print(id_tokens)
