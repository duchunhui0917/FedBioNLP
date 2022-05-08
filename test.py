import os.path

from stanfordcorenlp import StanfordCoreNLP

base_dir = os.path.expanduser('~/FedBioNLP')

nlp = StanfordCoreNLP(os.path.join(base_dir, 'stanford-corenlp-4.4.0'))
# nlp = StanfordCoreNLP('http://localhost', port=9000)  #通过服务器访问

sentence = 'ROTEIN1 or TPA-induced activation and tyrosine phosphorylation of p42 MAPK were completely blocked by down-regulation of PROTEIN2 betaI, epsilon, and delta, but still occurred, together with the cytosolic PLA2 mobility shift, in the absence of external Ca2+'

sentence = ['i', 'like', 'you']
print('Tokenize:', nlp.word_tokenize(sentence))  # 令牌化
print('Part of Speech:', nlp.pos_tag(sentence))  # 词性标注
print('Named Entities:', nlp.ner(sentence))  # 命名实体
print('Constituency Parsing:', nlp.parse(sentence))  # 语法树，成分句法把句子组织成短语的形式
print('Dependency Parsing:', nlp.dependency_parse(sentence))  # 依存句法 揭示句子中词的依赖关系

props = {'timeout': '5000000', 'annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true',
         'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
results = nlp.annotate(' '.join(sentence), properties=props)

nlp.close()
