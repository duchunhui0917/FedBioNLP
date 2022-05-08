import os.path
import pandas as pd
import xml.etree.ElementTree as ET
import xmltodict

base_dir = os.path.expanduser('~/cross_silo_FL')
xml_path = os.path.join(base_dir, 'data/data_files/gene_disease/CoMAGC/total.xml')
csv_path = os.path.join(base_dir, 'data/data_files/gene_disease/CoMAGC/total.csv')

xml_list = []
tree = ET.parse(xml_path)
xml_data = tree.getroot()
xml_str = ET.tostring(xml_data)
data_dict = dict(xmltodict.parse(xml_str))

ls = []
data_list = data_dict['gene_cancer_corpus']['annotation_unit']
for idx, val in enumerate(data_list):
    sentence = val['sentence']['#text']
    gen_text = val['gene']['#text']
    gen_range = val['gene']['@range']
    gen_start, gen_end = gen_range.split('-')
    try:
        disease_text = val['cancer_term']['#text']
        disease_range = val['cancer_term']['@range']
        disease_start, disease_end = disease_range.split('-')
    except:
        disease_text = None
        disease_start, disease_end = None, None
    label = val['expression_change_keyword_2']['@type']
    x = [sentence, gen_text, gen_start, gen_end, disease_text, disease_start, disease_end, label]
    ls.append(x)

column_name = ['sentence', 'gen_text', 'gen_start', 'gen_end', 'disease_text', 'disease_start', 'disease_end', 'label']
df = pd.DataFrame(ls, columns=column_name)
df.to_csv(csv_path, index=False, encoding='utf8')
