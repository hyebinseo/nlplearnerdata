import glob
import os

import pandas as pd

import spacy

# ==================================
# Saving paths, raw text, and names from the raw_text folder
# ==================================
# Folder path with learner texts
folder_path = r"C:\Users\seohy\nlplearnerdata\raw_text"

# Source: https://github.com/braj29/Text-to-Conll/blob/main/texts_to_conll.py
# Copy-pasted and adjusted the code for converting txt to conllu

# Read in and save content into lists:
#  - txt_paths: list of paths for each learner text
#  - texts: list of raw text for each learner 
#  - names: list of file name (i.e., the text id) for each learner
def get_paths(path):
    paths_ = glob.glob(path + "/*.txt")
    return paths_
def load_text(txt_path):
    with open(txt_path, "r", encoding="utf-8") as infile:
        content = infile.read()
    return content

txt_paths = get_paths(folder_path)
texts = []
names = []
for txt_path in txt_paths:
    texts.append(load_text(txt_path))
    # Source: https://medium.com/@emoome82/extracting-file-names-without-extensions-in-python-caabe8532f92
    name, extension = os.path.splitext(os.path.basename(txt_path))
    names.append(name)

# ==================================
# Run and save results from en_udv25_englishewt_trf model
# ==================================
spacy.prefer_gpu()
nlp = spacy.load("en_udv25_englishewt_trf")

en_udv25_englishewt_trf_results = {
    "FORM": [],
    "POS": [],
    "HEAD": [],
    "DEPREL": []
}

for path, text, name in zip(txt_paths, texts, names):
    doc = nlp(text)

    for token in doc:
        en_udv25_englishewt_trf_results["FORM"].append(token.text)
        en_udv25_englishewt_trf_results["POS"].append(token.tag_)
        en_udv25_englishewt_trf_results["HEAD"].append(token.head.i)
        en_udv25_englishewt_trf_results["DEPREL"].append(token.dep_)

    en_udv25_englishewt_trf_df = pd.DataFrame.from_dict(en_udv25_englishewt_trf_results)
    en_udv25_englishewt_trf_df.to_csv(f"C:\\Users\\seohy\\nlplearnerdata\\model_evaluation\\en_udv25_englishewt_trf\\{name}.csv")