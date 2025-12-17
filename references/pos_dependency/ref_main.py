import subprocess
import glob
import os
import pandas as pd

# ==================================
# Saving paths, raw text, and names from the raw_text folder
# ==================================
# Folder path with learner texts
folder_path = r"C:\Users\seohy\nlplearnerdata\raw_text"


# Source: https://github.com/braj29/Text-to-Conll/blob/main/texts_to_conll.py
# Copy-pasted and adjusted the code for converting txt to conllu
# Get paths of each learner text from folder_path
def get_paths(path):
    paths_ = glob.glob(path + "/*.txt")
    return paths_

def load_text(txt_path):
    with open(txt_path, "r", encoding="utf-8") as infile:
        content = infile.read()
    return content

# txt_paths: List of paths for each learner text
txt_paths = get_paths(folder_path)
# Save learner text into a list
# Get names of each txt file from path
texts = []
names = []
for txt_path in txt_paths:
    texts.append(load_text(txt_path))
    # Source: https://medium.com/@emoome82/extracting-file-names-without-extensions-in-python-caabe8532f92
    name, extension = os.path.splitext(os.path.basename(txt_path))
    names.append(name)

# ==================================
# Using subprocess to run each nlp model
# --> output each result into respective folders in csv format
# ==================================
# Running allen.py using subprocess
for path, text, name in zip(txt_paths, texts, names):
    subprocess.run([
        r"C:\Users\seohy\anaconda3\envs\nlplearnerdata-allen\python.exe", 
        "ref_allen.py",
        text,
        name
    ])

# Running stanza_atis.py using subprocess
for path, text, name in zip(txt_paths, texts, names):
    subprocess.run([
        r"C:\Users\seohy\anaconda3\envs\nlplearnerdata\python.exe", 
        "ref_stanza_atis.py", 
        text,
        name
    ])

# Running stanza_esl.py using subprocess
for path, text, name in zip(txt_paths, texts, names):
    subprocess.run([
        r"C:\Users\seohy\anaconda3\envs\nlplearnerdata\python.exe", 
        "ref_stanza_eslspok.py", 
        text,
        name
    ])

# Running udpipe.py using subprocess
for path, text, name in zip(txt_paths, texts, names):
    subprocess.run([
        r"C:\Users\seohy\anaconda3\envs\nlplearnerdata\python.exe", 
        "ref_udpipe.py", 
        text,
        name
    ])

# ==================================
# Combine each parser results into one df 
# ==================================
for name in names:
    # Read in
    df_allen = pd.read_csv(f"out_allen/{name}.csv")
    df_stanza_atis = pd.read_csv(f"out_stanza_atis/{name}.csv")
    df_stanza_esl = pd.read_csv(f"out_stanza_esl/{name}.csv")
    df_udpipe = pd.read_csv(f"out_udpipe/{name}.csv")

    # Add one to index column for atis and stanza_esl to match word index 
    df_stanza_atis.index = df_stanza_atis.index.astype(int) + 1
    df_stanza_esl.index = df_stanza_esl.index.astype(int) + 1

    # Combine all dataframes into one dataframe
    df_total = pd.concat([df_allen, df_stanza_atis, df_stanza_esl, df_udpipe], axis=1)

    # Output combined df into out_total folder
    df_total.to_csv(f"out_total/{name}.csv")