# Guidelines for using the Stanford parser on ONE TEXT:
# 1. EDIT THE BAT FILE INTO THE FOLLOWING -DONE
#    @echo off
#    java -mx512m -cp "\*;" edu.stanford.nlp.parser.lexparser.LexicalizedParser ^
#    -outputFormat "wordsAndTags, typedDependencies" ^
#    -writeOutputFiles ^
#    -outputFilesDirectory "output" ^
#    -outputFilesExtension "txt" ^
#    edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz %1
# 2. OPEN PARSER FOLDER IN VSCODE, CREATE FOLDER NAMED "OUTPUT":###
# 3. IN TERMINAL TYPE:###
#    .\lexparser.bat data\testsent.txt myresult
# 4. CHECK OUTPUT FOLDER FOR THE RESULTS ###
# 5. PARSE THE RESULTS INTO DFS USING CODE

# Import the libraries
import subprocess
import glob
import os
import pandas as pd

import re
import numpy as np

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
# Using subprocess to run the Stanford model on each text
# ==================================
# Run in cmd the following from the guidelines using subprocess
for path, text, name in zip(txt_paths, texts, names):
    # Set path for the Stanford parser
    parser_path = "C:\\Users\\seohy\\stanford-parser-4.2.0\\stanford-parser-full-2020-11-17"
    # Set cmd command for running Stanford parser
    command = f"""cd {parser_path} && .\lexparser.bat {path} {names}"""
    # Using subprocess for running Stanford parser
    subprocess.run(command, shell = True)

# ==================================
# Converting .txt output from the Stanford Parser into dataframes for evaluation
# ==================================
# Read a .txt file into a list, starting a new list item whenever an empty line occurs
for path, text, name in zip(txt_paths, texts, names):
    penn_pos = ""
    dependency = ""

    output_path = "C:\\Users\\seohy\\stanford-parser-4.2.0\\stanford-parser-full-2020-11-17\\output\\" + name + ".txt"
    
    blank_line = 0
    with open(output_path, "r", encoding = "utf-8") as file:
        for line in file:
            if line == "\n":
                blank_line += 1
            
            if (blank_line % 2 == 0) & (line != "\n"):
                penn_pos += line.strip("\n") + " "
            elif (blank_line % 2 == 1) & (line != "\n"):
                dependency += line.strip(" \n") + " "
        penn_pos = penn_pos.strip()
        dependency = dependency.strip()

        penn_pos_list = penn_pos.strip().split(" ")
        dependency_list = dependency.strip().split(") ")

        pos_results = {
            "FORM": [],
            "POS": [],
        }

        dependency_results = {
            "FORM": [],
            "HEAD": [],
            "DEPREL": [],
        }

        for penn in penn_pos_list:
            item = re.findall(r"[^/]+", penn)
            pos_results["FORM"].append(item[0])
            pos_results["POS"].append(item[1])

        pos_df = pd.DataFrame.from_dict(pos_results)

        # [0] --> relationship
        # [1] --> head
        # [2] --> head's index + ,
        # [3] --> word
        # [4] --> word's ID 

        for dependency in dependency_list:
            item = re.split(r"[(/\s/-]+", dependency)

            # Removing the additional comma for the index
            index = item[2]
            index = index[:-1]

            dependency_results["FORM"].append(item[3])
            dependency_results["HEAD"].append(index)
            dependency_results["DEPREL"].append(item[0])

        dependency_df = pd.DataFrame.from_dict(dependency_results)
        dependency_df = dependency_df.drop_duplicates(subset = "FORM")

        stanford_parser_df = pd.merge(pos_df, dependency_df, how = "left", on = "FORM")
        stanford_parser_df.index = np.arange(1, len(stanford_parser_df)+1)
        stanford_parser_df.to_csv(f"C:\\Users\\seohy\\nlplearnerdata\\model_evaluation\\stanford_parser\\{name}.csv")