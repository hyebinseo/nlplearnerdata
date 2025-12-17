# Function for running the biaffine parser on ONE text

# Installing and following post: https://medium.com/geekculture/how-to-use-allennlps-pretrained-ner-model-in-2021-b60335904fc3
# Using model based on official documentation: https://docs.allennlp.org/models/main/#pre-trained-models
# Using environment nlplearnerdata-allen (Python 3.8.20)

# Import neded libraries
from allennlp_models import pretrained
import pandas as pd

# Create function for running allen from main
def allen(text, name):
    # The model needs to be run, once per session
    model = pretrained.load_predictor("structured-prediction-biaffine-parser")
    data = model.predict(sentence = text)

    # Extract needed data, then save into a csv file
    allen_words = data["words"]
    allen_pos = data["pos"]
    allen_heads = data["predicted_heads"]
    allen_dependencies = data["predicted_dependencies"]

    df_allen = pd.DataFrame({
        "allen_words": allen_words,
        "allen_pos": allen_pos,
        "allen_heads": allen_heads,
        "allen_dependencies": allen_dependencies
    })
    df_allen.index.name = "allen_id"

    # Save df_allen to be read at main.py
    df_allen.to_csv(f"out_allen/{name}.csv")

# Testing the allen function
# Saves results of predicting text into name.csv
# text = "example text with name saving"
# name = "name"
# allen(text, name)

# Creating main blocker based on test
if __name__ == "__main__":
    import sys
    
    text = str(sys.argv[1])
    name = str(sys.argv[2])
    allen(text, name)