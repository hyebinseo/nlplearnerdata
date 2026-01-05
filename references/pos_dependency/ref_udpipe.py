# Function for outputting udpipe predictions for ONE text

import spacy_udpipe
import pandas as pd

def udpipe(text, name):
    # Download and run predictions through Spacy
    spacy_udpipe.download("en") 
    nlp = spacy_udpipe.load("en")
    doc = nlp(text)

    # Save results into a dataframe and return results
    rows = []
    for token in doc:
        rows.append({
            "udpipe_word": token.text,
            "udpipe_pos": token.tag_,
            "udpipe_head": token.head.i,
            "udpipe_dependencies": token.dep_
        })

    df_udpipe = pd.DataFrame(rows, columns=["udpipe_word", "udpipe_pos", "udpipe_head", "udpipe_dependencies"])
    df_udpipe.index.name = "udpipe_id"

    df_udpipe.to_csv(f"C:\\Users\\seohy\\nlplearnerdata\\references\\pos_dependency\\out_udpipe\\{name}.csv")

# Creating main blocker
if __name__ == "__main__":
    import sys
    
    text = str(sys.argv[1])
    name = str(sys.argv[2])
    udpipe(text, name)