import stanza
import pandas as pd

# Downloading the English model for Stanza
stanza.download("en", package="Atis")

def stanza_atis(text, name):
    nlp_atis = stanza.Pipeline(lang="en", package="Atis")
    doc_atis = nlp_atis(text) # run annotation over a sentence

    output = [f"atis_word: {word.text}\atis_pos: {word.xpos}\atis_head: {word.head}\atis_dependencies: {word.deprel}" for sent in doc_atis.sentences for word in sent.words]

    rows = [
        {
            "atis_word": word.text,
            "atis_pos (Note: in UD)": word.upos,
            "atis_head": word.head,
            "atis_dependencies": word.deprel,
        }
        for sent in doc_atis.sentences
        for word in sent.words
    ]

    df_atis = pd.DataFrame(rows)
    df_atis.index.name = "atis_id"
    df_atis.index = df_atis.index + 1

    df_atis.to_csv(f"out_stanza_atis/{name}.csv")

# Creating main blocker based on test
if __name__ == "__main__":
    import sys
    
    text = str(sys.argv[1])
    name = str(sys.argv[2])

    stanza_atis(text, name)