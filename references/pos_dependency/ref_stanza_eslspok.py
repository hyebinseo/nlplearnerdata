import stanza
import pandas as pd

# Downloading the English model for Stanza
stanza.download("en", package="esl")

def stanza_esl(text, name):
    nlp_esl = stanza.Pipeline(lang="en", package="ESLSpok")
    doc_esl = nlp_esl(text) # run annotation over a sentence

    output = [f'word: {word.text}\txpos: {word.xpos}\thead: {word.head}\tdeprel: {word.deprel}' for sent in doc_esl.sentences for word in sent.words]

    rows = [
        {
            "enesl_word": word.text,
            "enesl_pos": word.xpos,
            "enesl_head": word.head,
            "enesl_dependencies": word.deprel,
        }
        for sent in doc_esl.sentences
        for word in sent.words
    ]

    df_enesl = pd.DataFrame(rows)
    df_enesl.index.name = "enesl_id"
    df_enesl.index = df_enesl.index + 1


    df_enesl.to_csv(f"out_stanza_esl/{name}.csv")

# Creating main blocker based on test
if __name__ == "__main__":
    import sys
    
    text = str(sys.argv[1])
    name = str(sys.argv[2])

    stanza_esl(text, name)