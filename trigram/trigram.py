from spacy.lang.en import English
import nltk
import pandas as pd
import collections
import numpy as np

class trigram(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        print("Initializing")
        filename = "game_of_thrones.txt"
        raw_text = open(filename, 'r', encoding='utf-8').read()
        raw_text = raw_text.lower()
        nlp = English()
        tokenizer = nlp.tokenizer
        tokens = tokenizer(raw_text)
        tokens_list = [token.text for token in tokens if not token.is_punct]
        trigrams = list(nltk.trigrams(tokens_list))
        c = collections.Counter
        tri = [' '.join(i) for i in trigrams]
        x = c(tri)
        df = pd.DataFrame.from_dict(x, orient='index').reset_index()
        df = df.rename(columns={'index': 'trigrams', 0: 'count'})
        self._df = df

    def predict(self, X, features_names=None):
        print("Predict called - will run identity function")
        predictions = []
        for item in X:
            list_words=item.split(' ')
            while len(list_words)<20:
                t=self._df[self._df['trigrams'].str.startswith(' '.join(list_words[-2:]))]
                if len(t) == 0:
                    break
                t['count']=t['count']/sum(t['count'])
                word=np.random.choice(t['trigrams'], p=t['count'])
                list_words.append(word.split(' ')[-1])
            predictions.append(' '.join(list_words[:-1]))
        return(predictions)