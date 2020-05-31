import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd

class Classifier(object):
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.lem = WordNetLemmatizer()

    def sent_tokenisation(self, sent):
        return sent_tokenize(sent)

    def word_tokenisation(self, sent):
        return word_tokenize(sent)
    
    def freq_distribution(self, tokenized_word_array):
        return FreqDist(tokenized_word_array)

    def remove_stopwords(self, tokenized_word_array):
        filtered_sent=[]
        stop_words=set(stopwords.words("english"))

        for word in tokenized_word_array:
            if word not in stop_words:
                filtered_sent.append(word)
        return filtered_sent

    def word_lemmatisation(self, word):
        return self.lem.lemmatize(word,"v")

    def pos_tagging(self, tokenized_word_array):
        return nltk.pos_tag(tokenized_word_array)
