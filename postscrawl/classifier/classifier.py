import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import pandas as pd


class Classifier(object):
    def __init__(self):
        self.lem = WordNetLemmatizer()
        # ngram_range=(1,1) Bag of words (default) 
        # ngram_range=(2,2) Bigrams only 
        # ngram_range=(1,2) Both!
        self.basicvectorizer = CountVectorizer(ngram_range=(1,2))
        self.basicmodel = LogisticRegression(max_iter=1000)

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

    def read_csv_data(self, path):
        return pd.read_csv(path)

    def visualise_csv_data(self, tokenized_word_array):
        print(pd.DataFrame([[x,tokenized_word_array.count(x)] for x in set(tokenized_word_array)], 
                columns = ['Word', 'Count']))
    """
    [summary] generates a document term matrix to count the number of occurance of a single word
    """
    def generate_term_matrix(self, data):
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
        text_count = cv.fit_transform(data)
        return text_count

    def get_train_headlines(self, train_dataset):
        trainheadlines = []
        # dataset.iloc[...] does not include headers
        for row in range(0,len(train_dataset.index)):
            trainheadlines.append(' '.join(str(x) for x in train_dataset.iloc[row,2:27]))
        return trainheadlines

    def train_model(self, train_dataset):
        basictrain = self.basicvectorizer.fit_transform(self.get_train_headlines(train_dataset))
        self.get_train_headlines(train_dataset)
        # model training
        return self.basicmodel.fit(basictrain, train_dataset["Label"])

    def prediction_result(self, test_dataset, basic_model):
        testheadlines = []
        for row in range(0,len(test_dataset.index)):
            testheadlines.append(' '.join(str(x) for x in test_dataset.iloc[row,2:27]))
        basictest = self.basicvectorizer.transform(testheadlines)
        predictions = basic_model.predict(basictest)
        print(pd.crosstab(test_dataset["Label"], predictions, rownames=["Actual"], colnames=["Predicted"]))
        
        basicwords = self.basicvectorizer.get_feature_names()
        basiccoeffs = basic_model.coef_.tolist()[0]
        coeffdf = pd.DataFrame({'Word' : basicwords, 
                                'Coefficient' : basiccoeffs})
        coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print(coeffdf.head(10))
