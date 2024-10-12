import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import unicodedata
import re
import inflect
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class TextPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords_list=None):
        if stopwords_list is None:
            self.stopwords = stopwords.words('spanish')
        else:
            self.stopwords = stopwords_list

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in self.stopwords:
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        stemmer = SnowballStemmer('spanish')
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def stem_and_lemmatize(self, words):
        words = self.stem_words(words)
        words = self.lemmatize_verbs(words)
        return words

    def preproccessing(self, words):
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_train = pd.Series(X)
        X_train = X_train.apply(word_tokenize)
        X_train = X_train.apply(lambda x: self.preproccessing(x))
        X_train = X_train.apply(lambda x: self.stem_and_lemmatize(x))
        X_train = X_train.apply(lambda x: ' '.join(map(str, x)))
        return X_train


def entrenar_y_guardar_modelo():
    data = pd.read_excel("ODScat_345.xlsx")

    X = data['Textos_espanol']
    y = data['sdg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preproccess', TextPreprocessing()),
        ('vectorizer', CountVectorizer(lowercase=False)),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'modelo_svm.pkl')

    print("Modelo entrenado y guardado correctamente en 'modelo_svm.pkl'.")

if __name__ == "__main__":
    entrenar_y_guardar_modelo()
