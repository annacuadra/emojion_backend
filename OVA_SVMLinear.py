import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import joblib

dataset = pd.read_csv('datasets\story_emotion.csv', encoding='ISO-8859-1')
stemmer = pd.read_csv('datasets\stem_tl.csv')
word_to_stem = dict(zip(stemmer['word'], stemmer['stem']))

replace_patterns = {
    re.compile(r"\bngayo\'y\b"): 'ngayon ay',
    re.compile(r"\bhangga\'t\b"): 'hanggang',
    re.compile(r"\b\'?y\b"): ' ay',
    re.compile(r"\b\'?t\b"): ' at',
    re.compile(r"\b\'?yan\b"): 'iyan',
    re.compile(r"\b\'?yo\b"): 'iyo',
    re.compile(r"\b\'?yon\b"): 'iyon',
    re.compile(r"\b\'?yun\b"): 'iyun',
    re.compile(r"\b\'?pagkat\b"): 'sapagkat',
    re.compile(r"\b\'?di\b"): 'hindi',
    re.compile(r"\b\'?kaw\b"): "ikaw",
    re.compile(r"\b\'?to\b"): 'ito',
    re.compile(r"\b\'?wag\b"): 'huwag',
    re.compile(r"\bgano\'n\b"): 'ganoon'
}

class_names = {
    1: 'fear',
    2: 'anger',
    3: 'joy',
    4: 'sadness',
    5: 'disgust',
    6: 'surprise'
}

def data_preprocess(text, replace_patterns, word_to_stem):
    text = text.lower()

    for pattern, replacement in replace_patterns.items():
        text = pattern.sub(replacement, text)

    text = re.sub("[^a-zA-Z0-9\s?!.]", '', text)
    tokens = word_tokenize(text)
    text = ' '.join([word_to_stem.get(word, word) for word in tokens])

    return text

dataset['text_preprocessed'] = dataset['text'].apply(data_preprocess, replace_patterns=replace_patterns, word_to_stem=word_to_stem)

vectorizer = CountVectorizer()
sparse_matrix = vectorizer.fit_transform(dataset['text_preprocessed'])
sparse_matrix.todense()
CountVectorized = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names_out())
CountVectorized_result = pd.concat([dataset['text_preprocessed'], CountVectorized], axis=1)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(sparse_matrix)
X_tfidf.todense()
tfidf = pd.DataFrame(X_tfidf.toarray(), columns = vectorizer.get_feature_names_out())
tfidf_result = pd.concat([dataset['text_preprocessed'], tfidf], axis=1)

X = dataset['text_preprocessed']
Y = dataset['emotion']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_vectorized)
X_test_tfidf = tfidf_transformer.transform(X_test_vectorized)

svm = SVC(kernel='linear', C=10, random_state=42)
classifier = OneVsRestClassifier(svm)

classifier.fit(X_train_tfidf, Y_train)
Y_pred = classifier.predict(X_test_tfidf)

joblib.dump((classifier, vectorizer, tfidf_transformer), 'OVA_Linear_model.pkl')