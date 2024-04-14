import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
import pickle
from catboost import CatBoostClassifier

# Загрузка TF-IDF векторизатора
with open('data\\tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Загрузка модели CatBoost
model_catboost = CatBoostClassifier()
model_catboost.load_model('data\\catboost_sentiment_model.cbm')


def predict2(tokens):
    text = ' '.join(tokens)
    transformed = loaded_vectorizer.transform([text])
    predictions = model_catboost.predict(transformed)

    return predictions[0]


def predict(tokens):
    return predict2(tokens)