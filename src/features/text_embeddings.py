"""Text embedding wrappers (BERT / TF-IDF / SBERT)."""
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

TFIDF_PATH = "experiments/models/tfidf_vectorizer.joblib"

def fit_tfidf(texts, max_features=5000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    os.makedirs(os.path.dirname(TFIDF_PATH), exist_ok=True)
    joblib.dump(vectorizer, TFIDF_PATH)
    return X, vectorizer

def load_tfidf():
    if os.path.exists(TFIDF_PATH):
        return joblib.load(TFIDF_PATH)
    raise FileNotFoundError("TFIDF vectorizer not found")
