# src/models/svm_model.py

import os
import time
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROCESSED = "data/processed/"
MODELS = "models/"

def train_svm():
    # Load data
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))["clean_text"]
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    # Drop missing values
    mask_train = X_train.notna()
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_test = X_test.notna()
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # Convert to string
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    # TF-IDF Vectorizer (smaller feature set)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Linear SVM model
    model = LinearSVC(max_iter=3000)

    start = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start

    # Predictions
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("Linear SVM Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Save model and vectorizer
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS, "svm_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS, "tfidf.pkl"))

    return train_time, acc

if __name__ == "__main__":
    train_svm()

