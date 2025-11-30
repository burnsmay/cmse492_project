# src/models/logistic_regression.py

import os
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib

PROCESSED = "data/processed/"
MODELS = "models/"

def train_logistic_regression():
    # load data
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))["clean_text"]
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    # vectorizer
    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # model
    model = LogisticRegression(max_iter=3000, n_jobs=-1)

    start = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start

    # evaluate
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("Logistic Regression Accuracy:", acc)
    print(classification_report(y_test, preds))

    # save
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS, "logreg_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS, "tfidf.pkl"))

    return train_time, acc

if __name__ == "__main__":
    train_logistic_regression()
