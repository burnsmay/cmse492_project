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


    train_df = pd.concat([X_train, y_train], axis=1).dropna()
    X_train = train_df["clean_text"].astype(str)
    y_train = train_df[y_train.columns[0]].values.ravel()

    test_df = pd.concat([X_test, y_test], axis=1).dropna()
    X_test = test_df["clean_text"].astype(str)
    y_test = test_df[y_test.columns[0]].values.ravel()

    
    # vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
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
