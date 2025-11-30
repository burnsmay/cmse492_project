# src/models/svm_model.py

import os
import time
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

PROCESSED = "data/processed/"
MODELS = "models/"

def train_svm():
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))["clean_text"]
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # grid search
    params = {
        "C": [0.1, 1, 3],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    svc = SVC()

    grid = GridSearchCV(
        svc,
        params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    start = time.time()
    grid.fit(X_train_vec, y_train)
    train_time = time.time() - start

    best = grid.best_estimator_
    preds = best.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    print("Best SVM Accuracy:", acc)
    print("Best Params:", grid.best_params_)
    print(classification_report(y_test, preds))

    # save
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(best, os.path.join(MODELS, "svm_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS, "tfidf.pkl"))

    return train_time, acc

if __name__ == "__main__":
    train_svm()
