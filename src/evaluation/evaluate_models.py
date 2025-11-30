# src/evaluation/evaluate_models.py

import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROCESSED = "data/processed/"
MODELS = "models/"

def evaluate_logreg():
    model = joblib.load(os.path.join(MODELS, "logreg_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS, "tfidf.pkl"))

    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    print("\n=== Logistic Regression Evaluation ===")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, preds))


def evaluate_svm():
    model = joblib.load(os.path.join(MODELS, "svm_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS, "tfidf.pkl"))

    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    print("\n=== SVM Evaluation ===")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, preds))


if __name__ == "__main__":
    evaluate_logreg()
    evaluate_svm()
