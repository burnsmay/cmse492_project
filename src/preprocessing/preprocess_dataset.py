# src/preprocessing/preprocess_dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from clean_text import clean_text

RAW = "data/raw/"
PROCESSED = "data/processed/"

def load_and_prepare():
    """Load raw Fake/True news data, merge, clean, and split."""

    true_df = pd.read_csv(os.path.join(RAW, "True.csv"))
    fake_df = pd.read_csv(os.path.join(RAW, "Fake.csv"))

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)

    # clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # save processed
    X_train.to_csv(os.path.join(PROCESSED, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED, "y_test.csv"), index=False)

    print("✔ Preprocessing complete! Files saved in data/processed/")

if __name__ == "__main__":
    load_and_prepare()
