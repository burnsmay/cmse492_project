# src/models/bilstm_model.py
import os
import time
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

PROCESSED = "data/processed/"
MODELS = "models/"

MAX_WORDS = 20000
MAX_LEN = 300

def train_bilstm():

    # Load data
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    # Make sure column exists
    if "clean_text" not in X_train.columns:
        # fallback: use first column
        X_train = X_train.iloc[:, 0]
        X_test = X_test.iloc[:, 0]
    else:
        X_train = X_train["clean_text"]
        X_test = X_test["clean_text"]

    # Drop missing values
    mask_train = X_train.notna()
    mask_test = X_test.notna()
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # Convert to string
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    # Build model
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    start = time.time()
    history = model.fit(
        X_train_pad,
        y_train,
        batch_size=64,
        epochs=4,
        validation_split=0.2
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(X_test_pad, y_test)

    # Save model and tokenizer

    import joblib
    os.makedirs(MODELS, exist_ok=True)

    # Save BiLSTM model
    model.save(os.path.join(MODELS, "bilstm_model.h5"))      # legacy HDF5
    # or use newer format:
    # model.save(os.path.join(MODELS, "bilstm_model.keras"))

    # Save the tokenizer
    joblib.dump(tokenizer, os.path.join(MODELS, "tokenizer.pkl"))

# ===========================
# Then return results
# ===========================
    return train_time, test_acc, history.history


if __name__ == "__main__":
    train_bilstm()
