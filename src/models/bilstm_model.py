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

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))["clean_text"]
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))["clean_text"]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv"))

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    # Model
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

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

    # save
    os.makedirs(MODELS, exist_ok=True)
    model.save(os.path.join(MODELS, "bilstm_model.h5"))

    return train_time, test_acc, history.history

if __name__ == "__main__":
    train_bilstm()
