# src/preprocessing/clean_text.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK packages only once
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and normalize news article text."""
    
    if not isinstance(text, str):
        return ""
    
    # lowercase
    text = text.lower()
    
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    
    return " ".join(words)
