# modules/preprocessing.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data 
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """
    Cleans and normalizes raw text input.
    Steps:
    1. Lowercasing
    2. Removing special characters, punctuation, numbers
    3. Removing stopwords
    4. Lemmatization
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

    cleaned_text = " ".join(lemmatized)

    return cleaned_text


def preprocess_documents(docs):
    """
    Applies clean_text() to a list of text documents.
    Returns a list of cleaned text strings.
    """
    return [clean_text(doc) for doc in docs if isinstance(doc, str) and len(doc.strip()) > 0]


# Quick test section (optional)
if __name__ == "__main__":
    sample_texts = [
        "This is an Example! It includes numbers 123 and links: https://example.com",
        "Data preprocessing is essential for NLP tasks!!! #AI #ML"
    ]

    cleaned = preprocess_documents(sample_texts)
    for i, text in enumerate(cleaned, 1):
        print(f"🧹 Cleaned Text {i}:\n{text}\n")
