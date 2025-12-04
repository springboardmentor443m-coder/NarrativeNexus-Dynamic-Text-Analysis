# modules/preprocessing.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -----------------------------
# Download NLTK resources once
# -----------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

# Cached components (avoid re-initializing)
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text input.
    Steps:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation + numbers
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    """

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = text.lower()

    # Remove URLs and social media tags
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Lemmatization
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_documents(docs):
    """
    Applies clean_text() to a list of text documents.
    Returns cleaned documents list.
    """

    if not isinstance(docs, list):
        return []

    return [
        clean_text(doc)
        for doc in docs
        if isinstance(doc, str) and len(doc.strip()) > 0
    ]


# Debug test
if __name__ == "__main__":
    print(clean_text("Hello!!! This is a TEST with #hashtags and 123 numbers."))
