import re
import string
from typing import List


def clean_text(text: str) -> str:
    """
    Clean and preprocess text by:
    - Converting to lowercase
    - Removing URLs
    - Removing email addresses
    - Removing HTML tags
    - Removing special characters
    - Removing extra whitespace
    - Tokenizing and filtering
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    
    tokens = text.lower().split()
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords.
    """
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
        'the', 'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they',
        'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'same', 'such', 'no', 'nor', 'not', 'only', 'own', 'so',
        'than', 'too', 'very', 'can', 'just', 'should', 'now', 'about',
        'above', 'after', 'again', 'against', 'am', 'do', 'does', 'did',
        'have', 'had', 'having', 'there', 'here', 'up', 'down', 'out',
        'if', 'then', 'else', 'because', 'while', 'being', 'been'
    }
    
    return [token for token in tokens if token not in stopwords and len(token) > 2]


def filter_tokens(tokens: List[str]) -> List[str]:
    """
    Filter tokens to keep only meaningful words.
    Remove numbers, very short words, etc.
    """
    filtered = []
    for token in tokens:
        if token.isalpha() and len(token) > 2 and not token.isnumeric():
            filtered.append(token)
    return filtered


def preprocess_for_lda(text: str) -> str:
    """
    Preprocess text specifically for LDA topic modeling:
    - Clean the text
    - Tokenize
    - Remove stopwords
    - Filter tokens
    - Return cleaned document
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = filter_tokens(tokens)
    return " ".join(tokens)


def preprocess_for_sentiment(text: str) -> str:
    """
    Preprocess text for sentiment analysis.
    Lighter cleaning to preserve some linguistic features.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
