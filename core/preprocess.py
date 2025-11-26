import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

_nltk_ready = False
def _prep_nltk():
    """Ensure required NLTK data is available."""
    global _nltk_ready
    if _nltk_ready:
        return
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    _nltk_ready = True


# ✨ CUSTOM STOPWORDS (extend built-in stopwords)
CUSTOM_STOPWORDS = {
    'also', 'while', 'however', 'therefore', 'thus', 'one', 'two',
    'etc', 'may', 'use', 'used', 'using', 'would', 'could', 'make'
}


def clean_text(text: str) -> str:
    """Remove URLs, emojis, extra spaces, and non-text characters."""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove emojis & symbols
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Keep letters, numbers, and basic punctuation
    text = re.sub(r'[^A-Za-z0-9.,;:!?\'\"()\-\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split text into tokens (words)."""
    _prep_nltk()
    return nltk.word_tokenize(text)


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Lowercase, remove stopwords, and lemmatize."""
    _prep_nltk()
    stop = set(stopwords.words('english')) | CUSTOM_STOPWORDS
    lem = WordNetLemmatizer()

    cleaned_tokens = []
    for t in tokens:
        t = t.lower().strip(string.punctuation)
        if not t or t in stop or len(t) < 2:
            continue
        cleaned_tokens.append(lem.lemmatize(t))
    return cleaned_tokens


def preprocess(text: str):
    """Full preprocessing pipeline."""
    txt = clean_text(text)
    toks = tokenize(txt)
    toks = normalize_tokens(toks)
    return txt, toks


def compute_stats(raw_text: str, cleaned_text: str, tokens: List[str]) -> dict:
    """Return basic statistics about text."""
    return {
        "original_chars": len(raw_text),
        "cleaned_chars": len(cleaned_text),
        "original_words": len(raw_text.split()),
        "token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
    }
