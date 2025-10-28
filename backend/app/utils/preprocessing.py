import re
import html
import string
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK dependencies (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_html_tags(text: str) -> str:
    """Remove all HTML tags from text."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def decode_html_entities(text: str) -> str:
    """Convert HTML entities like &nbsp; and &amp; to normal characters."""
    return html.unescape(text)

def normalize_unicode(text: str) -> str:
    """Normalize accented characters (e.g., 'Thîs' → 'This')."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

def clean_text(text: str) -> str:
    """Perform basic text cleaning and normalization."""
    text = text.lower()
    text = decode_html_entities(text)
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"console\.log.*?;", "", text)
    text = re.sub(r"var\s+\w+\s*=\s*'.*?';", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\-\' ]", "", text)
    return text.strip()

def remove_stopwords(text: str) -> str:
    """Remove English stopwords from the text."""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(filtered)

def normalize_text(text: str) -> str:
    """Lemmatize words for consistency (e.g., running → run)."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def preprocess_pipeline(text: str) -> str:
    """Full preprocessing pipeline for raw text."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = normalize_text(text)
    return text

def split_into_chunks(text: str, max_words: int = 50):
    """Split text into roughly equal-sized chunks for topic modeling."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks
