# backend/text_analysis.py
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # simple, fast summarizer

# download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()


def strip_html_tags(text: str) -> str:
    """Remove HTML tags safely using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def clean_text(text: str) -> str:
    """Basic text cleaning: remove punctuation, lowercase, stopwords."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [w for w in word_tokenize(text) if w not in stop_words]
    return ' '.join(tokens)


def summarize_text(text: str, num_sentences: int = 3) -> str:
    """Generate a short summary using Sumy's LSA summarizer."""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)
    return ' '.join(str(s) for s in summary_sentences)


def analyze_sentiment(text: str) -> dict:
    """Use VADER to return sentiment label and score using compound value."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": round(compound, 3)}



def clean_summarize_analyze(text: str) -> dict:
    """
    Correct pipeline:
    1. Strip HTML safely (so tags aren't summarized).
    2. Summarize the cleaned version (with sentences).
    3. Clean separately for preview.
    4. Analyze sentiment on cleaned version.
    """
    no_html = strip_html_tags(text)            # <-- remove tags first
    summary = summarize_text(no_html)          # summarize actual text
    cleaned = clean_text(no_html)              # clean for preview
    sentiment = analyze_sentiment(cleaned)

    preview = cleaned[:500]
    return {
        "preview": preview,
        "summary": summary,
        "sentiment": sentiment,
    }

