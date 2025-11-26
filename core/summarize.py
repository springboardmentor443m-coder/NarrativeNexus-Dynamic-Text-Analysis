# ===============================
# core/summarize.py – Week 4
# ===============================

import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extractive_summary(text: str, sentences_max=3):
    """Simple extractive summary using TF-IDF."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sents) <= sentences_max:
        return " ".join(sents)
    vec = TfidfVectorizer().fit(sents)
    X = vec.transform(sents).toarray()
    scores = X.sum(axis=1)
    top = scores.argsort()[::-1][:sentences_max]
    top_sorted = sorted(top)
    return " ".join(sents[i] for i in top_sorted)


# Optional transformer-based abstractive summary
try:
    from transformers import pipeline
    abstractive_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    abstractive_pipe = None


def abstractive_summary(text: str, max_len=130):
    """Generate an abstractive summary using HuggingFace transformers."""
    if not abstractive_pipe:
        return extractive_summary(text)
    try:
        summary = abstractive_pipe(text[:2000], max_length=max_len,
                                   min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return extractive_summary(text)
