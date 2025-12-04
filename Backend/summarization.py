"""
Summarization module with both extractive and abstractive methods.
"""

import re
import nltk
from typing import Tuple

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

summarizer_abstractive = None


def load_abstractive_summarizer():
    """
    Load the abstractive summarization model (BART).
    Called once at startup.
    """
    global summarizer_abstractive
    try:
        from transformers import pipeline
        summarizer_abstractive = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        print("[OK] Abstractive summarizer (BART) loaded successfully.")
    except Exception as e:
        print(f"[WARN] Failed to load abstractive summarizer: {e}")
        summarizer_abstractive = None


def extract_summary(text: str, max_sentences: int = 3) -> str:
    """
    Extractive summarization: Select key sentences from the original text.
    Uses sentence scoring based on word frequency.
    
    Args:
        text: Input text to summarize
        max_sentences: Maximum number of sentences to include
        
    Returns:
        Extractive summary as string
    """
    text = text.strip()
    if not text:
        return ""
    
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    
    words = nltk.word_tokenize(text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3 and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words_in_sentence = nltk.word_tokenize(sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words_in_sentence if word.isalpha())
        sentence_scores[i] = score
    
    top_sentence_indices = sorted(
        sentence_scores.keys(),
        key=lambda x: sentence_scores[x],
        reverse=True
    )[:max_sentences]
    
    top_sentence_indices.sort()
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    
    return " ".join(summary_sentences)


def abstractive_summary(text: str, min_length: int = 50, max_length: int = 150) -> str:
    """
    Abstractive summarization: Currently disabled to use extractive only.
    Falls back to extractive summarization.
    
    Args:
        text: Input text to summarize
        min_length: Minimum summary length (tokens)
        max_length: Maximum summary length (tokens)
        
    Returns:
        Extractive summary as string (fallback)
    """
    return extract_summary(text, max_sentences=3)


def get_both_summaries(text: str) -> Tuple[str, str]:
    """
    Generate both extractive and abstractive summaries.
    
    Args:
        text: Input text to summarize
        
    Returns:
        Tuple of (extractive_summary, abstractive_summary)
    """
    extractive = extract_summary(text, max_sentences=3)
    abstractive = abstractive_summary(text, min_length=50, max_length=150)
    
    return extractive, abstractive
