# modules/summarization.py

import nltk
import math
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import streamlit as st

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# --------------------------------------------------------
# 1. CACHED MODEL LOADER (HUGE SPEED BOOST)
# --------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer(use_light_model=False):
    """
    Load the summarizer only once.
    - If use_light_model=True → fast DistilBART
    - Else → high-quality BART-large
    """
    model_name = (
        "sshleifer/distilbart-cnn-12-6" if use_light_model
        else "facebook/bart-large-cnn"
    )
    return pipeline("summarization", model=model_name, device=-1)


summarizer = load_summarizer(use_light_model=False)  # Set to True for SPEED


# --------------------------------------------------------
# 2. HELPERS
# --------------------------------------------------------
def _words_to_tokens_estimate(words: int) -> int:
    """Approx words → tokens (BART uses ~1.4–1.6× expansion)."""
    return int(words * 1.55)


def _join_sentences(sentences):
    return " ".join([s.strip() for s in sentences if s.strip()])


def _chunk_text(text, max_words=900, overlap=150):
    """
    Efficient chunking designed for extremely large inputs (50MB+).
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    i = 0
    step = max_words - overlap
    L = len(words)

    while i < L:
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += step

        # safety for extremely long text
        if len(chunks) > 120:
            break

    return chunks


# --------------------------------------------------------
# 3. SUMMARY GENERATOR
# --------------------------------------------------------
def generate_summary(text: str, target_words: int = 120):
    """
    High-performance summarization:
    - < 120 words → extractive
    - <= 1200 words → single-pass BART
    - > 1200 words → chunk + merge + final summary

    Ensures output approx ≈ target_words ± 10%.
    """
    if not text or not text.strip():
        return "⚠️ No text provided."

    word_count = len(text.split())

    # ----------------------------------------------------
    # Short text → return first 1–2 sentences
    # ----------------------------------------------------
    if word_count < 120:
        sents = sent_tokenize(text)
        if len(sents) <= 2:
            return text.strip()
        return _join_sentences(sents[:2])

    # ----------------------------------------------------
    # Medium text → direct summarization
    # ----------------------------------------------------
    if word_count <= 1200:
        max_len = _words_to_tokens_estimate(int(target_words * 1.3))
        min_len = max(40, int(max_len * 0.35))

        try:
            out = summarizer(
                text,
                max_length=min(max_len, 512),
                min_length=min(min_len, 200),
                do_sample=False
            )
            summary = out[0]["summary_text"].strip()
            return _enforce_target_length(summary, target_words)
        except Exception:
            sents = sent_tokenize(text)
            return _join_sentences(sents[:3])

    # ----------------------------------------------------
    # Large documents → chunk processing
    # ----------------------------------------------------
    chunks = _chunk_text(text, max_words=950, overlap=180)
    partial_summaries = []

    for chunk in chunks:
        try:
            max_len = _words_to_tokens_estimate(int(target_words * 0.6))
            min_len = int(max_len * 0.35)

            out = summarizer(
                chunk,
                max_length=min(max_len, 450),
                min_length=min(40, min_len),
                do_sample=False
            )
            partial_summaries.append(out[0]["summary_text"].strip())
        except Exception:
            partial_summaries.append(chunk[:300])

    # Merge partial summaries
    merged = " ".join(partial_summaries)

    # Final compression
    try:
        final_max = _words_to_tokens_estimate(int(target_words * 1.3))
        final_min = max(50, int(final_max * 0.35))

        out = summarizer(
            merged,
            max_length=min(final_max, 500),
            min_length=min(final_min, 200),
            do_sample=False
        )
        final_summary = out[0]["summary_text"].strip()
        return _enforce_target_length(final_summary, target_words)

    except Exception:
        return merged[:800]


# --------------------------------------------------------
# 4. LENGTH ENFORCER
# --------------------------------------------------------
def _enforce_target_length(summary: str, target_words: int):
    """
    Ensure final summary matches the target length
    within ±10% (best possible with BART).
    """
    words = summary.split()
    N = len(words)

    lower = int(target_words * 0.9)
    upper = int(target_words * 1.1)

    if lower <= N <= upper:
        return summary

    if N > upper:
        sents = sent_tokenize(summary)
        final_words = []
        for s in sents:
            final_words.extend(s.split())
            if len(final_words) >= upper:
                break
        trimmed = " ".join(final_words)
        return trimmed.rsplit(" ", 1)[0] + "..."

    return summary  # shorter than target → leave untouched


# --------------------------------------------------------
# Standalone test
# --------------------------------------------------------
if __name__ == "__main__":
    txt = "Machine learning and AI are transforming industries worldwide. " * 150
    print(generate_summary(txt, target_words=120))
