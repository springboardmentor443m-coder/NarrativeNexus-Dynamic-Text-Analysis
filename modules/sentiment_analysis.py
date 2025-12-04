# modules/sentiment_analysis.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# 1. Load tokenizer + model ONCE only (cached → speed boost)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device


tokenizer, model, DEVICE = load_sentiment_model()
LABELS = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------
# 2. Internal: Predict sentiment on a <=512-token fragment
# ---------------------------------------------------------
def _predict_chunk(text_chunk):
    """
    Predict sentiment of a chunk that fits within 512 tokens.
    Returns (label, confidence).
    """
    inputs = tokenizer(
        text_chunk,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()

    return LABELS[label_id], float(probs[label_id])


# ---------------------------------------------------------
# 3. Smart Chunking (token-safe) for large documents (50–200MB)
# ---------------------------------------------------------
def _chunk_text(text, max_chars=1800):
    """
    Creates approximately 512-token safe chunks.
    1800 chars ≈ 480–512 tokens depending on text density.
    """
    chunks = []
    L = len(text)

    for i in range(0, L, max_chars):
        chunks.append(text[i:i + max_chars])

    return chunks


# ---------------------------------------------------------
# 4. Main Public API — SAFE for ANY text size
# ---------------------------------------------------------
def get_sentiment(text: str):
    """
    Sentiment analysis for ANY size input, from 1 sentence to 200MB.
    Uses intelligent chunking + weighted confidence averaging.
    """

    if not text.strip():
        return "Neutral", 0.0  # safe default

    chunks = _chunk_text(text, max_chars=1800)
    label_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    conf_sums = {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}

    for ch in chunks:
        label, conf = _predict_chunk(ch)
        label_counts[label] += 1
        conf_sums[label] += conf

    # Final label by majority voting
    final_label = max(label_counts, key=label_counts.get)

    # Confidence = avg confidence of that label
    final_conf = conf_sums[final_label] / max(1, label_counts[final_label])

    return final_label, float(final_conf)


# ---------------------------------------------------------
# 5. Topic-level sentiment (short strings)
# ---------------------------------------------------------
def get_topic_sentiments(topics):
    """
    Efficient sentiment scores for topic words/phrases.
    Returns: {topic: sentiment_score}
      Positive → +confidence
      Negative → -confidence
      Neutral  → 0
    """
    sentiments = {}
    for topic in topics:
        label, conf = _predict_chunk(topic)

        if label == "Positive":
            score = conf
        elif label == "Negative":
            score = -conf
        else:
            score = 0

        sentiments[topic] = score

    return sentiments


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------
if __name__ == "__main__":
    text = "I love AI but sometimes it is extremely annoying and difficult."
    print(get_sentiment(text))
