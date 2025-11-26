# ===============================
# core/sentiment.py – Week 4
# ===============================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional transformer-based model
try:
    from transformers import pipeline
    transformer_senti = pipeline("sentiment-analysis")
except Exception:
    transformer_senti = None


def sentiment_label(text: str):
    """
    Perform sentiment analysis.
    Uses transformer pipeline if available, otherwise falls back to VADER.
    Returns {'label': str, 'compound': float, ...}
    """
    # --- Transformer path ---
    if transformer_senti:
        try:
            result = transformer_senti(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            return {"label": label, "compound": score}
        except Exception:
            pass

    # --- Fallback VADER path ---
    _an = SentimentIntensityAnalyzer()
    s = _an.polarity_scores(text)
    comp = s["compound"]
    if comp >= 0.05:
        lab = "positive"
    elif comp <= -0.05:
        lab = "negative"
    else:
        lab = "neutral"
    return {"compound": comp, "label": lab, **s}
