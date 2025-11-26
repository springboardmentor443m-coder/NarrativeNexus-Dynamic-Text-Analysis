# backend/topic_model.py

import os
import pickle
from bertopic import BERTopic


# -------------------------------------------------------
# 1. Paths
# -------------------------------------------------------

# Base path WITHOUT extension
MODEL_BASE = os.path.join("backend", "models", "topic_model")

# The actual saved model file (BERTopic format)
MODEL_FILE = MODEL_BASE + ".bertopic"


# -------------------------------------------------------
# 2. Load BERTopic model
# -------------------------------------------------------

print(f"🔄 Loading BERTopic model from: {MODEL_FILE}")

topic_model = BERTopic.load(MODEL_FILE)


# -------------------------------------------------------
# 3. Load metadata (keywords, names)
# -------------------------------------------------------

# Load keywords
try:
    with open(os.path.join("backend", "models", "topic_keywords.pkl"), "rb") as f:
        TOPIC_KEYWORDS = pickle.load(f)
except Exception:
    TOPIC_KEYWORDS = {}

# Load names
try:
    with open(os.path.join("backend", "models", "topic_names.pkl"), "rb") as f:
        TOPIC_NAMES = pickle.load(f)
except Exception:
    TOPIC_NAMES = {}



# -------------------------------------------------------
# 4. Infer topic
# -------------------------------------------------------

def infer_topic(text: str):
    """
    Infer topic from text using BERTopic.
    Returns: { id, name, keywords, probability }
    """
    if not text or not text.strip():
        return {
            "id": -1,
            "name": "Unknown Topic",
            "keywords": [],
            "probability": 0.0,
        }

    # BERTopic transform
    topics, probs = topic_model.transform([text])

    topic_id = int(topics[0]) if topics[0] is not None else -1
    probability = float(probs[0]) if probs is not None else 0.0

    # Get human-friendly name
    topic_name = TOPIC_NAMES.get(topic_id, None)

    if topic_name is None:
        # fallback: derive from BERTopic directly
        rep = topic_model.get_topic(topic_id) or []
        words = [w for w, _ in rep]
        topic_name = " / ".join(words[:3]).title() if words else "Unknown Topic"

    # Keywords
    topic_keywords = TOPIC_KEYWORDS.get(topic_id, None)

    if topic_keywords is None:
        rep = topic_model.get_topic(topic_id) or []
        topic_keywords = [w for w, _ in rep]

    return {
        "id": topic_id,
        "name": topic_name,
        "keywords": topic_keywords,
        "probability": probability,
    }
