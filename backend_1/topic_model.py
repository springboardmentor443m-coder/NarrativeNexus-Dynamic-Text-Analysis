# backend_1/topic_model.py

import os
import pickle
from bertopic import BERTopic

# Model paths
MODEL_BASE = os.path.join("backend_1", "models", "topic_model")
MODEL_FILE = MODEL_BASE + ".bertopic"

print(f"🔄 Loading BERTopic model from: {MODEL_FILE}")

topic_model = BERTopic.load(MODEL_FILE)


# -------------------------------------------------------
# LOAD METADATA
# -------------------------------------------------------

def _safe_load(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return default

TOPIC_KEYWORDS = _safe_load(
    os.path.join("backend_1", "models", "topic_keywords.pkl"),
    {}
)

TOPIC_NAMES = _safe_load(
    os.path.join("backend_1", "models", "topic_names.pkl"),
    {}
)


# -------------------------------------------------------
# TOPIC COUNT
# -------------------------------------------------------

def get_topic_count():
    info = topic_model.get_topic_info()
    return int(info[info["Topic"] != -1].shape[0])


# -------------------------------------------------------
# TOPIC INFERENCE
# -------------------------------------------------------

def infer_topic(text: str):
    if not text or not text.strip():
        return {
            "id": -1,
            "name": "Unknown Topic",
            "keywords": [],
            "probability": 0.0,
        }

    topics, probs = topic_model.transform([text])

    topic_id = int(topics[0]) if topics else -1
    probability = float(probs[0]) if probs is not None else 0.0

    name = TOPIC_NAMES.get(topic_id, "Unknown Topic")
    keywords = TOPIC_KEYWORDS.get(topic_id, [])

    return {
        "id": topic_id,
        "name": name,
        "keywords": keywords,
        "probability": probability,
    }
