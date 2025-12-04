# train_bertopic_news.py

import os
import pickle
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

LOCAL_FILE = "/mnt/data/nlp_preprocessing_2.txt"

MODEL_DIR = "backend_1/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_BASE = os.path.join(MODEL_DIR, "topic_model")
MODEL_FILE = MODEL_BASE + ".bertopic"

DESIRED_TOPICS = 40     # <-- Final number of topics after reduction
TRAIN_DOCS = 10000      # <-- Train on 10k documents


# -----------------------------------------------------------
# TEXT CLEANING
# -----------------------------------------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    return s.strip()


# -----------------------------------------------------------
# 1) LOAD & PREPARE AG NEWS (10K samples)
# -----------------------------------------------------------

print("📥 Loading AG News dataset...")
ds = load_dataset("ag_news")

documents = [clean_text(item["text"]) for item in ds["train"]]
documents = documents[:TRAIN_DOCS]      # <-- LIMIT TO 10K SAMPLES

print(f"Loaded {len(documents)} training documents.")


# -----------------------------------------------------------
# 2) OPTIONAL — Add your own uploaded text
# -----------------------------------------------------------

if os.path.exists(LOCAL_FILE):
    print(f"📄 Adding local file: {LOCAL_FILE}")
    with open(LOCAL_FILE, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    extra_docs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]
    if not extra_docs:
        if len(content.strip()) > 40:
            extra_docs = [content.strip()]

    documents.extend([clean_text(d) for d in extra_docs])
    print("Final document count:", len(documents))


# -----------------------------------------------------------
# 3) FILTER SHORT TEXTS
# -----------------------------------------------------------

documents = [d for d in documents if len(d) > 40]
print("Documents after filtering:", len(documents))


# -----------------------------------------------------------
# 4) LOAD EMBEDDING MODEL
# -----------------------------------------------------------

print("🔍 Loading MiniLM embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------------------
# 5) TRAIN BERTopic
# -----------------------------------------------------------

print("🚀 Training BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedder,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(documents)


# -----------------------------------------------------------
# 6) REDUCE TOPICS TO DESIRED NUMBER
# -----------------------------------------------------------

print(f"🔧 Reducing topics to {DESIRED_TOPICS}...")

topic_model = topic_model.reduce_topics(
    docs=documents,
    nr_topics=DESIRED_TOPICS
)

# Updated topic assignments after reduction
topics = topic_model.topics_

print(f"✅ Final number of topics: {DESIRED_TOPICS}")


# -----------------------------------------------------------
# 7) SAVE FINAL MODEL (.bertopic)
# -----------------------------------------------------------

print("💾 Saving BERTopic model to:", MODEL_FILE)
topic_model.save(MODEL_FILE)


# -----------------------------------------------------------
# 8) SAVE METADATA FOR BACKEND
# -----------------------------------------------------------

# Save documents
with open(os.path.join(MODEL_DIR, "documents.pkl"), "wb") as f:
    pickle.dump(documents, f)

# Save topic IDs
np.save(os.path.join(MODEL_DIR, "topic_labels.npy"), np.array(topics))

# Extract topic keywords
topic_keywords = {}
for tid in topic_model.get_topic_info()["Topic"].tolist():
    if tid == -1:
        continue
    words = topic_model.get_topic(int(tid))
    topic_keywords[int(tid)] = [w for w, _ in words]

with open(os.path.join(MODEL_DIR, "topic_keywords.pkl"), "wb") as f:
    pickle.dump(topic_keywords, f)

# Generate topic names
topic_names = {
    tid: " / ".join(topic_keywords[tid][:3]).title() if topic_keywords.get(tid) else "Unknown Topic"
    for tid in topic_keywords.keys()
}

with open(os.path.join(MODEL_DIR, "topic_names.pkl"), "wb") as f:
    pickle.dump(topic_names, f)


# -----------------------------------------------------------
# DONE
# -----------------------------------------------------------

print("🎉 Training Complete!")
print("📌 Model saved at:", MODEL_FILE)
print("📌 Topics discovered:", len(topic_keywords))

