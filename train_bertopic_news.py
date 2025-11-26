# train_bertopic_news.py

import os
import pickle
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np
import pandas as pd


# ----------------------------------------------
#  CONFIG
# ----------------------------------------------

# Optional: include your uploaded file in training
LOCAL_FILE = "/mnt/data/nlp_preprocessing_2.txt"

# Save model as a single .bertopic file — FIXED
MODEL_BASE = "backend/models/topic_model"      # no trailing slash
MODEL_FILE = MODEL_BASE + ".bertopic"          # final file = backend/models/topic_model.bertopic

# Ensure parent directory exists
os.makedirs("backend/models", exist_ok=True)


# ----------------------------------------------
#  CLEANING FUNCTION
# ----------------------------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    return s.strip()


# ----------------------------------------------
#  1) LOAD AG NEWS
# ----------------------------------------------

print("📥 Loading AG News dataset...")
ds = load_dataset("ag_news")
documents = [clean_text(item["text"]) for item in ds["train"]]

print("Loaded AG News train size:", len(documents))


# ----------------------------------------------
#  2) OPTIONAL — Add your own uploaded file
# ----------------------------------------------

if os.path.exists(LOCAL_FILE):
    print(f"📄 Found local file: {LOCAL_FILE} — appending to training data.")
    try:
        with open(LOCAL_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # break into paragraphs
        extra_docs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]
        if not extra_docs:
            if len(content.strip()) > 40:
                extra_docs = [content.strip()]

        documents.extend([clean_text(d) for d in extra_docs])
        print("After adding local documents:", len(documents))

    except Exception as e:
        print("⚠️ Could not load local file. Error:", e)


# ----------------------------------------------
#  3) FILTER SHORT DOCS (needed for BERTopic)
# ----------------------------------------------

documents = [d for d in documents if len(d) > 40]
print("Documents after filtering:", len(documents))


# ----------------------------------------------
#  4) EMBEDDING MODEL (MiniLM)
# ----------------------------------------------

print("🔍 Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------------------
#  5) TRAIN BERTopic
# ----------------------------------------------

print("🚀 Training BERTopic (this will take time)...")
topic_model = BERTopic(
    embedding_model=embedder,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(documents)


# ----------------------------------------------
#  6) SAVE BERTopic MODEL (FIXED)
# ----------------------------------------------

print("💾 Saving BERTopic model to:", MODEL_FILE)
topic_model.save(MODEL_FILE)      # <-- now saving as a single file (.bertopic)


# ----------------------------------------------
#  7) SAVE METADATA SEPARATELY
# ----------------------------------------------

# For compatibility with your backend
os.makedirs(MODEL_BASE.replace("topic_model", ""), exist_ok=True)
os.makedirs("backend/models", exist_ok=True)

# Save processed documents
with open("backend/models/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

np.save("backend/models/topic_labels.npy", np.array(topics))

# Extract keywords for each topic
topic_keywords = {}
for tid in topic_model.get_topic_info()["Topic"].tolist():
    if tid == -1:
        continue
    words = topic_model.get_topic(int(tid))      # list of (word, score)
    topic_keywords[int(tid)] = [w for w, _ in words]

with open("backend/models/topic_keywords.pkl", "wb") as f:
    pickle.dump(topic_keywords, f)

# Generate human-readable names
topic_names = {
    tid: " / ".join(topic_keywords[tid][:3]).title() if topic_keywords.get(tid) else "Unknown Topic"
    for tid in topic_keywords.keys()
}

with open("backend/models/topic_names.pkl", "wb") as f:
    pickle.dump(topic_names, f)


# ----------------------------------------------
#  DONE
# ----------------------------------------------

print("✅ BERTopic model saved.")
print("📌 Model file:", MODEL_FILE)
print("📌 Topics discovered:", len(topic_keywords))
print("🎉 Training complete!")

