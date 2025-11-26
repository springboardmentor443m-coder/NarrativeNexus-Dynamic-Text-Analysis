import os
import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from core.preprocess import clean_text

# ---------------------------
# CONFIG
# ---------------------------
N_TOPICS = 20          # full dataset
N_WORDS = 15           # top keywords per topic
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# LOAD 20 NEWSGROUPS DATASET
# ---------------------------
print("📥 Downloading 20 Newsgroups dataset...")
dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
docs = dataset.data
print(f"✅ Loaded {len(docs)} documents.")

# ---------------------------
# CLEAN TEXT
# ---------------------------
print("🧹 Cleaning text...")
cleaned_docs = [clean_text(d) for d in docs]

# ---------------------------
# TF-IDF VECTORIZATION
# ---------------------------
print("🔢 Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_df=0.95, 
    min_df=2, 
    stop_words="english"
)

X = vectorizer.fit_transform(cleaned_docs)
print("TF-IDF Matrix shape:", X.shape)

# ---------------------------
# TRAIN NMF
# ---------------------------
print("⚙️ Training NMF model...")
nmf_model = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    max_iter=500,
    random_state=42
)

W = nmf_model.fit_transform(X)
H = nmf_model.components_

print("✅ NMF training complete!")


# ---------------------------
# EXTRACT TOPIC KEYWORDS
# ---------------------------
print("📝 Extracting topic keywords...")

feature_names = vectorizer.get_feature_names_out()
topic_keywords = []

for i, topic in enumerate(H):
    top_indices = topic.argsort()[:-N_WORDS - 1:-1]
    words = [feature_names[j] for j in top_indices]
    topic_keywords.append(words)
    print(f"Topic {i}: {', '.join(words)}")


# ---------------------------
# SAVE MODEL ARTIFACTS
# ---------------------------
print("💾 Saving model files...")

joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")
joblib.dump(nmf_model, f"{MODEL_DIR}/nmf_model.pkl")

# Save topic keyword JSON for later UI display
import json
with open(f"{MODEL_DIR}/topic_keywords.json", "w") as f:
    json.dump(topic_keywords, f, indent=4)

print("🎉 All files saved in /models/")
print("Done!")
