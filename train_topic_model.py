import os
import json
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# -----------------------------------------
# Paths
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------------------
# Load 20 Newsgroups dataset
# -----------------------------------------
print("📥 Loading 20 Newsgroups dataset...")
dataset = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
documents = dataset.data
print(f"✅ Loaded {len(documents)} documents.")

# -----------------------------------------
# TF-IDF vectorization
# -----------------------------------------
print("🔠 Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    lowercase=True
)
X = vectorizer.fit_transform(documents)
print("✅ TF-IDF shape:", X.shape)

# -----------------------------------------
# Train NMF topic model
# -----------------------------------------
n_topics = 20  # 20 topics for 20 newsgroups
print(f"🧠 Training NMF model with {n_topics} topics...")
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_model.fit(X)
print("✅ NMF model trained.")

# -----------------------------------------
# Extract top keywords per topic
# -----------------------------------------
feature_names = vectorizer.get_feature_names_out()
topic_keywords = {}

for topic_idx, topic in enumerate(nmf_model.components_):
    top_indices = topic.argsort()[-10:][::-1]  # top 10
    top_keywords = [feature_names[i] for i in top_indices]
    topic_keywords[str(topic_idx)] = top_keywords

# -----------------------------------------
# Save model, vectorizer, and topic keywords
# -----------------------------------------
joblib.dump(nmf_model, os.path.join(MODELS_DIR, "nmf_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

with open(os.path.join(MODELS_DIR, "topic_names.json"), "w", encoding="utf-8") as f:
    json.dump(topic_keywords, f, indent=4)

print("💾 Saved:")
print("  - nmf_model.pkl")
print("  - tfidf_vectorizer.pkl")
print("  - topic_names.json")
print("📁 Location:", MODELS_DIR)
print("🎉 Training complete!")
