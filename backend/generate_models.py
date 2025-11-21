# backend/generate_models.py
import argparse
import os
import json
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# ensure nltk resources
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except Exception:
    nltk.download("wordnet")

STOP = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

def simple_preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    toks = [LEM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(toks)

def load_texts_from_csv(path, text_col):
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Columns: {list(df.columns)}")
    texts = df[text_col].fillna("").astype(str).tolist()
    return texts

def build_and_save_models(texts, out_dir, n_topics=10, max_features=2000):
    os.makedirs(out_dir, exist_ok=True)
    cleaned = [simple_preprocess(t) for t in texts]

    # Vectorizer (TF-IDF)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=max_features)
    X = vectorizer.fit_transform(cleaned)
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    print("Saved vectorizer.joblib")

    # NMF topic model
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=200)
    W = nmf.fit_transform(X)
    joblib.dump(nmf, os.path.join(out_dir, "topic_model.joblib"))
    print("Saved topic_model.joblib (NMF)")

    # Extract keywords per topic and save topics.json
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for t_idx, topic_vec in enumerate(nmf.components_):
        topn = topic_vec.argsort()[-15:][::-1]
        keywords = [feature_names[i] for i in topn]
        topics[str(t_idx)] = {"keywords": keywords}
    with open(os.path.join(out_dir, "topics.json"), "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
    print("Saved topics.json")

    # Also create a small LDA for compatibility (CountVectorizer -> LDA)
    cv = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features)
    Xc = cv.fit_transform(cleaned)
    lda = LatentDirichletAllocation(n_components=min(n_topics, 10), random_state=42)
    lda.fit(Xc)
    joblib.dump(lda, os.path.join(out_dir, "lda_model.joblib"))
    joblib.dump(cv, os.path.join(out_dir, "lda_vectorizer.joblib"))
    print("Saved lda_model.joblib and lda_vectorizer.joblib")

    # Dummy sentiment model (only for completeness). If you have labels, replace this.
    # We'll train a tiny logistic regression on synthetic labels (not for production).
    try:
        y = [1 if "good" in t or "positive" in t or "great" in t else 0 for t in cleaned]
        sent_clf = LogisticRegression(max_iter=200)
        sent_clf.fit(X, y)
        joblib.dump(sent_clf, os.path.join(out_dir, "sentiment_model.joblib"))
        print("Saved sentiment_model.joblib (dummy)")
    except Exception as e:
        print("Skipping sentiment model (could not train):", e)

    print("All models saved into:", out_dir)

def main():
    parser = argparse.ArgumentParser(description="Generate models for dynamic text analysis")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to CSV with text column")
    parser.add_argument("--text-column", type=str, default="text", help="Name of text column in CSV")
    parser.add_argument("--out-dir", type=str, default="backend/models", help="Output models directory")
    parser.add_argument("--n-topics", type=int, default=10, help="Number of topics to train")
    parser.add_argument("--max-features", type=int, default=2000, help="Max features for vectorizers")
    args = parser.parse_args()

    if args.input_csv:
        print("Loading texts from:", args.input_csv)
        texts = load_texts_from_csv(args.input_csv, args.text_column)
    else:
        print("No input CSV provided — using a small dummy corpus.")
        texts = [
            "This is a sample text analysis project for topic modeling and summarization",
            "Natural language processing and machine learning help extract topics",
            "Sentiment analysis detects positive or negative tone in text",
            "We will train simple models to demonstrate functionality"
        ]

    build_and_save_models(texts, args.out_dir, n_topics=args.n_topics, max_features=args.max_features)

if __name__ == "__main__":
    main()
