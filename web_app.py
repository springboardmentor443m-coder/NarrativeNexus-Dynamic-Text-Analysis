import os
import re
import json
from heapq import nlargest

from flask import Flask, request, jsonify, render_template

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import joblib

# ---------------------------------------------------
# Flask app
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")

app = Flask(__name__)

# ---------------------------------------------------
# NLTK setup
# ---------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------
# Load trained topic model & vectorizer
# ---------------------------------------------------
nmf_model_path = os.path.join(MODELS_DIR, "nmf_model.pkl")
vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
topics_json_path = os.path.join(MODELS_DIR, "topic_names.json")

nmf_model = joblib.load(nmf_model_path)
vectorizer = joblib.load(vec_path)

with open(topics_json_path, "r", encoding="utf-8") as f:
    topic_keywords_map = json.load(f)


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# 1. PREPROCESS API
@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    original_len = len(text)

    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).lower()

    tokens = [t for t in word_tokenize(cleaned) if t not in stop_words and t.strip()]

    return jsonify({
        "original_chars": original_len,
        "cleaned_chars": len(cleaned),
        "words": len(text.split()),
        "tokens": len(tokens),
        "cleaned_text": cleaned,
        "tokens_list": tokens
    })


# 2. SENTIMENT API
@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.5:
        label = "Strong Positive"
    elif compound >= 0.05:
        label = "Positive"
    elif compound <= -0.5:
        label = "Strong Negative"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return jsonify({
        "label": label,
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
        "interpretation": f"The paragraph expresses a {label.lower()} sentiment."
    })


# 3. TOPIC MODELING API (trained on 20 Newsgroups)
@app.route("/api/topics", methods=["POST"])
def api_topics():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    X_input = vectorizer.transform([text])
    topic_distribution = nmf_model.transform(X_input)
    topic_index = int(topic_distribution.argmax())

    keywords = topic_keywords_map.get(str(topic_index), [])
    topic_name = ", ".join(keywords[:4]) if keywords else "Unknown Topic"

    return jsonify({
        "topic_id": topic_index,
        "topic_name": topic_name,
        "keywords": keywords
    })


# 4. SUMMARIZATION API (simple extractive)
@app.route("/api/summary", methods=["POST"])
def api_summary():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    sentences = sent_tokenize(text)
    if len(sentences) <= 2:
        return jsonify({"summary": text})

    word_freq = {}
    for word in word_tokenize(text.lower()):
        if word.isalpha() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values())
    for w in word_freq:
        word_freq[w] /= max_freq

    scoring = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                scoring[sent] = scoring.get(sent, 0) + word_freq[word]

    summary_sentences = nlargest(3, scoring, key=scoring.get)
    summary_text = " ".join(summary_sentences)

    return jsonify({"summary": summary_text})


if __name__ == "__main__":
    app.run(debug=True)
