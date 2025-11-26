import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json

from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud

from core.preprocess import preprocess, compute_stats
from core.topics import nmf_topics, lda_topics
from core.sentiment import sentiment_label
from core.summarize import extractive_summary, abstractive_summary
from core.report import save_json_report, save_pdf_report

# -------------------------------------------------
# LOAD TRAINED TOPIC MODEL (TF-IDF + NMF)
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("📦 Loading trained topic model...")

try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    nmf_model = joblib.load(os.path.join(MODELS_DIR, "nmf_model.pkl"))

    with open(os.path.join(MODELS_DIR, "topic_keywords.json"), "r") as f:
        topic_keywords = json.load(f)

    print("✅ Trained NMF model loaded.")
except Exception as e:
    print("❌ Failed to load trained topic model:", e)
    vectorizer = None
    nmf_model = None
    topic_keywords = []


# -------------------------------------------------
# Predict Topic using NMF Model
# -------------------------------------------------
def predict_topic(text):
    if vectorizer is None or nmf_model is None:
        return {
            "topic_id": -1,
            "topic_keywords": [],
            "topic_score": 0.0
        }

    cleaned = text
    X = vectorizer.transform([cleaned])

    topic_scores = nmf_model.transform(X)[0]
    best_topic = int(topic_scores.argmax())

    return {
        "topic_id": best_topic,
        "topic_keywords": topic_keywords[best_topic],
        "topic_score": float(topic_scores[best_topic])
    }


# -------------------------------------------------
# FLASK APP
# -------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    model_type = data.get("model_type", "NMF")
    summary_mode = data.get("summary_mode", "Extractive")
    n_topics = int(data.get("n_topics", 5))
    n_words = int(data.get("n_words", 8))

    if not text:
        return jsonify({"error": "No text provided."}), 400

    # --- Preprocess ---
    cleaned, tokens = preprocess(text)
    stats = compute_stats(text, cleaned, tokens)

    if stats["token_count"] < 15:
        return jsonify({
            "error": "Input is too short after cleaning (need at least ~15 tokens).",
            "stats": stats
        }), 400

    # --- Topic Model (NMF/LDA keywords) ---
    docs = [cleaned]
    if model_type.upper() == "LDA":
        topics, doc_topic, model, vec = lda_topics(docs, n_topics=n_topics, n_words=n_words)
    else:
        topics, doc_topic, model, vec = nmf_topics(docs, n_topics=n_topics, n_words=n_words)

    # --- Topic Prediction (using trained model) ---
    topic_data = predict_topic(cleaned)

    # --- Sentiment ---
    senti = sentiment_label(cleaned)

    # --- Summary ---
    if summary_mode.lower() == "abstractive":
        summary = abstractive_summary(cleaned)
    else:
        summary = extractive_summary(cleaned, sentences_max=3)

    # --- Word Cloud ---
    wc_b64 = None
    if tokens:
        wc = WordCloud(width=900, height=400, background_color="white").generate(" ".join(tokens))
        fig = plt.figure(figsize=(6, 3))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        wc_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # --- Save Reports (JSON + PDF) ---
    payload = {
        "summary": summary,
        "sentiment": senti,
        "topics": topics,
        "topic_prediction": topic_data,
        "model_type": model_type,
        "summary_mode": summary_mode
    }

    json_path = save_json_report(payload)
    pdf_path = save_pdf_report(payload)

    # --- Response ---
    return jsonify({
        "stats": stats,
        "summary": summary,
        "sentiment": senti,
        "topics": topics,
        "topic_prediction": topic_data,
        "wordcloud": wc_b64,
        "json_report_path": json_path,
        "pdf_report_path": pdf_path
    })


if __name__ == "__main__":
    app.run(debug=True)
