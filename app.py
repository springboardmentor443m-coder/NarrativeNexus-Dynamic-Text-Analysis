from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import traceback
import pandas as pd
from docx import Document
import fitz
from utils import clean_text, clean_light

# ================================
# BERTopic (pretrained model)
# ================================
from bertopic import BERTopic

app = Flask(__name__)
CORS(app)

# ================================
# Models
# ================================
sid = SentimentIntensityAnalyzer()

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

# 🔹 Load the model you trained with train_topic_model.py
# Make sure the "topic_model" folder is in the same directory as app.py
topic_model = BERTopic.load("topic_model")


# ================================
# Helper Functions
# ================================
def safe_summarize(text, max_length=120, min_length=25):
    """Prevent crashes on short text"""
    if not text or len(text.split()) < 6:
        return text
    t = text[:2000]
    out = summarizer(t, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]["summary_text"].strip()


def model_topic_output(docs):
    """
    Use the PRETRAINED BERTopic model to get topic id, confidence and keywords.
    This does NOT re-train; it just calls .transform().
    """
    if not docs:
        return []

    # pretrained model: this works even with a single document
    topics, probs = topic_model.transform(docs)

    output = []
    for idx, topic_id in enumerate(topics):
        # get top keywords for this topic id
        keywords = topic_model.get_topic(topic_id)

        output.append({
            "doc_index": int(idx),
            "predicted_topic_id": int(topic_id),
            "confidence": float(probs[idx]) if probs is not None else None,
            "top_keywords": [str(k) for k, _ in keywords] if keywords else []
        })

    return output


# ================================
# Main analysis pipeline
# ================================
def analyze_text(text):
    # Split user input by paragraphs (double newline)
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]

    if not docs:
        docs = [text]

    cleaned_for_topics = [clean_text(d) for d in docs]
    cleaned_for_summary = [clean_light(d) for d in docs]

    # ======================
    # Sentiment scoring
    # ======================
    pos_sum = neg_sum = neu_sum = 0

    for d in cleaned_for_summary:
        s = sid.polarity_scores(d)
        pos_sum += s["pos"]
        neg_sum += s["neg"]
        neu_sum += s["neu"]

    total = pos_sum + neg_sum + neu_sum
    if total == 0:
        sentiment = {
            "positive": 0,
            "negative": 0,
            "neutral": 100,
            "dominant": "neutral",
        }
    else:
        pos_ratio = pos_sum / total
        neg_ratio = neg_sum / total
        neu_ratio = neu_sum / total

        dominant = max(
            [("positive", pos_ratio), ("negative", neg_ratio), ("neutral", neu_ratio)],
            key=lambda x: x[1]
        )[0]

        sentiment = {
            "positive": round(pos_ratio * 100, 2),
            "negative": round(neg_ratio * 100, 2),
            "neutral": round(neu_ratio * 100, 2),
            "dominant": dominant
        }

    # ======================
    # Topic Modeling (PRETRAINED BERTopic)
    # ======================
    topic_predictions = model_topic_output(cleaned_for_topics)

    # ======================
    # Summaries
    # ======================
    summaries = [safe_summarize(x) for x in cleaned_for_summary]

    return {
        "cleaned_text": cleaned_for_topics,
        "sentiment": sentiment,
        "topics": topic_predictions,
        "summaries": summaries,
    }


# ================================
# API ROUTES
# ================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        txt = data.get("text", "")

        if not txt:
            return jsonify({"error": "No text provided"}), 400

        return jsonify(analyze_text(txt))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        name = f.filename.lower()
        text = ""

        if name.endswith(".txt"):
            text = f.read().decode()

        elif name.endswith(".csv"):
            df = pd.read_csv(f)
            text = "\n\n".join(df.astype(str).agg(" ".join, axis=1).tolist())

        elif name.endswith(".docx"):
            doc = Document(f)
            text = "\n\n".join([p.text for p in doc.paragraphs])

        elif name.endswith(".pdf"):
            pdf = fitz.open(stream=f.read(), filetype="pdf")
            text = "\n\n".join([p.get_text() for p in pdf])

        else:
            return jsonify({"error": "Unsupported file format"}), 400

        res = analyze_text(text)
        res["filename"] = name
        return jsonify(res)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
