# --------------------------------------------
# NarrativeNexus – Full Flask Backend (Final)
# --------------------------------------------
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import time
import base64
import numpy as np

# ---- NLP Components ----
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from textblob import TextBlob
from wordcloud import WordCloud
from werkzeug.utils import secure_filename
from fpdf import FPDF

app = Flask(__name__)

# Ensure outputs folder exists
OUTPUT_DIR = os.path.join(app.root_path, "outputs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# -------------------------------------------------
#            HOME PAGE
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------
#        CLEAN TEXT FUNCTION
# -------------------------------------------------
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# -------------------------------------------------
#        TOPIC MODELING FUNCTION
# -------------------------------------------------
def extract_topics(model_type, cleaned_text, n_topics, n_words):
    docs = [cleaned_text]

    if model_type == "NMF":
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(docs)
        model = NMF(n_components=n_topics, random_state=42)
        model.fit(matrix)
        feature_names = vectorizer.get_feature_names_out()

    else:  # LDA
        vectorizer = CountVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(docs)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        model.fit(matrix)
        feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_words:]]
        topics.append(top_words)

    # Use topic 0 as the predicted-topic (simple version)
    predicted = {
        "topic_id": 0,
        "topic_keywords": topics[0],
        "topic_score": float(1.0)
    }

    return topics, predicted


# -------------------------------------------------
#      SUMMARIZATION FUNCTION
# -------------------------------------------------
def generate_summary(text, mode="Extractive"):
    sentences = text.split(".")
    summary = ". ".join(sentences[:2]) + "."

    if mode == "Abstractive":
        summary = "This text discusses key ideas and themes in the content."

    return summary


# -------------------------------------------------
#      WORD CLOUD FUNCTION
# -------------------------------------------------
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    img = wc.to_image()

    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# -------------------------------------------------
#      PDF REPORT GENERATOR
# -------------------------------------------------
def create_pdf(path, text, summary, topics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "NarrativeNexus Report", ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 7, f"Original Text:\n{text}")
    pdf.ln(4)

    pdf.multi_cell(0, 7, f"Summary:\n{summary}")
    pdf.ln(4)

    pdf.multi_cell(0, 7, f"Topics:\n{json.dumps(topics, indent=2)}")

    pdf.output(path)


# -------------------------------------------------
#      MAIN API ENDPOINT
# -------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "").strip()
    model_type = data.get("model_type", "NMF")
    summary_mode = data.get("summary_mode", "Extractive")
    n_topics = int(data.get("n_topics", 5))
    n_words = int(data.get("n_words", 8))

    if len(text) < 20:
        return jsonify({"error": "Input is too short!"})

    cleaned = clean_text(text)

    topics, predicted = extract_topics(model_type, cleaned, n_topics, n_words)

    blob = TextBlob(text)
    sentiment = {
        "label": blob.sentiment.polarity > 0 and "positive" or "negative" if blob.sentiment.polarity != 0 else "neutral",
        "score": float(blob.sentiment.polarity)
    }

    summary = generate_summary(text, summary_mode)
    wordcloud = generate_wordcloud(cleaned)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = f"report-{timestamp}.json"
    pdf_path = f"report-{timestamp}.pdf"

    full_json_path = os.path.join(OUTPUT_DIR, json_path)
    full_pdf_path = os.path.join(OUTPUT_DIR, pdf_path)

    # Save JSON report
    with open(full_json_path, "w") as f:
        json.dump({
            "text": text,
            "summary": summary,
            "topics": topics,
            "predicted": predicted,
            "sentiment": sentiment
        }, f, indent=4)

    # Save PDF report
    create_pdf(full_pdf_path, text, summary, topics)

    # Response
    return jsonify({
        "stats": {
            "original_chars": len(text),
            "cleaned_chars": len(cleaned),
            "word_count": len(text.split()),
            "token_count": len(cleaned.split()),
            "unique_tokens": len(set(cleaned.split()))
        },
        "sentiment": sentiment,
        "summary": summary,
        "summary_mode": summary_mode,
        "topics": topics,
        "topic_prediction": predicted,
        "wordcloud": wordcloud,
        "json_report_path": f"/outputs/{json_path}",
        "pdf_report_path": f"/outputs/{pdf_path}"
    })


# -------------------------------------------------
#      SERVE PDF / JSON FILES
# -------------------------------------------------
@app.route("/outputs/<path:filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


# -------------------------------------------------
# Start Application
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
