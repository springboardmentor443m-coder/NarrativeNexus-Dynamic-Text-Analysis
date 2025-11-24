from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import traceback
import io
import pandas as pd
from docx import Document
import fitz  # PyMuPDF
from utils import clean_text, clean_light

app = Flask(__name__)
CORS(app)

sid = SentimentIntensityAnalyzer()

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

def safe_summarize(text, max_length=120, min_length=25):
    if not text or len(text.split()) < 6:
        return text
    t = text[:2000]
    out = summarizer(t, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]["summary_text"].strip()


def get_topics_lda(docs, n_topics=3, n_top_words=6):
    if not docs:
        return [], []

    cv = CountVectorizer(stop_words="english")
    X = cv.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = cv.get_feature_names_out()
    topics = [
        {"topic_id": i,
         "words": [words[j] for j in comp.argsort()[:-n_top_words-1:-1]]
         }
        for i, comp in enumerate(lda.components_)
    ]
    return topics, lda.transform(X).tolist()


def get_topics_nmf(docs, n_topics=3, n_top_words=6):
    if not docs:
        return [], []

    tv = TfidfVectorizer(stop_words="english")
    X = tv.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(X)

    words = tv.get_feature_names_out()
    topics = [
        {"topic_id": i,
         "words": [words[j] for j in comp.argsort()[:-n_top_words-1:-1]]
         }
        for i, comp in enumerate(nmf.components_)
    ]
    return topics, nmf.transform(X).tolist()


def analyze_text(text):
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    if not docs:
        docs = [text]

    cleaned_topics = [clean_text(d) for d in docs]
    cleaned_summary = [clean_light(d) for d in docs]

    # ======================
    # SENTIMENT (REAL %)
    # ======================
    pos_sum = 0
    neg_sum = 0

    for d in cleaned_summary:
        s = sid.polarity_scores(d)
        pos_sum += s["pos"]
        neg_sum += s["neg"]

    total = pos_sum + neg_sum

    if total == 0:
        sentiment = {"emotion": "neutral", "percentage": 0}
    else:
        pos_ratio = pos_sum / total
        neg_ratio = neg_sum / total

        if pos_ratio >= neg_ratio:
            sentiment = {"emotion": "positive", "percentage": round(pos_ratio * 100, 2)}
        else:
            sentiment = {"emotion": "negative", "percentage": round(neg_ratio * 100, 2)}

    # ======================
    # TOPIC MODELING
    # ======================
    lda_topics, lda_dist = get_topics_lda(cleaned_topics)
    nmf_topics, nmf_dist = get_topics_nmf(cleaned_topics)

    # ======================
    # SUMMARIZATION
    # ======================
    summaries = [safe_summarize(x) for x in cleaned_summary]

    return {
        "cleaned_text": cleaned_topics,
        "sentiment": sentiment,
        "lda_topics": lda_topics,
        "nmf_topics": nmf_topics,
        "summaries": summaries
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        txt = data.get("text", "")
        if not txt:
            return jsonify({"error": "No text provided"}), 400
        return jsonify(analyze_text(txt))
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500


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
            return jsonify({"error": "Unsupported format"}), 400

        res = analyze_text(text)
        res["filename"] = name
        return jsonify(res)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
