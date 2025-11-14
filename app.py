# app.py
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

from utils import clean_html, clean_light, clean_text

app = Flask(__name__)
CORS(app)

# Initialize sentiment and summarization
sid = SentimentIntensityAnalyzer()

# --- UPDATED: USE SMALL, FAST, RELIABLE MODEL ---
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",  # FAST WORKING MODEL
    device=-1  # CPU
)


def safe_summarize(text: str, max_length=120, min_length=25):
    """
    Safely call the summarizer with a length guard and exception handling.
    Returns either the summary string or raises exception to caller.
    """
    if not text or len(text.strip().split()) < 8:
        return text.strip()

    trunc = text.strip()
    if len(trunc) > 2000:
        trunc = trunc[:2000]

    out = summarizer(trunc, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
        return out[0]["summary_text"].strip()
    else:
        return str(out)


def get_topics_lda(docs, n_topics=3, n_top_words=6):
    if len(docs) == 0:
        return [], []
    cv = CountVectorizer(stop_words='english', max_df=1, min_df=1.0)
    X = cv.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online', max_iter=10)
    lda.fit(X)
    feature_names = cv.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append({'topic_id': int(topic_idx), 'words': top_features})
    doc_topic_distr = lda.transform(X).tolist()
    return topics, doc_topic_distr


def get_topics_nmf(docs, n_topics=3, n_top_words=6):
    if len(docs) == 0:
        return [], []
    tv = TfidfVectorizer(stop_words='english', max_df=1, min_df=1.0)
    X = tv.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf.fit(X)
    feature_names = tv.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append({'topic_id': int(topic_idx), 'words': top_features})
    doc_topic_distr = nmf.transform(X).tolist()
    return topics, doc_topic_distr


def analyze_text(text):
    raw_docs = [d.strip() for d in text.split('\n\n') if d.strip()]
    if not raw_docs:
        raw_docs = [text]

    cleaned_for_topics = [clean_text(d) for d in raw_docs]   # heavy cleaning
    cleaned_for_summary = [clean_light(d) for d in raw_docs]  # light cleaning

    lda_topics, lda_distr = get_topics_lda(cleaned_for_topics)
    nmf_topics, nmf_distr = get_topics_nmf(cleaned_for_topics)

    # ---------- SENTIMENT ANALYSIS ----------
    sentiments = []
    pos_sum = neu_sum = neg_sum = compound_sum = 0.0

    for d in cleaned_for_summary:
        scores = sid.polarity_scores(d)

        # Convert to percentages
        pos_pct = round(scores["pos"] * 100, 2)
        neu_pct = round(scores["neu"] * 100, 2)
        neg_pct = round(scores["neg"] * 100, 2)
        compound = round(scores["compound"], 4)

        # Identify strongest emotion
        emotion_map = {
            "positive": pos_pct,
            "neutral": neu_pct,
            "negative": neg_pct
        }
        strongest_emotion = max(emotion_map, key=emotion_map.get)
        strongest_value = emotion_map[strongest_emotion]

        # Store per-document result
        sentiments.append({
            "emotion": strongest_emotion,
            "percentage": strongest_value
        })

        # Accumulate for final emotion
        pos_sum += scores["pos"]
        neu_sum += scores["neu"]
        neg_sum += scores["neg"]
        compound_sum += scores["compound"]

    # ---------- FINAL OVERALL EMOTION ----------
    n = max(1, len(cleaned_for_summary))
    avg_compound = round(compound_sum / n, 4)

    if avg_compound >= 0.05:
        final_emotion = "positive"
    elif avg_compound <= -0.05:
        final_emotion = "negative"
    else:
        final_emotion = "neutral"

    overall_sentiment = {
        "final_emotion": final_emotion,
        "avg_compound": avg_compound
    }

    summaries = []
    for d in cleaned_for_summary:
        try:
            if not d or len(d.strip().split()) < 6:
                summaries.append(d.strip())
            else:
                s = safe_summarize(d, max_length=120, min_length=25)
                summaries.append(s)
        except Exception as e:
            summaries.append(f"Summarization failed: {str(e)[:200]}")

    result = {
        "num_documents": len(raw_docs),
        "cleaned_text": cleaned_for_topics,
        "lda_topics": lda_topics,
        "lda_doc_topic_distribution": lda_distr,
        "nmf_topics": nmf_topics,
        "nmf_doc_topic_distribution": nmf_distr,
        "sentiments": sentiments,
        "overall_sentiment": overall_sentiment,
        "summaries": summaries
    }
    return result


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = analyze_text(text)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        filename = file.filename.lower()
        text = ""

        if filename.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(file.read().decode("utf-8", errors="ignore")), low_memory=False)
            text_cols = [c for c in df.columns if df[c].dtype == object or "text" in c.lower() or "review" in c.lower()]
            if text_cols:
                text = "\n\n".join(df[text_cols].astype(str).agg(" ".join, axis=1).tolist())
            else:
                text = "\n\n".join(df.astype(str).agg(" ".join, axis=1).tolist())
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file.read()))
            text = "\n\n".join([p.text for p in doc.paragraphs])
        elif filename.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n\n".join([page.get_text() for page in doc])
        else:
            return jsonify({"error": "Unsupported file format. Supported: .txt, .csv, .docx, .pdf"}), 400

        result = analyze_text(text)
        result["filename"] = filename
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)