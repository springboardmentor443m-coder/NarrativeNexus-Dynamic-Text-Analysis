# ================================================
# app/app.py – Week 4 Final Version
# ================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from core.preprocess import preprocess, compute_stats
from core.topics import nmf_topics, lda_topics
from core.sentiment import sentiment_label
from core.summarize import extractive_summary, abstractive_summary
from core.report import save_json_report, save_pdf_report
from visuals import topic_bar_chart


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="NarrativeNexus", layout="wide")

st.title("🧠 NarrativeNexus — Week 4 Final AI Text Analysis")
st.caption("Analyze text with advanced topic modeling, sentiment, and summarization.")


# ---- Input Area ----
uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
raw = st.text_area("Or paste text here", height=200, placeholder="Paste any article, post, or report...")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    n_topics = st.slider("Number of Topics", 3, 10, 5, 1)
with col2:
    n_words = st.slider("Keywords per Topic", 5, 15, 8, 1)
with col3:
    model_type = st.radio("Topic Model:", ["NMF", "LDA"], horizontal=True)

summary_mode = st.radio("Choose Summary Type:", ["Extractive", "Abstractive"], horizontal=True)


# ---- Analyze Button ----
if st.button("Analyze", type="primary"):

    topics, senti, summary = [], {}, ""

    # ---- Input Handling ----
    if not (uploaded or (raw and raw.strip())):
        st.warning("⚠️ Please upload or paste some text before analyzing.")
        st.stop()

    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        text = raw

    # ---- Preprocess & Stats ----
    cleaned, tokens = preprocess(text)
    stats = compute_stats(text, cleaned, tokens)

    if stats["token_count"] < 15:
        st.warning("⚠️ Input too short after cleaning (need at least ~15 tokens).")
        st.stop()
    if stats["cleaned_chars"] > 200_000:
        st.warning("⚠️ Input very large. Please trim for faster processing.")
        st.stop()

    # ---- Display Stats ----
    with st.expander("📊 Input & Preprocessing Stats", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Original chars", stats["original_chars"])
        c2.metric("Cleaned chars", stats["cleaned_chars"])
        c3.metric("Original words", stats["original_words"])
        c4.metric("Tokens", stats["token_count"])
        c5.metric("Unique tokens", stats["unique_tokens"])
        if st.checkbox("Show cleaned text preview", value=False):
            preview = cleaned[:2000] + ("..." if len(cleaned) > 2000 else "")
            st.text_area("Cleaned text", preview, height=200)

    # ---- Topic Modeling ----
    st.subheader("🔍 Topics")
    if model_type == "NMF":
        topics, doc_topic, model, vec = nmf_topics([cleaned], n_topics=n_topics, n_words=n_words)
    else:
        topics, doc_topic, model, vec = lda_topics([cleaned], n_topics=n_topics, n_words=n_words)

    if topics:
        for t in topics:
            st.write(f"**Topic {t['topic_id']}** — " + ", ".join(t['keywords']))
    else:
        st.info("No topics generated. Provide a richer text sample.")

    # ---- Visualization ----
    st.subheader("📈 Topic Visualization")
    topic_bar_chart(topics)

    # ---- Word Cloud ----
    st.subheader("☁️ Word Cloud")
    if tokens:
        wc = WordCloud(width=900, height=400).generate(" ".join(tokens))
        fig = plt.figure(figsize=(9, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
    else:
        st.info("No tokens to display in word cloud.")

    # ---- Sentiment ----
    st.subheader("💬 Sentiment")
    try:
        senti = sentiment_label(cleaned)
        st.write(f"**Label:** {senti.get('label','N/A')}  |  **Score:** {senti.get('compound',0):.3f}")
    except Exception as e:
        st.error(f"Sentiment error: {e}")

    # ---- Summary ----
    st.subheader("📝 Summary")
    try:
        if summary_mode == "Extractive":
            summary = extractive_summary(cleaned, sentences_max=3)
        else:
            summary = abstractive_summary(cleaned)
        st.write(summary)
    except Exception as e:
        st.error(f"Summary error: {e}")

    # ---- Save Reports ----
    payload = {
        "summary": summary,
        "sentiment": senti,
        "topics": topics,
        "model_type": model_type,
        "summary_mode": summary_mode
    }

    json_path = save_json_report(payload)
    pdf_path = save_pdf_report(payload)

    st.success(f"✅ JSON report saved → {json_path}")
    st.success(f"📄 PDF report saved → {pdf_path}")
