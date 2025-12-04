import os
import streamlit as st
import plotly.graph_objects as go

# MODULE IMPORTS
from modules import (
    data_input,
    preprocessing,
    sentiment_analysis,
    summarization,
    visualization,
    topic_modeling,
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Text Analysis Editor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# LOAD EXTERNAL CSS
# ------------------------------------------------------------
def load_css(path="style.css"):
    if os.path.exists(path):
        with open(path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ style.css not found! UI will not be styled.")

load_css()

# ------------------------------------------------------------
# THEMES
# ------------------------------------------------------------
THEMES = {
    "Ocean": {
        "bg": "linear-gradient(180deg, #0a192f 0%, #102a43 100%)",
        "primary": "#0078d7",
        "accent": "#00b4d8",
        "card": "rgba(28, 37, 65, 0.9)",
    },
    "Grape": {
        "bg": "linear-gradient(180deg, #2b0f39 0%, #4e2777 100%)",
        "primary": "#9d4edd",
        "accent": "#ff79c6",
        "card": "rgba(50, 25, 70, 0.9)",
    },
    "Sunset": {
        "bg": "linear-gradient(180deg, #1a0f0b 0%, #432818 100%)",
        "primary": "#ff7a59",
        "accent": "#ffd166",
        "card": "rgba(50, 25, 15, 0.9)",
    },
    "Emerald": {
        "bg": "linear-gradient(180deg, #071713 0%, #0b2720 100%)",
        "primary": "#19c09b",
        "accent": "#34e1b9",
        "card": "rgba(10, 35, 28, 0.9)",
    },
}

def apply_theme(theme):
    t = THEMES.get(theme, THEMES["Ocean"])
    css = f"""
    <style>
      html, body, [class*="stApp"] {{
        background: {t['bg']} !important;
      }}
      :root {{
        --primary: {t['primary']};
        --accent: {t['accent']};
        --card-bg: {t['card']};
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("⚙️ Settings")

    features = st.sidebar.multiselect(
        "Select features to run:",
        ["Summary", "Sentiment", "Key Topics", "Topic Modeling"],
        default=["Summary", "Sentiment", "Key Topics"],
    )

    theme_choice = st.sidebar.selectbox(
        "Theme", list(THEMES.keys()), index=0
    )

    summary_words = st.sidebar.slider(
        "Summary target length:",
        60, 220, 120, step=10
    )

    st.sidebar.info("🛈 Models may take time to load the first time.")

    return set(features), theme_choice, summary_words


selected_features, chosen_theme, summary_target = render_sidebar()
apply_theme(chosen_theme)

# ------------------------------------------------------------
# HERO SECTION
# ------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <h1>Transform Text into <span class="highlight">Actionable Insights</span></h1>
    <p>Upload, analyze, and visualize — uncover themes, sentiments & summaries instantly.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# INPUT SECTION
# ------------------------------------------------------------
st.markdown("<h3 class='section-title'>📂 Data Input</h3>", unsafe_allow_html=True)

st.markdown("<div class='input-container'>", unsafe_allow_html=True)

uploaded_text = data_input.handle_file_upload()

pasted_text = st.text_area(
    "Paste your text here...",
    height=220,
    placeholder="Paste large documents, reports, articles...",
)

analyze = st.button("✨ Analyze Text", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# ANALYSIS PIPELINE
# ------------------------------------------------------------

if analyze:
    raw_text = uploaded_text or pasted_text.strip()
    if not raw_text:
        st.warning("⚠️ Please upload a file or paste text first.")
        st.stop()

    # Preprocess
    with st.spinner("🧹 Preparing text..."):
        cleaned = preprocessing.clean_text(raw_text)

    # Feature flags
    want_summary = "Summary" in selected_features
    want_sentiment = "Sentiment" in selected_features
    want_topics = "Key Topics" in selected_features or "Topic Modeling" in selected_features

    summary_output = None
    sentiment_label = None
    sentiment_conf = None
    topic_words = []
    topic_readable = []
    topic_distribution = None

    # TOPICS
    if want_topics:
        with st.spinner("🔍 Extracting key topics..."):
            topic_readable, topic_words, topic_distribution = topic_modeling.perform_hybrid_lda(cleaned)

    # SENTIMENT
    if want_sentiment:
        with st.spinner("😊 Running Sentiment Analysis..."):
            sentiment_label, sentiment_conf = sentiment_analysis.get_sentiment(raw_text)

    # SUMMARY
    if want_summary:
        with st.spinner("🧾 Generating Summary..."):
            summary_output = summarization.generate_summary(raw_text, target_words=summary_target)

    # WORDCLOUD (always)
    with st.spinner("☁️ Generating Word Cloud..."):
        wordcloud_fig = visualization.create_wordcloud(cleaned)

    st.success("🎉 Analysis Complete!")

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------
    st.markdown("<h2 class='section-title'>📊 Detailed Analysis</h2>", unsafe_allow_html=True)

    tab_sum, tab_topics, tab_sent, tab_wc, tab_clean = st.tabs(
        ["Summary", "Themes", "Sentiment", "Word Cloud", "Processed Text"]
    )

    # Summary Tab
    with tab_sum:
        st.subheader("🧾 Summary")
        if summary_output:
            st.write(summary_output)
        else:
            st.info("Summary not selected or could not be generated.")

    # Topics Tab
    with tab_topics:
        st.subheader("🎯 Key Topics")
        if topic_words:
            for i, words in enumerate(topic_words, 1):
                st.markdown(f"**Topic {i}:** {words}")
        else:
            st.info("No topics found or not selected.")

    # Sentiment Tab
    with tab_sent:
        st.subheader("🧠 Sentiment")
        if sentiment_label:
            score = float(sentiment_conf)
            color = {
                "Positive": "#00C853",
                "Negative": "#D50000",
                "Neutral": "#FFD600",
            }.get(sentiment_label, "#999")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": f"{sentiment_label}", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": color},
                }
            ))
            fig.update_layout(height=260, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sentiment analysis not selected.")

    # Word Cloud Tab
    with tab_wc:
        st.subheader("☁️ Word Cloud")
        st.pyplot(wordcloud_fig)

    # Processed Text Tab
    with tab_clean:
        st.subheader("🧹 Cleaned Text")
        st.text_area("Processed Text", cleaned, height=250)

    # --------------------------------------------------------
    # QUICK INSIGHTS
    # --------------------------------------------------------
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>✨ Quick Insights</h2>", unsafe_allow_html=True)

    a, b, c = st.columns(3)

    # Summary box
    with a:
        short = summary_output[:220] + "..." if summary_output else "No summary"
        st.markdown(f"<div class='info-box'><h4>🧾 Summary</h4><p>{short}</p></div>", unsafe_allow_html=True)

    # Sentiment box
    with b:
        if sentiment_label:
            st.markdown(
                f"<div class='info-box'><h4>🎯 Sentiment</h4><p><b>{sentiment_label}</b> ({sentiment_conf:.2f})</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='info-box'><h4>🎯 Sentiment</h4><p>No sentiment analyzed</p></div>", unsafe_allow_html=True)

    # Topics box
    with c:
        if topic_words:
            st.markdown(
                f"<div class='info-box'><h4>🗂 Topics</h4><p>{topic_words[0]}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='info-box'><h4>🗂 Topics</h4><p>No topics extracted</p></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    Powered by Infosys NLP Suite | Smart Text Analytics Platform
</div>
""", unsafe_allow_html=True)
