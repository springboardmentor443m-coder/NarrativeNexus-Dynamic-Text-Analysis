import streamlit as st
import pandas as pd
from typing import List, Optional, Tuple
import io

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_file(file) -> str:
    """Extract text from uploaded file based on type"""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return ""

@st.cache_resource(show_spinner=False)
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@st.cache_resource(show_spinner=False)
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_resource(show_spinner=False)
def load_keybert():
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return KeyBERT(model=model)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_bertopic():
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True, calculate_probabilities=True)
        return topic_model
    except Exception:
        # BERTopic not installed or failed to load
        return None


def summarize_text(text: str, max_words: int = 120) -> str:
    if not text.strip():
        return ""
    summarizer = load_summarizer()
    max_len = min(max(64, min(len(text.split()) * 2, 512)), 1024)
    result = summarizer(text, max_length=max_len, min_length=max(32, max_len // 4), do_sample=False)
    return result[0]["summary_text"].strip()


def summarize_text_fast(text: str, target_words: int = 100) -> str:
    if not text.strip():
        return ""
    summarizer = load_summarizer()
    max_length = max(56, min(256, target_words * 2))
    min_length = max(24, min(max_length // 3, target_words))
    result = summarizer(text[:3500], max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"].strip()


def analyze_sentiment(text: str) -> Tuple[str, float]:
    if not text.strip():
        return ("NEUTRAL", 0.0)
    sentiment = load_sentiment()
    res = sentiment(text[:4000])[0]
    label = res["label"].upper()
    score = float(res["score"])
    return label, score


def extract_key_topics(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    if not text.strip():
        return []
    kw_model = load_keybert()
    if kw_model is not None:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.6,
            top_n=top_n,
        )
        return keywords
    # Fallback: simple TF-IDF based keyphrase extraction
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=5000,
    )
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]
    terms = np.array(vectorizer.get_feature_names_out())
    if scores.size == 0:
        return []
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in top_idx]


def run_topic_modeling(docs: List[str]):
    topic_model = load_bertopic()
    if topic_model is None:
        raise ImportError("BERTopic is not installed. Please install it with: pip install bertopic")
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def render_sidebar():
    st.sidebar.header("Text Analysis Editor")
    st.sidebar.markdown("Select analysis features and provide input text or dataset.")
    
    features = st.sidebar.multiselect(
        "Choose features",
        ["Key Topics", "Summary", "Sentiment", "Topic Modeling"],
        default=["Key Topics", "Summary", "Sentiment"],
    )
    st.sidebar.divider()

    theme_choice = st.sidebar.selectbox(
        "Theme",
        ["Ocean", "Grape", "Sunset", "Emerald"],
        index=0,
    )
    
    # Show selected theme
    st.sidebar.markdown(f"**Selected:** {theme_choice}")

    fast_mode = st.sidebar.toggle("Fast summarization", value=True, help="Speeds up with a shorter summary")
    summary_words = st.sidebar.slider("Summary length (words)", 50, 220, 100, 10)

    st.sidebar.info(
        "First run downloads models (30–120s). Subsequent runs are faster."
    )
    return set(features), theme_choice, fast_mode, summary_words


def inject_theme(theme_name: str) -> None:
    themes = {
        "Ocean": {
            "bg": "#0b1220",
            "surface": "#121a2a",
            "text": "#e8eefc",
            "muted": "#9fb3d1",
            "primary": "#2e8cff",
            "accent": "#00d4ff",
        },
        "Grape": {
            "bg": "#12091a",
            "surface": "#1b0f26",
            "text": "#f0e9f6",
            "muted": "#c6b6d9",
            "primary": "#8a4fff",
            "accent": "#ff6bd6",
        },
        "Sunset": {
            "bg": "#1a0f0b",
            "surface": "#221510",
            "text": "#fff3ec",
            "muted": "#f1c7ad",
            "primary": "#ff7a59",
            "accent": "#ffd166",
        },
        "Emerald": {
            "bg": "#071713",
            "surface": "#0c241e",
            "text": "#e6fff8",
            "muted": "#b8e7d9",
            "primary": "#19c09b",
            "accent": "#34e1b9",
        },
    }

    t = themes.get(theme_name, themes["Ocean"])
    css = f"""
    <style>
      :root {{
        --bg: {t['bg']};
        --surface: {t['surface']};
        --text: {t['text']};
        --muted: {t['muted']};
        --primary: {t['primary']};
        --accent: {t['accent']};
      }}
      .stApp {{
        background: var(--bg);
        color: var(--text);
      }}
      h1, h2, h3, h4, h5, h6 {{
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.35);
      }}
      .stMarkdown, .stText, .stCaption, .stDataFrame {{ color: var(--text); }}
      [data-testid="stHeader"] {{ background: linear-gradient(180deg, rgba(0,0,0,0.3), rgba(0,0,0,0)); }}
      [data-testid="stSidebar"] {{ background: var(--surface); }}
      
      /* Sidebar text colors - make all text visible */
      [data-testid="stSidebar"] label, 
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] div,
      [data-testid="stSidebar"] span {{
        color: #ffffff !important;
      }}
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] input {{
        color: #ffffff !important;
        background: rgba(255,255,255,0.1) !important;
      }}
      [data-testid="stSidebar"] .stSelectbox label,
      [data-testid="stSidebar"] .stMultiselect label {{
        color: #ffffff !important;
        font-weight: 500;
      }}

      .stTextArea textarea {{
        background: var(--surface);
        color: var(--text);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
      }}

      .stButton>button {{
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: #0b0e14;
        border: 0;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        box-shadow: 0 6px 16px rgba(0, 212, 255, 0.18);
        transition: transform .06s ease, box-shadow .2s ease;
      }}
      .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(0, 212, 255, 0.28);
      }}
      .stButton>button:focus, .stButton>button:active {{
        outline: none;
        box-shadow: 0 0 0 4px rgba(46, 140, 255, 0.35), 0 16px 40px rgba(0, 212, 255, 0.35);
      }}

      .glow-card {{
        background: var(--surface);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 12px 32px rgba(0,0,0,0.25);
      }}
      .glow-card.active {{
        box-shadow: 0 0 0 4px rgba(0, 212, 255, 0.22), 0 18px 48px rgba(0, 212, 255, 0.28);
      }}

      .metric-container [data-testid="stMetricValue"], .metric-container [data-testid="stMetricDelta"] {{ color: var(--text) !important; }}
      
      .tag-cloud {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px;
      }}
      .tag {{
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: #0b0e14;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_key_topics(keywords: List[Tuple[str, float]]):
    """Render key topics in multiple formats"""
    st.markdown('<div class="glow-card active"><h4>Key Topics / Phrases</h4></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["List View", "Tag Cloud"])
    
    with tab1:
        for phrase, _ in keywords:
            st.markdown(f"• **{phrase}**")
    
    with tab2:
        # Tag cloud visualization
        tags_html = '<div class="tag-cloud">'
        for phrase, score in keywords:
            # Scale font size based on score (if available)
            tags_html += f'<span class="tag">{phrase}</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)


def render_sentiment_analysis(label: str, score: float):
    """Render sentiment analysis with multiple visualizations"""
    st.markdown('<div class="glow-card active"><h4>Sentiment Analysis</h4></div>', unsafe_allow_html=True)
    
    # Calculate scores for visualization
    if label == "NEGATIVE":
        neg_score = score
        pos_score = 1 - score
        neutral_score = 0.05
    elif label == "POSITIVE":
        pos_score = score
        neg_score = 1 - score
        neutral_score = 0.05
    else:  # NEUTRAL
        neutral_score = score
        pos_score = (1 - score) / 2
        neg_score = (1 - score) / 2
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {label}")
        st.markdown(f"**Confidence:** {score:.1%}")
    
    # Sentiment visualizations
    with col2:
        try:
            import plotly.graph_objects as go
        except Exception:
            st.info("Plotly is not installed. Run: pip install plotly")
            return
        
        tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
        
        with tab1:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["NEGATIVE", "NEUTRAL", "POSITIVE"],
                y=[neg_score, neutral_score, pos_score],
                marker_color=["#ff4444", "#888888", "#44ff44"],
                text=[f"{neg_score:.1%}", f"{neutral_score:.1%}", f"{pos_score:.1%}"],
                textposition="outside",
                name="Sentiment"
            ))
            fig_bar.update_layout(
                title=dict(text="Sentiment Distribution", font=dict(color="#ffffff", size=16, family="Arial Black")),
                xaxis_title="",
                yaxis_title="Confidence",
                yaxis_range=[0, 1],
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                showlegend=False,
                title_x=0.5
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing before pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
                values=[neg_score, neutral_score, pos_score],
                marker_colors=["#ff4444", "#888888", "#44ff44"],
                hole=0.4,
                textinfo="label+percent",
                textfont=dict(color="#ffffff", size=12)
            )])
            fig_pie.update_layout(
                title=dict(text="Sentiment Distribution", font=dict(color="#ffffff", size=16, family="Arial Black")),
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                showlegend=True,
                title_x=0.5,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)


def render_analysis_panel(text: str, selected_features: set, fast_mode: bool, summary_words: int):
    """Render analysis buttons and results for given text"""
    if not text.strip():
        st.info("Please provide text or upload a file to analyze.")
        return
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    buttons = {}
    with col1:
        buttons["key_topics"] = "Key Topics" in selected_features and st.button("Extract Key Topics", use_container_width=True)
    with col2:
        buttons["summary"] = "Summary" in selected_features and st.button("Summarize", use_container_width=True)
    with col3:
        buttons["sentiment"] = "Sentiment" in selected_features and st.button("Analyze Sentiment", use_container_width=True)
    with col4:
        buttons["topic_modeling"] = "Topic Modeling" in selected_features and st.button("Topic Modeling", use_container_width=True)
    
    # Key Topics
    if buttons["key_topics"]:
        with st.spinner("Extracting key topics..."):
            keywords = extract_key_topics(text)
        if keywords:
            render_key_topics(keywords)
    
    # Summary
    if buttons["summary"]:
        with st.spinner("Summarizing text..."):
            summary = summarize_text_fast(text, summary_words) if fast_mode else summarize_text(text, summary_words)
        st.markdown('<div class="glow-card active"><h4>Summary</h4></div>', unsafe_allow_html=True)
        st.write(summary)
    
    # Sentiment
    if buttons["sentiment"]:
        with st.spinner("Analyzing sentiment..."):
            label, score = analyze_sentiment(text)
        render_sentiment_analysis(label, score)
    
    # Topic Modeling (for single document, split into sentences/chunks)
    if buttons["topic_modeling"]:
        with st.spinner("Running topic modeling (this may take a while)..."):
            # Split text into sentences for topic modeling
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 5:
                st.warning("Text is too short for topic modeling. Please provide a longer document or multiple documents.")
            else:
                model, topics, probs = run_topic_modeling(sentences)
                st.markdown("**Top Topics**")
                topic_info = model.get_topic_info()
                st.dataframe(topic_info.head(20), use_container_width=True)
                
                try:
                    fig = model.visualize_topics()
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Interactive topic visualization requires plotly support.")


def main():
    st.set_page_config(page_title="Text Analysis Editor", layout="wide")
    st.title("Text Analysis Editor")
    st.caption(
        "Transformer-powered analysis: key topics, summarization, sentiment, and scalable topic modeling."
    )

    selected, theme_choice, fast_mode, summary_words = render_sidebar()
    inject_theme(theme_choice)

    # Input method selection
    st.subheader("Input Source")
    input_method = st.radio(
        "Choose input method:",
        ["Upload File (PDF, DOCX, TXT)", "Paste Text"],
        horizontal=True
    )
    
    text_content = ""
    
    if input_method == "Upload File (PDF, DOCX, TXT)":
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=["pdf", "docx", "txt"],
            help="Upload PDF, DOCX, or TXT files for analysis"
        )
        
        if uploaded_file is not None:
            with st.spinner("Reading file..."):
                text_content = extract_text_from_file(uploaded_file)
            
            if text_content:
                st.success(f"File loaded: {uploaded_file.name} ({len(text_content)} characters)")
                # Show preview
                with st.expander("Preview extracted text"):
                    st.text_area("Text preview", text_content[:1000], height=150, disabled=True)
            else:
                st.error("Could not extract text from file. Please try another file.")
    
    else:  # Paste Text
        text_content = st.text_area(
            "Enter or paste text",
            height=220,
            placeholder="Type or paste your document here..."
        )
    
    st.divider()
    
    # Analysis panel - works for both file upload and pasted text
    if text_content:
        render_analysis_panel(text_content, selected, fast_mode, summary_words)
    else:
        st.info("Please upload a file or paste text to begin analysis.")


if __name__ == "__main__":
    main()
