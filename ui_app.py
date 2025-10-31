import streamlit as st
import re
import os
import io
import pandas as pd
import numpy as np
from typing import List, Tuple

# File readers
from docx import Document
import PyPDF2

# NLP & ML
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt

# Download NLTK resources if needed
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()


# ========== Utility Functions ==========

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def read_docx(file) -> str:
    doc = Document(io.BytesIO(file.read()))
    return " ".join([p.text for p in doc.paragraphs])


def read_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_csv(file) -> str:
    df = pd.read_csv(file)
    for col in df.columns:
        if df[col].dtype == object:
            return "\n".join(df[col].astype(str).tolist())
    return "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())


def clean_text_regex(
    text: str,
    remove_emojis: bool = False,
    remove_hashtags_mentions: bool = True,
    remove_numbers: bool = False,
    custom_remove: List[str] = None,
    lowercase: bool = True,
    remove_punct_keep_single: bool = True
) -> str:

    # Remove <script>, <style> and tags
    text = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)

    # URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Hashtags & mentions
    if remove_hashtags_mentions:
        text = re.sub(r'[@#]\w+', ' ', text)

    # Emojis
    if remove_emojis:
        emoji_pattern = re.compile(
            "["u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002700-\U000027BF"
            u"\U0001F900-\U0001F9FF"
            u"\U00002600-\U000026FF"
            u"\U0001FA70-\U0001FAFF"
            u"\U00002500-\U00002BEF"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' ', text)

    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)

    # Normalize punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)

    # Custom word removal
    if custom_remove:
        for w in custom_remove:
            text = re.sub(rf'\b{re.escape(w)}\b', ' ', text, flags=re.IGNORECASE)

    if lowercase:
        text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove symbols
    if remove_punct_keep_single:
        text = re.sub(r"[^a-zA-Z0-9 \.\,\!\?\:\;\-']", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def analyze_document_sentiment(text: str) -> dict:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return {**scores, "label": label}


def run_lda(docs: List[str], n_topics: int = 3, max_iter: int = 10):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=42)
    lda.fit(X)
    return lda, vectorizer, X


def run_nmf_tfidf(docs: List[str], n_topics: int = 3, max_iter: int = 200):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
    X = vectorizer.fit_transform(docs)
    nmf = NMF(n_components=n_topics, max_iter=max_iter, random_state=42)
    nmf.fit(X)
    return nmf, vectorizer, X


def get_top_words(model, feature_names, n_top_words=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append((idx, top_words))
    return topics


# ========== Streamlit App ==========

st.set_page_config(page_title="NarrativeNexus – Multi-Document NLP", layout="wide")
st.title("NarrativeNexus – Multi-Document Text Analysis Platform")

# Sidebar: Upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple text documents (.txt, .csv, .docx, .pdf)",
    type=['txt', 'csv', 'docx', 'pdf'],
    accept_multiple_files=True
)

# Sidebar: Cleaning Options
st.sidebar.header("Cleaning Options")
remove_emojis = st.sidebar.checkbox("Remove emojis", False)
remove_hashtags_mentions = st.sidebar.checkbox("Remove hashtags/mentions", True)
remove_numbers = st.sidebar.checkbox("Remove numbers", False)
lowercase = st.sidebar.checkbox("Convert to lowercase", True)
remove_punct = st.sidebar.checkbox("Clean punctuation", True)
custom_words_raw = st.sidebar.text_area("Custom words to remove (comma-separated)")
custom_remove = [w.strip() for w in custom_words_raw.split(",") if w.strip()]

st.sidebar.header("Topic Modeling Settings")
topic_method = st.sidebar.selectbox("Method", ["LDA (Count)", "NMF (TF-IDF)"])
n_topics = st.sidebar.slider("Number of topics", 2, 10, 3)
max_iter = st.sidebar.slider("Max iterations", 10, 300, 50)

# Load and read
texts, file_names = [], []
if uploaded_files:
    for file in uploaded_files:
        fname = file.name.lower()
        try:
            if fname.endswith('.txt'):
                text = read_txt(file)
            elif fname.endswith('.docx'):
                text = read_docx(file)
            elif fname.endswith('.pdf'):
                text = read_pdf(file)
            elif fname.endswith('.csv'):
                text = read_csv(file)
            else:
                st.sidebar.warning(f"Unsupported type: {file.name}")
                continue
            texts.append(text)
            file_names.append(file.name)
        except Exception as e:
            st.sidebar.error(f"Error reading {file.name}: {e}")

if not texts:
    st.warning("Upload one or more files to begin.")

# Cleaning + Sentiment
if st.button("Run Cleaning & Sentiment Analysis"):
    cleaned_texts = []
    sentiments = []

    for name, raw_text in zip(file_names, texts):
        cleaned = clean_text_regex(
            raw_text,
            remove_emojis=remove_emojis,
            remove_hashtags_mentions=remove_hashtags_mentions,
            remove_numbers=remove_numbers,
            custom_remove=custom_remove,
            lowercase=lowercase,
            remove_punct_keep_single=remove_punct
        )

        cleaned_texts.append(cleaned)
        scores = analyze_document_sentiment(cleaned)
        sentiments.append({
            "Document": name,
            "Positive": scores['pos'],
            "Neutral": scores['neu'],
            "Negative": scores['neg'],
            "Compound": scores['compound'],
            "Overall Sentiment": scores['label']
        })

    st.session_state['cleaned_texts'] = cleaned_texts
    st.session_state['file_names'] = file_names
    st.session_state['sentiments'] = sentiments
    st.success("✅ All files cleaned and analyzed successfully!")

# Display sentiment comparison
if 'sentiments' in st.session_state:
    df = pd.DataFrame(st.session_state['sentiments'])
    st.subheader("📊 Sentiment Comparison Across Documents")
    st.dataframe(df)

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(df['Document'], df['Compound'], color=[
        'green' if x >= 0.05 else 'red' if x <= -0.05 else 'gray' for x in df['Compound']
    ])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Compound Sentiment Score")
    ax.set_title("Document Sentiment Comparison")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Topic Modeling
st.header("Topic Modeling (Combined Documents)")

if st.button("Run Topic Modeling"):
    if 'cleaned_texts' not in st.session_state:
        st.warning("Run cleaning first.")
    else:
        docs = [t for t in st.session_state['cleaned_texts'] if len(t.strip()) > 0]
        try:
            if topic_method.startswith("LDA"):
                model, vec, X = run_lda(docs, n_topics=n_topics, max_iter=max_iter)
                topics = get_top_words(model, vec.get_feature_names_out())
                method = "LDA"
            else:
                model, vec, X = run_nmf_tfidf(docs, n_topics=n_topics, max_iter=max_iter)
                topics = get_top_words(model, vec.get_feature_names_out())
                method = "NMF"

            st.session_state['topics'] = (method, topics)
            st.success(f"{method} topic modeling complete.")
        except Exception as e:
            st.error(f"Topic modeling failed: {e}")

# Display topics
if 'topics' in st.session_state:
    method, topics = st.session_state['topics']
    st.subheader(f"🔍 Topics Discovered ({method})")
    for idx, words in topics:
        st.markdown(f"**Topic {idx+1}:** " + ", ".join(words))

# Cleaned preview
if 'cleaned_texts' in st.session_state:
    st.header("📄 Cleaned Documents Preview")
    for name, text in zip(st.session_state['file_names'], st.session_state['cleaned_texts']):
        with st.expander(name):
            st.text_area(f"Cleaned content of {name}", text[:3000], height=200)

st.markdown("---")
st.caption("NarrativeNexus — Multi-document NLP Platform using regex cleaning, VADER sentiment, and topic modeling (LDA/NMF).")


