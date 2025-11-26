import streamlit as st
import pandas as pd
import re
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# --- 1. Page Configuration & Theming ---
st.set_page_config(
    page_title="NarrativeNexus | Neural Insights",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (Dark Mode & Decorations) ---
st.markdown("""
    <style>
    /* Global Background: Deep Midnight Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Neon Header Typography */
    h1, h2, h3 {
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 15px rgba(0, 198, 255, 0.3);
    }
    
    /* Glassmorphism Containers */
    div[data-testid="stExpander"], div[data-testid="stContainer"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        padding: 15px;
        margin-bottom: 15px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0,0,0,0.3);
        border-radius: 5px;
        color: #fff;
        border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 114, 255, 0.3) !important;
        border: 1px solid #00c6ff !important;
    }
    
    /* Metrics & Stats */
    div[data-testid="stMetric"] {
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00c6ff;
    }
    label { color: #cfcfcf !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Caching NLTK & Models ---
@st.cache_resource
def load_resources():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    # Lightweight summarizer for speed
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_resources()
lemmatizer = WordNetLemmatizer()

# --- 4. Logic from Notebook (Gensim LDA) ---

def preprocess_text(text):
    """
    Applies the exact preprocessing steps from your notebook:
    1. Tokenize & remove accents (simple_preprocess)
    2. Remove Stopwords
    3. Filter short words (len > 3)
    4. Lemmatize
    """
    tokens = simple_preprocess(text, deacc=True)
    clean_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in STOPWORDS and len(token) > 3
    ]
    return clean_tokens

def train_gensim_lda(clean_docs, num_topics=5):
    """
    Trains LDA model using Gensim logic.
    """
    # 1. Create Dictionary
    id2word = corpora.Dictionary(clean_docs)
    
    # Note: We skip 'filter_extremes' here because users usually upload 
    # small datasets (1-5 files). Filtering <15 docs would delete everything.
    
    # 2. Create Corpus
    corpus = [id2word.doc2bow(text) for text in clean_docs]
    
    # 3. Train Model (Logic from notebook cell 4)
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=1000,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Calculate Coherence (from notebook cell 5)
    coherence_model_lda = CoherenceModel(
        model=lda_model, 
        texts=clean_docs, 
        dictionary=id2word, 
        coherence='c_v'
    )
    coherence_score = coherence_model_lda.get_coherence()
    
    return lda_model, corpus, id2word, coherence_score

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --- 5. Main App Layout ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    num_topics = st.slider("Number of Topics", 2, 10, 4, help="How many themes to extract from the corpus.")
    st.info("Upload .txt files to activate the pipeline.")
    st.markdown("---")
    st.caption("Logic: Gensim LDA | UI: Streamlit")

# Main Content
st.title("🧠 NarrativeNexus AI")
st.markdown("### Advanced Document Intelligence Pipeline")
st.markdown("---")

# File Upload
uploaded_files = st.file_uploader("📂 Upload Corpus", type="txt", accept_multiple_files=True)

if uploaded_files:
    # --- Data Processing ---
    raw_data = []
    processed_data = [] # For LDA
    filenames = []

    for file in uploaded_files:
        text = file.getvalue().decode("utf-8")
        raw_data.append(text)
        filenames.append(file.name)
        # Apply notebook preprocessing
        processed_data.append(preprocess_text(text))

    # --- TAB 1: GLOBAL TOPIC MODELING ---
    st.subheader("🔍 Corpus-Wide Topic Modeling")
    
    with st.spinner("Training Gensim Neural Network..."):
        try:
            lda_model, corpus, id2word, coherence = train_gensim_lda(processed_data, num_topics)
            
            # Display Coherence Score
            st.metric("Model Coherence Score", f"{coherence:.4f}", help="Higher is better. Indicates semantic consistency.")

            # Display Topics visually
            topic_cols = st.columns(num_topics)
            
            # Iterate through topics found by Gensim
            for idx, topic in lda_model.print_topics(-1):
                # Parse the Gensim string: '0.05*"word" + 0.03*"word2"'
                clean_topic = topic.split('+')
                words = []
                weights = []
                for term in clean_topic:
                    weight, word = term.split('*')
                    words.append(word.strip().replace('"', ''))
                    weights.append(float(weight))
                
                # Render Chart in Column
                with topic_cols[idx % num_topics]:
                    with st.container():
                        st.markdown(f"**Topic {idx+1}**")
                        fig = px.bar(x=weights[:5], y=words[:5], orientation='h',
                                     color=weights[:5], color_continuous_scale='Viridis')
                        fig.update_layout(
                            showlegend=False, 
                            height=200, 
                            margin=dict(l=0,r=0,t=0,b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Topic Modeling Error: {e}. Try uploading more text data.")

    st.markdown("---")

    # --- TAB 2: INDIVIDUAL DOCUMENT ANALYSIS ---
    st.subheader("📄 Document Insights")
    
    doc_tabs = st.tabs(filenames)

    for i, tab in enumerate(doc_tabs):
        with tab:
            col1, col2 = st.columns([1.5, 1])

            # --- LEFT COLUMN: Summary & Text ---
            with col1:
                st.markdown("#### 📝 Abstractive Summary")
                with st.spinner("Generating Summary..."):
                    try:
                        # Truncate to 1500 chars for performance
                        summary = summarizer(raw_data[i][:1500], max_length=130, min_length=30, do_sample=False)
                        st.success(summary[0]['summary_text'])
                    except:
                        st.warning("Text too short to summarize.")

                with st.expander("📜 View Full Source Text"):
                    st.write(raw_data[i])

            # --- RIGHT COLUMN: Sentiment & Visuals ---
            with col2:
                st.markdown("#### 📊 Sentiment Analysis")
                score = get_sentiment(raw_data[i])
                
                # Gauge Chart for Sentiment
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Polarity"},
                    gauge = {
                        'axis': {'range': [-1, 1], 'tickcolor': "white"},
                        'bar': {'color': "#00c6ff"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-1, -0.1], 'color': 'rgba(255, 0, 0, 0.3)'},
                            {'range': [-0.1, 0.1], 'color': 'rgba(255, 255, 255, 0.1)'},
                            {'range': [0.1, 1], 'color': 'rgba(0, 255, 0, 0.3)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("#### ☁️ Word Cloud")
                try:
                    # Dark theme word cloud
                    wc = WordCloud(background_color='#0f2027', width=400, height=200, colormap='cool').generate(raw_data[i])
                    fig_wc, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    # Make matplotlib background transparent
                    fig_wc.patch.set_alpha(0)
                    st.pyplot(fig_wc)
                except:
                    st.caption("Not enough words for cloud.")

else:
    # Empty State
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #6c757d;'>
        <h2>🚀 Ready to Analyze</h2>
        <p>Please upload your dataset text files in the sidebar to start the Gensim engine.</p>
    </div>
    """, unsafe_allow_html=True)
