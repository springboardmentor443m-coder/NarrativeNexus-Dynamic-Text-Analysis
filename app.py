"""
Streamlit Frontend for AI Narrative Nexus v3.0
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="AI Narrative Nexus v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Initialize session state for storing analysis history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "app"
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def save_analysis(result, files_info):
    """Save analysis to session state history"""
    analysis_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'files': files_info,
        'result': result,
        'id': len(st.session_state.analysis_history)
    }
    st.session_state.analysis_history.append(analysis_entry)
    st.session_state.last_analysis = analysis_entry

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sentiment cards */
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Topic cards */
    .topic-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* History card */
    .history-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Check API connection
api_status, health_data = check_api()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 AI Narrative Nexus")
    st.markdown("**v3.0 Advanced**")
    
    if api_status:
        st.success("✅ Backend Connected")
        if health_data:
            with st.expander("📊 System Info"):
                st.json(health_data)
    else:
        st.error("⚠️ Backend Offline")
        st.code("uvicorn backend.main:app --reload")
    
    st.markdown("---")
    
    # Navigation Menu
    st.markdown("### 📂 Navigation")
    if st.button("🏠 App", use_container_width=True):
        st.session_state.current_page = "app"
        st.rerun()
    
    if st.button("📤 Data Upload", use_container_width=True):
        st.session_state.current_page = "upload"
        st.rerun()
    
    if st.button("📊 Analysis", use_container_width=True):
        st.session_state.current_page = "analysis"
        st.rerun()
    
    if st.button("📈 Visualization", use_container_width=True):
        st.session_state.current_page = "visualization"
        st.rerun()
    
    st.markdown("---")
    
    # Analysis History Count
    st.markdown("### 📚 Analysis History")
    st.info(f"**{len(st.session_state.analysis_history)}** saved reports")
    
    if st.button("🗂️ View History", use_container_width=True):
        st.session_state.current_page = "history"
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Features Info
    st.markdown("### ✨ Features")
    st.markdown("""
    🤖 **RoBERTa Sentiment**
    📊 **BERTopic Modeling**
    💾 **Save Reports**
    📜 **View History**
    ⚡ **Fast Analysis**
    """)

# ============================================================================
# PAGE: APP (Main Analysis Page)
# ============================================================================
if st.session_state.current_page == "app":
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🧠 AI Narrative Nexus v3.0</h1>
            <p>Advanced Text Analysis with Transformers + BERTopic + Groq LLM</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not api_status:
        st.error("⚠️ Backend API is not running. Please start it first.")
        st.stop()
    
    # Analysis Mode Selection
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Full Analysis", use_container_width=True):
            st.session_state.analysis_mode = "full"
    with col2:
        if st.button("😊 Sentiment Only", use_container_width=True):
            st.session_state.analysis_mode = "sentiment"
    with col3:
        if st.button("📊 Topics Only", use_container_width=True):
            st.session_state.analysis_mode = "topics"
    
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "full"
    
    st.markdown("---")
    
    # Full Analysis Mode
    if st.session_state.analysis_mode == "full":
        st.markdown("## 🎯 Full Advanced Analysis")
        st.info("📝 Uses RoBERTa for sentiment + BERTopic for topics + Groq for summaries")
        
        uploaded_files = st.file_uploader(
            "📁 Upload Files for Analysis",
            type=["txt", "csv", "docx"],
            accept_multiple_files=True,
            help="Supported: TXT, CSV, DOCX | Max 200MB per file"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Display file info
            for file in uploaded_files:
                st.text(f"📄 {file.name} ({file.size} bytes)")
            
            if st.button("🚀 Analyze with Advanced Models", type="primary", use_container_width=True):
                with st.spinner("🔄 Running advanced analysis..."):
                    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    response = requests.post(f"{API_URL}/api/analyze/files", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Save to history
                        files_info = [f.name for f in uploaded_files]
                        save_analysis(result, files_info)
                        
                        st.success(f"✅ {result.get('analysis_type', 'Analysis')} Complete!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📁 Files", result['num_files'])
                        with col2:
                            st.metric("📄 Documents", result['num_documents'])
                        with col3:
                            st.metric("🤖 Model", result.get('sentiment', {}).get('model', 'N/A'))
                        with col4:
                            if result.get('topics'):
                                st.metric("📊 Topics", result['topics'].get('num_topics', 0))
                        
                        st.markdown("---")
                        
                        # Results Tabs
                        tabs = st.tabs(["😊 Sentiment", "📝 Summary", "🎯 Topics", "💾 Save & Export"])
                        
                        with tabs[0]:
                            st.markdown("### Sentiment Analysis")
                            sentiment = result.get('sentiment', {})
                            if sentiment:
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=sentiment.get('confidence', 0.5) * 100,
                                        title={'text': f"Sentiment: {sentiment.get('sentiment', 'neutral').upper()}"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "#667eea"},
                                            'steps': [
                                                {'range': [0, 33], 'color': "#ffcccc"},
                                                {'range': [33, 66], 'color': "#ffeecc"},
                                                {'range': [66, 100], 'color': "#ccffcc"}
                                            ]
                                        }
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown(f"**Model:** {sentiment.get('model', 'N/A')}")
                                    st.markdown(f"**Emotion:** {sentiment.get('emotion', 'N/A')}")
                                    st.markdown(f"**Confidence:** {sentiment.get('confidence', 0):.2%}")
                                    
                                    if sentiment.get('reasoning'):
                                        st.info(sentiment['reasoning'])
                                    
                                    if sentiment.get('key_phrases'):
                                        st.markdown("**Key Phrases:**")
                                        st.write(", ".join(sentiment['key_phrases']))
                                
                                with col2:
                                    st.json(sentiment)
                        
                        with tabs[1]:
                            st.markdown("### Text Summary")
                            st.write(result.get('summary', 'No summary available'))
                        
                        with tabs[2]:
                            st.markdown("### Topic Modeling")
                            if result.get('topics') and result['topics'].get('topics'):
                                topics = result['topics']
                                for idx, topic in enumerate(topics['topics']):
                                    with st.expander(f"📌 Topic {idx + 1}: {topic.get('topic_name', 'Unnamed')}"):
                                        st.markdown(f"**Description:** {topic.get('description', 'N/A')}")
                                        if topic.get('enhanced_description'):
                                            st.markdown(f"**Enhanced:** {topic['enhanced_description']}")
                                        st.markdown(f"**Relevance:** {topic.get('relevance_score', 0):.1%}")
                                        st.markdown(f"**Keywords:** {', '.join(topic.get('keywords', []))}")
                            else:
                                st.info("Upload 2+ documents to enable topic modeling")
                        
                        with tabs[3]:
                            st.markdown("### Save & Export Analysis")
                            st.success("✅ Analysis automatically saved to history!")
                            st.info(f"📊 Report ID: {st.session_state.last_analysis['id']}")
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.text}")

# ============================================================================
# PAGE: DATA UPLOAD
# ============================================================================
elif st.session_state.current_page == "upload":
    st.markdown("""
        <div class="main-header">
            <h1>📤 Data Upload</h1>
            <p>Upload and manage your documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "csv", "docx"],
        accept_multiple_files=True,
        key="upload_page"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        for idx, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                st.text(f"{file.size} bytes")
            with col3:
                st.text(file.type)
        
        if st.button("🚀 Analyze These Files", type="primary"):
            st.session_state.current_page = "app"
            st.rerun()

# ============================================================================
# PAGE: ANALYSIS
# ============================================================================
elif st.session_state.current_page == "analysis":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Analysis Dashboard</h1>
            <p>View and manage your analysis results</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_analysis:
        st.markdown("### 📋 Latest Analysis")
        analysis = st.session_state.last_analysis
        
        st.info(f"**Timestamp:** {analysis['timestamp']}")
        st.info(f"**Files:** {', '.join(analysis['files'])}")
        
        result = analysis['result']
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment = result.get('sentiment', {}).get('sentiment', 'N/A')
            st.metric("😊 Sentiment", sentiment.upper())
        with col2:
            confidence = result.get('sentiment', {}).get('confidence', 0)
            st.metric("📊 Confidence", f"{confidence:.1%}")
        with col3:
            topics_count = len(result.get('topics', {}).get('topics', []))
            st.metric("🎯 Topics Found", topics_count)
        
        # Full results
        with st.expander("📄 View Full Results"):
            st.json(result)
    else:
        st.warning("⚠️ No analysis performed yet. Go to App page to analyze documents.")

# ============================================================================
# PAGE: VISUALIZATION
# ============================================================================
elif st.session_state.current_page == "visualization":
    st.markdown("""
        <div class="main-header">
            <h1>📈 Visualization Dashboard</h1>
            <p>Interactive charts and graphs</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.analysis_history:
        st.markdown("### 📊 Analysis Overview")
        
        # Sentiment distribution chart
        sentiments = []
        for analysis in st.session_state.analysis_history:
            sent = analysis['result'].get('sentiment', {}).get('sentiment', 'unknown')
            sentiments.append(sent)
        
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                marker_color=['#38ef7d', '#f45c43', '#4b6cb7']
            )
        ])
        fig.update_layout(
            title="Sentiment Distribution Across All Analyses",
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        st.markdown("### 📅 Analysis Timeline")
        for analysis in reversed(st.session_state.analysis_history[-5:]):
            st.markdown(f"""
                <div class="history-card">
                    <strong>🕒 {analysis['timestamp']}</strong><br>
                    📁 Files: {', '.join(analysis['files'])}<br>
                    😊 Sentiment: {analysis['result'].get('sentiment', {}).get('sentiment', 'N/A').upper()}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data to visualize. Perform some analysis first!")

# ============================================================================
# PAGE: HISTORY
# ============================================================================
elif st.session_state.current_page == "history":
    st.markdown("""
        <div class="main-header">
            <h1>📚 Analysis History</h1>
            <p>View all your previous analysis reports</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.analysis_history:
        st.info(f"📊 Total Reports: {len(st.session_state.analysis_history)}")
        
        for analysis in reversed(st.session_state.analysis_history):
            with st.expander(f"📋 Report #{analysis['id']} - {analysis['timestamp']}"):
                st.markdown(f"**Files:** {', '.join(analysis['files'])}")
                
                result = analysis['result']
                
                # Quick metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment", result.get('sentiment', {}).get('sentiment', 'N/A').upper())
                with col2:
                    st.metric("Confidence", f"{result.get('sentiment', {}).get('confidence', 0):.1%}")
                with col3:
                    st.metric("Topics", len(result.get('topics', {}).get('topics', [])))
                
                # View full report
                if st.button(f"📄 View Full Report", key=f"view_{analysis['id']}"):
                    st.session_state.last_analysis = analysis
                    st.session_state.current_page = "analysis"
                    st.rerun()
                
                # Export option
                st.download_button(
                    label="💾 Download Report (JSON)",
                    data=json.dumps(result, indent=2),
                    file_name=f"analysis_report_{analysis['id']}.json",
                    mime="application/json",
                    key=f"download_{analysis['id']}"
                )
        
        # Clear history option
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.analysis_history = []
            st.session_state.last_analysis = None
            st.success("✅ History cleared!")
            st.rerun()
    else:
        st.warning("⚠️ No analysis history yet. Start analyzing documents!")
        if st.button("🚀 Go to Analysis"):
            st.session_state.current_page = "app"
            st.rerun()
