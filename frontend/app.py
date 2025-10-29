# Streamlit app for AI text analysis
# Built this to analyze sentiment, topics, and generate summaries
# Uses RoBERTa, BERTopic, and Groq LLM under the hood

import streamlit as st
import requests
import json
import plotly.graph_objects as go
from datetime import datetime

# Basic setup
st.set_page_config(
    page_title="AI Narrative Nexus v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Set up session state to store results between page refreshes
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None

def check_api():
    # Quick check to see if backend is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def save_analysis(result, files_info):
    # Store analysis in session for later viewing
    analysis_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'files': files_info,
        'result': result,
        'id': len(st.session_state.analysis_history)
    }
    st.session_state.analysis_history.append(analysis_entry)
    st.session_state.last_analysis = analysis_entry

# Some basic styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Check if backend API is up
api_status, health_data = check_api()

# Sidebar stuff
with st.sidebar:
    st.markdown("### 🧠 AI Narrative Nexus")
    st.markdown("**v3.0 Advanced**")
    st.markdown("---")
    
    # Show connection status
    if api_status:
        st.success("✅ Backend Connected")
        if health_data:
            with st.expander("📊 System Info"):
                st.json(health_data)
    else:
        st.error("⚠️ Backend Offline")
        st.code("uvicorn backend.main:app --reload")
    
    st.markdown("---")
    
    # History section
    st.markdown("### 📚 Analysis History")
    st.info(f"**{len(st.session_state.analysis_history)}** saved reports")
    
    if st.button("🗂️ View History", use_container_width=True):
        st.session_state.show_history = True
    
    st.markdown("---")
    
    # Feature list
    st.markdown("### ✨ Features")
    st.markdown("""
    🤖 **RoBERTa Sentiment**
    📊 **BERTopic Modeling**
    💾 **Save Reports**
    📜 **View History**
    ⚡ **Fast Analysis**
    """)

# Main page header
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">🧠 AI Narrative Nexus v3.0</h1>
        <p style="color: white; margin: 10px 0 0 0;">Advanced Text Analysis with Transformers + BERTopic + Groq LLM</p>
    </div>
""", unsafe_allow_html=True)

# Check if user wants to see history
if 'show_history' in st.session_state and st.session_state.show_history:
    st.header("📚 Analysis History")
    
    if st.session_state.analysis_history:
        # Show last 10 analyses
        for analysis in reversed(st.session_state.analysis_history[-10:]):
            with st.expander(f"📋 Report #{analysis['id']} - {analysis['timestamp']}"):
                st.markdown(f"**Files:** {', '.join(analysis['files'])}")
                st.json(analysis['result'])
    else:
        st.info("No analysis history yet")
    
    if st.button("← Back to Analysis"):
        st.session_state.show_history = False
        st.rerun()

# Main analysis interface
else:
    st.markdown("## 🎯 Full Advanced Analysis")
    st.info("📝 Uses RoBERTa for sentiment + BERTopic for topics + Groq for summaries")
    
    st.write("")
    
    # Two ways to input data
    tab1, tab2 = st.tabs(["📁 Upload Files", "📝 Paste Text"])
    
    # File upload tab
    with tab1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["txt", "csv", "docx"],
            accept_multiple_files=True,
            help="Supported: TXT, CSV, DOCX | Max 200MB per file"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show what was uploaded
            for file in uploaded_files:
                st.text(f"📄 {file.name} ({file.size} bytes)")
            
            if st.button("🚀 Analyze Now", type="primary", use_container_width=True, key="analyze_files"):
                with st.spinner("🔄 Running advanced analysis..."):
                    # Prepare files for upload
                    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    response = requests.post(f"{API_URL}/api/analyze/files", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        files_info = [f.name for f in uploaded_files]
                        save_analysis(result, files_info)
                        
                        st.success(f"✅ {result.get('analysis_type', 'Analysis')} Complete!")
                        
                        # Show key metrics
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
                        
                        # Results organized in tabs
                        tabs = st.tabs(["😊 Sentiment", "📝 Summary", "🎯 Topics", "📊 Raw Data"])
                        
                        # Sentiment tab
                        with tabs[0]:
                            st.markdown("### Sentiment Analysis")
                            sentiment = result.get('sentiment', {})
                            
                            if sentiment:
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    sent_label = sentiment.get('sentiment', 'neutral')
                                    confidence = sentiment.get('confidence', 0.5)
                                    model_used = sentiment.get('model', 'Unknown')
                                    
                                    # Create gauge chart
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=confidence * 100,
                                        title={'text': f"Sentiment: {sent_label.upper()}"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "#667eea"},
                                            'steps': [
                                                {'range': [0, 33], 'color': "#ffcccc"},
                                                {'range': [33, 66], 'color': "#ffeecc"},
                                                {'range': [66, 100], 'color': "#ccffcc"}
                                            ],
                                        }
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown(f"**Model:** {model_used}")
                                    st.markdown(f"**Emotion:** {sentiment.get('emotion', 'N/A')}")
                                    st.markdown(f"**Confidence:** {confidence:.2%}")
                                    
                                    if sentiment.get('reasoning'):
                                        st.info(sentiment['reasoning'])
                                    
                                    if sentiment.get('key_phrases'):
                                        st.markdown("**Key Phrases:**")
                                        st.write(", ".join(sentiment['key_phrases']))
                                
                                with col2:
                                    st.json(sentiment)
                        
                        # Summary tab
                        with tabs[1]:
                            st.markdown("### Text Summary")
                            st.write(result.get('summary', 'No summary available'))
                        
                        # Topics tab
                        with tabs[2]:
                            st.markdown("### Topic Modeling")
                            if result.get('topics') and result['topics'].get('topics'):
                                topics = result['topics']
                                
                                for idx, topic in enumerate(topics['topics']):
                                    with st.expander(f"📌 Topic {idx + 1}: {topic.get('topic_name', 'Unnamed')}"):
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.markdown(f"**Description:** {topic.get('description', 'N/A')}")
                                            if topic.get('enhanced_description'):
                                                st.markdown(f"**Enhanced:** {topic['enhanced_description']}")
                                            st.markdown("**Keywords:**")
                                            keywords = topic.get('keywords', [])
                                            st.write(", ".join(keywords[:10]))
                                        
                                        with col2:
                                            st.metric("Relevance", f"{topic.get('relevance_score', 0):.1%}")
                                            st.metric("Docs", topic.get('document_count', 0))
                            else:
                                st.info("Upload 2+ documents for topic modeling")
                        
                        # Raw data tab
                        with tabs[3]:
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.text}")
    
    # Text paste tab
    with tab2:
        st.subheader("📝 Paste Text")
        text_input = st.text_area(
            "Paste your text here:",
            height=250,
            placeholder="Enter text for analysis..."
        )
        
        if text_input.strip():
            st.success("✅ Text ready for analysis")
            
            if st.button("🚀 Analyze Now", type="primary", use_container_width=True, key="analyze_paste"):
                with st.spinner("🔄 Running analysis..."):
                    # Convert to file format for API
                    text_bytes = text_input.encode('utf-8')
                    files = [("files", ("pasted_text.txt", text_bytes, "text/plain"))]
                    
                    response = requests.post(f"{API_URL}/api/analyze/files", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        save_analysis(result, ["Pasted Text"])
                        
                        st.success(f"✅ {result.get('analysis_type', 'Analysis')} Complete!")
                        
                        # Quick metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📁 Files", 1)
                        with col2:
                            st.metric("📄 Documents", result.get('num_documents', 1))
                        with col3:
                            st.metric("🤖 Model", result.get('sentiment', {}).get('model', 'N/A'))
                        with col4:
                            st.metric("📊 Status", "Complete")
                        
                        st.markdown("---")
                        
                        # Show results
                        tabs = st.tabs(["😊 Sentiment", "📝 Summary", "🎯 Topics", "📊 Raw Data"])
                        
                        with tabs[0]:
                            st.markdown("### Sentiment Analysis")
                            sentiment = result.get('sentiment', {})
                            
                            if sentiment:
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    sent_label = sentiment.get('sentiment', 'neutral')
                                    confidence = sentiment.get('confidence', 0.5)
                                    
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=confidence * 100,
                                        title={'text': f"Sentiment: {sent_label.upper()}"},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "#667eea"},
                                            'steps': [
                                                {'range': [0, 33], 'color': "#ffcccc"},
                                                {'range': [33, 66], 'color': "#ffeecc"},
                                                {'range': [66, 100], 'color': "#ccffcc"}
                                            ],
                                        }
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown(f"**Model:** {sentiment.get('model', 'N/A')}")
                                    st.markdown(f"**Emotion:** {sentiment.get('emotion', 'N/A')}")
                                    st.markdown(f"**Confidence:** {confidence:.2%}")
                                    
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
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.markdown(f"**Description:** {topic.get('description', 'N/A')}")
                                            if topic.get('enhanced_description'):
                                                st.markdown(f"**Enhanced:** {topic['enhanced_description']}")
                                            st.markdown("**Keywords:**")
                                            keywords = topic.get('keywords', [])
                                            st.write(", ".join(keywords[:10]))
                                        
                                        with col2:
                                            st.metric("Relevance", f"{topic.get('relevance_score', 0):.1%}")
                                            st.metric("Docs", topic.get('document_count', 0))
                            else:
                                st.info("Topic modeling info not available")
                        
                        with tabs[3]:
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.text}")

st.markdown("---")
st.caption("📊 AI Narrative Nexus v3.0 | Powered by Streamlit, RoBERTa, BERTopic, and Groq LLM")
