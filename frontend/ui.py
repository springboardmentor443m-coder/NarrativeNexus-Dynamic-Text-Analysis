import streamlit as st
import requests
import os

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="DyNarrative – Text Analysis", layout="wide")

st.markdown("""
<style>
    /* Base theme */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #151b28 100%);
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #151b28 100%);
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0e17;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #00d9ff;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    p, span, div {
        color: #cbd5e1;
    }
    
    /* Tab styling - enhanced */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(37, 45, 61, 0.3);
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid rgba(0, 217, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(37, 45, 61, 0.6) 0%, rgba(58, 69, 86, 0.6) 100%);
        border-radius: 6px;
        padding: 0.8rem 1.5rem;
        color: #cbd5e1;
        font-weight: 600;
        border: 1px solid rgba(58, 69, 86, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(0, 217, 255, 0.3);
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
    }
    
    .stTabs [aria-selected="true"] [data-baseweb="tab"] {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        color: #0a0e17;
        border: 1px solid #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
    }
    
    /* Button styling - MATCHING TAB STYLE */
    .stButton > button {
        background: linear-gradient(135deg, rgba(37, 45, 61, 0.6) 0%, rgba(58, 69, 86, 0.6) 100%);
        color: #cbd5e1;
        border: 1px solid rgba(58, 69, 86, 0.8);
        border-radius: 6px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.2), transparent);
        transition: left 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton > button:hover {
        border-color: rgba(0, 217, 255, 0.3);
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        color: #0a0e17;
        border: 1px solid #00d9ff;
    }
    
    /* Input fields */
    .stTextArea textarea {
        background-color: rgba(15, 20, 25, 0.8);
        color: #00d9ff;
        border: 2px solid rgba(0, 217, 255, 0.2);
        border-radius: 6px;
        transition: all 0.3s;
        font-family: 'Monaco', 'Courier New', monospace;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d9ff;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3);
    }
    
    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(37, 45, 61, 0.5) 0%, rgba(58, 69, 86, 0.3) 100%);
        border: 2px dashed rgba(0, 217, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Messages */
    .stSuccess {
        background: rgba(6, 95, 70, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 6px;
    }
    
    .stError {
        background: rgba(127, 29, 29, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
    }
    
    .stInfo {
        background: rgba(12, 74, 110, 0.3);
        border-left: 4px solid #00d9ff;
        border-radius: 6px;
    }
    
    .stWarning {
        background: rgba(120, 53, 15, 0.3);
        border-left: 4px solid #fbbf24;
        border-radius: 6px;
    }
    
    /* Divider */
    hr {
        background: linear-gradient(90deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.3) 50%, rgba(0, 217, 255, 0.1) 100%);
        border: none;
        height: 1px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(37, 45, 61, 0.6) 0%, rgba(58, 69, 86, 0.4) 100%);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        transition: all 0.3s;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(0, 217, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.2);
    }
    
    /* JSON output */
    .stJsonData {
        background: rgba(15, 20, 25, 0.8);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

CUSTOM_TOPIC_NAMES = {
    0:  "Narrative / story-style text",
    1:  "General explanations and statements",
    2:  "Cars and transportation discussions",
    3:  "Israel and Middle East politics",
    4:  "X Window System / GUI widgets",
    5:  "Cryptography and Clipper key debate",
    6:  "Space exploration and astronomy",
    7:  "Image density and P1–P3 graphics",
    8:  "Guns, weapons and gun control",
    9:  "Video cards and computer screens",
    10: "Basic descriptive sentences / misc text",
    11: "MS-DOS, Windows and disk files",
    12: "DOS software versions and quality reviews",
    13: "Personal experiences and anecdotes",
    14: "Operating systems and software versions (edu/comp)",
    15: "US federal budget and Clinton-era politics",
    16: "Waco siege fire and David Koresh",
    17: "Food, MSG and vitamin B6 health effects",
    18: "CPU speed, 68040 processor and FPU performance",
    19: "Sexuality and homosexuality discussions",
    20: "Disk drives, cables and storage hardware",
    21: "Stereo systems, speakers and audio channels",
    22: "Printers, fonts and TrueType printing",
    23: "SCSI / IDE disk controllers and interface chips",
    24: "Water, steam and heat transfer (thermodynamics)",
    25: "Computer memory (SIMMs, RAM capacity and access)",
    26: "Serial ports, modems and COM/IRQ settings",
    27: "Moral vs immoral behaviour and ethics",
    28: "Health risks of smokeless tobacco and E. coli",
    29: "Illegal drugs and substance abuse (cocaine, LSD)",
    -1: "Outlier / uncategorized documents",
}

st.title("🧠 DyNarrative - The Dynamic Text Analysis Platform")
st.markdown(
        "**Upload a `.txt` file and unlock AI-powered insights.**"
)
st.divider()

st.markdown(
    "Use the tabs below to run **Preprocessing**, **Summarization**, **Sentiment Analysis**, and **Topic Modelling** independently."
)

uploaded_file = st.file_uploader("📁 Upload a .txt file", type=["txt"])

shared_file_path = None
if uploaded_file is not None:
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)
    shared_file_path = os.path.join(save_dir, uploaded_file.name)
    with open(shared_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File uploaded: `{uploaded_file.name}`")

st.divider()

tab_pre, tab_sum, tab_sent, tab_topic = st.tabs(
    ["🧹 Preprocess", "📝 Summarize", "😊 Sentiment", "🏷️ Topic"]
)


with tab_pre:
    st.subheader("🧹 Text Preprocessing")
    pre_result = None

    if st.button("Run Preprocessing", key="btn_pre"):
        if not shared_file_path:
            st.error("Please upload a .txt file above first.")
        else:
            try:
                st.info("Calling /preprocess on shared file...")
                resp = requests.post(
                    f"{API_BASE}/preprocess",
                    params={"file_path": shared_file_path},
                    timeout=120,
                )
                resp.raise_for_status()
                pre_result = resp.json().get("preprocessing_result")
                st.success("Preprocessing completed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if pre_result:
        st.markdown("### Preprocessing Result")
        st.json(pre_result)


with tab_sum:
    st.subheader("📝 Summarization")
    summary = None

    if st.button("Run Summarization", key="btn_sum"):
        if not shared_file_path:
            st.error("Please upload a .txt file above first.")
        else:
            try:
                st.info("Calling /summarize on shared file...")
                resp = requests.post(
                    f"{API_BASE}/summarize",
                    json={"file_path": shared_file_path},
                    timeout=120,
                )
                resp.raise_for_status()
                summary = resp.json().get("summary")
                st.success("Summarization completed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if summary:
        st.markdown("### Summarization Result")
        st.text_area("Summary", value=summary, height=260, disabled=True)


with tab_sent:
    st.subheader("😊 Sentiment Analysis")
    sent_result = None

    if st.button("Run Sentiment Analysis", key="btn_sent"):
        if not shared_file_path:
            st.error("Please upload a .txt file above first.")
        else:
            try:
                st.info("Calling /sentiment on shared file...")
                resp = requests.post(
                    f"{API_BASE}/sentiment",
                    params={"file_path": shared_file_path},
                    timeout=120,
                )
                resp.raise_for_status()
                sent_result = resp.json().get("sentiment_result")
                st.success("Sentiment analysis completed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if sent_result:
        st.markdown("### Sentiment Analysis Result")
        st.json(sent_result)

with tab_topic:
    st.subheader("🏷️ Topic Modelling")

    topic_text = ""
    topic_data = None

    if shared_file_path and os.path.exists(shared_file_path):
        try:
            resp_pre = requests.post(
                f"{API_BASE}/preprocess",
                params={"file_path": shared_file_path},
                timeout=120,
            )
            resp_pre.raise_for_status()
            pre_result = resp_pre.json().get("preprocessing_result")
            topic_text = pre_result.get("processed_text", "")
        except:
            with open(shared_file_path, "r", encoding="utf-8", errors="ignore") as f:
                topic_text = f.read()
        
        st.info("Review or edit the text that will be sent to /infer-topics.")
        topic_text = st.text_area("Text for Topic Modelling", value=topic_text, height=260, key=f"topic_text_{shared_file_path}")
    else:
        st.warning("Upload a .txt file above to enable topic modelling.")

    if st.button("Run Topic Detection", key="btn_topic"):
        if not topic_text or len(topic_text.strip()) < 20:
            st.error("Text is too short. Please upload/provide richer content (20+ characters).")
        else:
            try:
                st.info("Calling /infer-topics on shared file text...")
                resp = requests.post(
                    f"{API_BASE}/infer-topics",
                    params={"text": topic_text},
                    timeout=120,
                )
                resp.raise_for_status()
                topic_data = resp.json()
                st.success("Topic Detection completed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if topic_data:
        st.markdown("### Topic Detection Result")
        
        detected_topic_id = topic_data.get('topic_id')
        detected_topic_name = CUSTOM_TOPIC_NAMES.get(detected_topic_id, "Unknown")
        confidence = topic_data.get('confidence')
        keywords = topic_data.get('top_keywords', [])
        
        st.write(f"**Topic ID:** {detected_topic_id}")
        st.write(f"**Topic Name:** {detected_topic_name}")
        st.write(f"**Confidence:** {confidence}")
        st.write("**Top Keywords:** " + ", ".join(keywords))
        
        st.divider()
        st.markdown("### 📚 All Available Topics Reference")
        
        cols = st.columns(2)
        col_idx = 0
        for topic_id in sorted(CUSTOM_TOPIC_NAMES.keys()):
            topic_name = CUSTOM_TOPIC_NAMES[topic_id]
            is_matched = topic_id == detected_topic_id
            
            with cols[col_idx % 2]:
                if is_matched:
                    st.success(f"✓ **#{topic_id}** — {topic_name}")
                else:
                    st.info(f"**#{topic_id}** — {topic_name}")
            col_idx += 1
