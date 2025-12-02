import streamlit as st
import requests
import json
import os
import pandas as pd
import time

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
TOPIC_MAP_PATH = os.path.join("backend", "models", "bertopic_20newsgroups", "topic_name_mapping.json")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nexus Intelligence",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- 🎨 ADVANCED CSS STYLING (The "Web App" Look) ---
st.markdown("""
    <style>
        /* 1. GOOGLE FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');
        
        /* 2. GLOBAL TEXT STYLES */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E2E8F0;
        }
        
        h1, h2, h3, .nav-logo {
            font-family: 'Rajdhani', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* 3. BACKGROUND IMAGE (The "Wow" Factor) */
        /* We use a dark overlay + an image URL for a professional look */
        .stApp {
            background: 
                linear-gradient(to bottom, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)),
                url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }

        /* 4. REMOVE STREAMLIT BRANDING */
        #MainMenu, footer, header {visibility: hidden;}

        /* 5. GLASSMORPHISM CONTAINERS */
        /* This targets st.container(border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            padding: 24px;
            transition: border 0.3s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* 6. PROFESSIONAL BUTTONS (Gradients & Hover Effects) */
        div.stButton > button {
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            width: 100%;
            position: relative;
            overflow: hidden;
        }
        
        /* Button Hover Animation */
        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        }
        div.stButton > button:active {
            transform: translateY(1px);
        }
        
        /* Disabled Button Style */
        div.stButton > button:disabled {
            background: #334155;
            color: #94A3B8;
            box-shadow: none;
            cursor: not-allowed;
        }

        /* Secondary Button (Ghost Style) */
        button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            box-shadow: none !important;
        }
        button[kind="secondary"]:hover {
            border-color: #60A5FA !important;
            color: #60A5FA !important;
        }

        /* 7. CUSTOM TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            padding: 0 20px;
            background-color: transparent;
            color: #94A3B8;
            font-weight: 500;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #E2E8F0;
            background-color: rgba(255,255,255,0.05);
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* 8. ALERTS & BADGES */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            backdrop-filter: blur(4px);
        }
        .status-online {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.2);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# --- LOGIC ---

def check_health():
    try:
        requests.get(f"{API_URL}/health", timeout=0.5)
        return True
    except:
        return False

def fetch_history():
    try:
        return requests.get(f"{API_URL}/api/history").json()
    except:
        return {"results": []}

def load_topics():
    if os.path.exists(TOPIC_MAP_PATH):
        try:
            with open(TOPIC_MAP_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "view" not in st.session_state:
    st.session_state.view = "analyzer" 

# --- HEADER ROW ---
col_brand, col_status, col_nav = st.columns([5, 2, 1.5])

with col_brand:
    # Using a custom styled H1
    st.markdown("""
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:2.5rem;">⚡</span>
            <div>
                <h1 style="margin:0; padding:0; font-size:1.8rem; background: linear-gradient(to right, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">NEXUS INTELLIGENCE</h1>
                <p style="margin:0; font-size:0.9rem; color:#64748B;">Analysis Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_status:
    is_up = check_health()
    status_class = "status-online" if is_up else "status-offline"
    status_text = "SYSTEM OPERATIONAL" if is_up else "SYSTEM OFFLINE"
    st.markdown(f"""
        <div style="text-align:right; padding-top:15px;">
            <span class="status-badge {status_class}">
                ● &nbsp; {status_text}
            </span>
        </div>
    """, unsafe_allow_html=True)

with col_nav:
    st.write("") 
    if st.session_state.view == "analyzer":
        if st.button("📜 History", type="secondary"):
            st.session_state.view = "history"
            st.rerun()
    else:
        if st.button("🏠 Dashboard", type="secondary"):
            st.session_state.view = "analyzer"
            st.rerun()

st.write("") # Spacer

# =========================================
# 🕰️ HISTORY VIEW
# =========================================
if st.session_state.view == "history":
    st.subheader("🗂️ Analysis Archives")
    
    with st.container(border=True):
        history = fetch_history()
        items = history.get("results", [])

        if not items:
            st.info("No analysis records found in database.")
        else:
            # Header
            c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
            c1.markdown("**ID**")
            c2.markdown("**Date**")
            c3.markdown("**Files**")
            c4.markdown("**Load**")
            st.divider()

            for item in items:
                c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
                c1.write(f"#{item['id']}")
                c2.write(item['timestamp'].replace('T', ' ').split('.')[0])
                c3.write(str(item['num_files']))
                if c4.button("Open", key=f"load_{item['id']}", use_container_width=True):
                    try:
                        full_data = requests.get(f"{API_URL}/api/history/{item['id']}").json()
                        st.session_state.analysis_result = full_data
                        st.session_state.view = "analyzer"
                        st.rerun()
                    except:
                        st.error("Error loading.")
                st.divider()

# =========================================
# 🚀 ANALYZER VIEW
# =========================================
else:
    # --- 1. INPUT CARD ---
    with st.container(border=True):
        st.markdown("### 📥 Data Ingestion")
        
        tabs = st.tabs(["📄 File Upload", "✍️ Direct Input", "ℹ️ Capability Map"])

        # TAB 1: FILES
        with tabs[0]:
            uploaded_files = st.file_uploader(
                "Drop files here",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if st.button("✨ Run Analysis", type="primary", disabled=not uploaded_files):
                with st.spinner("Processing Pipeline..."):
                    files_payload = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in uploaded_files]
                    try:
                        res = requests.post(f"{API_URL}/api/analyze/files", files=files_payload, timeout=300)
                        if res.status_code == 200:
                            st.session_state.analysis_result = res.json()
                            st.rerun()
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(str(e))

        # TAB 2: TEXT
        with tabs[1]:
            txt_input = st.text_area("Paste text", height=150, label_visibility="collapsed", placeholder="Enter text to analyze...")
            if st.button("✨ Analyze Text", type="primary", disabled=not txt_input):
                with st.spinner("Analyzing..."):
                    try:
                        payload = [("files", ("input.txt", txt_input.encode("utf-8"), "text/plain"))]
                        res = requests.post(f"{API_URL}/api/analyze/files", files=payload, timeout=60)
                        if res.status_code == 200:
                            st.session_state.analysis_result = res.json()
                            st.rerun()
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(str(e))

        # TAB 3: LEGEND
        with tabs[2]:
            tmap = load_topics()
            if tmap:
                st.caption("The system can automatically classify documents into these BERTopic categories:")
                html_tags = ""
                for k, v in tmap.items():
                    if k != "-1":
                        html_tags += f"""
                        <span style="
                            display:inline-block; 
                            background:rgba(59, 130, 246, 0.15); 
                            border:1px solid rgba(59, 130, 246, 0.3); 
                            padding:4px 10px; 
                            border-radius:6px; 
                            margin:4px; 
                            font-size:0.85em; 
                            font-weight:500;
                            color:#E2E8F0;">
                            <span style="color:#60A5FA; font-weight:bold;">#{k}</span> {v}
                        </span>
                        """
                st.markdown(html_tags, unsafe_allow_html=True)
            else:
                st.warning("Topic map not found on server.")

    # --- 2. RESULTS CARD ---
    if st.session_state.analysis_result:
        data = st.session_state.analysis_result
        st.write("") # Spacer

        # Metrics Row (Custom styled HTML for better look)
        st.markdown(f"""
            <div style="display:flex; gap:20px; margin-bottom:20px;">
                <div style="flex:1; background:rgba(30,41,59,0.6); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                    <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">Files Processed</div>
                    <div style="color:#F8FAFC; font-size:24px; font-weight:700;">{data.get('num_files', 0)}</div>
                </div>
                <div style="flex:1; background:rgba(30,41,59,0.6); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1); text-align:center;">
                    <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">Documents Segments</div>
                    <div style="color:#F8FAFC; font-size:24px; font-weight:700;">{data.get('num_documents', 0)}</div>
                </div>
                <div style="flex:1; background:rgba(16,185,129,0.1); padding:15px; border-radius:10px; border:1px solid rgba(16,185,129,0.2); text-align:center;">
                    <div style="color:#34D399; font-size:12px; text-transform:uppercase;">Pipeline Status</div>
                    <div style="color:#34D399; font-size:24px; font-weight:700;">Completed</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Main Results Container
        with st.container(border=True):
            r_tabs = st.tabs(["🏷️ Topic Analysis", "📄 Summary", "📊 Sentiment", "💾 JSON Data"])
            file_results = data.get("per_file", [])

            # 1. TOPICS
            with r_tabs[0]:
                if not file_results:
                    st.warning("No data.")
                else:
                    for item in file_results:
                        t_info = item.get("assigned_topic", {})
                        t_name = t_info.get("topic_name", "Unknown")
                        t_id = t_info.get("topic_id", -1)
                        t_conf = t_info.get("similarity", 0.0)
                        
                        # Styled Result Row
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, rgba(30,41,59,0.4) 0%, rgba(15,23,42,0.4) 100%);
                            border: 1px solid rgba(255,255,255,0.05);
                            border-left: 4px solid {('#10B981' if t_id != -1 else '#64748B')}; 
                            padding: 20px; 
                            margin-bottom: 15px; 
                            border-radius: 0 12px 12px 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <div style="font-size:0.85em; color:#94A3B8; margin-bottom:4px;">DOCUMENT NAME</div>
                                    <div style="font-size:1.1em; font-weight:600; color:#F1F5F9;">{item.get('file_name')}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:0.85em; color:#94A3B8; margin-bottom:4px;">DETECTED TOPIC</div>
                                    <div style="font-size:1.2em; font-weight:700; color:#60A5FA;">{t_name}</div>
                                    <div style="font-size:0.8em; color:#94A3B8;">ID: {t_id if t_id != -1 else 'N/A'} • Confidence: {t_conf:.1%}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            # 2. SUMMARIES
            with r_tabs[1]:
                for item in file_results:
                    with st.expander(f"📄 {item.get('file_name')}", expanded=False):
                        st.markdown(item.get("summary", "_No summary generated._"))

            # 3. SENTIMENT
            with r_tabs[2]:
                if file_results:
                    sent_rows = []
                    for item in file_results:
                        s = item.get("sentiment", {})
                        conf = s.get("confidence", 0.0)
                        label = s.get("sentiment", "neutral").capitalize()
                        
                        indicator = "⚪"
                        if label == "Positive": indicator = "🟢"
                        elif label == "Negative": indicator = "🔴"
                        
                        sent_rows.append({
                            "File": item.get("file_name"),
                            "Indicator": indicator,
                            "Sentiment": label,
                            "Confidence Score": f"{conf:.4f}"
                        })
                    st.dataframe(pd.DataFrame(sent_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No sentiment data available.")

            # 4. RAW DATA
            with r_tabs[3]:
                st.json(data)