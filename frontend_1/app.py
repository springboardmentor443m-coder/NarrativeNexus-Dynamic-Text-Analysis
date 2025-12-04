import streamlit as st
import requests

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="AI Narrative Nexus",
    layout="wide",
    page_icon="🧠"
)

# ====================================================================
# NEON CYBERPUNK THEME (NO TOGGLING)
# ====================================================================
CYBERPUNK_CSS = """
<style>
/* -------- GLOBAL BACKGROUND -------- */
.stApp {
    background: radial-gradient(circle at 20% 20%, #220035, #05020a 70%) !important;
    color: #EEE !important;
    font-family: 'Inter', sans-serif;
}

/* -------- TOP STREAMLIT HEADER (DEPLOY/STOP) -------- */
header, .st-emotion-cache-18ni7ap, .st-emotion-cache-6qob1r {
    background: linear-gradient(90deg, #2b004d, #000000) !important;
    color: #EEE !important;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
header * {
    color: #EEE !important;
}

/* -------- CONTAINER SPACING -------- */
.main > div.block-container {
    max-width: 1150px;
    padding-top: 20px;
}

/* -------- HERO -------- */
.hero {
    text-align: center;
    padding: 20px 0 10px 0;
}
.hero-title {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(90deg, #d54aff, #00eaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 0px 22px rgba(213, 74, 255, 0.4);
}
.hero-sub {
    font-size: 15px;
    color: #cfc9dd;
    margin-top: -6px;
    margin-bottom: 30px;
}

/* -------- INPUT CARD (NEON GLOW) -------- */
.margin-line {
    background: rgba(255,255,255,0.03);
    padding: 2px;
    border-radius: 9px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 25px rgba(213, 74, 255, 0.35);
    backdrop-filter: blur(12px);
    margin-bottom: 26px;
}

/* =============================
   NEON GLOW — UPLOAD + TEXTAREA
   ============================= */

/* Wraps the file uploader box */
.stFileUploader {
    border-radius: 14px !important;
    padding: 22px !important;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);  /* purple neon */
    box-shadow: 0 0 25px rgba(213, 74, 255, 0.35);
    backdrop-filter: blur(12px);
    margin-bottom: 26px;
}

/* Dropzone container */
[data-testid="stFileUploadDropzone"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    background: rgba(255,255,255,0.06) !important;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.55),
                0 0 30px rgba(0, 234, 255, 0.25) inset !important;
}

/* Hover glow effect */
[data-testid="stFileUploadDropzone"]:hover {
    border: 1px solid rgba(213, 74, 255, 0.7) !important;
    box-shadow: 0 0 26px rgba(213, 74, 255, 0.8),
                0 0 36px rgba(0, 234, 255, 0.45) inset !important;
}

/* TEXTAREA NEON */
.stTextArea textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    background: rgba(255,255,255,0.03);
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.55),
                0 0 30px rgba(0, 234, 255, 0.25) inset !important;
}

/* Hover glow on textarea */
.stTextArea textarea:hover {
    border-color: rgba(255,255,255,0.2) !important;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.55),
                0 0 30px rgba(0, 234, 255, 0.25) inset !important;
}

/* Label text above textarea */
.stTextArea label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* -------- RESULT CARD -------- */
.result-card {
    background: rgba(255,255,255,0.04);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
    box-shadow: 0 0 15px rgba(160, 0, 255, 0.18);
}

/* Fix browse file button transparency */
.stFileUploader label div[data-testid="stFileUploadDropzone"] button {
    color: #ffffff !important;
    font-weight: 600 !important;
    background: rgba(255,255,255,0.2) !important;
    border: 1px solid rgba(255,255,255,0.45) !important;
    border-radius: 8px !important;
    padding: 4px 14px !important;
    backdrop-filter: blur(6px) !important;
}

/* Also fix the text above dropzone */
.stFileUploader label {
    color: #ffffff !important;
    font-weight: 600;
}

/* -------- TABS -------- */
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
    color: #EEE !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    border-bottom: 3px solid #d54aff !important;
}

/* Make the 'Or paste text here' label white */
.stTextArea label {
    color: #ffffff !important;
    font-weight: 600;
}

/* -------- KEYWORD BADGES -------- */
/* KEYWORD CONTAINER (flex row wrap) */
.keyword-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px; /* spacing between pills */
    margin-top: 10px;
}

/* Individual keyword pill */
.keyword-pill {
    padding: 6px 12px;       /* auto width = based on word length */
    border-radius: 12px;
    font-weight: 600;
    font-size: 13px;
    color: #000;
    background: linear-gradient(135deg, #d54aff, #00eaff);
    box-shadow: 0px 0px 10px rgba(0,234,255,0.35);
    white-space: nowrap;     /* prevents wrapping inside pill */
    display: inline-block;
}


/* -------- TOPICS GRID -------- */
.topic-tile {
    padding: 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 12px;
    box-shadow: 0 0 8px rgba(255,255,255,0.08);
}
.topic-selected {
    background: linear-gradient(90deg,#d54aff,#00eaff) !important;
    color: #000 !important;
    border: none;
    font-weight: 800;
    box-shadow: 0px 0px 18px rgba(213,74,255,0.8);
}

/* -------- FOOTER -------- */
.footer-note {
    color: rgba(200,200,210,0.65);
    text-align: center;
    margin-top: 25px;
    font-size: 13px;
}
</style>
"""

st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# ====================================================================
# HERO
# ====================================================================
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Saikrishna's AI Narrative Nexus</div>", unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Document Intelligence — Summary • Sentiment • Topic Modelling</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ====================================================================
# INPUT AREA (Upload + Text) — STACKED COLUMN LAYOUT
# ====================================================================
st.markdown('<div class="margin-line">', unsafe_allow_html=True)
st.subheader("Upload a Document or Type Text")

# Stacked vertical layout (column direction)
uploaded_file = st.file_uploader(
    "Choose a file",
    type=None,
    label_visibility="visible"  # show label above
)

typed_text = st.text_area(
    "Or paste text here",
    height=220
)

st.markdown("</div>", unsafe_allow_html=True)


# ====================================================================
# PROCESS BUTTON
# ====================================================================
process = st.button("Analyze Document", type="primary")

if process and (uploaded_file or typed_text.strip()):

    with st.spinner("⚙️ Running AI Pipeline..."):

        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        else:
            files = {"file": ("manual.txt", typed_text, "text/plain")}

        resp = requests.post("http://127.0.0.1:8000/api/process", files=files)
        if resp.status_code != 200:
            st.error("Processing error.")
            st.stop()
        data = resp.json()

    st.success("Processing complete!")
    st.write("---")

    # ====================================================================
    # TABS
    # ====================================================================
    tabs = st.tabs(["Cleaned Text", "Summary", "Sentiment", "Topics"])

    with tabs[0]:
        st.markdown(f'<div class="result-card">{data["cleaned_preview"]}</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(f'<div class="result-card">{data["summary"]}</div>', unsafe_allow_html=True)

    with tabs[2]:
        sent = data["sentiment"]
        st.markdown(
            f'<div class="result-card"><h4>{sent["label"]}</h4>Confidence: {sent["score"]*100:.1f}%</div>',
            unsafe_allow_html=True
        )

    with tabs[3]:
        tid = data["topic_id"]
        tname = data["topic_name"]
        tkeywords = data["topic_keywords"]
        tcount = data["topic_count"]
        tprob = data["topic_probability"]

        st.markdown(f'### Assigned Topic: **{tid} / {tcount}** — {tname}')
        st.markdown(f'<div class="result-card"><b>Probability:</b> {tprob:.2f}</div>', unsafe_allow_html=True)

        st.write("### Keywords")
        # Create one combined HTML block for flex layout
        keyword_html = '<div class="keyword-container">'

        for kw in tkeywords:
            keyword_html += f"<div class='keyword-pill'>{kw}</div>"

        keyword_html += "</div>"

        st.markdown(keyword_html, unsafe_allow_html=True)



        # Fetch all topics
        all_topics = requests.get("http://127.0.0.1:8000/api/topics").json()["topics"]

        st.write("---")
        st.write("### All Topics")

        assigned_topic = int(data.get("topic_id"))

        keys = list(all_topics.items())
        for i in range(0, len(keys), 2):
            c1, c2 = st.columns(2)

            # LEFT TILE
            tid1, name1 = keys[i]
            tid1 = int(tid1)   # <-- IMPORTANT FIX
            css1 = "topic-tile topic-selected" if tid1 == assigned_topic else "topic-tile"
            c1.markdown(
                f'<div class="{css1}"><b>{tid1}</b> — {name1}</div>',
                unsafe_allow_html=True
            )

            # RIGHT TILE
            if i + 1 < len(keys):
                tid2, name2 = keys[i+1]
                tid2 = int(tid2)   # <-- IMPORTANT FIX
                css2 = "topic-tile topic-selected" if tid2 == assigned_topic else "topic-tile"
                c2.markdown(
                    f'<div class="{css2}"><b>{tid2}</b> — {name2}</div>',
                    unsafe_allow_html=True
                )


# ====================================================================
# FOOTER
# ====================================================================
st.markdown('<div class="footer-note">AI Narrative Nexus • Built with Love • Built by Ramoju Saikrishna</div>', unsafe_allow_html=True)
