import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Narrative Nexus", layout="wide", page_icon="🧠")


# ------------------- Health Check -------------------
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return True, r.json()
    except Exception:
        return False, {}


healthy, health_data = check_api()

# ------------------- Sidebar ------------------------
with st.sidebar:
    st.title("🧠 AI Narrative Nexus")
    if healthy:
        st.success("Backend Connected")
        with st.expander("System Info"):
            st.json(health_data)
    else:
        st.error("Backend Offline")
        st.code("Run: uvicorn backend.main:app --reload")

    st.markdown("---")
    if st.button("📚 View History"):
        st.session_state.show_history = True


# ------------------- History Page -------------------
if st.session_state.get("show_history"):
    st.header("📚 Analysis History")

    try:
        resp = requests.get(f"{API_URL}/api/history").json()

        for item in resp.get("results", []):
            with st.expander(f"#{item['id']} — {item['timestamp']}"):
                full = requests.get(f"{API_URL}/api/history/{item['id']}").json()

                cleaned = {
                    "id": full.get("id"),
                    "timestamp": full.get("timestamp"),
                    "file_names": full.get("file_names"),
                    "num_files": full.get("num_files"),
                    "num_documents": full.get("num_documents"),
                    "per_file": full.get("per_file"),
                    "topics": full.get("topics")
                }
                st.json(cleaned)

    except Exception as e:
        st.error("Could not load history.")

    if st.button("← Back"):
        st.session_state.show_history = False
        st.experimental_rerun()


# ------------------- Main UI -------------------------
st.title("Advanced Text Analysis Suite")

tabs = st.tabs(["Upload Files", "Paste Text"])


# -------- Upload tab ----------
with tabs[0]:
    uploaded = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["txt", "csv", "docx", "pdf"]
    )

    if uploaded and st.button("Analyze Files"):
        files_for_api = [
            ("files", (f.name, f.getvalue(), "application/octet-stream"))
            for f in uploaded
        ]

        resp = requests.post(f"{API_URL}/api/analyze/files", files=files_for_api)

        if resp.status_code == 200:
            st.session_state.last_analysis = resp.json()
            st.success("Analysis Complete")
        else:
            st.error(resp.text)


# -------- Paste text tab ----------
with tabs[1]:
    txt = st.text_area("Paste text here", height=300)

    if txt and st.button("Analyze Text"):
        file = ("files", ("input.txt", txt.encode(), "text/plain"))
        resp = requests.post(f"{API_URL}/api/analyze/files", files=[file])

        if resp.status_code == 200:
            st.session_state.last_analysis = resp.json()
            st.success("Analysis Complete")
        else:
            st.error(resp.text)


# ------------------- After Analysis -------------------
if st.session_state.get("last_analysis"):
    rs = st.session_state.last_analysis

    st.markdown("---")
    st.header("Analysis Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Files", rs.get("num_files", 0))
    c2.metric("Documents", rs.get("num_documents", 0))

    topics_count = len(rs.get("topics", {}).get("topics", []))
    c3.metric("Topics Found", topics_count)

    # ------------------- Result Tabs -------------------
    res_tabs = st.tabs(["Sentiment", "Summary", "Topics", "Raw Data"])


    # -------------- Sentiment Tab --------------
    with res_tabs[0]:
        st.subheader("Per-file Sentiment")

        per_file = rs.get("per_file", [])

        if not per_file:
            st.info("No per-file sentiment available.")
        else:
            for item in per_file:
                fname = item.get("file_name", "Unknown")
                sent = item.get("sentiment", {})
                s_label = sent.get("sentiment", "neutral").capitalize()
                s_conf = sent.get("confidence", 0.0)

                st.write(f"**{fname}** — {s_label} ({s_conf*100:.1f}%)")
                st.markdown("---")


    # -------------- Summary Tab --------------
    with res_tabs[1]:
        st.subheader("Per-file Summaries")

        per_file = rs.get("per_file", [])
        if not per_file:
            st.info("No per-file summaries available.")
        else:
            for item in per_file:
                fname = item.get("file_name", "Unknown")

                with st.expander(f"{fname}"):
                    sent = item.get("sentiment", {})
                    s_label = sent.get("sentiment", "neutral").capitalize()
                    s_conf = sent.get("confidence", 0.0)

                    st.markdown(f"**Sentiment:** {s_label} ({s_conf*100:.1f}%)")
                    st.markdown("**Summary:**")
                    st.write(item.get("summary", ""))


    # -------------- Topics Tab --------------
    with res_tabs[2]:
        st.subheader("Topic Modeling")

        topics = rs.get("topics", {}).get("topics", [])

        if not topics:
            st.info("No topics found.")
        else:
            for i, t in enumerate(topics, start=1):
                st.markdown(f"### Topic {i}: {t.get('topic_name','')}")
                st.write("**Keywords:**", ", ".join(t.get("keywords", [])))
                st.write("**Count:**", t.get("count", 0))
                st.markdown("---")


    # -------------- Raw Data Tab --------------
    with res_tabs[3]:
        st.subheader("Raw JSON Output (cleaned)")

        cleaned = {
            "status": rs.get("status"),
            "timestamp": rs.get("timestamp"),
            "file_names": rs.get("file_names"),
            "num_files": rs.get("num_files"),
            "num_documents": rs.get("num_documents"),
            "per_file": rs.get("per_file"),
            "topics": rs.get("topics"),
        }

        st.json(cleaned)
