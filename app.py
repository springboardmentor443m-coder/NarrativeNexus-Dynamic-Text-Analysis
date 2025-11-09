# app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Narrative Nexus", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.title("🧠 Narrative Nexus — Semantic Topic Discovery Engine")
st.markdown(
    "Upload **any document or text**, and this app will automatically find meaningful topics and name them using "
    "a transformer-based summarization model (*facebook/bart-large-cnn*)."
)

# ---------------------------------------------------------------------
# SECTION 1: Choose Input Source
# ---------------------------------------------------------------------
st.sidebar.header("📂 Choose Input Source")
mode = st.sidebar.radio(
    "Select what you want to analyze:",
    ["Upload your own document", "Analyze Hugging Face Financial Dataset"],
)

# ---------------------------------------------------------------------
# SECTION 2A: Upload and Analyze Your Own File
# ---------------------------------------------------------------------
if mode == "Upload your own document":
    uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=["txt", "docx"])
    text_input = st.text_area("Or paste text manually below:", height=200)

    if st.button("🔍 Analyze My Text"):
        if not uploaded_file and not text_input.strip():
            st.warning("Please upload a file or enter text.")
        else:
            with st.spinner("Analyzing your document... this may take a few seconds ⏳"):
                files = {"file": uploaded_file.getvalue()} if uploaded_file else None
                data = {"text": text_input} if text_input else None

                try:
                    response = requests.post(f"{API_URL}/analyze", files=files, data=data)
                    result = response.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if "topics" in result and result["topics"]:
                st.success("✅ Analysis Complete!")

                topics = result["topics"]
                st.subheader("🧩 Detected Topics")
                for tid, info in topics.items():
                    st.markdown(f"### Topic {tid}: **{info['label']}**")
                    st.write(", ".join(info["keywords"]))
            else:
                st.warning("No topics were detected. Try with a larger text.")

# ---------------------------------------------------------------------
# SECTION 2B: Analyze Hugging Face Dataset
# ---------------------------------------------------------------------
else:
    limit = st.slider("Number of samples to analyze", min_value=200, max_value=2000, step=200, value=1000)

    if st.button("📊 Analyze Dataset"):
        with st.spinner("Analyzing Hugging Face dataset... please wait ⏳"):
            try:
                response = requests.get(f"{API_URL}/test-dataset?limit={limit}")
                data = response.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success("✅ Dataset Analysis Complete!")
            st.metric("Analyzed Entries", data["sample_size"])
            st.metric("Detected Topics", len(data["predicted_topics"]))

            # True Label Distribution
            labels_df = pd.DataFrame(list(data["true_labels"].items()), columns=["Label", "Count"])
            fig = px.bar(labels_df, x="Label", y="Count", title="True Label Distribution", color="Label")
            st.plotly_chart(fig, use_container_width=True)

            # Extracted Topics
            st.subheader("🧩 Extracted Topics (Auto-Labeled)")
            for tid, topic_info in data["predicted_topics"].items():
                st.markdown(f"### Topic {tid}: **{topic_info['label']}**")
                st.write(", ".join(topic_info["keywords"]))
