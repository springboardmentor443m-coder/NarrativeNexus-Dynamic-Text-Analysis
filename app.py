import streamlit as st
import requests

st.set_page_config(page_title="NarrativeNexus", layout="wide")
st.title("NarrativeNexus — Dynamic Text Analyzer")

tabs = st.tabs([
    "📊 Topic Modeling",
    "🧹 Clean, Summarize & Analyze",
    "💬 HF Dataset Sentiment & Summary"
])

with tabs[0]:

    option = st.radio(
        "Choose Analysis Mode:",
        ["Upload your own text", "Use Twitter Financial News Dataset (Hugging Face)"],
    )

    # ----------------------------
    # Option 1: Manual Upload/Text
    # ----------------------------
    if option == "Upload your own text":
        text_input = st.text_area("Paste your text here:", height=200)
        file_upload = st.file_uploader("Or upload a text or DOCX file:", type=["txt", "docx"])

        if st.button("Analyze Text"):
            with st.spinner("Analyzing your text..."):
                files, data = {}, {"text": text_input}
                if file_upload:
                    files = {"file": (file_upload.name, file_upload.getvalue())}
                try:
                    res = requests.post("http://localhost:8000/analyze", data=data, files=files)
                    if res.status_code == 200:
                        result = res.json()
                        st.success("✅ Analysis completed successfully!")

                        if result.get("coherence") is not None:
                            st.metric("Model Coherence", f"{result['coherence']:.3f}")

                        st.subheader("🧩 Tokens (cleaned & lemmatized)")
                        st.write(result["tokens"][:50])

                        st.subheader("📊 Detected Topics")
                        for t in result.get("topics", []):
                            if isinstance(t, dict):
                                st.write(f"**Topic {t.get('topic_id', '?')}**: {', '.join(t.get('keywords', []))}")
                    else:
                        st.error(f"Backend error: {res.status_code}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    # ----------------------------
    # Option 2: Hugging Face Dataset
    # ----------------------------
    elif option == "Use Twitter Financial News Dataset (Hugging Face)":
        limit = st.slider("Number of tweets to analyze", 500, 5000, 2000, 500)
        if st.button("Run Dataset Analysis"):
            with st.spinner("Analyzing Hugging Face dataset..."):
                try:
                    res = requests.get(f"http://localhost:8000/test-dataset?limit={limit}")
                    if res.status_code == 200:
                        result = res.json()
                        st.success(f"✅ Analyzed {result['sample_size']} tweets")

                        if "coherence" in result:
                            st.metric("Model Coherence", f"{result['coherence']:.3f}")

                        st.subheader("📊 Detected Topics")
                        topics = result.get("topics", [])
                        if isinstance(topics, list):
                            for t in topics:
                                if isinstance(t, dict):
                                    st.write(f"**Topic {t.get('topic_id', '?')}**: {', '.join(t.get('keywords', []))}")
                        else:
                            st.error("Unexpected topic format returned by backend.")
                    else:
                        st.error(f"Backend error: {res.status_code}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tabs[1]:
    st.header("🧹 Text Cleaner, Summarizer & Sentiment Analyzer")

    uploaded_file = st.file_uploader("Upload a text or HTML file:", type=["txt", "html"])
    manual_text = st.text_area("Or paste text manually:", height=200)
    analyze_btn = st.button("Clean, Summarize & Analyze")

    if analyze_btn:
        with st.spinner("Analyzing..."):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post("http://localhost:8000/clean-and-summarize", files=files)
            elif manual_text.strip():
                from io import BytesIO
                fake_file = BytesIO(manual_text.encode("utf-8"))
                files = {"file": ("input.txt", fake_file)}
                res = requests.post("http://localhost:8000/clean-and-summarize", files=files)
            else:
                st.warning("Please upload a file or enter some text.")
                st.stop()

            if res.status_code == 200:
                data = res.json()
                st.subheader("🧾 Cleaned Preview")
                st.code(data.get("preview", "—"), language="text")

                st.subheader("📰 Summary")
                st.write(data.get("summary", "—"))

                sentiment = data.get("sentiment", {})
                label = sentiment.get("label", "N/A")
                score = sentiment.get("score", 0)

                st.subheader("💬 Sentiment Analysis")
                if label.lower() == "positive":
                    st.success(f"Positive ({score*100:.1f}%)")
                elif label.lower() == "negative":
                    st.error(f"Negative ({score*100:.1f}%)")
                else:
                    st.warning(f"Neutral ({score*100:.1f}%)")
            else:
                st.error(f"Backend error: {res.status_code}")

with tabs[2]:
    st.header("💬 Hugging Face Dataset — Cleaning, Summarization & Sentiment")

    # Step 1: Choose dataset size
    limit = st.slider("Number of tweets to analyze", 50, 500, 100, 50)

    # Step 2: Run backend analysis only once
    if st.button("Run Analysis on Dataset"):
        with st.spinner("Analyzing dataset..."):
            res = requests.get(f"http://localhost:8000/analyze-hf-dataset?limit={limit}")
            if res.status_code == 200:
                data = res.json()
                st.session_state.results = data["results"]
                st.session_state.total = len(data["results"])
                st.session_state.page_num = 1  # reset pagination
                st.success(f"✅ Processed {st.session_state.total} tweets")
            else:
                st.error("Backend error. Could not fetch data.")

    # Step 3: Only show pagination if data exists
    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        total = st.session_state.total

        # --- Pagination setup ---
        page_size = 10
        total_pages = (total - 1) // page_size + 1

        # Ensure page_num always valid
        if "page_num" not in st.session_state:
            st.session_state.page_num = 1

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.page_num <= 1):
                st.session_state.page_num -= 1
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.page_num >= total_pages):
                st.session_state.page_num += 1

        current_page = st.session_state.page_num
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total)

        st.info(f"Showing tweets {start_idx + 1}–{end_idx} of {total} (Page {current_page}/{total_pages})")

        # --- Show paginated tweets ---
        for r in results[start_idx:end_idx]:
            st.markdown(f"**Tweet {r['index'] + 1}:** {r['original']}")
            st.write(f"🧹 *Cleaned:* {r['cleaned_preview']}")
            st.write(f"📰 *Summary:* {r['summary']}")
            st.write(f"💬 *Sentiment:* {r['sentiment_label']} ({r['sentiment_score']})")
            st.divider()
    else:
        st.info("👆 Choose how many tweets to analyze and click **Run Analysis on Dataset**.")
