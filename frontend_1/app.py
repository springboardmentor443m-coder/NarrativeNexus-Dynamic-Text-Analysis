import streamlit as st
import requests

st.set_page_config(page_title="AI Narrative Nexus", layout="centered")
st.title("Infosys x AI Narrative Nexus")
st.subheader("Dynamic Text Analysis Platform")

uploaded_file = st.file_uploader("Choose a file", type=None)

if uploaded_file:
    with st.spinner("Processing file..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        # Call backend
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/process",
                files=files,
                timeout=200
            )
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            st.stop()

        if response.status_code != 200:
            try:
                err = response.json().get("detail", response.text)
            except Exception:
                err = response.text
            st.error(f"Error: {err}")
            st.stop()

        data = response.json()
        st.success("✅ File processed successfully!")

        # Cleaned preview
        st.subheader("🧹 Cleaned Text Preview")
        st.text_area("Preview", data["cleaned_preview"], height=200)

        # Summary
        st.subheader("📝 Summary")
        st.write(data["summary"])

        # Sentiment
        st.subheader("💬 Sentiment Analysis")
        sentiment = data["sentiment"]
        st.metric(
            "Sentiment",
            sentiment["label"],
            delta=f"{sentiment['score']*100:.1f}% confidence"
        )

        # Topic Section
        st.subheader("🧭 Topic Classification")
        st.write(f"**Topic ID:** {data['topic']}")

        # Name
        if "topic_name" in data:
            st.write(f"**Topic Name:** {data['topic_name']}")

        # Keywords
        if "topic_keywords" in data and data["topic_keywords"]:
            st.write("**Top Keywords:**")
            for kw in data["topic_keywords"]:
                st.markdown(f"- {kw}")

