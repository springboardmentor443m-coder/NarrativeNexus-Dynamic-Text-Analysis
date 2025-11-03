import streamlit as st
import pandas as pd
import docx
import os

def read_text_file(file):
    """Read a .txt file and return text"""
    return file.read().decode("utf-8")

def read_csv_file(file):
    """Read a .csv file and return text joined from all cells"""
    df = pd.read_csv(file)
    text_data = " ".join(df.astype(str).fillna("").values.flatten())
    return text_data

def read_docx_file(file):
    """Read a .docx file and return text"""
    doc = docx.Document(file)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)

def handle_file_upload():
    """
    Streamlit file uploader with validation and preview.
    Returns uploaded text as string.
    """
    st.header("📂 Data Input Module")
    st.write("Upload your text data file (.txt, .csv, or .docx).")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "csv", "docx"],
        help="Supported formats: .txt, .csv, .docx"
    )

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == ".txt":
                text_data = read_text_file(uploaded_file)
            elif file_extension == ".csv":
                text_data = read_csv_file(uploaded_file)
            elif file_extension == ".docx":
                text_data = read_docx_file(uploaded_file)
            else:
                st.error("❌ Unsupported file format.")
                return None

            if not text_data.strip():
                st.warning("⚠️ File is empty or unreadable.")
                return None

            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

            with st.expander("📄 Preview Uploaded Content"):
                st.text_area("File Content Preview", text_data[:2000], height=300)

            return text_data

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    else:
        st.info("⬆️ Please upload a file to begin.")
        return None
