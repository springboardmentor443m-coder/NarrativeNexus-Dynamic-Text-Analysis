# modules/data_input.py

import streamlit as st
import docx
import io
import PyPDF2


MAX_READ_BYTES = 50 * 1024 * 1024   # read max 50MB text from file


def _read_txt(file):
    """
    Read large TXT files safely using streaming.
    """
    text = []
    total = 0

    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        total += len(chunk)
        if total > MAX_READ_BYTES:
            text.append("\n\n[Truncated due to size limit]\n")
            break
        try:
            text.append(chunk.decode("utf-8", errors="ignore"))
        except Exception:
            text.append(chunk.decode("latin1", errors="ignore"))

    return "".join(text)


def _read_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = []
    for page in pdf_reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(text)


def handle_file_upload():
    """
    Unified file handler for TXT, CSV, DOCX, PDF.
    Auto-detects size and reads efficiently.
    """
    uploaded = st.file_uploader(
        "📂 Upload file (.txt, .csv, .docx, .pdf)",
        type=["txt", "csv", "docx", "pdf"],
    )

    if not uploaded:
        return None

    file_size = uploaded.size

    if file_size > 150 * 1024 * 1024:
        st.error("❌ File too large! Max allowed is 150MB.")
        return None

    st.info(f"📄 File uploaded: **{uploaded.name}** ({file_size/1024/1024:.1f} MB)")

    if uploaded.type == "text/plain":
        return _read_txt(uploaded)

    if uploaded.type in ["text/csv", "application/vnd.ms-excel"]:
        return _read_txt(uploaded)

    if uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return _read_docx(uploaded)

    if uploaded.type == "application/pdf":
        return _read_pdf(uploaded)

    st.warning("⚠ Unsupported file type.")
    return None
