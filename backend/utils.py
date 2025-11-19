# backend/utils.py
import re
import io
import csv
import chardet
from typing import List, Union

# docx
try:
    from docx import Document
    _DOCX_OK = True
except Exception:
    _DOCX_OK = False

# pdf
try:
    import pypdf
    _PDF_LIB = "pypdf"
except Exception:
    try:
        import PyPDF2
        _PDF_LIB = "pypdf2"
    except Exception:
        _PDF_LIB = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"txt", "csv", "docx", "pdf"}


def extract_text_from_file(filename: str, content: Union[bytes, str]) -> Union[str, List[str]]:
    fn = filename.lower()
    if fn.endswith(".txt"):
        if isinstance(content, bytes):
            detected = chardet.detect(content)
            enc = detected.get("encoding") or "utf-8"
            return content.decode(enc, errors="ignore")
        return str(content)

    if fn.endswith(".docx") and _DOCX_OK:
        try:
            doc = Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
        except Exception:
            return ""

    if fn.endswith(".csv"):
        try:
            text_content = content.decode("utf-8", errors="ignore")
            rows = list(csv.DictReader(io.StringIO(text_content)))
            if not rows:
                return ""
            possible_cols = ["text", "content", "message", "review", "comment", "description"]
            cols = rows[0].keys()
            col = next((c for c in possible_cols if c in cols), None)
            if not col:
                col = list(cols)[0]
            return "\n".join([r[col] for r in rows if r.get(col)])
        except Exception:
            return ""

    if fn.endswith(".pdf") and _PDF_LIB:
        try:
            if _PDF_LIB == "pypdf":
                reader = pypdf.PdfReader(io.BytesIO(content))
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n\n".join([p for p in pages if p.strip()])
            else:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n\n".join([p for p in pages if p.strip()])
        except Exception:
            return ""

    return ""
    
# Simple fallback keyword extractor used only as utility (UI hides this by default)
def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    stopwords = {
        "the", "is", "and", "are", "to", "in", "of", "for", "with", "on",
        "a", "an", "it", "this", "that", "as", "at", "by", "be", "from",
        "was", "were", "has", "had", "have", "their", "his", "her",
        "you", "your", "our", "they", "them", "its", "now", "more", "new"
    }
    if not text:
        return []
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]
