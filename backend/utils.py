import re
import io
import csv
import chardet
from typing import List, Union

# Check for docx support
try:
    from docx import Document
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False

# Check for PDF support
try:
    import pypdf
    _PDF_LIB = "pypdf"
except ImportError:
    try:
        import PyPDF2
        _PDF_LIB = "pypdf2"
    except ImportError:
        _PDF_LIB = None

def allowed_file(filename: str) -> bool:
    """Check if the file extension is supported."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"txt", "csv", "docx", "pdf"}

def extract_text_from_file(filename: str, content: Union[bytes, str]) -> Union[str, List[str]]:
    """Extract raw text from various file formats."""
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
        except Exception as e:
            print(f"Error parsing DOCX: {e}")
            return ""

    if fn.endswith(".csv"):
        try:
            # Decode bytes to string for CSV reader
            text_content = content.decode("utf-8", errors="ignore")
            rows = list(csv.DictReader(io.StringIO(text_content)))
            if not rows:
                return ""
            
            # Try to find a text-heavy column
            possible_cols = ["text", "content", "message", "review", "comment", "description", "body"]
            cols = rows[0].keys()
            col = next((c for c in possible_cols if c in cols), None)
            
            if col:
                return "\n".join([r[col] for r in rows if r.get(col)])
            else:
                # Fallback: Join all values in the row
                return "\n".join([" ".join(r.values()) for r in rows])
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return ""

    if fn.endswith(".pdf") and _PDF_LIB:
        try:
            reader = None
            stream = io.BytesIO(content)
            
            if _PDF_LIB == "pypdf":
                reader = pypdf.PdfReader(stream)
            else:
                reader = PyPDF2.PdfReader(stream)

            if reader:
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n\n".join([p for p in pages if p.strip()])
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return ""

    return ""

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Simple frequency-based keyword extraction."""
    stopwords = {
        "the", "is", "and", "are", "to", "in", "of", "for", "with", "on",
        "a", "an", "it", "this", "that", "as", "at", "by", "be", "from",
        "was", "were", "has", "had", "have", "their", "his", "her",
        "you", "your", "our", "they", "them", "its", "now", "more", "new"
    }
    if not text:
        return []
    
    # Clean and tokenize
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
            
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]