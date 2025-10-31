import pandas as pd
from docx import Document
import requests

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_csv(path: str, column: str) -> str:
    df = pd.read_csv(path)
    return " ".join(df[column].dropna().astype(str).tolist())

def load_docx(path: str) -> str:
    doc = Document(path)
    return " ".join([p.text for p in doc.paragraphs])

def load_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    return response.text
