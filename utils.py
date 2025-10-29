"""
Utility Functions
"""

import os
from typing import List
import pandas as pd
from docx import Document
import io

ALLOWED_EXTENSIONS = {'.txt', '.csv', '.docx'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def extract_text_from_file(filename: str, content: bytes) -> List[str]:
    """Extract text from various file formats"""
    texts = []
    
    if filename.endswith('.txt'):
        text_data = content.decode('utf-8', errors='ignore')
        texts = [t.strip() for t in text_data.split('\n\n') if t.strip()]
    
    elif filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(content))
        text_column = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'review', 'comment']):
                text_column = col
                break
        if text_column is None:
            text_column = df.columns[0]
        texts = df[text_column].dropna().astype(str).tolist()
    
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    
    return texts
