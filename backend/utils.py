
import re
from typing import List, Union
import chardet
from docx import Document
import csv
import io


def allowed_file(filename: str) -> bool:
    """Check if file is something we can actually process"""
    allowed_extensions = {'txt', 'csv', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def extract_text_from_file(filename: str, content: Union[bytes, str]) -> str:
    """
    Pull text out of various file types
    Works with txt, csv, and docx
    """
    
    # Handle TXT files
    if filename.endswith('.txt'):
        if isinstance(content, bytes):
            # Try to figure out encoding
            detected = chardet.detect(content)
            encoding = detected.get('encoding', 'utf-8')
            try:
                return content.decode(encoding)
            except:
                return content.decode('utf-8', errors='ignore')
        return content
    
    # Handle DOCX files
    elif filename.endswith('.docx'):
        try:
            doc = Document(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return "Could not extract text from DOCX file"
    
    # Handle CSV files
    elif filename.endswith('.csv'):
        try:
            text_content = content.decode('utf-8')
            rows = csv.DictReader(io.StringIO(text_content))
            
            # Try to find text column automatically
            text_columns = ['text', 'content', 'description', 'review', 'message', 'comment']
            
            if rows.fieldnames:
                text_col = None
                for col in text_columns:
                    if col in rows.fieldnames:
                        text_col = col
                        break
                
                if not text_col:
                    text_col = rows.fieldnames[0]
                
                texts = []
                for row in rows:
                    if text_col in row and row[text_col]:
                        texts.append(row[text_col])
                
                return '\n\n'.join(texts)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return "Could not extract text from CSV file"
    
    return ""


def remove_urls(text: str) -> str:
    """Get rid of web links"""
    return re.sub(r'http\S+|www\S+', '', text)


def remove_emails(text: str) -> str:
    """Remove email addresses"""
    return re.sub(r'\S+@\S+', '', text)

