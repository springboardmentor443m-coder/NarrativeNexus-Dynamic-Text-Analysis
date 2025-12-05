"""File parsing utilities for extracting text from various file formats"""
import os
from io import BytesIO

def extract_text_from_file(file, filename):
    """
    Extract text content from various file formats
    
    Args:
        file: File object from Flask request
        filename: Name of the file
        
    Returns:
        str: Extracted text content
    """
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.txt':
        return extract_txt(file)
    elif file_ext == '.pdf':
        return extract_pdf(file)
    elif file_ext == '.docx':
        return extract_docx(file)
    else:
        raise ValueError(f'Unsupported file format: {file_ext}')

def extract_txt(file):
    """Extract text from TXT file"""
    try:
        content = file.read()
        if isinstance(content, bytes):
            # Try UTF-8 first
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1
                return content.decode('latin-1')
        return content
    except Exception as e:
        raise Exception(f'Error reading TXT file: {str(e)}')

def extract_pdf(file):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text_pages = []
        for page in pdf_reader.pages:
            text_pages.append(page.extract_text())
        return '\n'.join(text_pages)
    except ImportError:
        raise ImportError('PyPDF2 is required for PDF processing. Install it with: pip install PyPDF2')
    except Exception as e:
        raise Exception(f'Error reading PDF file: {str(e)}')

def extract_docx(file):
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(BytesIO(file.read()))
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs)
    except ImportError:
        raise ImportError('python-docx is required for DOCX processing. Install it with: pip install python-docx')
    except Exception as e:
        raise Exception(f'Error reading DOCX file: {str(e)}')


