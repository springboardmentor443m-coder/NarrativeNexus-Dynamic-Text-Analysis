"""
Text Preprocessing Utilities
"""

import re
from typing import List

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text.strip()
    
    @staticmethod
    def split_into_chunks(text: str, max_length: int = 2000) -> List[str]:
        """Split long text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
