import re
from typing import List

class TextPreprocessor:
    """Cleans up text before sending to AI models"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning - be careful not to remove too much
        We want to keep the actual content intact
        """
        if not text:
            return ""

        # Just collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)

        # Remove only the really bad stuff
        # Keep letters, numbers, spaces, and basic punctuation
        return text.strip()

    @staticmethod
    def split_into_chunks(text: str, max_length: int = 2000) -> List[str]:
        """
        Split long texts into smaller pieces
        Some models have limits on how much they can handle at once
        """
        if not text or len(text.strip()) < 10:
            return [text] if text else []

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            current_length += word_length

            # If chunk is getting too big, save it and start a new one
            if current_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]