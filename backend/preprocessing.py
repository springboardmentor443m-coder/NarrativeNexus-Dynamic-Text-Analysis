import re

# Optional dependencies
try:
    import emoji as _emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False

try:
    import contractions as _contractions_lib
    _HAS_CONTRACTIONS = True
except ImportError:
    _HAS_CONTRACTIONS = False


class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def expand_contractions(self, text: str) -> str:
        """Expands common contractions (e.g., 'don't' -> 'do not')."""
        if _HAS_CONTRACTIONS:
            try:
                return _contractions_lib.fix(text)
            except Exception:
                pass
        
        # Fallback dictionary for common cases
        common_contractions = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "i'm": "i am", "it's": "it is", "that's": "that is",
            "you're": "you are", "they're": "they are", "we're": "we are",
            "isn't": "is not", "aren't": "are not"
        }
        for k, v in common_contractions.items():
            text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
        return text

    def remove_emojis(self, text: str) -> str:
        """Removes emoji characters."""
        if _HAS_EMOJI:
            try:
                return _emoji_lib.replace_emoji(text, replace="")
            except Exception:
                pass
        # Conservative fallback: Remove characters outside basic multilingual plane
        # This regex avoids stripping standard punctuation
        return re.sub(r"[^\x00-\x7F]+", " ", text)

    def clean_text(self, text: str) -> str:
        """Master cleaning function pipeline."""
        if not isinstance(text, str):
            return ""
            
        text = self.expand_contractions(text)
        text = self.remove_emojis(text)
        
        # Remove URLs and Emails
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        
        # Keep alphanumeric, punctuation, and whitespace
        # This is safer than the previous [^a-zA-Z0-9] which stripped foreign accents
        text = re.sub(r"[^a-zA-Z0-9.,!?;:\s'\-]", " ", text)
        
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text