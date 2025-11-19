# backend/preprocessing.py
import re

# optional libs
try:
    import emoji as _emoji_lib
    _HAS_EMOJI = True
except Exception:
    _HAS_EMOJI = False

try:
    import contractions as _contractions_lib
    _HAS_CONTRACTIONS = True
except Exception:
    _HAS_CONTRACTIONS = False


class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def expand_contractions(self, text: str) -> str:
        if _HAS_CONTRACTIONS:
            try:
                return _contractions_lib.fix(text)
            except Exception:
                return text
        # small fallback dictionary
        small = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "i'm": "i am", "it's": "it is", "that's": "that is",
            "you're": "you are", "they're": "they are"
        }
        for k, v in small.items():
            text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
        return text

    def remove_emojis(self, text: str) -> str:
        if _HAS_EMOJI:
            try:
                return _emoji_lib.replace_emoji(text, replace="")
            except Exception:
                pass
        # basic fallback (may remove some symbols)
        return re.sub(r"[^\w\s.,!?;:()'-]", " ", text)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.expand_contractions(text)
        text = self.remove_emojis(text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9.,!?;:\s'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
