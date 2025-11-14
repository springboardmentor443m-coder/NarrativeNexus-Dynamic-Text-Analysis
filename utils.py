#utils.py
import re
from bs4 import BeautifulSoup
import string

# emoji regex (covers large Unicode ranges)
RE_EMOJI = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

URL_MENTION_HASHTAG_RE = re.compile(r"http\S+|www\.\S+|@\w+|#\w+")
MULTISPACE_RE = re.compile(r"\s+")


def remove_emojis(text: str) -> str:
    if not text:
        return ""
    return RE_EMOJI.sub(" ", text)


def clean_html(text: str) -> str:
    """Remove HTML tags and entities."""
    if not text:
        return ""
    # BeautifulSoup gets text content and handles entities
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def clean_light(text: str) -> str:
    """
    Light cleaning for summarization:
    - remove HTML & emojis
    - remove URLs/mentions/hashtags
    - keep punctuation (so sentence boundaries remain)
    - collapse whitespace
    """
    if not text:
        return ""
    t = clean_html(text)
    t = remove_emojis(t)
    t = URL_MENTION_HASHTAG_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def clean_text(text: str) -> str:
    """
    Aggressive normalization for topic modeling / sentiment:
    - remove HTML & emojis
    - lowercase
    - remove URLs/mentions/hashtags and digits
    - remove punctuation
    - collapse whitespace
    """
    if not text:
        return ""
    t = clean_html(text)
    t = remove_emojis(t)
    t = t.lower()
    # remove URLs/mentions/hashtags and numbers
    t = re.sub(r"http\S+|www\.\S+|@\w+|#\w+|\d+", " ", t)
    # remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t