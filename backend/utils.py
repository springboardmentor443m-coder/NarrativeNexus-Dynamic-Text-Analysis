import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from heapq import nlargest
from textblob import TextBlob
import numpy as np

# download required nltk data
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [LEM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)

def extractive_summary(text: str, n_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n_sentences:
        return text

    try:
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(sentences)
        scores = X.sum(axis=1).A1

        top_idx = nlargest(n_sentences, range(len(scores)), key=lambda i: scores[i])
        top_idx = sorted(top_idx)

        return " ".join([sentences[i] for i in top_idx])
    except:
        return " ".join(sentences[:n_sentences])

def predict_sentiment(text: str) -> dict:
    tb = TextBlob(text or "")
    polarity = tb.sentiment.polarity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {"polarity": float(polarity), "label": label}

def match_topic(clean_text: str, vectorizer, topic_model, topics_meta):
    vec = vectorizer.transform([clean_text])
    topic_dist = topic_model.transform(vec)

    top_idx = int(np.argmax(topic_dist))
    score = float(topic_dist[0, top_idx])

    meta = topics_meta.get(str(top_idx), {})
    return {
        "topic_id": top_idx,
        "score": score,
        "keywords": meta.get("keywords", [])
    }
