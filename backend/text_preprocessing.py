import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import re


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


nlp = spacy.load('en_core_web_sm')


stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def preprocess_text(text: str):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    doc = nlp(' '.join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return lemmatized