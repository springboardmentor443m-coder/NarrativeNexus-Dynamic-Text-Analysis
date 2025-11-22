import re
import string
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextProcessor:
    def _init_(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process_for_lda(self, text: str) -> List[str]:
        """Process text specifically for LDA topic modeling"""
        processed_text = self.preprocess(text)
        tokens = self.tokenize(processed_text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, float]]:
        """Extract important keywords using TF-IDF-like approach"""
        processed_text = self.preprocess(text)
        tokens = self.tokenize(processed_text)
        tokens = self.remove_stopwords(tokens)
        
        # Count word frequencies
        word_freq = {}
        for token in tokens:
            if len(token) > 2:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords with their frequencies
        total_words = sum(word_freq.values())
        keywords = [
            {"word": word, "frequency": freq, "weight": freq / total_words}
            for word, freq in sorted_words[:top_n]
        ]
        
        return keywords
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences based on word frequency
        processed_text = self.preprocess(text)
        tokens = self.tokenize(processed_text)
        tokens = self.remove_stopwords(tokens)
        
        word_freq = {}
        for token in tokens:
            if len(token) > 2:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Score each sentence
        sentence_scores = {}
        for sentence in sentences:
            sentence_tokens = self.tokenize(self.preprocess(sentence))
            sentence_tokens = self.remove_stopwords(sentence_tokens)
            score = sum(word_freq.get(token, 0) for token in sentence_tokens)
            sentence_scores[sentence] = score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences = [sent for sent, _ in sorted(top_sentences, key=lambda x: sentences.index(x[0]))]
        
        return ' '.join(top_sentences)
    
    def detect_themes(self, text: str) -> List[str]:
        """Detect themes based on keyword clusters"""
        keywords = self.extract_keywords(text, top_n=20)
        # Simple theme detection based on top keywords
        themes = [kw["word"] for kw in keywords[:5]]
        return themes
