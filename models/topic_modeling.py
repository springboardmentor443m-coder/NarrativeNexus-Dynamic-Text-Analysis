import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

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

class TopicModeler:
    def __init__(self, num_topics=5, num_words=10):
        self.num_topics = num_topics
        self.num_words = num_words
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lda_model = None
        self.vectorizer = None
    
    def preprocess_text(self, text):
        """Preprocess text for topic modeling"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_topics(self, text, num_topics=None, num_words=None):
        """Extract topics from text using LDA"""
        if num_topics is None:
            num_topics = self.num_topics
        if num_words is None:
            num_words = self.num_words
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if len(processed_text.split()) < 10:
            return {
                'topics': [],
                'message': 'Text is too short for topic modeling. Please provide more content.'
            }
        
        # Create document-term matrix
        self.vectorizer = CountVectorizer(max_features=100, ngram_range=(1, 2))
        doc_term_matrix = self.vectorizer.fit_transform([processed_text])
        
        # Apply LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        self.lda_model.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_weight = float(topic.sum())
            
            topics.append({
                'topic_id': topic_idx + 1,
                'words': top_words,
                'weight': round(topic_weight, 4)
            })
        
        return {
            'topics': topics,
            'num_topics': num_topics
        }

