"""
Topic modeling module using pre-trained LDA model on 20 Newsgroups dataset
"""
import os
import pickle
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
    def __init__(self, model_path='models_data/lda_model_20newsgroups.pkl'):
        """
        Initialize topic modeler with pre-trained LDA model
        
        Args:
            model_path: Path to the saved LDA model pickle file
        """
        self.model_path = model_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lda_model = None
        self.vectorizer = None
        self.topic_names = None
        self.num_topics = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained LDA model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Warning: Model file not found at {self.model_path}")
                print("Please run train_lda_model.py first to train the model.")
                self.model_loaded = False
                return
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.vectorizer = model_data['vectorizer']
            self.topic_names = model_data['topic_names']
            self.num_topics = model_data['num_topics']
            self.model_loaded = True
            print(f"✓ Loaded LDA model with {self.num_topics} topics from 20 Newsgroups dataset")
            
        except Exception as e:
            print(f"Error loading LDA model: {e}")
            self.model_loaded = False
    
    def preprocess_text(self, text):
        """Preprocess text for topic modeling"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses and URLs
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def classify_topic(self, text):
        """
        Classify a document to one of the topics from 20 Newsgroups dataset
        
        Args:
            text: Input text to classify
            
        Returns:
            dict: Topic classification results
        """
        if not self.model_loaded:
            return {
                'error': 'LDA model not loaded. Please train the model first by running train_lda_model.py',
                'topics': []
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if len(processed_text.split()) < 10:
            return {
                'topics': [],
                'message': 'Text is too short for topic classification. Please provide more content.'
            }
        
        try:
            # Transform the document using the same vectorizer
            doc_term_matrix = self.vectorizer.transform([processed_text])
            
            # Get topic distribution for the document
            topic_distribution = self.lda_model.transform(doc_term_matrix)[0]
            
            # Get top topics
            top_topic_idx = np.argmax(topic_distribution)
            top_topic_score = float(topic_distribution[top_topic_idx])
            
            # Get top 5 topics with their scores
            top_indices = np.argsort(topic_distribution)[-5:][::-1]
            
            topics = []
            for idx in top_indices:
                score = float(topic_distribution[idx])
                if score > 0.01:  # Only include topics with significant probability
                    # Get top words for this topic
                    feature_names = self.vectorizer.get_feature_names_out()
                    topic_weights = self.lda_model.components_[idx]
                    top_words_idx = topic_weights.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    
                    topics.append({
                        'topic_id': int(idx + 1),
                        'topic_name': self.topic_names[idx] if self.topic_names else f'Topic {idx + 1}',
                        'score': round(score, 4),
                        'top_words': top_words[:5]  # Top 5 words
                    })
            
            # Primary topic (most likely)
            primary_topic = {
                'topic_id': int(top_topic_idx + 1),
                'topic_name': self.topic_names[top_topic_idx] if self.topic_names else f'Topic {top_topic_idx + 1}',
                'confidence': round(top_topic_score * 100, 2),
                'top_words': topics[0]['top_words'] if topics else []
            }
            
            return {
                'primary_topic': primary_topic,
                'all_topics': topics,
                'num_topics': self.num_topics,
                'message': f'Document classified as: {primary_topic["topic_name"]} (confidence: {primary_topic["confidence"]}%)'
            }
            
        except Exception as e:
            return {
                'error': f'Error classifying topic: {str(e)}',
                'topics': []
            }
    
    def get_topics(self, text, num_topics=None, num_words=None):
        """
        Extract topics from text - wrapper for classify_topic for backward compatibility
        """
        return self.classify_topic(text)
