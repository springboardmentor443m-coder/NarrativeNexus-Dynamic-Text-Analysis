"""
Theme extraction module using pre-trained LDA model on 20 Newsgroups dataset
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

class ThemeExtractor:
    def __init__(self, model_path='model_storage/lda_model_20newsgroups.pkl'):
        """
        Initialize theme extractor with pre-trained LDA model
        
        Args:
            model_path: Path to the saved LDA model pickle file
        """
        self.model_path = model_path
        self.word_lemmatizer = WordNetLemmatizer()
        self.filter_words = set(stopwords.words('english'))
        self.lda_model = None
        self.text_vectorizer = None
        self.theme_labels = None
        self.theme_count = None
        self.is_ready = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the pre-trained LDA model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Warning: Model file not found at {self.model_path}")
                print("Please run build_theme_model.py first to train the model.")
                self.is_ready = False
                return
            
            with open(self.model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.lda_model = saved_data['lda_model']
            self.text_vectorizer = saved_data['vectorizer']
            self.theme_labels = saved_data['topic_names']
            self.theme_count = saved_data['num_topics']
            self.is_ready = True
            print(f"✓ Loaded LDA model with {self.theme_count} themes from 20 Newsgroups dataset")
            
        except Exception as e:
            print(f"Error loading LDA model: {e}")
            self.is_ready = False
    
    def clean_and_prepare_text(self, text):
        """Clean and prepare text for theme extraction"""
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
        tokens = [self.word_lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.filter_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_themes(self, text):
        """
        Extract themes from a document using 20 Newsgroups dataset model
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Theme extraction results
        """
        if not self.is_ready:
            return {
                'error': 'LDA model not loaded. Please train the model first by running build_theme_model.py',
                'topics': []
            }
        
        # Clean and prepare text
        cleaned_text = self.clean_and_prepare_text(text)
        
        if len(cleaned_text.split()) < 10:
            return {
                'topics': [],
                'message': 'Text is too short for theme extraction. Please provide more content.'
            }
        
        try:
            # Transform the document using the same vectorizer
            doc_matrix = self.text_vectorizer.transform([cleaned_text])
            
            # Get theme distribution for the document
            theme_dist = self.lda_model.transform(doc_matrix)[0]
            
            # Get top themes
            top_theme_idx = np.argmax(theme_dist)
            top_theme_score = float(theme_dist[top_theme_idx])
            
            # Get top 5 themes with their scores
            top_indices = np.argsort(theme_dist)[-5:][::-1]
            
            themes = []
            for idx in top_indices:
                score = float(theme_dist[idx])
                if score > 0.01:  # Only include themes with significant probability
                    # Get top words for this theme
                    feature_names = self.text_vectorizer.get_feature_names_out()
                    theme_weights = self.lda_model.components_[idx]
                    top_words_idx = theme_weights.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    
                    themes.append({
                        'topic_id': int(idx + 1),
                        'topic_name': self.theme_labels[idx] if self.theme_labels else f'Theme {idx + 1}',
                        'score': round(score, 4),
                        'top_words': top_words[:5]  # Top 5 words
                    })
            
            # Primary theme (most likely)
            primary_theme = {
                'topic_id': int(top_theme_idx + 1),
                'topic_name': self.theme_labels[top_theme_idx] if self.theme_labels else f'Theme {top_theme_idx + 1}',
                'confidence': round(top_theme_score * 100, 2),
                'top_words': themes[0]['top_words'] if themes else []
            }
            
            return {
                'primary_topic': primary_theme,
                'all_topics': themes,
                'num_topics': self.theme_count,
                'message': f'Document classified as: {primary_theme["topic_name"]} (confidence: {primary_theme["confidence"]}%)'
            }
            
        except Exception as e:
            return {
                'error': f'Error extracting themes: {str(e)}',
                'topics': []
            }
    
    def get_topics(self, text, num_topics=None, num_words=None):
        """
        Extract themes from text - wrapper for extract_themes for backward compatibility
        """
        return self.extract_themes(text)
