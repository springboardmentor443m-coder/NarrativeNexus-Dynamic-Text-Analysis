from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load BERT model for sentiment analysis"""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Falling back to default sentiment analysis...")
            self.classifier = None
    
    def analyze(self, text, max_length=512):
        """Analyze sentiment of text using BERT"""
        if not text or len(text.strip()) == 0:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'sentiment': 'neutral'
            }
        
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            if self.classifier:
                result = self.classifier(text)[0]
                
                # Map BERT output to sentiment
                label = result['label']
                score = result['score']
                
                # Convert to standard format
                if 'POSITIVE' in label.upper() or '5' in label or '4' in label:
                    sentiment = 'positive'
                elif 'NEGATIVE' in label.upper() or '1' in label or '2' in label:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'label': label,
                    'score': round(score, 4),
                    'sentiment': sentiment,
                    'confidence': round(score * 100, 2)
                }
            else:
                # Fallback to simple rule-based analysis
                return self._simple_sentiment(text)
        
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text):
        """Simple rule-based sentiment analysis as fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'love', 'happy', 'pleased', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 
                         'disappointed', 'angry', 'sad', 'frustrated', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(0.7 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = min(0.7 + (negative_count * 0.1), 0.95)
        else:
            sentiment = 'neutral'
            score = 0.5
        
        return {
            'label': sentiment.upper(),
            'score': round(score, 4),
            'sentiment': sentiment,
            'confidence': round(score * 100, 2)
        }

