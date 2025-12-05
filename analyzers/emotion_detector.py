from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

class EmotionDetector:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = None
        self.model = None
        self.emotion_classifier = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Load BERT model for emotion detection"""
        try:
            self.emotion_classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Falling back to default emotion detection...")
            self.emotion_classifier = None
    
    def detect_emotion(self, text, max_length=512):
        """Detect emotion in text using BERT"""
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
            if self.emotion_classifier:
                result = self.emotion_classifier(text)[0]
                
                # Map BERT output to emotion
                label = result['label']
                score = result['score']
                
                # Convert to standard format
                if 'POSITIVE' in label.upper() or '5' in label or '4' in label:
                    emotion = 'positive'
                elif 'NEGATIVE' in label.upper() or '1' in label or '2' in label:
                    emotion = 'negative'
                else:
                    emotion = 'neutral'
                
                return {
                    'label': label,
                    'score': round(score, 4),
                    'sentiment': emotion,
                    'confidence': round(score * 100, 2)
                }
            else:
                # Fallback to simple rule-based detection
                return self._basic_emotion_detection(text)
        
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return self._basic_emotion_detection(text)
    
    def _basic_emotion_detection(self, text):
        """Simple rule-based emotion detection as fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'love', 'happy', 'pleased', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 
                         'disappointed', 'angry', 'sad', 'frustrated', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            emotion = 'positive'
            score = min(0.7 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            emotion = 'negative'
            score = min(0.7 + (negative_count * 0.1), 0.95)
        else:
            emotion = 'neutral'
            score = 0.5
        
        return {
            'label': emotion.upper(),
            'score': round(score, 4),
            'sentiment': emotion,
            'confidence': round(score * 100, 2)
        }
