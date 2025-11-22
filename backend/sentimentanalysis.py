from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict


class SentimentAnalyzer:
    def _init_(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, any]:
        """Analyze sentiment using both VADER and TextBlob"""
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores (weighted average)
        compound_score = vader_scores['compound']
        polarity_score = (vader_scores['compound'] + textblob_polarity) / 2
        
        # Determine label
        if compound_score >= 0.05:
            label = "positive"
        elif compound_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "sentiment_score": float(compound_score),
            "polarity": float(polarity_score),
            "subjectivity": float(textblob_subjectivity),
            "sentiment_label": label,
            "vader_scores": {
                "positive": float(vader_scores['pos']),
                "neutral": float(vader_scores['neu']),
                "negative": float(vader_scores['neg']),
                "compound": float(vader_scores['compound'])
            }
        }
