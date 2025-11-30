from textblob import TextBlob
from nltk.tokenize import sent_tokenize

class SentimentAnalyzer:
    """Sentiment analysis using TextBlob"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.model_loaded = True
    
    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of text"""
        try:
            sentences = sent_tokenize(text)
            
            sentence_sentiments = []
            polarity_scores = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 5:
                    blob = TextBlob(sentence)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    if polarity > 0.1:
                        label = "POSITIVE"
                    elif polarity < -0.1:
                        label = "NEGATIVE"
                    else:
                        label = "NEUTRAL"
                    
                    polarity_scores.append(polarity)
                    
                    sentence_sentiments.append({
                        "text": sentence,
                        "label": label,
                        "polarity": round(polarity, 4),
                        "subjectivity": round(subjectivity, 4)
                    })
            
            if len(sentence_sentiments) == 0:
                return {
                    "error": "No sentences to analyze",
                    "status": "error"
                }
            
            avg_polarity = sum(polarity_scores) / len(polarity_scores)
            
            if avg_polarity > 0.1:
                overall_sentiment = "POSITIVE"
                confidence = abs(avg_polarity)
            elif avg_polarity < -0.1:
                overall_sentiment = "NEGATIVE"
                confidence = abs(avg_polarity)
            else:
                overall_sentiment = "NEUTRAL"
                confidence = 0.5
            
            positive_count = sum(1 for s in sentence_sentiments if s["label"] == "POSITIVE")
            negative_count = sum(1 for s in sentence_sentiments if s["label"] == "NEGATIVE")
            neutral_count = sum(1 for s in sentence_sentiments if s["label"] == "NEUTRAL")
            total_sentences = len(sentence_sentiments)
            
            positive_percentage = round((positive_count / total_sentences) * 100, 2)
            negative_percentage = round((negative_count / total_sentences) * 100, 2)
            neutral_percentage = round((neutral_count / total_sentences) * 100, 2)
            
            return {
                "status": "success",
                "overall_sentiment": overall_sentiment,
                "sentiment_breakdown": {
                    "positive": positive_percentage,
                    "negative": negative_percentage,
                    "neutral": neutral_percentage
                },
                "sentence_count": total_sentences,
                "positive_sentences": positive_count,
                "negative_sentences": negative_count,
                "neutral_sentences": neutral_count,
                "average_polarity": round(avg_polarity, 4),
                "confidence": round(confidence, 4),
                "detailed_sentiment": sentence_sentiments
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }


sentiment_analyzer = SentimentAnalyzer()
