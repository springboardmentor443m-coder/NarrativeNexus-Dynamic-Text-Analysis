from transformers import pipeline
from nltk.tokenize import sent_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

def remove_code_artifacts(text: str) -> str:
    """Remove code snippets & HTML artifacts."""
    text = re.sub(r'console\.log\([^)]*\);?', '', text)
    text = re.sub(r'var\s+\w+\s*=\s*[^;]*;', '', text)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'[©®™•""'']', '', text)
    text = re.sub(r'^\s*[^\w\s]{2,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def extract_quotes(text: str):
    """Extract quoted text separately for better analysis."""
    quotes = re.findall(r'"([^"]+)"', text)
    text_without_quotes = re.sub(r'"[^"]+"', '', text)
    return text_without_quotes, quotes

def smart_sentence_split(text: str):
    """Sentence splitting with quote protection."""
    text = re.sub(r'"\s*\.', '"###QUOTEEND###', text)
    sents = sent_tokenize(text)
    sents = [s.replace('###QUOTEEND###', '".') for s in sents]
    return sents

class BertSentimentAnalyzer:
    """BERT sentiment with improved NEUTRAL detection"""

    def __init__(self):
        try:
            print("Loading BERT model...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            self.model_loaded = True
            print("✓ BERT model loaded!")
        except Exception as e:
            print(f"✗ Error loading BERT model: {e}")
            self.model_loaded = False

    def minimal_clean(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = remove_code_artifacts(text)
        return text

    def analyze_sentence(self, sentence: str):
        """Analyze a single sentence with improved neutral detection."""
        if len(sentence) > 512:
            sentence = sentence[:512]
        
        result = self.sentiment_pipeline(sentence)[0]
        label = result["label"]
        score = result["score"]
        

        neutral_keywords = ['mixed', 'talks about', 'article', 'discusses', 'mentions', 'wrote']
        has_neutral_keyword = any(kw in sentence.lower() for kw in neutral_keywords)
        
        if score < 0.65 or (has_neutral_keyword and score < 0.75):
            final_label = "NEUTRAL"
            polarity = 0
        elif label == "POSITIVE":
            final_label = "POSITIVE"
            polarity = score
        else:
            final_label = "NEGATIVE"
            polarity = -score
        
        return final_label, polarity, score

    def analyze_text(self, text: str) -> dict:
        if not self.model_loaded:
            return {"error": "BERT model not loaded", "status": "error"}
        
        try:
            text = self.minimal_clean(text)
            
            text_without_quotes, quotes = extract_quotes(text)
            
            sentences = smart_sentence_split(text_without_quotes)
            
            all_sentences = [s.strip() for s in sentences if s.strip()] + quotes
            
            sentence_sentiments = []
            polarity_scores = []
            
            for sentence in all_sentences:
                if len(sentence) <= 4 or not re.search(r'\w', sentence):
                    continue
                
                final_label, polarity, score = self.analyze_sentence(sentence)
                
                polarity_scores.append(polarity)
                sentence_sentiments.append({
                    "text": sentence,
                    "label": final_label,
                    "polarity": round(polarity, 4),
                    "confidence": round(score, 4)
                })
            
            if not sentence_sentiments:
                return {"error": "No sentences to analyze", "status": "error"}
            
            avg_polarity = sum(polarity_scores) / len(polarity_scores)
            
            if avg_polarity > 0.1:
                overall = "POSITIVE"
            elif avg_polarity < -0.1:
                overall = "NEGATIVE"
            else:
                overall = "NEUTRAL"
            
            pos = sum(1 for s in sentence_sentiments if s["label"] == "POSITIVE")
            neg = sum(1 for s in sentence_sentiments if s["label"] == "NEGATIVE")
            neu = sum(1 for s in sentence_sentiments if s["label"] == "NEUTRAL")
            total = len(sentence_sentiments)
            
            return {
                "status": "success",
                "overall_sentiment": overall,
                "sentiment_breakdown": {
                    "positive": round((pos/total)*100, 2),
                    "negative": round((neg/total)*100, 2),
                    "neutral": round((neu/total)*100, 2),
                },
                "sentence_count": total,
                "positive_sentences": pos,
                "negative_sentences": neg,
                "neutral_sentences": neu,
                "average_polarity": round(avg_polarity, 4),
                "confidence": round(abs(avg_polarity), 4),
                "detailed_sentiment": sentence_sentiments,
                "model": "BERT (DistilBERT + Enhanced Neutral)"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

bert_sentiment_analyzer = BertSentimentAnalyzer()














