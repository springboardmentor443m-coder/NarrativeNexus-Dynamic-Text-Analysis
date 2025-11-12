
from groq import Groq
import json
from typing import Dict, Set
import nltk
from nltk.corpus import opinion_lexicon

# Auto-download NLTK data if needed
try:
    opinion_lexicon.positive()
except LookupError:
    nltk.download('opinion_lexicon')

# Try to load RoBERTa
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """Analyzes text sentiment using RoBERTa or Groq."""

    def __init__(self, groq_client: Groq, model_name: str):
        self.client = groq_client
        self.model = model_name
        self.roberta = None
        
        # Load comprehensive sentiment lexicons from NLTK (2000+ and 4700+ words!)
        try:
            self.positive_words = set(opinion_lexicon.positive())
            self.negative_words = set(opinion_lexicon.negative())
        except:
            # Fallback only if NLTK completely unavailable
            self.positive_words = {"good", "great", "excellent", "amazing", "awesome"}
            self.negative_words = {"bad", "terrible", "awful", "horrible", "dreadful"}
        
        print(f"📊 Loaded {len(self.positive_words)} positive, {len(self.negative_words)} negative words")

        # Load RoBERTa if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.roberta = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    return_all_scores=True
                )
                print("✅ RoBERTa loaded")
            except Exception as e:
                print(f"⚠️ RoBERTa not available: {e}")

    def analyze_with_roberta(self, text: str) -> Dict:
        """Analyze using RoBERTa model."""
        if not self.roberta:
            return None

        try:
            results = self.roberta(text[:512])[0]
            label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
            
            best = max(results, key=lambda x: x['score'])
            sentiment = label_map[best['label']]
            confidence = best['score']
            
            # Extract key phrases using full lexicon
            words = text.lower().split()
            if sentiment == "positive":
                key_phrases = [w for w in words if w in self.positive_words][:5]
            elif sentiment == "negative":
                key_phrases = [w for w in words if w in self.negative_words][:5]
            else:
                key_phrases = [w for w in words[:5] if len(w) > 3]

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "emotion": {"positive": "joy", "negative": "disappointment", "neutral": "calm"}[sentiment],
                "key_phrases": key_phrases,
                "model": "RoBERTa"
            }
        except Exception as e:
            print(f"RoBERTa failed: {e}")
            return None

    def analyze_with_groq(self, text: str) -> Dict:
        """Analyze using Groq LLM as fallback."""
        instructions = (
            'Analyze sentiment. Reply with ONLY JSON: '
            '{"sentiment": "positive/negative/neutral", "confidence": 0.85, '
            '"emotion": "joy/disappointment/calm", "key_phrases": ["word1", "word2"], "reasoning": "reason"}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Analyze: {text[:2000]}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            return {"sentiment": "neutral", "confidence": 0.5, "emotion": "unknown", "key_phrases": [], "reasoning": f"Error: {str(e)}"}

    def analyze(self, text: str) -> Dict:
        """Main method: Try RoBERTa first, fall back to Groq."""
        result = self.analyze_with_roberta(text)
        if result:
            return result
        
        result = self.analyze_with_groq(text)
        result["model"] = "Groq"

        return result
