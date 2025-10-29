# The brain of the operation
# Uses RoBERTa for sentiment, BERTopic for topics, and Groq LLM for summaries

from groq import Groq
import json
import os
from typing import List, Dict

# Try to load advanced models
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not installed - falling back to Groq only")

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    from nltk.corpus import stopwords
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: BERTopic not installed - using Groq fallback")


class GroqAnalyzer:
    """Main analyzer class that handles sentiment, topics, and summarization"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        
        # Set up sentiment analyzer if available
        self.sentiment_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    return_all_scores=True
                )
                print("✅ RoBERTa sentiment analyzer loaded")
            except Exception as e:
                print(f"⚠️ Failed to load RoBERTa: {e}")
        
        # Set up topic modeler if available
        self.topic_model = None
        if BERTOPIC_AVAILABLE:
            try:
                # Download stopwords if needed
                try:
                    stopwords.words('english')
                except LookupError:
                    nltk.download('stopwords')
                
                # Set up custom stopwords for better topics
                stop_words = list(stopwords.words('english'))
                stop_words.extend(['http', 'https', 'amp', 'com', 'said', 'says', 'would', 'could'])
                
                vectorizer = CountVectorizer(
                    ngram_range=(1, 2),
                    stop_words=stop_words,
                    min_df=2,
                    max_df=0.95,
                    max_features=1000
                )
                
                self.topic_model = BERTopic(
                    language="english",
                    vectorizer_model=vectorizer,
                    top_n_words=8,
                    nr_topics="auto",
                    calculate_probabilities=True,
                    verbose=False
                )
                print("✅ BERTopic initialized")
            except Exception as e:
                print(f"⚠️ BERTopic setup failed: {e}")
    
    def _try_parse_json(self, content: str) -> Dict:
        """Try to parse JSON, return fallback if it fails"""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"JSON parsing failed: {str(e)}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "unknown",
                "key_phrases": [],
                "reasoning": "Couldn't parse response properly"
            }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis using Groq"""
        system_prompt = (
            "You're a sentiment analyzer. Reply ONLY with JSON like this:\n"
            '{"sentiment": "positive|negative|neutral", "confidence": 0-1, '
            '"emotion": "emotion name", "key_phrases": ["phrase1", "phrase2"], '
            '"reasoning": "brief reason"}'
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this:\n\n{text[:2000]}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            result = response.choices[0].message.content.strip()
            return self._try_parse_json(result)
        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "unknown",
                "key_phrases": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def analyze_sentiment_advanced(self, text: str) -> Dict:
        """Use RoBERTa if possible, fall back to Groq"""
        if self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(text[:512])[0]
                
                # Map results
                sentiment_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
                primary = max(results, key=lambda x: x['score'])
                
                sentiment = sentiment_map.get(primary['label'], 'neutral')
                confidence = primary['score']
                
                # Basic emotion detection
                emotions = {
                    "positive": ["joy", "happiness", "optimism"],
                    "negative": ["disappointment", "anger", "frustration"],
                    "neutral": ["calm", "indifference", "objectivity"]
                }
                emotion = emotions.get(sentiment, ["unknown"])[min(2, int(confidence * 3))]
                
                # Extract some key phrases
                positive_words = {"good", "great", "excellent", "amazing", "love", "awesome"}
                negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor"}
                word_set = positive_words if sentiment == "positive" else (negative_words if sentiment == "negative" else set())
                
                words = text.lower().split()
                key_phrases = [w for w in words if w in word_set][:3]
                if not key_phrases:
                    key_phrases = [w for w in words[:3] if len(w) > 3]
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "emotion": emotion,
                    "key_phrases": key_phrases,
                    "reasoning": f"RoBERTa analysis with {confidence:.1%} confidence",
                    "model": "RoBERTa-Advanced",
                    "all_scores": {r['label']: r['score'] for r in results}
                }
            except Exception as e:
                print(f"RoBERTa failed: {e}")
        
        # Fallback to Groq
        result = self.analyze_sentiment(text)
        result["model"] = "Groq-Fallback"
        return result
    
    def generate_summary(self, text: str, max_sentences: int = 5) -> str:
        """Ask Groq to summarize text"""
        prompt = (
            f"Summarize this text in {max_sentences} clear, concise sentences. "
            "Reply with only the summary, no extra commentary."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\n{text[:3000]}"}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Couldn't generate summary: {str(e)}"
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict:
        """Use Groq to extract topics"""
        system_prompt = (
            f"Extract {num_topics} distinct topics. Return JSON with 'topics' key containing array of "
            "objects with: topic_name, keywords (7-10), description (1 sentence), relevance_score (0-1)"
        )
        
        try:
            combined = "\n\n".join([f"Doc {i+1}: {t[:600]}" for i, t in enumerate(texts[:15])])
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract topics from:\n\n{combined}"}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            result = response.choices[0].message.content.strip()
            return self._try_parse_json(result)
        except Exception as e:
            return {"topics": [], "error": f"Topic extraction failed: {str(e)}"}
    
    def extract_topics_advanced(self, texts: List[str], num_topics: int = 5) -> Dict:
        """Use BERTopic if available, otherwise fall back to Groq"""
        if self.topic_model and len(texts) >= 2:
            try:
                # Clean texts first
                cleaned = [t for t in texts if len(t.strip()) > 20]
                if len(cleaned) < 2:
                    return {"topics": [], "error": "Need at least 2 documents", "model": "BERTopic-Error"}
                
                # Run BERTopic
                topics, probs = self.topic_model.fit_transform(cleaned)
                topic_info = self.topic_model.get_topic_info()
                
                # Convert to our format
                extracted = []
                for idx, row in topic_info.iterrows():
                    if row['Topic'] == -1:  # Skip outliers
                        continue
                    
                    keywords = row['Representation'][:8]
                    topic_name = " & ".join(keywords[:2]).title()
                    desc = f"Topic focused on: {', '.join(keywords[:5])}"
                    relevance = float(row['Count'] / len(texts)) if len(texts) > 0 else 0.0
                    
                    # Try to enhance with Groq
                    try:
                        enhanced = self._enhance_topic(keywords)
                    except:
                        enhanced = desc
                    
                    extracted.append({
                        "topic_id": int(row['Topic']),
                        "topic_name": topic_name,
                        "keywords": list(keywords),
                        "description": desc,
                        "enhanced_description": enhanced,
                        "document_count": int(row['Count']),
                        "relevance_score": relevance,
                    })
                
                return {
                    "topics": extracted,
                    "num_topics": len(extracted),
                    "num_documents": len(texts),
                    "model": "BERTopic-Advanced"
                }
            except Exception as e:
                print(f"BERTopic failed: {e}")
        
        # Fallback to Groq
        result = self.extract_topics(texts, num_topics)
        if "error" not in result:
            result["model"] = "Groq-Fallback"
        return result
    
    def _enhance_topic(self, keywords: List[str]) -> str:
        """Use Groq to create a better description of a topic"""
        try:
            keywords_str = ", ".join(keywords[:5])
            prompt = f"Given these keywords: {keywords_str}. Write a 1-sentence topic description:"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        except:
            return f"Topic: {', '.join(keywords[:3])}"
    
    def analyze_comprehensive(self, texts: List[str]) -> Dict:
        """Do everything: sentiment + summary + topics"""
        combined = " ".join(texts)
        
        result = {
            "sentiment": self.analyze_sentiment_advanced(combined),
            "summary": self.generate_summary(combined)
        }
        
        # Only do topics if we have multiple documents
        if len(texts) >= 2:
            result["topics"] = self.extract_topics_advanced(texts)
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> Dict:
        """Basic batch analysis"""
        combined = " ".join(texts)
        result = {
            "sentiment": self.analyze_sentiment(combined),
            "summary": self.generate_summary(combined)
        }
        
        if len(texts) >= 2:
            result["topics"] = self.extract_topics(texts)
        
        return result
