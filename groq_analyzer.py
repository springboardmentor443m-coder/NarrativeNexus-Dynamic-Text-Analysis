"""
Advanced Groq LLM Analyzer with Transformers & BERTopic Integration
Handles sentiment, summarization, and topic extraction with improved logging
"""

from groq import Groq
import json
import os
from typing import List, Dict

# Advanced sentiment analysis with transformers
try:
    from transformers import pipeline
    import torch
    import numpy as np
    from scipy.special import softmax
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] Transformers not installed. Using Groq-only sentiment analysis.")

# Advanced topic modeling with BERTopic
try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import nltk
    from nltk.corpus import stopwords
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("[WARNING] BERTopic not available. Using Groq-only topic modeling.")


class GroqAnalyzer:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        
        # Initialize advanced sentiment analyzer
        self.sentiment_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    return_all_scores=True
                )
                print("[INFO] ✅ Advanced transformer sentiment analyzer initialized (RoBERTa)")
            except Exception as e:
                print(f"[WARNING] Failed to initialize transformer sentiment: {e}")
        
        # Initialize advanced topic modeler
        self.topic_model = None
        if BERTOPIC_AVAILABLE:
            try:
                # Download stopwords if not available
                try:
                    stopwords.words('english')
                except LookupError:
                    nltk.download('stopwords')
                
                stop_words = list(stopwords.words('english'))
                stop_words.extend(['http', 'https', 'amp', 'com', 'said', 'says', 'would', 'could'])
                
                vectorizer_model = CountVectorizer(
                    ngram_range=(1, 2),
                    stop_words=stop_words,
                    min_df=2,
                    max_df=0.95,
                    max_features=1000
                )
                
                self.topic_model = BERTopic(
                    language="english",
                    vectorizer_model=vectorizer_model,
                    top_n_words=8,
                    nr_topics="auto",
                    calculate_probabilities=True,
                    verbose=False
                )
                print("[INFO] ✅ BERTopic initialized for advanced topic modeling")
            except Exception as e:
                print(f"[WARNING] BERTopic initialization failed: {e}")

    def try_load_json(self, content: str) -> Dict:
        """Safely parse JSON content, fallback with error and log raw content"""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[GroqAnalyzer] JSON parsing failed: {str(e)}")
            print(f"[GroqAnalyzer] Raw API response content:\n{content}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "unknown",
                "key_phrases": [],
                "reasoning": "No valid JSON response from Groq API"
            }

    def analyze_sentiment(self, text: str) -> Dict:
        """Advanced sentiment analysis with reasoning using Groq"""
        system_prompt = (
            "You are an expert sentiment analyzer. "
            "Reply ONLY with a valid JSON object exactly like:\n"
            '{"sentiment": "positive|negative|neutral", '
            '"confidence": float (0-1), '
            '"emotion": "primary emotion", '
            '"key_phrases": [list of 3 key phrases], '
            '"reasoning": "brief explanation"}.\n'
            "No extra text or explanation."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze sentiment:\n\n{text[:2000]}"}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            result = response.choices[0].message.content.strip()
            if not result:
                raise ValueError("Empty response from Groq API")
            return self.try_load_json(result)
        except Exception as e:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotion": "unknown",
                "key_phrases": [],
                "reasoning": f"Error during sentiment analysis: {str(e)}"
            }

    def analyze_sentiment_advanced(self, text: str) -> Dict:
        """Advanced sentiment analysis using transformers with fallback to Groq"""
        if self.sentiment_analyzer:
            try:
                # Use transformer model
                results = self.sentiment_analyzer(text[:512])[0]
                
                # Process results
                sentiment_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
                primary_result = max(results, key=lambda x: x['score'])
                
                sentiment = sentiment_map.get(primary_result['label'], primary_result['label'].lower())
                confidence = primary_result['score']
                
                # Generate emotion based on sentiment and confidence
                emotion_map = {
                    "positive": ["joy", "happiness", "optimism"][min(2, int(confidence * 3))],
                    "negative": ["disappointment", "anger", "frustration"][min(2, int(confidence * 3))],
                    "neutral": ["calm", "indifference", "objectivity"][min(2, int(confidence * 3))]
                }
                
                emotion = emotion_map.get(sentiment, "unknown")
                
                # Extract key phrases (simple approach)
                words = text.lower().split()
                positive_words = {"good", "great", "excellent", "amazing", "love", "awesome", "perfect", "wonderful"}
                negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor", "disappointing", "horrible"}
                
                word_set = positive_words if sentiment == "positive" else (
                    negative_words if sentiment == "negative" else set()
                )
                
                key_phrases = [w for w in words if w in word_set][:3]
                if not key_phrases:
                    key_phrases = [w for w in words[:3] if len(w) > 3]
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "emotion": emotion,
                    "key_phrases": key_phrases,
                    "reasoning": f"Advanced transformer analysis with {confidence:.1%} confidence using RoBERTa model",
                    "model": "RoBERTa-Advanced",
                    "all_scores": {r['label']: r['score'] for r in results}
                }
                
            except Exception as e:
                print(f"[ERROR] Transformer sentiment failed: {e}")
        
        # Fallback to original Groq method
        result = self.analyze_sentiment(text)
        result["model"] = "Groq-Fallback"
        return result

    def generate_summary(self, text: str, max_sentences: int = 5) -> str:
        """Generate concise, accurate summary with strict JSON prompt"""
        system_prompt = (
            f"You are an expert at creating concise summaries. "
            f"Summarize the text in {max_sentences} clear sentences. "
            "Reply ONLY with the plain summary text, no extra commentary."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize:\n\n{text[:3000]}"}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            if not result:
                return "Summary generation failed: empty response from Groq API."
            return result
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict:
        """Extract topics from multiple documents with strict JSON output"""
        system_prompt = (
            f"You are a topic modeling expert. Extract {num_topics} distinct topics from the provided documents. "
            "Reply ONLY with a valid JSON object with a 'topics' key that maps to an array of objects, each object containing: "
            "'topic_name' (string), 'keywords' (array of 7-10 strings), "
            "'description' (1 sentence string), 'relevance_score' (float between 0 and 1). "
            "No extra text."
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
            if not result:
                return {"topics": [], "error": "Empty response from Groq API"}
            return self.try_load_json(result)
        except Exception as e:
            return {"topics": [], "error": f"Topic extraction failed: {str(e)}"}

    def extract_topics_advanced(self, texts: List[str], num_topics: int = 5) -> Dict:
        """Advanced topic modeling using BERTopic with fallback to Groq"""
        if self.topic_model and len(texts) >= 2:
            try:
                # Clean texts
                cleaned_texts = [t for t in texts if len(t.strip()) > 20]
                if len(cleaned_texts) < 2:
                    return {"topics": [], "error": "Need at least 2 documents with sufficient content", "model": "BERTopic-Error"}
                
                # Fit BERTopic
                topics, probs = self.topic_model.fit_transform(cleaned_texts)
                topic_info = self.topic_model.get_topic_info()
                
                # Extract topics
                extracted_topics = []
                for idx, row in topic_info.iterrows():
                    if row['Topic'] == -1:  # Skip outliers
                        continue
                    
                    topic_id = int(row['Topic'])
                    keywords = row['Representation'][:8]
                    
                    topic_name = " & ".join(keywords[:2]).title()
                    description = f"Topic focused on: {', '.join(keywords[:5])}"
                    relevance_score = float(row['Count'] / len(texts)) if len(texts) > 0 else 0.0
                    
                    # Use Groq to enhance topic description
                    try:
                        enhanced_desc = self.enhance_topic_with_groq(keywords)
                    except:
                        enhanced_desc = description
                    
                    extracted_topics.append({
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "keywords": list(keywords),
                        "description": description,
                        "enhanced_description": enhanced_desc,
                        "document_count": int(row['Count']),
                        "relevance_score": relevance_score,
                        "top_words": list(keywords[:5])
                    })
                
                return {
                    "topics": extracted_topics,
                    "num_topics": len(extracted_topics),
                    "num_documents": len(texts),
                    "model": "BERTopic-Advanced"
                }
                
            except Exception as e:
                print(f"[ERROR] BERTopic failed: {e}")
        
        # Fallback to original Groq method
        result = self.extract_topics(texts, num_topics)
        if "error" not in result:
            result["model"] = "Groq-Fallback"
        return result

    def enhance_topic_with_groq(self, keywords: List[str]) -> str:
        """Use Groq to enhance topic description"""
        try:
            keywords_str = ", ".join(keywords[:5])
            prompt = f"Given these keywords: {keywords_str}. Write a brief 1-sentence topic description:"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        except:
            return f"Topic related to: {', '.join(keywords[:3])}"

    def analyze_comprehensive(self, texts: List[str]) -> Dict:
        """Comprehensive analysis using advanced methods with Groq fallback"""
        combined = " ".join(texts)
        
        result = {
            "sentiment": self.analyze_sentiment_advanced(combined),
            "summary": self.generate_summary(combined)
        }
        
        if len(texts) >= 2:
            result["topics"] = self.extract_topics_advanced(texts)
        
        return result

    def analyze_batch(self, texts: List[str]) -> Dict:
        """Comprehensive batch analysis"""
        combined = " ".join(texts)
        result = {
            "sentiment": self.analyze_sentiment(combined),
            "summary": self.generate_summary(combined)
        }
        
        if len(texts) >= 2:
            result["topics"] = self.extract_topics(texts)
        
        return result
