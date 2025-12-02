from typing import List, Dict, Any, Optional
from groq import Groq

from backend.text_summarizer import TextSummarizer
from backend.sentiment_analyzer import SentimentAnalyzer
from backend.topic_modeler import TopicModeler

class GroqAnalyzer:
    """
    Orchestrator for the document analysis pipeline.
    Combines summarization (LLM), sentiment analysis (Transformer), 
    and topic modeling (BERTopic/Embeddings) into a single result.
    """

    def __init__(self, api_key: str, llm_model: str, embed_model: str):
        self.client = Groq(api_key=api_key)
        
        # Initialize sub-modules
        self.summarizer = TextSummarizer(groq_client=self.client, model=llm_model)
        self.sentiment = SentimentAnalyzer()
        self.topics = TopicModeler(embed_model_name=embed_model)

    def _get_empty_topic(self) -> Dict[str, Any]:
        """Returns default structure for failed or unmatched topic assignment."""
        return {
            "topic_id": -1,
            "topic_name": "Uncategorized",
            "similarity": 0.0
        }

    def analyze_single(self, text: str, file_name: str) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline on a single document.
        """
        # 1. Summarization
        try:
            summary = self.summarizer.summarize_text(text)
        except Exception as e:
            print(f"[Analyzer] Summary failed for {file_name}: {e}")
            summary = "Summary unavailable."

        # 2. Sentiment Analysis
        try:
            sentiment_data = self.sentiment.analyze_text(text)
        except Exception as e:
            print(f"[Analyzer] Sentiment failed for {file_name}: {e}")
            sentiment_data = {"sentiment": "Neutral", "confidence": 0.0}

        # 3. Topic Classification
        try:
            topic_id, topic_name, score = self.topics.find_closest_topic(text)
            
            if topic_name is None:
                topic_data = self._get_empty_topic()
            else:
                topic_data = {
                    "topic_id": int(topic_id),
                    "topic_name": topic_name,
                    "similarity": float(score)
                }
        except Exception as e:
            print(f"[Analyzer] Topic modeling failed for {file_name}: {e}")
            topic_data = self._get_empty_topic()

        return {
            "file_name": file_name,
            "summary": summary,
            "sentiment": sentiment_data,
            "assigned_topic": topic_data
        }

    def analyze_comprehensive(self, cleaned_texts: List[str], file_names: List[str]) -> Dict[str, Any]:
        """
        Processes a batch of documents and structures the response for the API.
        """
        results = [
            self.analyze_single(text, fname) 
            for text, fname in zip(cleaned_texts, file_names)
        ]

        return {
            "per_file": results,
            # Global assigned_topic is deprecated in favor of per-file topics,
            # but kept as empty dict if required by legacy DB schema.
            "assigned_topic": {} 
        }

def create_analyzer(api_key: str, llm_model: str, embed_model: str) -> GroqAnalyzer:
    """Factory function to instantiate the analyzer with config."""
    return GroqAnalyzer(api_key=api_key, llm_model=llm_model, embed_model=embed_model)