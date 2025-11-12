from groq import Groq
import os
from typing import List, Dict

# Import our specialized modules
from backend.sentiment_analyzer import SentimentAnalyzer
from backend.topic_modeler import TopicModeler
from backend.text_summarizer import TextSummarizer


class GroqAnalyzer:
    """
    Main text analyzer that coordinates everything.
    Uses specialized modules for each task:
    - SentimentAnalyzer: Analyzes sentiment
    - TopicModeler: Discovers topics
    - TextSummarizer: Creates summaries
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Set up the analyzer.
        api_key: Your Groq API key
        model: Groq model to use (default is good)
        """
        # Create Groq client
        self.client = Groq(api_key=api_key)
        self.model = model

        # Initialize our specialized analyzers
        print("🔧 Setting up analyzers...")
        self.sentiment = SentimentAnalyzer(self.client, self.model)
        self.topics = TopicModeler(self.client, self.model)
        self.summarizer = TextSummarizer(self.client, self.model)
        print("✅ All analyzers ready!")

    # ==================================
    # SIMPLE METHODS (one task at a time)
    # ==================================

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Just analyze sentiment.
        text: Text to analyze
        Returns: Sentiment results
        """
        return self.sentiment.analyze(text)

    def analyze_sentiment_advanced(self, text: str) -> Dict:
        """
        Advanced sentiment analysis.
        Returns detailed sentiment with confidence scores.
        """
        return self.sentiment.analyze(text)

    def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict:
        """
        Just extract topics.
        texts: List of documents
        num_topics: How many topics to find
        Returns: Topic results
        """
        return self.topics.extract(texts, num_topics)

    def extract_topics_advanced(self, texts: List[str], num_topics: int = 5) -> Dict:
        """
        Advanced topic extraction with BERTopic.
        texts: List of documents to analyze
        num_topics: How many topics to find
        Returns: Detailed topic modeling results
        """
        return self.topics.extract(texts, num_topics)

    def generate_summary(self, text: str, max_sentences: int = 5) -> str:
        """
        Just generate a summary.
        text: Text to summarize
        max_sentences: Length of summary
        Returns: Summary text
        """
        return self.summarizer.summarize(text, max_sentences)

    # ==================================
    # COMPREHENSIVE METHOD (do everything)
    # ==================================

    def analyze_comprehensive(self, texts: List[str]) -> Dict:
        """
        Do complete analysis: sentiment + topics + summary.
        This is the most useful method!
        texts: List of documents to analyze
        Returns: Dictionary with all results
        """
        print("🔍 Starting comprehensive analysis...")

        # Combine all text for sentiment and summary
        combined_text = " ".join(texts)

        # Run sentiment analysis
        print(" - Analyzing sentiment...")
        sentiment_results = self.sentiment.analyze(combined_text)

        # Generate summary
        print(" - Creating summary...")
        summary_text = self.summarizer.summarize(combined_text)

        # Prepare results
        results = {
            "sentiment": sentiment_results,
            "summary": summary_text,
            "document_count": len(texts)
        }

        # Extract topics (only if we have 2+ documents)
        if len(texts) >= 2:
            print(" - Discovering topics...")
            topic_results = self.topics.extract(texts)
            results["topics"] = topic_results
        else:
            results["topics"] = {"topics": [], "note": "Need 2+ documents for topics"}

        print("✅ Analysis complete!")
        return results


# ==================================
# HELPER FUNCTION (easy setup)
# ==================================

def create_analyzer(api_key: str = None) -> GroqAnalyzer:
    """
    Easy way to create an analyzer.
    Gets API key from environment if not provided.
    Usage:
    analyzer = create_analyzer()
    results = analyzer.analyze_comprehensive(texts)
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please provide api_key or set GROQ_API_KEY environment variable")

    return GroqAnalyzer(api_key)
