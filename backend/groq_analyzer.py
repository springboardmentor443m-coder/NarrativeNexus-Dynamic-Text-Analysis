# backend/groq_analyzer.py

from typing import List, Dict, Any, Optional
from groq import Groq

from backend.sentiment_analyzer import SentimentAnalyzer
from backend.topic_modeler import TopicModeler
from backend.text_summarizer import TextSummarizer


class GroqAnalyzer:
    
    def __init__(
        self,
        api_key: str,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.client = Groq(api_key=api_key)
        self.llm_model = llm_model

        # Sentiment now uses HuggingFace only
        self.sentiment = SentimentAnalyzer()

        # Your existing summarizer (Groq-based)
        self.summarizer = TextSummarizer(self.client, model=llm_model)

        # Your topic modeler
        self.topics = TopicModeler(
            groq_client=self.client,
            llm_model=llm_model,
            embedding_general="all-mpnet-base-v2",
            embedding_technical="all-distilroberta-v1"
        )

    # ------------------------ MAIN FUNCTION ------------------------
    def analyze_comprehensive(
        self,
        documents: List[str],
        file_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        if isinstance(documents, str):
            documents = [documents]

        if not file_names:
            file_names = [f"Document {i+1}" for i in range(len(documents))]

        # ---------- 1. PER-FILE SENTIMENT ----------
        per_file_sent = self.sentiment.analyze_per_file(documents, file_names)


        # ---------- 2. PER-FILE SUMMARIES ----------
        per_file_summaries = self.summarizer.summarize_documents(
            documents, file_names
        )

        # Merge sentiment + summary
        merged_output = []
        for i, name in enumerate(file_names):
            merged_output.append({
                "file_name": name,
                "sentiment": per_file_sent[i],
                "summary": per_file_summaries[i]["summary"]
            })

        # ---------- 3. TOPICS (combined corpus) ----------
        combined_text = " ".join(documents)
        topics = self.topics.extract_topics(combined_text)

        return {
            "per_file": merged_output,
            "topics": topics
        }


def create_analyzer(
    api_key: str,
    llm_model: str = "llama-3.3-70b-versatile",
    embedding_model_name: str = "all-MiniLM-L6-v2"
):
    return GroqAnalyzer(
        api_key,
        llm_model=llm_model,
        embedding_model_name=embedding_model_name
    )
