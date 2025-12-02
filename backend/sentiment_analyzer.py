import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["Negative", "Neutral", "Positive"]
        self.max_tokens = 512

    def _split_into_chunks(self, text: str, max_words: int = 400) -> List[str]:
        """Splits text into chunks that fit within the model's token limit."""
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def _analyze_chunk(self, text: str) -> Dict[str, Any]:
        """Runs inference on a single chunk of text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_tokens
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
        label_idx = np.argmax(scores)
        
        return {
            "label": self.labels[label_idx],
            "score": float(scores[label_idx])
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Aggregates sentiment scores from chunks to determine overall document sentiment.
        """
        if not text or not text.strip():
            return {"sentiment": "Neutral", "confidence": 0.0}

        chunks = self._split_into_chunks(text)
        chunk_results = [self._analyze_chunk(chunk) for chunk in chunks]

        # Aggregate scores
        score_sums = {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
        count = len(chunk_results)

        for res in chunk_results:
            score_sums[res["label"]] += res["score"]

        # Determine dominant sentiment
        final_sentiment = max(score_sums, key=score_sums.get)
        
        # Calculate average confidence for that sentiment
        # (We normalize by count to keep it between 0.0 and 1.0)
        avg_confidence = score_sums[final_sentiment] / count

        return {
            "sentiment": final_sentiment,
            "confidence": round(avg_confidence, 4)
        }