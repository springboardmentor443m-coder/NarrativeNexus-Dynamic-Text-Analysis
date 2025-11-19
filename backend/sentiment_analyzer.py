# backend/sentiment_analyzer.py
import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:

    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # HF model labels
        self.labels = ["Negative", "Neutral", "Positive"]

        # Max tokens allowed by RoBERTa
        self.max_tokens = 512

    # --------------------------------------
    # Split long text into smaller chunks
    # --------------------------------------
    def _chunk_text(self, text: str, max_len: int = 450) -> List[str]:
        """
        Splits long text into manageable word chunks (<450 tokens).
        Ensures no overflow and better accuracy.
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_len):
            chunk = " ".join(words[i:i + max_len]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    # --------------------------------------
    # Run sentiment on a single chunk
    # --------------------------------------
    def _analyze_chunk(self, text: str) -> Dict:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = self.model(**tokens)

        scores = torch.softmax(output.logits, dim=1).numpy()[0]
        idx = int(np.argmax(scores))
        confidence = float(scores[idx])

        return {
            "label": self.labels[idx],
            "confidence": confidence
        }

    # --------------------------------------
    # Public: analyze full document
    # --------------------------------------
    def analyze_text(self, text: str) -> Dict:
        """
        Chunk-based sentiment:
        - Split text
        - Analyze each chunk
        - Average confidence for final result
        """
        text = text.strip()
        if not text:
            return {"sentiment": "Neutral", "confidence": 0.0}

        chunks = self._chunk_text(text)

        results = [self._analyze_chunk(c) for c in chunks]

        # Aggregate
        label_scores = {"Negative": 0, "Neutral": 0, "Positive": 0}
        for r in results:
            label_scores[r["label"]] += r["confidence"]

        final_label = max(label_scores, key=label_scores.get)
        total_conf = label_scores[final_label] / len(results)

        return {
            "sentiment": final_label,
            "confidence": round(total_conf, 4)
        }

    # --------------------------------------
    # Analyze list of documents
    # --------------------------------------
    def analyze_per_file(self, docs: List[str], file_names: List[str]):
        output = []

        for text, fname in zip(docs, file_names):
            result = self.analyze_text(text)
            output.append({
                "file_name": fname,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })

        return output
