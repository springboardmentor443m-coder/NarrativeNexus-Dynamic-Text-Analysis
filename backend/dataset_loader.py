# backend/dataset_loader.py
from datasets import load_dataset
from text_preprocessing import preprocess_text
from semantic_topic_modeling import build_semantic_topics
from collections import Counter

def load_twitter_financial_sample(limit=2000):
    dataset = load_dataset("zeroshot/twitter-financial-news-topic", split=f"train[:{limit}]")
    texts, labels = [], []
    for item in dataset:
        if isinstance(item, dict) and "text" in item and "label" in item:
            texts.append(item["text"])
            labels.append(str(item["label"]))
    return texts, labels


def analyze_twitter_financial_dataset(limit=2000, num_topics=5):
    texts, labels = load_twitter_financial_sample(limit)
    tokenized_texts = [preprocess_text(t) for t in texts]
    joined_docs = [" ".join(toks) for toks in tokenized_texts if toks]
    topics = build_semantic_topics(joined_docs, num_topics=num_topics)
    return {
        "sample_size": len(texts),
        "true_labels": dict(Counter(labels)),
        "predicted_topics": topics,
    }
