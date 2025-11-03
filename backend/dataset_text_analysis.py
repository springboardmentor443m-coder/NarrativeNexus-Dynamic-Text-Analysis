# backend/dataset_text_analysis.py

from datasets import load_dataset
from text_analysis import clean_summarize_analyze

def analyze_twitter_financial_dataset_text(limit=200):
    """
    Runs text cleaning, summarization, and sentiment analysis
    on the Hugging Face Twitter Financial News dataset.
    """
    print(f"Loading {limit} samples from Hugging Face dataset...")
    dataset = load_dataset("zeroshot/twitter-financial-news-topic", split=f"train[:{limit}]")
    
    results = []
    for i, item in enumerate(dataset):
        text = item.get("text", "").strip()
        if not text:
            continue
        analysis = clean_summarize_analyze(text)
        results.append({
            "index": i,
            "original": text,
            "cleaned_preview": analysis["preview"],
            "summary": analysis["summary"],
            "sentiment_label": analysis["sentiment"]["label"],
            "sentiment_score": analysis["sentiment"]["score"]
        })
    
    return results
