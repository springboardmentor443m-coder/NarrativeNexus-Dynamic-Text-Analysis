from datasets import load_dataset
from text_preprocessing import preprocess_text
from topic_modeling import build_lda_model

def load_twitter_financial_sample(limit=2000):
    print(f"Loading {limit} tweets from Twitter Financial News dataset...")
    dataset = load_dataset("zeroshot/twitter-financial-news-topic", split=f"train[:{limit}]")
    texts = [item["text"] for item in dataset if item.get("text")]
    return texts

def analyze_twitter_financial_dataset(limit=2000, num_topics=5):
    texts = load_twitter_financial_sample(limit)
    tokenized_texts = [preprocess_text(t) for t in texts]

    lda_result = build_lda_model(tokenized_texts, num_topics=num_topics)

    # Ensure consistent structured output
    return {
        "sample_size": len(texts),
        "topics": lda_result["topics"],
        "coherence": lda_result["coherence"]
    }

if __name__ == "__main__":
    result = analyze_twitter_financial_dataset(limit=500)
    print(f"Analyzed {result['sample_size']} tweets.")
    print(f"Coherence Score: {result['coherence']:.3f}")
    for t in result["topics"]:
        print(f"Topic {t['topic_id']}: {', '.join(t['keywords'])}")

