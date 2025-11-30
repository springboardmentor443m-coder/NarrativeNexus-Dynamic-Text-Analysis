from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pickle
import os


def main():
    print("📥 Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    docs = [d for d in newsgroups.data if len(d.split()) >= 50][:2000]
    print(f"✅ Using {len(docs)} documents for training")

    print("🤖 Loading sentence transformer model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🧠 Training BERTopic model (this can take a few minutes)...")
    topic_model = BERTopic(
        embedding_model=embed_model,
        min_topic_size=10,
        language="english",
    )
    topics, _ = topic_model.fit_transform(docs)

    print("✅ Training complete")
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"📊 Discovered {n_topics} topics (excluding outliers)")

    model_dir = os.path.join("app", "models")
    os.makedirs(model_dir, exist_ok=True)

    print("💾 Saving BERTopic model...")
    topic_model.save(os.path.join(model_dir, "bertopic_model"))

    print("💾 Creating topic name mapping...")
    topic_names = {}
    unique_topics = sorted(set(topics))
    for tid in unique_topics:
        if tid == -1:
            topic_names[tid] = "Outliers"
            continue
        words = topic_model.get_topic(tid)
        if words:
            top_words = [w for w, _ in words[:3]]
            topic_names[tid] = "_".join(top_words)
        else:
            topic_names[tid] = f"Topic_{tid}"

    mapping = {"topic_names": topic_names}
    with open(os.path.join(model_dir, "topic_mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)

    print("✅ Model and mapping saved to app/models/")
    print("🎯 Now you can use TopicPredictor from app.models.topic_model")


if __name__ == "__main__":
    main()