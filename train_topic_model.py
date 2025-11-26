# train_topic_model.py
import os
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def main():
    print("▶ Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),  # cleaner text
    )

    texts = dataset.data          # list of documents
    labels = dataset.target       # integer labels
    target_names = dataset.target_names  # list of topic names

    print(f"Loaded {len(texts)} documents across {len(target_names)} topics.")

    # TF-IDF + Logistic Regression pipeline
    print("▶ Building TF-IDF + Logistic Regression pipeline...")
    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=None,  # keep it simple & portable
                    verbose=0,
                ),
            ),
        ]
    )

    print("▶ Training classifier (this may take a minute)...")
    clf.fit(texts, labels)

    # Quick training accuracy check
    preds = clf.predict(texts)
    acc = accuracy_score(labels, preds)
    print(f"Training accuracy: {acc:.3f}")
    print(classification_report(labels, preds, target_names=target_names))

    # Save pipeline + label names
    os.makedirs("models", exist_ok=True)

    pipeline_path = os.path.join("models", "topic_pipeline.pkl")
    labels_path = os.path.join("models", "topic_labels.pkl")

    with open(pipeline_path, "wb") as f:
        pickle.dump(clf, f)

    with open(labels_path, "wb") as f:
        pickle.dump(target_names, f)

    print(f"✅ Saved pipeline to {pipeline_path}")
    print(f"✅ Saved topic labels to {labels_path}")
    print("Done.")


if __name__ == "__main__":
    main()
