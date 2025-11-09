# backend/semantic_topic_modeling.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import numpy as np

# Initialize models once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
label_generator = pipeline("summarization", model="facebook/bart-large-cnn")


def build_semantic_topics(docs, num_topics=5):
    """
    Build semantic clusters using embeddings and automatically label each cluster.
    """
    if not docs or all(not isinstance(d, str) or not d.strip() for d in docs):
        return {}

    # Step 1: Embedding
    embeddings = embedding_model.encode(docs, convert_to_tensor=True)
    docs = [d for d in docs if isinstance(d, str) and len(d.strip()) > 5]

    # Step 2: Clustering
    kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Step 3: Extract representative words via TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X_tfidf = tfidf.fit_transform(docs)
    terms = tfidf.get_feature_names_out()

    topic_keywords = {}
    for i in range(num_topics):
        cluster_docs_idx = np.where(cluster_labels == i)[0]
        if len(cluster_docs_idx) == 0:
            continue

        # Get mean TF-IDF score for words in this cluster
        cluster_tfidf = np.asarray(X_tfidf[cluster_docs_idx].mean(axis=0)).ravel()
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        keywords = [terms[idx] for idx in top_indices]

        # Step 4: Automatic topic label generation (BART summarizer)
        joined_text = " ".join(keywords)
        try:
            summary = label_generator(joined_text, max_length=12, min_length=3, do_sample=False)[0]["summary_text"]
        except Exception:
            summary = "General Topic"

        topic_keywords[i] = {
            "topic_id": i,
            "label": summary.strip(),
            "keywords": keywords,
        }

    return topic_keywords


