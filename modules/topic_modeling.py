# modules/topic_modeling.py

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def chunk_text(text, chunk_words=1000, overlap=200, max_chunks=60):
    """
    Splits large documents into overlapping word chunks.
    """
    tokens = text.split()
    chunks = []
    i = 0

    while i < len(tokens):
        chunk = tokens[i:i + chunk_words]
        if not chunk:
            break

        chunks.append(" ".join(chunk))

        i += chunk_words - overlap
        if max_chunks and len(chunks) >= max_chunks:
            break

    return chunks


def perform_hybrid_lda(
        text,
        n_topics=4,
        top_n=6,
        chunk_words=1000,
        overlap=200,
        max_chunks=60
):
    """
    Performs scalable LDA topic modeling:
    - Cleaning
    - Chunking for large docs
    - TF-IDF + LDA hybrid
    Returns:
        topics_readable, topic_keywords, doc_topic distribution
    """

    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    chunks = chunk_text(cleaned, chunk_words, overlap, max_chunks)

    if len(chunks) == 0:
        return [], [], []

    # Vectorizer
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=1
    )
    X = vec.fit_transform(chunks)

    # LDA Model
    k = min(n_topics, len(chunks))
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="online",
        max_iter=12
    )
    lda.fit(X)

    terms = np.array(vec.get_feature_names_out())
    topic_keywords = []

    for comp in lda.components_:
        top_idx = comp.argsort()[-top_n:][::-1]
        top_terms = terms[top_idx].tolist()
        topic_keywords.append(", ".join(top_terms))

    # Document-wide topic distribution
    doc_topic_dist = lda.transform(X).mean(axis=0).tolist()

    topics_readable = [f"Topic {i + 1}" for i in range(len(topic_keywords))]

    return topics_readable, topic_keywords, doc_topic_dist
