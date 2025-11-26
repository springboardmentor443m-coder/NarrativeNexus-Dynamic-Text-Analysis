from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def nmf_topics(docs, n_topics=5, n_words=8):
    """
    NMF-based topic modeling using TF-IDF.
    Works well for smaller, cleaner text corpora.
    """
    n_docs = len(docs)
    vec = TfidfVectorizer(min_df=1, max_df=1.0)
    X = vec.fit_transform(docs)
    n_samples, n_features = X.shape
    if n_features == 0:
        return [], [0] * n_docs, None, vec

    n_components = max(1, min(n_topics, n_features, n_samples))
    init_method = "nndsvda" if n_components <= min(n_samples, n_features) else "random"

    try:
        model = NMF(n_components=n_components, random_state=42, init=init_method, max_iter=400)
        W = model.fit_transform(X)
    except ValueError:
        model = NMF(n_components=n_components, random_state=42, init="random", max_iter=400)
        W = model.fit_transform(X)

    H = model.components_
    terms = vec.get_feature_names_out()

    topics = []
    for i, comp in enumerate(H):
        top_k = min(n_words, comp.size)
        top_idx = comp.argsort()[::-1][:top_k]
        topics.append({
            "topic_id": i,
            "keywords": [terms[j] for j in top_idx],
            "weight": float(W[:, i].mean())
        })
    doc_topic = W.argmax(axis=1) if W.size else [0] * n_docs
    return topics, doc_topic, model, vec


def lda_topics(docs, n_topics=5, n_words=8):
    """
    LDA-based topic modeling using word counts.
    Works safely for single-document and multi-document input.
    """
    n_docs = len(docs)

    # Choose vectorizer settings dynamically
    if n_docs <= 1:
        vec = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')  # single-doc safe
    else:
        vec = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    X = vec.fit_transform(docs)
    n_samples, n_features = X.shape

    if n_features == 0:
        return [], [0] * n_docs, None, vec

    # Limit number of components to avoid overfitting on tiny corpora
    n_components = max(1, min(n_topics, n_features, n_docs))

    model = LatentDirichletAllocation(
        n_components=n_components,
        random_state=42,
        learning_method='batch'
    )
    W = model.fit_transform(X)
    H = model.components_
    terms = vec.get_feature_names_out()

    topics = []
    for i, comp in enumerate(H):
        top_k = min(n_words, comp.size)
        top_idx = comp.argsort()[::-1][:top_k]
        topics.append({
            "topic_id": i,
            "keywords": [terms[j] for j in top_idx],
            "weight": float(W[:, i].mean())
        })

    doc_topic = W.argmax(axis=1) if W.size else [0] * n_docs
    return topics, doc_topic, model, vec

