
# ---------------------------TOPIC MOEDLING USING LDA AND NMF-----------------------------

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation, NMF

# def perform_topic_modeling(cleaned_docs, num_topics=5, num_words=10, method="lda"):
#     if not cleaned_docs:
#         return {"error": "No text data provided"}
#     if isinstance(cleaned_docs, str):
#         cleaned_docs = [cleaned_docs]

#     # Convert token lists to strings if needed
#     if isinstance(cleaned_docs[0], list):
#         cleaned_docs = [" ".join(doc) for doc in cleaned_docs]

#     # ✅ Adjust thresholds dynamically
#     min_df_value = 1 if len(cleaned_docs) < 5 else 2
#     max_df_value = 1.0 if len(cleaned_docs) < 5 else 0.95

#     # Select vectorizer/model
#     if method.lower() == "lda":
#         vectorizer = CountVectorizer(max_df=max_df_value, min_df=min_df_value, stop_words='english')
#         model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#     else:
#         vectorizer = TfidfVectorizer(max_df=max_df_value, min_df=min_df_value, stop_words='english')
#         model = NMF(n_components=num_topics, random_state=42)

#     # ✅ Try fitting safely
#     try:
#         X = vectorizer.fit_transform(cleaned_docs)
#     except ValueError as e:
#         return {"error": f"Vectorization failed: {str(e)}"}

#     if X.shape[1] == 0:
#         return {"error": "No valid terms found in input text after preprocessing."}

#     model.fit(X)
#     terms = vectorizer.get_feature_names_out()

#     topics = []
#     for idx, topic in enumerate(model.components_):
#         top_words = [terms[i] for i in topic.argsort()[:-num_words - 1:-1]]
#         topics.append({
#             "topic_id": idx + 1,
#             "keywords": top_words
#         })

#     return {
#         "method": method.upper(),
#         "num_topics": num_topics,
#         "topics": topics
#     }

# ---------------------------- TOPIC MODELING USING TRANSFORMERS ----------------------------   

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import numpy as np
from app.utils.preprocessing import split_into_chunks

# Load transformer-based embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def perform_topic_modeling_transformers(text, num_topics=5, num_words=10):
    """
    Perform topic modeling using transformer embeddings + HDBSCAN clustering.
    Returns topics with representative text and keywords.
    """
    # Step 1: Split and deduplicate text chunks
    sentences = split_into_chunks(text, max_words=50)
    sentences = list(dict.fromkeys(sentences))
    if not sentences:
        return {"error": "No valid text chunks for topic modeling."}

    # Adjust topic count based on text size
    num_topics = min(num_topics, max(1, len(sentences)))

    # Step 2: Generate sentence embeddings
    embeddings = embedding_model.encode(sentences, batch_size=16, show_progress_bar=False)

    # Step 3: Cluster embeddings using HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, len(sentences) // num_topics),
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    topics = []

    # Handle noisy data points
    if cluster_labels.tolist().count(-1) > len(sentences) * 0.1:
        misc_text = " ".join([sentences[i] for i, l in enumerate(cluster_labels) if l == -1])
        topics.append({"topic_id": 0, "keywords": ["miscellaneous"], "text": misc_text})

    # Fallback to KMeans if HDBSCAN fails
    if len(set(cluster_labels)) <= 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_topics, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

    # Step 4: Extract top keywords per cluster
    unique_labels = [label for label in set(cluster_labels) if label != -1]
    for topic_id, label in enumerate(unique_labels, start=1):
        cluster_sentences = [sentences[i] for i, lbl in enumerate(cluster_labels) if lbl == label]
        if not cluster_sentences:
            continue

        vectorizer = TfidfVectorizer(stop_words='english', max_features=num_words)
        X = vectorizer.fit_transform(cluster_sentences)
        tfidf_scores = np.array(X.mean(axis=0)).flatten()
        keywords = [word for _, word in sorted(zip(tfidf_scores, vectorizer.get_feature_names_out()), reverse=True)]
        representative_text = " ".join(cluster_sentences[:3])

        topics.append({"topic_id": topic_id, "keywords": keywords, "text": representative_text})

    return {"method": "Transformers", "num_topics": len(topics), "topics": topics}
