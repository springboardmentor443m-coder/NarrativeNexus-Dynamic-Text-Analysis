# modules/topic_modeling.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

def perform_lda(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Handle both short and long text cases
    vectorizer_params = {
        "stop_words": "english"
    }

    # Adjust df params only if multiple documents exist
    if isinstance(text, list) and len(text) > 1:
        vectorizer_params["max_df"] = 0.95
        vectorizer_params["min_df"] = 2

    vectorizer = CountVectorizer(**vectorizer_params)
    doc_term_matrix = vectorizer.fit_transform([cleaned_text])

    lda = LatentDirichletAllocation(
        n_components=3,
        random_state=42,
        learning_method='online'
    )
    lda.fit(doc_term_matrix)

    topics = lda.transform(doc_term_matrix)
    topic_keywords = []

    for idx, topic in enumerate(lda.components_):
        top_features_indices = topic.argsort()[:-6:-1]
        top_features = [vectorizer.get_feature_names_out()[i] for i in top_features_indices]
        topic_keywords.append(", ".join(top_features))

    return topics, topic_keywords
