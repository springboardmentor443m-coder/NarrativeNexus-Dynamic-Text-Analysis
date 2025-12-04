"""
Topic modeling module using LDA (Latent Dirichlet Allocation).
Based on the notebook's LDA implementation.
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def create_lda_model(texts: List[str], n_topics: int = 10, max_features: int = 5000) -> Tuple:
    """
    Create and train an LDA topic model with improved hyperparameters.
    
    Args:
        texts: List of documents
        n_topics: Number of topics to extract
        max_features: Maximum number of features for vectorizer
        
    Returns:
        Tuple of (vectorizer, lda_model)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.85,
        stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method='batch',
        learning_offset=50.0,
        n_jobs=-1,
        verbose=0
    )
    
    lda_model.fit(doc_term_matrix)
    
    return vectorizer, lda_model


def get_topics(vectorizer, lda_model, n_words: int = 10) -> List[List[str]]:
    """
    Extract top words for each topic.
    
    Args:
        vectorizer: Fitted CountVectorizer
        lda_model: Fitted LDA model
        n_words: Number of top words per topic
        
    Returns:
        List of lists, where each inner list contains top words for a topic
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    
    return topics


def infer_topics_for_text(
    text: str,
    vectorizer,
    lda_model,
    top_k_topics: int = 3,
    top_n_words: int = 6
) -> List[List[str]]:
    """
    Infer topics for a single document and return top words from dominant topics.
    
    Args:
        text: Document text to analyze
        vectorizer: Fitted CountVectorizer
        lda_model: Fitted LDA model
        top_k_topics: Number of top topics to return
        top_n_words: Number of top words per topic
        
    Returns:
        List of lists containing top words from dominant topics
    """
    try:
        doc_term_matrix = vectorizer.transform([text])
        
        if doc_term_matrix.nnz == 0:
            return [["no", "meaningful", "content"]]
        
        topic_dist = lda_model.transform(doc_term_matrix)[0]
        
        top_topic_indices = np.argsort(topic_dist)[-top_k_topics:][::-1]
        
        feature_names = vectorizer.get_feature_names_out()
        result_topics = []
        
        for topic_idx in top_topic_indices:
            if topic_dist[topic_idx] > 0.01:
                top_words_idx = lda_model.components_[topic_idx].argsort()[-top_n_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                result_topics.append(top_words)
        
        if not result_topics:
            return [["general", "topic", "discussion"]]
        
        return result_topics
    except Exception as e:
        print(f"Error inferring topics: {e}")
        return [["unable", "to", "infer"]]


def get_document_topics(vectorizer, lda_model, texts: List[str]) -> np.ndarray:
    """
    Get topic distribution for multiple documents.
    
    Args:
        vectorizer: Fitted CountVectorizer
        lda_model: Fitted LDA model
        texts: List of documents
        
    Returns:
        Array of shape (n_documents, n_topics)
    """
    doc_term_matrix = vectorizer.transform(texts)
    doc_topics = lda_model.transform(doc_term_matrix)
    return doc_topics


def get_topic_similarity(vectorizer, lda_model, text1: str, text2: str) -> float:
    """
    Calculate similarity between topic distributions of two texts.
    
    Args:
        vectorizer: Fitted CountVectorizer
        lda_model: Fitted LDA model
        text1: First document
        text2: Second document
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    topics1 = lda_model.transform(vectorizer.transform([text1]))[0]
    topics2 = lda_model.transform(vectorizer.transform([text2]))[0]
    
    cosine_sim = np.dot(topics1, topics2) / (np.linalg.norm(topics1) * np.linalg.norm(topics2) + 1e-10)
    
    return float(max(0, min(1, cosine_sim)))


def create_dummy_lda_model(n_topics: int = 10):
    """
    Create a dummy LDA model with sample data for demo purposes.
    
    Args:
        n_topics: Number of topics
        
    Returns:
        Tuple of (vectorizer, lda_model)
    """
    sample_texts = [
        "health care medicine doctor patient hospital",
        "technology computer software programming development",
        "sports football basketball game team",
        "politics government election president law",
        "science research experiment discovery study",
        "business company market economy finance",
        "education school student learning teacher",
        "culture art music entertainment movie",
        "environment nature climate weather ecology",
        "food cooking recipe restaurant kitchen",
    ] * 3
    
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=1,
        stop_words='english'
    )
    
    doc_term_matrix = vectorizer.fit_transform(sample_texts)
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        n_jobs=-1
    )
    
    lda_model.fit(doc_term_matrix)
    
    return vectorizer, lda_model


def load_training_texts_from_directory(directory_path: str, max_files: int = None) -> List[str]:
    """
    Load all TXT files from a directory for training data.
    
    Args:
        directory_path: Path to directory containing TXT files
        max_files: Maximum number of files to load (None for all)
        
    Returns:
        List of text documents loaded from files
    """
    texts = []
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    txt_files = list(path.rglob("*.txt"))
    
    if max_files:
        txt_files = txt_files[:max_files]
    
    print(f"[OK] Found {len(txt_files)} TXT files in {directory_path}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"[WARN] Error reading {file_path}: {e}")
    
    print(f"[OK] Loaded {len(texts)} documents for training")
    return texts


def create_lda_model_from_external_data(
    directory_path: str,
    n_topics: int = 10,
    max_features: int = 5000,
    max_files: int = None
) -> Tuple:
    """
    Create and train an LDA model using external TXT files with improved hyperparameters.
    
    Args:
        directory_path: Path to directory containing training TXT files
        n_topics: Number of topics to extract
        max_features: Maximum number of features for vectorizer
        max_files: Maximum number of files to load
        
    Returns:
        Tuple of (vectorizer, lda_model)
    """
    texts = load_training_texts_from_directory(directory_path, max_files)
    
    if not texts:
        raise ValueError("No text data found in the specified directory")
    
    print(f"[OK] Training LDA model with {len(texts)} documents...")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.85,
        stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    print(f"[OK] Document-term matrix shape: {doc_term_matrix.shape}")
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method='batch',
        learning_offset=50.0,
        n_jobs=-1,
        verbose=0
    )
    
    lda_model.fit(doc_term_matrix)
    print(f"[OK] LDA model trained successfully with {n_topics} topics")
    
    return vectorizer, lda_model
