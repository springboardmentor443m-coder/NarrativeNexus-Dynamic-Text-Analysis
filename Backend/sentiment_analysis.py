"""
Sentiment analysis module using TF-IDF and Logistic Regression.
Based on the notebook's sentiment classification approach.
"""

import joblib
import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_model(model_path: str) -> dict:
    """
    Load a pre-trained sentiment model (TF-IDF + Logistic Regression pipeline).
    
    Args:
        model_path: Path to the saved joblib model
        
    Returns:
        Dictionary containing 'vectorizer' and 'classifier'
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def predict_labels(model: dict, texts: List[str]) -> List[str]:
    """
    Predict sentiment labels for a list of texts.
    
    Args:
        model: Dictionary with 'vectorizer' and 'classifier' keys
        texts: List of text strings to classify
        
    Returns:
        List of predicted sentiment labels
    """
    if not texts or all(not t for t in texts):
        return ["neutral"] * len(texts)
    
    try:
        vectorizer = model.get('vectorizer')
        classifier = model.get('classifier')
        
        if vectorizer is None or classifier is None:
            return ["neutral"] * len(texts)
        
        X = vectorizer.transform(texts)
        predictions = classifier.predict(X)
        
        # Convert numeric predictions to labels
        labels = []
        for pred in predictions:
            if pred == 0:
                labels.append("negative")
            elif pred == 1:
                labels.append("neutral")
            else:
                labels.append("positive")
        
        return labels
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        return ["neutral"] * len(texts)


def predict_probabilities(model: dict, texts: List[str]) -> np.ndarray:
    """
    Get confidence scores for sentiment predictions.
    
    Args:
        model: Dictionary with 'vectorizer' and 'classifier' keys
        texts: List of text strings
        
    Returns:
        Array of prediction probabilities
    """
    try:
        vectorizer = model.get('vectorizer')
        classifier = model.get('classifier')
        
        if vectorizer is None or classifier is None:
            return np.array([[0.33, 0.34, 0.33]] * len(texts))
        
        X = vectorizer.transform(texts)
        probabilities = classifier.predict_proba(X)
        return probabilities
    except Exception as e:
        print(f"Error during probability prediction: {e}")
        return np.array([[0.33, 0.34, 0.33]] * len(texts))


def create_dummy_model() -> dict:
    """
    Create a sentiment model with expanded training data for demo/testing.
    Uses more comprehensive examples to detect sentiment patterns.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    dummy_texts = [
        "this is great and wonderful",
        "excellent work very good",
        "i love this amazing",
        "this is bad terrible",
        "awful horrible poor",
        "i hate this disgusting",
        "this is okay neutral",
        "neither good nor bad",
        "average reasonable acceptable",
        "God atheism religion faith",
        "logical reasoning evidence",
        "philosophical discussion argument",
        "church state separation",
        "belief skeptical question",
    ]
    dummy_labels = [2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1]
    
    vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
    X = vectorizer.fit_transform(dummy_texts)
    
    classifier = LogisticRegression(random_state=42, max_iter=200)
    classifier.fit(X, dummy_labels)
    
    return {
        'vectorizer': vectorizer,
        'classifier': classifier
    }


def rule_based_sentiment(text: str) -> str:
    """
    Rule-based sentiment analysis as fallback for better semantic understanding.
    Analyzes text for sentiment keywords and patterns.
    """
    text_lower = text.lower()
    
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'perfect',
        'best', 'awesome', 'fantastic', 'brilliant', 'outstanding', 'superb',
        'logical', 'rational', 'evidence', 'clear', 'strong', 'valid',
        'reason', 'knowledge', 'understand', 'learn', 'improve', 'better'
    }
    
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting',
        'useless', 'stupid', 'dumb', 'poor', 'weak', 'failing', 'reject',
        'irrational', 'illogical', 'wrong', 'false', 'contradictory', 'nonsense',
        'contradiction', 'absurd', 'lack', 'fail', 'problem', 'issue'
    }
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"
