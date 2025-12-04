#!/usr/bin/env python
"""
Quick test script to verify modules work correctly.
"""

import sys
sys.path.insert(0, '.')

print("Testing modules...")

print("\n1. Testing preprocessing module...")
from preprocessing import clean_text, preprocess_for_lda, tokenize, remove_stopwords

text = "Hello! This is a TEST text with URLs http://example.com and special chars!"
cleaned = clean_text(text)
print(f"   Original: {text}")
print(f"   Cleaned: {cleaned}")

preprocessed = preprocess_for_lda(text)
print(f"   Preprocessed: {preprocessed}")

print("\n2. Testing sentiment_analysis module...")
from sentiment_analysis import create_dummy_model, predict_labels

model = create_dummy_model()
texts = ["this is great", "this is bad", "this is okay"]
predictions = predict_labels(model, texts)
print(f"   Texts: {texts}")
print(f"   Predictions: {predictions}")

print("\n3. Testing topic_modeling module...")
from topic_modeling import create_dummy_lda_model, infer_topics_for_text

vectorizer, lda_model = create_dummy_lda_model(n_topics=5)
test_text = "technology computer software programming development"
topics = infer_topics_for_text(test_text, vectorizer, lda_model, top_k_topics=2, top_n_words=3)
print(f"   Text: {test_text}")
print(f"   Topics: {topics}")

print("\n4. Testing summarization module...")
from summarization import extract_summary, get_both_summaries

summary_text = "This is a test document about machine learning. Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Deep learning is a part of machine learning that uses neural networks."
extractive = extract_summary(summary_text, max_sentences=2)
print(f"   Original: {summary_text}")
print(f"   Extractive Summary: {extractive}")

print("\n[OK] All module tests passed!")
