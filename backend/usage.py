"""
Example usage script for the Dynamic Text Analysis Platform API
Run this after starting the server with: python main.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def analyze_text_example():
    """Example: Analyze a single text"""
    print("=" * 50)
    print("Example 1: Analyzing a single text")
    print("=" * 50)
    
    text = """
    Artificial intelligence is transforming the way we work and live. 
    Machine learning algorithms can now process vast amounts of data 
    and make predictions with remarkable accuracy. However, there are 
    concerns about privacy and the ethical implications of AI systems. 
    We must ensure that AI is developed responsibly and benefits all of humanity.
    """
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        json={
            "content": text,
            "source": "example"
        }
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Document ID: {result['document_id']}")
    print(f"Status: {result['sentiment_label']}")
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Get the analysis results
    doc_id = result['document_id']
    analysis_response = requests.get(f"{BASE_URL}/api/analysis/{doc_id}")
    analysis = analysis_response.json()
    
    print(f"\nSentiment: {analysis['sentiment_label']} (score: {analysis['sentiment_score']:.2f})")
    print(f"\nSummary:\n{analysis['summary']}")
    print(f"\nThemes: {', '.join(analysis['themes'])}")
    print(f"\nTop Keywords:")
    for kw in analysis['keywords'][:5]:
        print(f"  - {kw['word']}: {kw['weight']:.3f}")


def batch_analyze_example():
    """Example: Analyze multiple texts"""
    print("\n" + "=" * 50)
    print("Example 2: Batch analysis")
    print("=" * 50)
    
    texts = [
        {
            "content": "I love this product! It works perfectly and exceeded my expectations.",
            "source": "review_positive"
        },
        {
            "content": "This is terrible. The quality is poor and it broke after one day.",
            "source": "review_negative"
        },
        {
            "content": "The product is okay. Nothing special, but it does the job.",
            "source": "review_neutral"
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/batch",
        json={"texts": texts}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Message: {result['message']}")
    print(f"Document IDs: {result['document_ids']}")
    print(f"Status: {result['status']}")
    
    # Wait for processing
    time.sleep(3)
    
    # Get results for each document
    for doc_id in result['document_ids']:
        analysis_response = requests.get(f"{BASE_URL}/api/analysis/{doc_id}")
        analysis = analysis_response.json()
        print(f"\nDocument {doc_id}:")
        print(f"  Sentiment: {analysis['sentiment_label']} ({analysis['sentiment_score']:.2f})")


def train_lda_example():
    """Example: Train LDA topic model"""
    print("\n" + "=" * 50)
    print("Example 3: Training LDA topic model")
    print("=" * 50)
    
    # First, make sure we have some documents
    sample_texts = [
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Data science involves statistics, programming, and domain expertise.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Computer vision enables machines to interpret visual information.",
        "Cloud computing provides scalable infrastructure for applications.",
        "Cybersecurity protects systems from threats and attacks.",
        "Blockchain technology enables secure decentralized transactions.",
        "Internet of Things connects everyday devices to the internet.",
        "Quantum computing promises to solve complex problems faster."
    ]
    
    # Add documents
    print("Adding sample documents...")
    for text in sample_texts:
        requests.post(
            f"{BASE_URL}/api/analyze",
            json={"content": text, "source": "lda_training"}
        )
    
    # Wait for processing
    print("Waiting for documents to be processed...")
    time.sleep(5)
    
    # Train LDA model
    response = requests.post(
        f"{BASE_URL}/api/lda/train",
        json={
            "model_name": "tech_topics",
            "num_topics": 3
        }
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"\nModel trained successfully!")
    print(f"Model ID: {result['model_id']}")
    print(f"Number of topics: {result['num_topics']}")
    print(f"Coherence Score: {result['coherence_score']:.4f}")
    print(f"Documents used: {result['documents_used']}")
    
    print("\nTopics discovered:")
    for topic in result['topics']:
        print(f"\nTopic {topic['topic_id']}:")
        print(f"  Top words: {', '.join(topic['top_words'])}")


def get_statistics_example():
    """Example: Get platform statistics"""
    print("\n" + "=" * 50)
    print("Example 4: Platform statistics")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/stats")
    stats = response.json()
    
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Processed Documents: {stats['processed_documents']}")
    print(f"Pending Documents: {stats['pending_documents']}")
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Total Models: {stats['total_models']}")
    print(f"Average Sentiment: {stats['average_sentiment']:.3f}")


def get_topics_example():
    """Example: Get LDA topics"""
    print("\n" + "=" * 50)
    print("Example 5: Get LDA topics")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/lda/topics")
    
    if response.status_code == 404:
        print("No LDA model found. Train a model first using Example 3.")
        return
    
    result = response.json()
    print(f"Model: {result['model_name']}")
    print(f"Number of topics: {result['num_topics']}")
    
    print("\nTopics:")
    for topic in result['topics']:
        print(f"\nTopic {topic['topic_id']}:")
        words_str = ", ".join([f"{w['word']} ({w['weight']:.3f})" 
                              for w in topic['words'][:5]])
        print(f"  {words_str}")


if _name_ == "_main_":
    print("Dynamic Text Analysis Platform - Example Usage")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/")
        print("✓ Server is running\n")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server.")
        print("Please start the server first with: python main.py")
        exit(1)
    
    # Run examples
    analyze_text_example()
    batch_analyze_example()
    train_lda_example()
    get_statistics_example()
    get_topics_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
