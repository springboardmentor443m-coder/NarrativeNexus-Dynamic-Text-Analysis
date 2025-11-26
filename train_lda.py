"""
Training script for LDA Topic Modeling on 20 Newsgroups Dataset
"""
import os
import json
from utils.data_loader import NewsgroupsDataLoader
from models.topic_modeling import TopicModeler


def main():
    """Main training function"""
    print("=" * 60)
    print("LDA Topic Modeling Training on 20 Newsgroups Dataset")
    print("=" * 60)
    
    # Configuration
    NUM_TOPICS = 20  # Match the number of newsgroups
    NUM_WORDS = 15
    MAX_FEATURES = 2000
    MAX_ITER = 50
    USE_TFIDF = False
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    data_loader = NewsgroupsDataLoader(data_dir='data')
    
    # Load training data
    train_data = data_loader.load_data(
        subset='train',
        categories=None,  # Use all categories
        remove_headers=True,
        remove_footers=True,
        remove_quotes=True,
        shuffle=True,
        random_state=42
    )
    
    # Display dataset statistics
    stats = data_loader.get_data_stats(train_data)
    print(f"\nDataset Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Number of categories: {stats['num_categories']}")
    print(f"  Average document length: {stats['avg_doc_length']:.1f} words")
    print(f"\nCategory distribution:")
    for cat, count in list(stats['category_distribution'].items())[:5]:
        print(f"  {cat}: {count}")
    print(f"  ... and {len(stats['category_distribution']) - 5} more categories")
    
    # Initialize topic modeler
    print(f"\n[2/4] Initializing Topic Modeler...")
    print(f"  Number of topics: {NUM_TOPICS}")
    print(f"  Words per topic: {NUM_WORDS}")
    print(f"  Max features: {MAX_FEATURES}")
    print(f"  Max iterations: {MAX_ITER}")
    
    topic_modeler = TopicModeler(
        num_topics=NUM_TOPICS,
        num_words=NUM_WORDS,
        max_features=MAX_FEATURES,
        use_tfidf=USE_TFIDF,
        random_state=42
    )
    
    # Train model
    print(f"\n[3/4] Training LDA model...")
    training_results = topic_modeler.train_on_dataset(
        documents=train_data['data'],
        num_topics=NUM_TOPICS,
        max_iter=MAX_ITER,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    print(f"\nTraining Results:")
    print(f"  Documents processed: {training_results['num_documents']}")
    print(f"  Vocabulary size: {training_results['vocabulary_size']}")
    print(f"  Perplexity: {training_results['perplexity']}")
    print(f"  Log-likelihood: {training_results['log_likelihood']}")
    
    # Calculate coherence
    print(f"\n[4/4] Calculating topic coherence...")
    coherence = topic_modeler.calculate_coherence(top_n=10)
    print(f"  Average coherence: {coherence:.4f}")
    
    # Extract and display topics
    print(f"\n{'=' * 60}")
    print("Extracted Topics:")
    print("=" * 60)
    topics_result = topic_modeler.get_topics()
    
    for topic in topics_result['topics']:
        print(f"\nTopic {topic['topic_id']} (weight: {topic['weight']}):")
        print(f"  Top words: {', '.join(topic['words'][:10])}")
    
    # Save model
    model_path = 'models/lda_20newsgroups.pkl'
    print(f"\nSaving model to {model_path}...")
    topic_modeler.save_model(model_path)
    
    # Save training results
    results_path = 'data/training_results.json'
    results = {
        'training_results': training_results,
        'coherence': round(coherence, 4),
        'topics': topics_result['topics'],
        'config': {
            'num_topics': NUM_TOPICS,
            'num_words': NUM_WORDS,
            'max_features': MAX_FEATURES,
            'max_iter': MAX_ITER,
            'use_tfidf': USE_TFIDF
        }
    }
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training results saved to {results_path}")
    
    # Test prediction on a few documents
    print(f"\n{'=' * 60}")
    print("Testing predictions on sample documents:")
    print("=" * 60)
    
    test_data = data_loader.load_data(subset='test', shuffle=True, random_state=42)
    sample_docs = test_data['data'][:5]
    predictions = topic_modeler.predict_topics(sample_docs)
    
    for i, pred in enumerate(predictions):
        print(f"\nDocument {i+1}:")
        print(f"  Dominant topic: {pred['dominant_topic']}")
        print(f"  Confidence: {pred['confidence']:.4f}")
        print(f"  Preview: {sample_docs[i][:100]}...")
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

