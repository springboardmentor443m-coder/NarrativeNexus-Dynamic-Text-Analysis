#!/usr/bin/env python
"""
Script to train LDA model on the atheism dataset.
This script trains the topic modeling and saves the model for use by the application.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from topic_modeling import create_lda_model_from_external_data
import joblib

def main():
    """Train LDA model on atheism dataset"""
    
    training_dir = Path(__file__).parent / "training_data" / "atheism"
    models_dir = Path(__file__).parent / "models"
    
    if not training_dir.exists():
        print(f"[ERROR] Training directory not found: {training_dir}")
        return False
    
    txt_files = list(training_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No TXT files found in {training_dir}")
        return False
    
    print(f"\n{'='*60}")
    print("LDA Topic Model Training")
    print(f"{'='*60}")
    print(f"Training Directory: {training_dir}")
    print(f"Found {len(txt_files)} training documents")
    print(f"Models will be saved to: {models_dir}\n")
    
    models_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        print("[*] Training LDA model on atheism dataset...")
        print("    Parameters:")
        print("      - n_topics: 10")
        print("      - max_features: 5000")
        print("      - max_iter: 50")
        print("      - learning_method: batch")
        print()
        
        vectorizer, lda_model = create_lda_model_from_external_data(
            directory_path=str(training_dir),
            n_topics=10,
            max_features=5000,
            max_files=None
        )
        
        vectorizer_path = models_dir / "lda_vectorizer.joblib"
        model_path = models_dir / "lda_model.joblib"
        
        print(f"\n[*] Saving trained models...")
        joblib.dump(vectorizer, str(vectorizer_path))
        joblib.dump(lda_model, str(model_path))
        
        print(f"[OK] Vectorizer saved: {vectorizer_path}")
        print(f"[OK] Model saved: {model_path}")
        
        from topic_modeling import get_topics
        
        print(f"\n[*] Extracting topics...")
        topics = get_topics(vectorizer, lda_model, n_words=10)
        
        print(f"\n{'='*60}")
        print("Discovered Topics")
        print(f"{'='*60}\n")
        
        for i, topic_words in enumerate(topics, 1):
            print(f"Topic {i}:")
            print(f"  {' / '.join(topic_words)}\n")
        
        topics_summary = {
            "n_topics": 10,
            "topics": [" / ".join(words) for words in topics],
            "training_documents": len(txt_files),
            "max_features": 5000
        }
        
        summary_path = models_dir / "lda_topics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(topics_summary, f, indent=2)
        
        print(f"[OK] Topics summary saved: {summary_path}")
        print(f"\n{'='*60}")
        print("[OK] Training completed successfully!")
        print(f"{'='*60}\n")
        print("Next steps:")
        print("1. Restart the FastAPI application")
        print("2. The application will automatically load the trained models")
        print("3. Test by submitting text to the /analyze endpoint")
        print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
