import requests
import json

BASE_URL = "http://localhost:8000"

def test_retrain_lda():
    """Test the LDA retraining endpoint"""
    
    payload = {
        "directory_path": r"C:\Users\Pavan\OneDrive\20news-18828",
        "n_topics": 10,
        "max_features": 5000,
        "max_files": None
    }
    
    print("[OK] Sending retrain request to /retrain-lda...")
    print(f"[OK] Using directory: {payload['directory_path']}")
    
    try:
        response = requests.post(f"{BASE_URL}/retrain-lda", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Response received:")
            print(json.dumps(result, indent=2))
            
            if result.get("success"):
                print("\n[OK] LDA model retrained successfully!")
                print(f"  - Topics: {result['n_topics']}")
                print(f"  - Max features: {result['max_features']}")
                return True
            else:
                print(f"\n[WARN] Error: {result.get('error')}")
                return False
        else:
            print(f"\n[WARN] HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n[WARN] Could not connect to FastAPI server.")
        print("[WARN] Make sure to run: uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\n[WARN] Error: {e}")
        return False


def test_analyze_after_retraining():
    """Test analyze endpoint after retraining"""
    
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables 
    computers to learn from data without being explicitly programmed. 
    Neural networks are inspired by biological neurons and can process 
    complex patterns in data. Deep learning has revolutionized computer 
    vision and natural language processing.
    """
    
    payload = {
        "title": "ML and AI Overview",
        "content": sample_text
    }
    
    print("\n\n[OK] Testing /analyze endpoint after retraining...")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[OK] Analysis complete:")
            print(f"  - Title: {result['title']}")
            print(f"  - Sentiment: {result['sentiment']}")
            print(f"  - Topics: {result['topics']}")
            print(f"  - Word count: {result['word_count']}")
            return True
        else:
            print(f"\n[WARN] HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n[WARN] Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing External Training Data Integration")
    print("=" * 60)
    
    success = test_retrain_lda()
    
    if success:
        test_analyze_after_retraining()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
