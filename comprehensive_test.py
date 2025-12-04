import urllib.request
import urllib.error
import json

print("="*60)
print("COMPREHENSIVE API TEST")
print("="*60)

test_cases = [
    {
        "title": "Atheism and Logic",
        "content": "Atheism is based on logical reasoning and evidence. Religious beliefs lack rational justification. The absence of evidence for God combined with logical inconsistencies makes atheism the most rational position."
    },
    {
        "title": "Church State Separation",
        "content": "The separation of church and state is a fundamental principle. Religious freedom requires government neutrality. Constitutional protections ensure no religious bias in state matters."
    },
    {
        "title": "Biblical Scholarship",
        "content": "Biblical texts require scholarly analysis. Historical evidence contradicts many religious claims. Critical examination reveals authorship and dating issues. Textual criticism shows inconsistencies and additions."
    }
]

print("\n1. TESTING /analyze ENDPOINT")
print("-"*60)
for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['title']}")
    try:
        data = {
            "title": test['title'],
            "content": test['content']
        }
        req = urllib.request.Request(
            'http://127.0.0.1:8000/analyze',  # Fixed: Using port 8000
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode('utf-8'))
        
        print(f"  Status: OK")
        print(f"  Sentiment: {result.get('sentiment')}")
        print(f"  Topics: {', '.join(result.get('topics', [])[:2])}")
        print(f"  Summary: {result.get('summary_extractive', '')[:80]}...")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\n2. TESTING /saved-narratives ENDPOINT")
print("-"*60)
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/saved-narratives', timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print(f"Status: OK")
    print(f"Total saved: {result.get('total')}")
    if result.get('narratives'):
        for n in result.get('narratives', []):
            print(f"  - {n['title']} [{n['sentiment']}] ({n['word_count']} words)")
except Exception as e:
    print(f"ERROR: {e}")

print("\n\n3. TESTING /statistics/sentiment ENDPOINT")
print("-"*60)
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/statistics/sentiment', timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print(f"Status: OK")
    print(f"Positive: {result.get('positive', 0)}")
    print(f"Neutral: {result.get('neutral', 0)}")
    print(f"Negative: {result.get('negative', 0)}")
    print(f"Total: {result.get('total', 0)}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
