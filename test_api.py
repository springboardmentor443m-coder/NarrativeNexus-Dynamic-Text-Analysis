import urllib.request
import urllib.error
import json

print("Testing /analyze endpoint...")
try:
    data = {
        "title": "Test Analysis",
        "content": "This is a test narrative to analyze. It contains some text about atheism and religion. The separation of church and state is important."
    }
    
    req = urllib.request.Request(
        'http://127.0.0.1:8000/analyze',
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    response = urllib.request.urlopen(req, timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print("SUCCESS: /analyze endpoint works")
    print(f"Response keys: {list(result.keys())}")
    if 'error' in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Title: {result.get('title')}")
        print(f"Sentiment: {result.get('sentiment')}")
        print(f"Topics: {result.get('topics')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting /saved-narratives endpoint...")
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/saved-narratives', timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print("SUCCESS: /saved-narratives endpoint works")
    print(f"Total: {result.get('total')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting /statistics/sentiment endpoint...")
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/statistics/sentiment', timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print("SUCCESS: /statistics/sentiment endpoint works")
    print(f"Stats: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
