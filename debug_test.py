import urllib.request
import json

test = {
    "title": "Biblical Scholarship",
    "content": "Biblical texts require scholarly analysis. Historical evidence contradicts many religious claims. Critical examination reveals authorship and dating issues. Textual criticism shows inconsistencies and additions."
}

try:
    req = urllib.request.Request(
        'http://127.0.0.1:8001/analyze',
        data=json.dumps(test).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    response = urllib.request.urlopen(req, timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print("SUCCESS")
    print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(f"Response: {e.read().decode('utf-8')}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
