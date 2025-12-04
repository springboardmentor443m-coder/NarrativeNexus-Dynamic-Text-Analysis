import requests
import json

url = "http://localhost:8000/analyze"
payload = {
    "title": "Test Narrative",
    "content": "This is a test text about atheism and logic. The arguments are logical and based on evidence. The reasoning is sound and rational."
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
