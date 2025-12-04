import urllib.request
import urllib.error

urls = [
    'http://127.0.0.1:8000',
    'http://localhost:8000'
]

for url in urls:
    try:
        response = urllib.request.urlopen(url, timeout=5)
        print(f"SUCCESS: {url} - Status {response.status}")
        content = response.read(500).decode()
        print(f"Content preview: {content[:100]}")
    except urllib.error.URLError as e:
        print(f"ERROR: {url} - {e}")
    except Exception as e:
        print(f"ERROR: {url} - {type(e).__name__}: {e}")
