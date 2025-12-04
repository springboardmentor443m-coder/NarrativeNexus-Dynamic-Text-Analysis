import urllib.request
import json
import time

time.sleep(2)

try:
    print("Testing HTTP connection to http://127.0.0.1:8000")
    response = urllib.request.urlopen('http://127.0.0.1:8000', timeout=5)
    print(f"Status: {response.status}")
    print("[OK] Server is running!")
    
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print(f"Type: {type(e).__name__}")
