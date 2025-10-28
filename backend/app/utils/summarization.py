import os
import requests
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Access your GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def summarize_with_groq(text: str, max_length: int = 100, min_length: int = 30):
    """
    Summarize text using Groq API. Generates concise summaries for given text.
    """
    if not text.strip():
        return ""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a concise summarization assistant."},
            {"role": "user", "content": f"Summarize the following text in {min_length}-{max_length} words:\n{text}"}
        ],
        "temperature": 0.5,
        "max_tokens": 200
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Groq summarization failed:", e)
        return "[Error generating summary]"
