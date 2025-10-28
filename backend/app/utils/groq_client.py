import os
from groq import Groq

from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# Access your GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load API key securely from environment variable
client = Groq(api_key=GROQ_API_KEY)

def call_groq_model(prompt: str, model="llama-3.1-8b-instant", max_tokens=300):
    """
    Calls the Groq API to generate a model response for a given prompt.
    Returns the response text.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise NLP assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq model call failed: {e}")
        return "Error: Unable to fetch response from Groq."
