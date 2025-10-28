from app.utils.groq_client import call_groq_model
import json

def analyze_topic_sentiment(topics):
    """
    Analyzes sentiment of each topic using Groq API.
    The model returns approximate percentages for positive, neutral, and negative sentiments.
    Expected JSON format from Groq:
    {
        "positive": 65,
        "neutral": 25,
        "negative": 10
    }
    """
    results = []

    for topic in topics:
        text = topic.get("text", "").strip()

        # Handle empty or invalid text
        if not text:
            results.append({
                **topic,
                "sentiment": {"positive": 0, "neutral": 100, "negative": 0}
            })
            continue

        # Prompt for Groq
        prompt = (
            "Analyze the sentiment of the following text and return percentages "
            "for 'positive', 'neutral', and 'negative'. Ensure the total equals 100. "
            "Respond strictly in valid JSON format.\n\n"
            f"Text:\n{text}"
        )

        # Call Groq model
        response = call_groq_model(prompt)

        # Try to parse JSON safely
        try:
            sentiment_data = json.loads(response)
        except json.JSONDecodeError:
            # If Groq returns text instead of JSON, fallback to default
            sentiment_data = {"positive": 0, "neutral": 100, "negative": 0}

        results.append({
            **topic,
            "sentiment": sentiment_data
        })

    return results
