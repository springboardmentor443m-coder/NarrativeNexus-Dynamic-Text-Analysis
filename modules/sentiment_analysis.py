import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the RoBERTa sentiment model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_sentiment(text):
    """
    Analyzes sentiment of given text and returns label and confidence score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    label = labels[torch.argmax(scores)]
    confidence = scores[0][torch.argmax(scores)].item()
    return label, confidence


def get_topic_sentiments(topics):
    """
    Computes sentiment for each topic in a list.
    Returns a dictionary mapping topic -> sentiment score (-1 to +1).
    """
    sentiments = {}
    for topic in topics:
        label, confidence = get_sentiment(topic)
        if label == "Positive":
            score = confidence
        elif label == "Negative":
            score = -confidence
        else:
            score = 0
        sentiments[topic] = score
    return sentiments


# Standalone run (for testing)
if __name__ == "__main__":
    text = "Artificial Intelligence is transforming industries worldwide."
    label, conf = get_sentiment(text)
    print(f"Sentiment: {label} (Confidence: {conf:.4f})")
