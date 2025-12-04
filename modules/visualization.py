# modules/visualization.py

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_wordcloud(text: str):
    """
    Generates a WordCloud figure for Streamlit.
    """
    if not text or len(text.strip()) == 0:
        text = "no data available"

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=150
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_sentiment(score: float):
    """
    Displays a simple horizontal bar for sentiment.
    """
    if not isinstance(score, (float, int)):
        score = 0.5

    fig, ax = plt.subplots(figsize=(5, 1.5))
    color = (
        "green" if score > 0.6 else
        "red" if score < 0.4 else
        "gray"
    )

    ax.barh(["Sentiment"], [score], color=color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Sentiment Score")
    plt.tight_layout()

    return fig


def plot_topics(topic_keywords):
    """
    Basic topic visualization: bar chart showing word count per topic.
    """
    if not topic_keywords or not any(topic_keywords):
        return None

    lengths = [len(t.split(",")) for t in topic_keywords]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(lengths)), lengths, color="skyblue")
    ax.set_title("Topic Keyword Count")
    ax.set_xlabel("Topic Number")
    ax.set_ylabel("Keyword Count")
    plt.tight_layout()

    return fig
