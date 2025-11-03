import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(text):
    if not text:
        text = "no data available"
    wordcloud = WordCloud(width=600, height=400, background_color='white', max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout()
    return fig

def plot_sentiment(score):
    if not isinstance(score, (int, float)):
        score = 0.5
    plt.figure(figsize=(4, 1))
    plt.barh(["Sentiment"], [score], color="green" if score > 0.6 else "red" if score < 0.4 else "gray")
    plt.xlim(0, 1)
    plt.tight_layout()
    return plt

def plot_topics(topic_keywords):
    if not topic_keywords or not any(topic_keywords):
        return None
    topic_lengths = [len(words) for words in topic_keywords]
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(topic_lengths)), topic_lengths, color='skyblue')
    plt.title("Topic Word Count Overview")
    plt.tight_layout()
    return plt
