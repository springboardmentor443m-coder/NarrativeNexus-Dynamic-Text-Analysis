import matplotlib.pyplot as plt
import streamlit as st

def topic_bar_chart(topics):
    """Draw a simple bar chart showing topic weights."""
    if not topics:
        st.info("No topics to visualize.")
        return

    labels = [f"Topic {t['topic_id']}" for t in topics]
    weights = [t["weight"] for t in topics]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, weights)
    ax.set_title("Topic Weights")
    ax.set_ylabel("Weight (average importance)")
    ax.set_xlabel("Topics")
    st.pyplot(fig)
