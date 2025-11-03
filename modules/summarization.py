import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

# Load transformer summarizer once (cached for performance)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text, max_chunk_size=1000):
    """
    Generates a summary for text of any length.
    
    - Short text (<100 words): extractive sentence summary
    - Medium text (100–1000 words): direct transformer summary
    - Long text (>1000 words): chunk-based summarization and merging
    """
    if not text or len(text.strip()) == 0:
        return "⚠️ No text provided for summarization."

    # --- Handle very short text quickly ---
    word_count = len(text.split())
    if word_count < 100:
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:2]) if len(sentences) > 2 else text
        return f"📝 *Short text summary:* {summary}"

    # --- Medium text (direct summarization) ---
    if word_count <= 1000:
        summary = summarizer(
            text, 
            max_length=180, 
            min_length=50, 
            do_sample=False
        )[0]['summary_text']
        return summary

    # --- Long text (chunked summarization) ---
    sentences = sent_tokenize(text)
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    chunks.append(current_chunk.strip())

    partial_summaries = []
    for chunk in chunks:
        try:
            part_summary = summarizer(
                chunk,
                max_length=150,
                min_length=40,
                do_sample=False
            )[0]['summary_text']
            partial_summaries.append(part_summary)
        except Exception:
            partial_summaries.append(chunk[:500])  # fallback in rare failure

    # Combine partial summaries and create a final meta-summary
    combined_summary = " ".join(partial_summaries)
    final_summary = summarizer(
        combined_summary,
        max_length=200,
        min_length=60,
        do_sample=False
    )[0]['summary_text']

    return final_summary
