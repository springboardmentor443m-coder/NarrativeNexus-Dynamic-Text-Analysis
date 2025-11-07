# Text Analysis Editor (Transformer-powered)

A modern Streamlit app for text analysis at scale: key topics, summarization, sentiment, and topic modeling using BERTopic.

- Topic modeling: BERTopic with transformer embeddings (robust for large datasets)
- Summary: DistilBART CNN
- Sentiment: DistilBERT SST-2
- Key topics/phrases: KeyBERT with MiniLM embeddings

This replaces classic LDA flows as seen in the 20-Newsgroup project and is better suited for larger datasets and richer semantics.

Reference inspiration: `20-Newsgroup-Dataset-Analysis` ([GitHub repository](https://github.com/Sameeksharajsb/20-Newsgroup-Dataset-Analysis/tree/main?tab=readme-ov-file)).

## Quickstart (Windows)

1. Create and activate a virtual environment
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Run the app
```powershell
streamlit run app.py
```

4. Open the local URL shown by Streamlit.

## Usage

- Editor panel: paste text, click buttons for Key Topics, Summary, Sentiment.
- Dataset panel: upload CSV, choose the text column, run BERTopic. View top topics and interactive plots.

## Notes

- BERTopic benefits from `umap-learn` and `hdbscan` for clustering structure; both are included.
- First run will download transformer models; ensure internet access the first time.
- For very large datasets, consider precomputing embeddings offline and passing them into BERTopic.

## Why BERTopic instead of LDA?

- Uses contextual transformer embeddings → higher-quality, coherent topics.
- Scales to larger corpora with better semantics than bag-of-words LDA.
- Offers insightful interactive visualizations.

## License

MIT


