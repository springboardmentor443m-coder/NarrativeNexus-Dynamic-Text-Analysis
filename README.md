# AI Narrative Nexus

A comprehensive text analysis platform built with Flask that provides advanced NLP capabilities including topic modeling, sentiment analysis, and text summarization.

## Features

- **Topic Modeling (LDA)**: Extract key topics from text using Latent Dirichlet Allocation
- **Sentiment Analysis (BERT)**: Analyze sentiment using BERT-based multilingual model
- **Text Summarization (BART)**: Generate concise summaries using BART transformer model
- **File Upload Support**: Analyze text from uploaded files
- **Modern UI**: Beautiful, responsive web interface

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **NLP Models**:
  - LDA (scikit-learn) for topic modeling
  - BERT (nlptown/bert-base-multilingual-uncased-sentiment) for sentiment analysis
  - BART (facebook/bart-large-cnn) for summarization

## Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```



## Usage

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Enter text** or **upload a file** and click "Analyze"

## Project Structure

```
AI-Narrative-Nexus/
│
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── setup.py              # Setup script for NLTK data
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── models/               # NLP models
│   ├── __init__.py
│   ├── topic_modeling.py    # LDA topic modeling
│   ├── sentiment_analyzer.py # BERT sentiment analysis
│   └── summarizer.py        # BART summarization
│
├── templates/            # HTML templates
│   └── index.html
│
├── static/               # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── uploads/             # Uploaded files (created automatically)
```



## Model Details

### Topic Modeling (LDA)
- Uses Latent Dirichlet Allocation from scikit-learn
- Extracts 5 topics by default
- Returns top 10 words per topic
- Includes topic weights

### Sentiment Analysis (BERT)
- Uses `nlptown/bert-base-multilingual-uncased-sentiment`
- Supports multiple languages
- Returns sentiment label, score, and confidence
- Falls back to rule-based analysis if model fails to load

### Summarization (BART)
- Uses `facebook/bart-large-cnn` model
- Generates abstractive summaries
- Falls back to extractive summarization if model fails
- Provides compression ratio statistics



