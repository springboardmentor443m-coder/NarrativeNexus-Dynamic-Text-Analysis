# AI Narrative Nexus

A FastAPI-based web application that analyzes text using NLP techniques including sentiment analysis and topic modeling with LDA.

## Features

- **Text Preprocessing**: Cleans and tokenizes input text
- **Sentiment Analysis**: Classifies text sentiment (positive, neutral, negative) using TF-IDF + Logistic Regression
- **Topic Modeling**: Extracts key topics using Latent Dirichlet Allocation (LDA)
- **Text Summarization**: Generates concise summaries of input text
- **Statistics**: Provides word count, character count, and reading time estimates
- **Interactive UI**: Beautiful web interface for text analysis

## Project Structure

```
Backend/
  ├── main.py                 # FastAPI application
  ├── preprocessing.py        # Text preprocessing functions
  ├── sentiment_analysis.py   # Sentiment classification module
  ├── topic_modeling.py       # LDA topic modeling module
  ├── templates/              # HTML templates
  │   └── index.html
  ├── static/                 # CSS and static files
  │   └── style.css
  ├── models/                 # Pre-trained models (optional)
  ├── requirements.txt        # Python dependencies
  └── test_modules.py         # Module tests

Frontend/
  ├── index.html             # HTML (original)
  └── styles.css             # CSS (original)
```

## Installation

1. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI server:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: `http://localhost:8000`

### Command-line Options

- `--reload`: Auto-reload on code changes (development mode)
- `--host`: Bind to specified host (default: 127.0.0.1)
- `--port`: Bind to specified port (default: 8000)

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Enter a title for your narrative (optional)
3. Paste your text in the textarea
4. Click "Generate Narrative"
5. View the analysis results including:
   - Quick statistics (word count, reading time, sentiment)
   - Key themes (LDA topics)
   - Summary of the text
   - Narrative analysis

## API Endpoints

### GET /
Returns the main HTML interface.

### POST /analyze
Analyzes input text and returns analysis results.

**Request Body:**
```json
{
  "title": "Optional Title",
  "content": "Your text content here..."
}
```

**Response:**
```json
{
  "title": "Your Title",
  "word_count": 150,
  "char_count": 890,
  "reading_time_minutes": 0.75,
  "sentiment": "positive",
  "topics": ["theme1 / word1 / word2", "theme2 / word3 / word4"],
  "summary": "Brief summary of the text...",
  "narrative": "Detailed narrative analysis..."
}
```

## Modules

### preprocessing.py
- `clean_text()`: General text cleaning and preprocessing
- `tokenize()`: Split text into tokens
- `remove_stopwords()`: Filter out common stopwords
- `filter_tokens()`: Keep only meaningful words
- `preprocess_for_lda()`: Aggressive preprocessing for LDA
- `preprocess_for_sentiment()`: Light preprocessing for sentiment analysis

### sentiment_analysis.py
- `load_model()`: Load pre-trained sentiment model
- `predict_labels()`: Predict sentiment for texts
- `predict_probabilities()`: Get confidence scores
- `create_dummy_model()`: Create demo model for testing

### topic_modeling.py
- `create_lda_model()`: Train LDA model on documents
- `get_topics()`: Extract top words per topic
- `infer_topics_for_text()`: Get topics for a single document
- `get_document_topics()`: Get topic distributions
- `get_topic_similarity()`: Calculate similarity between documents
- `create_dummy_lda_model()`: Create demo model for testing

## Training Custom Models

To use custom pre-trained models instead of demo models:

1. Train your models on your dataset
2. Save them using joblib:
   ```python
   import joblib
   joblib.dump(your_model, 'Backend/models/sentiment_model.joblib')
   joblib.dump(your_lda_model, 'Backend/models/lda_model.joblib')
   joblib.dump(your_vectorizer, 'Backend/models/lda_vectorizer.joblib')
   ```
3. Restart the application - it will automatically load your models

## Testing

Run the test script to verify all modules:
```bash
cd Backend
python test_modules.py
```

## Technologies

- **FastAPI**: Web framework
- **Scikit-learn**: Machine learning (TF-IDF, Logistic Regression, LDA)
- **Pydantic**: Data validation
- **Jinja2**: Template rendering
- **HTML/CSS**: Frontend

## Demo Notes

The application includes dummy models for demonstration purposes. When you start the app without pre-trained models, it uses default models trained on sample data. For production use, train custom models on your specific dataset.

## Future Enhancements

- Support for multiple languages
- Advanced text summarization (abstractive)
- Named Entity Recognition (NER)
- Document clustering
- Text similarity comparison
- Export results to PDF
- Model training interface
- Custom vocabulary management
