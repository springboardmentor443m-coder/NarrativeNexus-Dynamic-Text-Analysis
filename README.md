# TextInsight Pro

An intelligent content analysis platform built with Flask that provides advanced NLP capabilities including theme extraction, emotion detection, and content condensation.

## Features

- **Theme Extraction (LDA)**: Extract key themes from text using Latent Dirichlet Allocation
- **Emotion Detection (BERT)**: Detect emotional tone using BERT-based multilingual model
- **Content Condensation (BART)**: Generate concise summaries using BART transformer model
- **Document Upload Support**: Analyze content from uploaded files (TXT, PDF, DOCX)
- **Modern UI**: Beautiful, responsive web interface with intuitive design

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **NLP Models**:
  - LDA (scikit-learn) for theme extraction
  - BERT (nlptown/bert-base-multilingual-uncased-sentiment) for emotion detection
  - BART (facebook/bart-large-cnn) for content condensation

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

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Build the theme extraction model** (first time setup):

```bash
python build_theme_model.py
```

## Usage

1. **Start the Flask server**:

```bash
python server.py
```

2. **Open your browser** and navigate to:

```
http://localhost:5000
```

3. **Enter text** or **upload a document** and click "Process Content"

## Project Structure

```
TextInsight-Pro/
│
├── server.py                 # Main Flask application
├── config.py                 # Configuration settings
├── build_theme_model.py      # Script to build theme extraction model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── analyzers/                # NLP analyzers
│   ├── __init__.py
│   ├── theme_extractor.py    # LDA theme extraction
│   ├── emotion_detector.py   # BERT emotion detection
│   └── content_condenser.py  # BART content condensation
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   └── file_parser.py        # File parsing utilities
│
├── views/                    # HTML templates
│   └── index.html
│
├── static/                   # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── model_storage/           # Trained model storage (created after running build_theme_model.py)
└── uploads/                  # Uploaded files (created automatically)
```

## Model Details

### Theme Extraction (LDA)

- Uses Latent Dirichlet Allocation from scikit-learn
- Trained on 20 Newsgroups dataset
- Extracts 20 themes by default
- Returns top 5 keywords per theme
- Includes theme confidence scores

### Emotion Detection (BERT)

- Uses `nlptown/bert-base-multilingual-uncased-sentiment`
- Supports multiple languages
- Returns emotion label, score, and confidence
- Falls back to rule-based detection if model fails to load

### Content Condensation (BART)

- Uses `facebook/bart-large-cnn` model
- Generates abstractive summaries
- Falls back to extractive condensation if model fails
- Provides compression ratio statistics

## API Endpoints

- `GET /` - Home page
- `POST /process` - Process text content
- `POST /process-file` - Process uploaded document

## Development

To contribute or modify:

1. Theme extraction model can be rebuilt using `build_theme_model.py`
2. Analyzers are modular and can be extended independently
3. Frontend uses vanilla JavaScript - no framework dependencies
4. All configuration is in `config.py`

## License

This project is open source and available for educational purposes.
