# Quick Start Guide - TextInsight Pro

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Step-by-Step Setup Instructions

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This may take 10-15 minutes as it downloads large ML models (BERT, BART).

### Step 4: Build Theme Extraction Model (First Time Only)

```bash
python build_theme_model.py
```

**Note:** This step downloads the 20 Newsgroups dataset and trains the LDA model. It may take 15-30 minutes depending on your system.

### Step 5: Start the Server

```bash
python server.py
```

You should see output like:

```
 * Running on http://0.0.0.0:5000
```

### Step 6: Open in Browser

Open your web browser and navigate to:

```
http://localhost:5000
```

## Testing the Application

1. **Text Input Mode:**

   - Click "Direct Text" tab
   - Paste or type at least 50 characters of text
   - Click "Process Content"
   - Wait for results (Theme Extraction, Emotion Detection, Content Condensation)

2. **File Upload Mode:**
   - Click "Document Upload" tab
   - Select a TXT, PDF, or DOCX file
   - Click "Process Document"
   - View the analysis results

## Troubleshooting

**If models fail to load:**

- Make sure you completed Step 4 (build_theme_model.py)
- Check that `model_storage/lda_model_20newsgroups.pkl` exists

**If port 5000 is already in use:**

- Edit `config.py` and change `PORT = 5000` to another port (e.g., `PORT = 5001`)
- Then access at `http://localhost:5001`

**If you get import errors:**

- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Project Structure Summary

- `server.py` - Main Flask application
- `analyzers/` - NLP analysis modules (ThemeExtractor, EmotionDetector, ContentCondenser)
- `views/` - HTML templates
- `static/` - CSS and JavaScript files
- `utils/` - File parsing utilities
- `config.py` - Configuration settings
- `build_theme_model.py` - Model training script


