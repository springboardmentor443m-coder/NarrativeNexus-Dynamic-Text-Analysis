# Narrative Nexus — File Upload Enabled (TXT, CSV, DOCX, PDF)
- Preprocessing (cleaned text)
- Sentiment Analysis (VADER)
- Topic Modeling (LDA & NMF)
- Summarization (Hugging-Face transformer. Model name: sshleifer/distilbart-cnn-12-6)

## How to run
1. Backend: create venv, install requirements, download NLTK data.
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon
   python app.py
   ```
2. Frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```
3. Open http://localhost:3000 and upload a file or paste text, then click Analyze / Upload & Analyze.

## Supported file types
- .txt, .csv, .docx, .pdf

