# Narrative Nexus — File Upload Enabled (TXT, CSV, DOCX, PDF)
This upgraded version supports file uploads and displays results in organized sections with auto-scrolling:
- Preprocessing (cleaned text)
- Sentiment Analysis (VADER)
- Topic Modeling (LDA & NMF)
- Summarization (extractive)

## How to run
1. Unzip and open the folder in VS Code.
2. Backend: create venv, install requirements, download NLTK data.
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon
   python app.py
   ```
3. Frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```
4. Open http://localhost:3000 and upload a file or paste text, then click Analyze / Upload & Analyze.

## Supported file types
- .txt, .csv, .docx, .pdf

