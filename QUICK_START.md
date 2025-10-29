# AI Narrative Nexus v3.0 - Quick Start Guide

## Prerequisites
- Python 3.8+
- Groq API Key (get it free from https://console.groq.com)

## Step 1: Setup Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Create .env File

Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Groq API key.

## Step 4: Run Backend (Terminal 1)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Wait for:
```
Uvicorn running on http://0.0.0.0:8000
```

## Step 5: Run Frontend (Terminal 2)

Open a **NEW terminal**, activate venv again, then:

```bash
streamlit run frontend/app.py
```

Your browser will open automatically to: `http://localhost:8501`

---

## That's it! 🚀

The app is now running. You can:
- Upload TXT, CSV, or DOCX files
- Paste text directly
- Get sentiment analysis
- Generate summaries
- Discover topics
- View analysis history

## Stop Everything
Press `Ctrl+C` in both terminals

## Deactivate Virtual Environment
```bash
deactivate
```
