# ⚡ Nexus Intelligence: Analysis Platform

## 🚀 Key Features

- **Topic Classification:** Automatically clusters documents into 40+ pre-trained categories (e.g., Space, Hardware, Politics) using **BERTopic** and **Sentence-Transformers**.
- **AI Summarization:** Generates concise executive summaries using **Llama-3-70b** via the Groq API.
- **Sentiment Analysis:** Detects emotional tone and confidence scores using the `twitter-roberta-base-sentiment` model.
- **Analysis History:** Persists all analysis results in a local **SQLite** database for future reference.
- **Pro-Level UI:** A fully custom-styled dashboard with dark mode, custom metrics, and interactive visualizations.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom CSS injection, Session State Management)
- **Backend:** FastAPI (Async, RESTful), Uvicorn
- **Database:** SQLite + SQLModel (ORM)
- **ML Pipeline:**
  - *Topic Modeling:* BERTopic, Sentence-Transformers (`all-MiniLM-L6-v2`), UMAP, HDBSCAN
  - *LLM:* Groq API (`llama-3.3-70b-versatile`)
  - *Sentiment:* Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment`)
- **Data Processing:** Pandas, NumPy, PyPDF2, Python-docx

---

## 📂 Project Structure

```
PROJECT/
├── backend/
│   ├── models/
│   │   └── bertopic_20newsgroups/   # Model assets (Generated via training script)
│   │       ├── topic_centroids.npy
│   │       ├── topic_info.json
│   │       ├── topic_name_mapping.json
│   │       └── trained_bertopic.pkl
│   ├── groq_analyzer.py             # Main pipeline orchestration
│   ├── main.py                      # FastAPI entry point
│   ├── models.py                    # Pydantic & DB schemas
│   ├── preprocessing.py             # Text cleaning utility
│   ├── sentiment_analyzer.py        # Sentiment engine (RoBERTa)
│   ├── text_summarizer.py           # Summarization engine (Groq/Llama-3)
│   ├── topic_modeler.py             # Topic inference engine
│   ├── train_bertopic.py            # Model training script
│   └── utils.py                     # File extraction helpers (PDF/DOCX/CSV)
├── data/
│   └── analysis_history.db          # Local database (Auto-generated)
├── .env                             # API Configuration
├── app.py                           # Streamlit Frontend
├── requirements.txt                 # Project Dependencies
└── README.md                        # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone & Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd PROJECT

# Create a virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have PyTorch installed suitable for your system hardware. The requirements file will handle the standard CPU version by default.

### 3. Environment Configuration

Create a `.env` file in the root directory and add your API key:

```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
TOPIC_EMBEDDER=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
```

### 4. Initialize AI Models (Crucial Step)

The project requires pre-trained model assets to function. You must generate these locally by running the training script once before starting the app. This script downloads the 20 Newsgroups dataset, trains the BERTopic model, and uses the Groq API to generate smart topic labels.

```bash
python -m backend.train_bertopic
```

**What this does:**
- Trains the model
- Calculates centroids
- Generates AI topic labels
- Saves assets to `backend/models/bertopic_20newsgroups/`

**Time:** Approx. 10–20 minutes depending on hardware.

**Success Message:** You will see `✅ Success! AI-Labeled Model assets saved...` when finished.

---

## 🏃‍♂️ Running the Application

This project requires two terminal windows running simultaneously.

### Terminal 1: Start the Backend API

```bash
uvicorn backend.main:app --reload
```

**Status:** The API will start at `http://localhost:8000`.

**Docs:** You can view the interactive Swagger UI at `http://localhost:8000/docs`.

### Terminal 2: Launch the Frontend UI

```bash
streamlit run app.py
```

**Status:** The Application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Usage Guide

1. **Check Status:** Ensure the "SYSTEM ONLINE" indicator is green in the dashboard header.

2. **Upload:** Drag and drop documents (`.txt`, `.pdf`, `.docx`, `.csv`) into the upload zone.

3. **Analyze:** Click "Run Analysis". The pipeline will:
   - Clean and preprocess text
   - Calculate semantic embeddings to match specific Topics
   - Generate executive summaries via Llama-3
   - Compute Sentiment confidence scores

4. **View Results:** Use the tabs to view:
   - Topic Classification
   - Summaries
   - Sentiment data
   - Raw JSON

5. **History:** Click "📜 History" in the top right to view past analyses stored in the database.

---

## 📝 Notes for Developers

- **Database:** The `data/` folder and `analysis_history.db` file are automatically created on the first run via SQLModel. You do not need to create them manually.

- **Model Updates:** If you wish to retrain the model with different parameters, edit `backend/train_bertopic.py` and re-run the training command.

- **Customization:** UI styling is handled via CSS injection in `app.py`. Backend logic is modularized in the `backend/` package.


