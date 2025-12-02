import os
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select
from dotenv import load_dotenv

# Local imports
from backend.models import (
    HealthResponse, 
    AnalysisResponse, 
    HistoryListItem, 
    HistoryItem
)
from backend.groq_analyzer import create_analyzer
from backend.utils import allowed_file, extract_text_from_file
from backend.preprocessing import TextPreprocessor

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = os.getenv("TOPIC_EMBEDDER", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    # Warning instead of crash, allows app to start but analysis will fail gracefully
    print("⚠️ WARNING: GROQ_API_KEY not found in .env file.")

# --- Database Setup (SQLModel) ---
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "analysis_history.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

class AnalysisEntry(SQLModel, table=True):
    """Database Table Definition"""
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    per_file_json: str      # Stores List[PerFileResult] as JSON
    num_files: int
    num_documents: int
    file_names_json: str    # Stores List[str] as JSON

# Create tables if they don't exist
SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- FastAPI App ---
app = FastAPI(title="Nexus Intelligence API", version="6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core Services ---
analyzer = create_analyzer(
    api_key=GROQ_API_KEY or "missing_key",
    llm_model=LLM_MODEL,
    embed_model=EMBED_MODEL
)

preprocessor = TextPreprocessor(remove_stopwords=False)

# --- Helper Functions ---

def save_to_history(session: Session, response_data: AnalysisResponse) -> int:
    """Persists the analysis results to SQLite."""
    # Serialize complex objects to JSON for storage
    entry = AnalysisEntry(
        timestamp=response_data.timestamp,
        per_file_json=json.dumps([item.model_dump() for item in response_data.per_file]),
        num_files=response_data.num_files,
        num_documents=response_data.num_documents,
        file_names_json=json.dumps(response_data.file_names)
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry.id

# --- Routes ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="online",
        timestamp=datetime.now().isoformat(),
        groq_configured=bool(GROQ_API_KEY),
        version="6.1"
    )

@app.post("/api/analyze/files", response_model=AnalysisResponse)
async def analyze_files(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_session)
):
    """
    Main pipeline: Upload -> Extract -> Clean -> Analyze -> Save -> Return
    """
    raw_texts = []
    valid_filenames = []

    # 1. Extraction
    for f in files:
        if not allowed_file(f.filename):
            continue
            
        content = await f.read()
        text = extract_text_from_file(f.filename, content)
        
        if text and text.strip():
            raw_texts.append(text)
            valid_filenames.append(f.filename)

    if not raw_texts:
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded files.")

    # 2. Preprocessing
    cleaned_texts = [preprocessor.clean_text(t) for t in raw_texts]

    # 3. Analysis (Groq + BERTopic + Sentiment)
    # Returns dict with keys: 'per_file', 'assigned_topic' (deprecated)
    analysis_result = analyzer.analyze_comprehensive(cleaned_texts, valid_filenames)

    # 4. Construct Response Object
    response_obj = AnalysisResponse(
        status="success",
        timestamp=datetime.now().isoformat(),
        num_files=len(files),
        num_documents=len(cleaned_texts),
        file_names=valid_filenames,
        per_file=analysis_result["per_file"],
        history_id=None 
    )

    # 5. Save History
    db_id = save_to_history(db, response_obj)
    response_obj.history_id = db_id

    return response_obj

@app.get("/api/history", response_model=dict)
def get_history_list(limit: int = 50, db: Session = Depends(get_session)):
    """Returns recent analysis metadata for the sidebar list."""
    stmt = select(AnalysisEntry).order_by(AnalysisEntry.id.desc()).limit(limit)
    rows = db.exec(stmt).all()

    results = []
    for r in rows:
        results.append(HistoryListItem(
            id=r.id,
            timestamp=r.timestamp,
            file_names=json.loads(r.file_names_json),
            num_files=r.num_files,
            num_documents=r.num_documents
        ))

    return {"status": "success", "results": results}

@app.get("/api/history/{hid}", response_model=HistoryItem)
def get_history_item(hid: int, db: Session = Depends(get_session)):
    """Returns full details for a specific analysis ID."""
    row = db.get(AnalysisEntry, hid)
    if not row:
        raise HTTPException(status_code=404, detail="History item not found")

    return HistoryItem(
        id=row.id,
        timestamp=row.timestamp,
        file_names=json.loads(row.file_names_json),
        num_files=row.num_files,
        num_documents=row.num_documents,
        per_file=json.loads(row.per_file_json)
    )