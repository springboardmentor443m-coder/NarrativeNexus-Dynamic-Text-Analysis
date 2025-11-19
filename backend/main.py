# backend/main.py
import os
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select
from dotenv import load_dotenv

# local imports
from backend.models import HealthResponse
from backend.groq_analyzer import create_analyzer
from backend.utils import allowed_file, extract_text_from_file
from backend.preprocessing import TextPreprocessor


# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = os.getenv("TOPIC_EMBEDDER", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in environment (.env)")


# FastAPI app
app = FastAPI(title="AI Narrative Nexus", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------- DB Setup ----------------
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "analysis_history.db")
os.makedirs(DB_DIR, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class AnalysisEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: str

    # per-file results
    per_file_json: str       # contains summaries + sentiment per file
    topics_json: str         # topics only

    num_files: int
    num_documents: int
    file_names_json: str
    analysis_type: str


SQLModel.metadata.create_all(engine)


# Session dependency
def get_session():
    with Session(engine) as s:
        yield s


def save_history(session: Session, payload: dict) -> AnalysisEntry:
    entry = AnalysisEntry(
        timestamp=payload["timestamp"],
        per_file_json=json.dumps(payload.get("per_file", [])),
        topics_json=json.dumps(payload.get("topics", {})),
        num_files=payload["num_files"],
        num_documents=payload["num_documents"],
        file_names_json=json.dumps(payload["file_names"]),
        analysis_type="Advanced Per-File Analysis"
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


# Initialize analyzer + preprocessing
analyzer = create_analyzer(api_key=GROQ_API_KEY, llm_model=LLM_MODEL)
preprocessor = TextPreprocessor(remove_stopwords=False)


# ---------------- Health Endpoint ----------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        groq_configured=True,
        version="6.0"
    )


# ---------------- Analysis Endpoint ----------------
@app.post("/api/analyze/files")
async def analyze_files(files: List[UploadFile] = File(...), db: Session = Depends(get_session)):
    raw_texts = []
    file_names = []

    # Extract text
    for f in files:
        if not allowed_file(f.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")

        content = await f.read()
        extracted = extract_text_from_file(f.filename, content)
        if isinstance(extracted, list):
            raw_texts.extend([t for t in extracted if t.strip()])
        else:
            if extracted and extracted.strip():
                raw_texts.append(extracted)

        file_names.append(f.filename)

    if not raw_texts:
        raise HTTPException(status_code=400, detail="No usable text found in uploaded files")

    cleaned = [preprocessor.clean_text(t) for t in raw_texts]

    # Run analyzer
    analysis = analyzer.analyze_comprehensive(cleaned, file_names=file_names)

    response = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "num_files": len(files),
        "num_documents": len(cleaned),
        "file_names": file_names,
        "per_file": analysis.get("per_file", []),
        "topics": analysis.get("topics", {})
    }

    saved = save_history(db, response)
    response["history_id"] = saved.id

    return response


# ---------------- History Endpoints ----------------
@app.get("/api/history")
def history_list(limit: int = 50, db: Session = Depends(get_session)):
    stmt = select(AnalysisEntry).order_by(AnalysisEntry.id.desc()).limit(limit)
    rows = db.exec(stmt).all()

    return {
        "status": "success",
        "count": len(rows),
        "results": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "file_names": json.loads(r.file_names_json),
                "num_files": r.num_files,
                "num_documents": r.num_documents
            }
            for r in rows
        ]
    }


@app.get("/api/history/{hid}")
def history_item(hid: int, db: Session = Depends(get_session)):
    row = db.get(AnalysisEntry, hid)
    if not row:
        raise HTTPException(status_code=404, detail="History item not found")

    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "file_names": json.loads(row.file_names_json),
        "num_files": row.num_files,
        "num_documents": row.num_documents,
        "per_file": json.loads(row.per_file_json),
        "topics": json.loads(row.topics_json)
    }
