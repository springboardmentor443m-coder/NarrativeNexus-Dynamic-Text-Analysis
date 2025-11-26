from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import tempfile
import shutil
import os

from backend.text_processing import (
    clean_text,
    summarize_text,
    analyze_sentiment,
    extract_text_from_file,
)
from backend.topic_model import infer_topic

# Load the keywords dictionary
with open("backend/models/topic_keywords.pkl", "rb") as f:
    topic_keywords = pickle.load(f)

# Request MODEL for /api/topic
class TextRequest(BaseModel):
    text: str

app = FastAPI(title="AI Narrative Nexus API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "AI Narrative Nexus backend is running 🚀"}


@app.post("/api/process")
async def process_text(file: UploadFile = File(...)):
    """Handles uploaded documents (pdf/docx/pptx/txt/html)."""
    try:
        suffix = os.path.splitext(file.filename)[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        raw_text = extract_text_from_file(temp_path)

        if not raw_text.strip():
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Could not extract text from file. Unsupported or empty file.")

        cleaned_text = clean_text(raw_text)
        summary = summarize_text(cleaned_text)
        sentiment = analyze_sentiment(summary)

        topic_out = infer_topic(cleaned_text)

        os.remove(temp_path)

        return {
            "message": "File processed successfully!",
            "cleaned_preview": cleaned_text[:400] + "...",
            "summary": summary,
            "sentiment": sentiment,
            "topic": topic_out["id"],
            "topic_name": topic_out["name"],
            "topic_keywords": topic_out["keywords"],
            "topic_probability": topic_out["probability"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/topic")
def get_topic(req: TextRequest):
    topic_id = infer_topic(req.text)
    return {"topic": topic_id}

