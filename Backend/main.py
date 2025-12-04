from __future__ import annotations

import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import joblib

from pydantic import BaseModel

from preprocessing import clean_text, preprocess_for_lda
from topic_modeling import infer_topics_for_text, create_dummy_lda_model, create_lda_model_from_external_data
from sentiment_analysis import load_model, predict_labels, create_dummy_model, rule_based_sentiment
from summarization import get_both_summaries, load_abstractive_summarizer
from database import init_db, get_db
from sqlalchemy.orm import Session
import db_operations
from schemas import SavedNarrativeCreate


app = FastAPI(title="AI Narrative Nexus (LDA + ML Sentiment)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TextPayload(BaseModel):
    title: str | None = None
    content: str


class TrainLDAPayload(BaseModel):
    directory_path: str
    n_topics: int = 10
    max_features: int = 5000
    max_files: int = None


# ---------- Utility functions ----------

def build_topic_strings(topic_word_lists: List[List[str]]) -> List[str]:
    """
    Convert list of topic word lists into compact strings
    for the UI chips.

    Example:
        [["health", "patient", "diagnosis"],
         ["model", "accuracy", "training"]]

        -> ["health / patient / diagnosis",
            "model / accuracy / training"]
    """
    topic_strings = []
    for words in topic_word_lists:
        # keep it short for the frontend chips
        short_words = words[:3]
        topic_strings.append(" / ".join(short_words))
    return topic_strings


# ---------- Model loading at startup ----------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
LDA_MODEL_PATH = MODELS_DIR / "lda_model.joblib"
LDA_VECTORIZER_PATH = MODELS_DIR / "lda_vectorizer.joblib"

sentiment_model = None
lda_model = None
lda_vectorizer = None


@app.on_event("startup")
def startup_event():
    """Initialize database and load models on startup"""
    init_db()
    load_models()
    print("[OK] Application startup complete")


def load_models():
    """
    Load all trained models once when the FastAPI app starts.
    Falls back to dummy models if real models don't exist.
    """
    global sentiment_model, lda_model, lda_vectorizer

    MODELS_DIR.mkdir(exist_ok=True)

    # Sentiment model (TF-IDF + Logistic Regression pipeline)
    if SENTIMENT_MODEL_PATH.exists():
        try:
            sentiment_model = load_model(str(SENTIMENT_MODEL_PATH))
            print("[OK] Sentiment model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load sentiment model, using dummy: {e}")
            sentiment_model = create_dummy_model()
    else:
        print("[WARN] Sentiment model not found, using dummy model for demo.")
        sentiment_model = create_dummy_model()

    # LDA model
    if LDA_MODEL_PATH.exists() and LDA_VECTORIZER_PATH.exists():
        try:
            lda_model = joblib.load(str(LDA_MODEL_PATH))
            lda_vectorizer = joblib.load(str(LDA_VECTORIZER_PATH))
            print("[OK] LDA model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load LDA model, using dummy: {e}")
            lda_vectorizer, lda_model = create_dummy_lda_model(n_topics=10)
    else:
        print("[WARN] LDA models not found, creating dummy model for demo.")
        lda_vectorizer, lda_model = create_dummy_lda_model(n_topics=10)

    print("[OK] All models ready!")


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_text(payload: TextPayload, db: Session = Depends(get_db)):
    try:
        raw_text = (payload.content or "").strip()
        title = (payload.title or "").strip() or "Untitled Narrative"

        if not raw_text:
            return {"error": "No text provided."}

        # 1. Clean / preprocess text for general use
        cleaned = clean_text(raw_text)
        if not cleaned:
            cleaned = raw_text
        
        # 2. Prepare text for LDA (more aggressive cleaning)
        lda_text = preprocess_for_lda(raw_text)
        if not lda_text:
            lda_text = cleaned
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return {"error": f"Error processing text: {str(e)}"}

    # 3. Basic statistics
    tokens = cleaned.split()
    word_count = len(tokens)
    char_count = len(raw_text)
    reading_time_minutes = round(word_count / 200, 2)

    # 4. ML-based sentiment prediction with rule-based fallback
    try:
        sentiment_label = predict_labels(sentiment_model, [cleaned])[0]
    except Exception as e:
        print(f"Error predicting sentiment with ML: {e}")
        sentiment_label = "neutral"
    
    if sentiment_label == "neutral":
        rule_sentiment = rule_based_sentiment(cleaned)
        if rule_sentiment != "neutral":
            sentiment_label = rule_sentiment

    # 5. LDA topic inference
    topic_word_lists: List[List[str]] = []
    try:
        if lda_text.strip():
            topic_word_lists = infer_topics_for_text(
                text=lda_text,
                vectorizer=lda_vectorizer,
                lda_model=lda_model,
                top_k_topics=3,
                top_n_words=6,
            )
        topics_for_ui = build_topic_strings(topic_word_lists) if topic_word_lists else ["no clear themes"]
    except Exception as e:
        topics_for_ui = ["topics unavailable"]
        print(f"Error inferring topics: {e}")

    # 6. Get both extractive and abstractive summaries
    summary_extractive, summary_abstractive = get_both_summaries(raw_text)

    # 7. Narrative explanation
    topic_desc = (
        ", ".join(topics_for_ui)
        if topics_for_ui
        else "no clear recurring themes"
    )

    narrative = (
        f"In \"{title}\", you explore a narrative of about {word_count} words, "
        f"which would take roughly {reading_time_minutes} minutes to read. "
        f"The machine-learning sentiment classifier interprets the overall tone as "
        f"**{sentiment_label}**.\n\n"
        f"From the LDA topic model's perspective, the key thematic clusters appear as: "
        f"{topic_desc}.\n\n"
        f"**Extractive Summary:**\n{summary_extractive}\n\n"
        f"**Abstractive Summary:**\n{summary_abstractive}"
    )

    # 8. Save to database
    try:
        saved_narrative = SavedNarrativeCreate(
            title=title,
            content=raw_text,
            word_count=word_count,
            sentiment=sentiment_label,
            topics_json=topic_word_lists,
            summary=summary_extractive,
            narrative=narrative
        )
        db_narrative = db_operations.create_saved_narrative(db, saved_narrative)
        narrative_id = db_narrative.id
    except Exception as e:
        print(f"Error saving to database: {e}")
        narrative_id = None

    try:
        return {
            "id": narrative_id,
            "title": title,
            "word_count": word_count,
            "char_count": char_count,
            "reading_time_minutes": reading_time_minutes,
            "sentiment": sentiment_label,
            "topics": topics_for_ui,
            "summary_extractive": summary_extractive,
            "summary_abstractive": summary_abstractive,
            "narrative": narrative,
        }
    except Exception as e:
        print(f"Error returning response: {e}")
        return {
            "error": f"Analysis complete but response formatting failed: {str(e)}",
            "title": title,
            "sentiment": sentiment_label,
            "topics": topics_for_ui
        }


@app.get("/saved-narratives")
async def get_saved_narratives(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Get all saved narratives"""
    narratives = db_operations.get_all_saved_narratives(db, skip=skip, limit=limit)
    total = db_operations.get_saved_narratives_count(db)
    return {
        "total": total,
        "narratives": [
            {
                "id": n.id,
                "title": n.title,
                "word_count": n.word_count,
                "sentiment": n.sentiment,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "updated_at": n.updated_at.isoformat() if n.updated_at else None
            }
            for n in narratives
        ]
    }


@app.get("/saved-narratives/{narrative_id}")
async def get_saved_narrative(narrative_id: int, db: Session = Depends(get_db)):
    """Get a specific saved narrative"""
    narrative = db_operations.get_saved_narrative_by_id(db, narrative_id)
    if not narrative:
        return {"error": "Narrative not found"}
    return {
        "id": narrative.id,
        "title": narrative.title,
        "content": narrative.content,
        "word_count": narrative.word_count,
        "sentiment": narrative.sentiment,
        "topics_json": narrative.topics_json,
        "summary": narrative.summary,
        "narrative": narrative.narrative,
        "created_at": narrative.created_at.isoformat() if narrative.created_at else None,
        "updated_at": narrative.updated_at.isoformat() if narrative.updated_at else None
    }


@app.get("/saved-narratives/search/{query}")
async def search_narratives(query: str, db: Session = Depends(get_db)):
    """Search saved narratives"""
    narratives = db_operations.search_saved_narratives(db, query)
    return {
        "total": len(narratives),
        "narratives": [
            {
                "id": n.id,
                "title": n.title,
                "word_count": n.word_count,
                "sentiment": n.sentiment,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "updated_at": n.updated_at.isoformat() if n.updated_at else None
            }
            for n in narratives
        ]
    }


@app.get("/saved-narratives-by-sentiment/{sentiment}")
async def get_narratives_by_sentiment(sentiment: str, db: Session = Depends(get_db)):
    """Filter narratives by sentiment"""
    narratives = db_operations.filter_saved_narratives_by_sentiment(db, sentiment)
    return {
        "total": len(narratives),
        "sentiment": sentiment,
        "narratives": [
            {
                "id": n.id,
                "title": n.title,
                "word_count": n.word_count,
                "sentiment": n.sentiment,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "updated_at": n.updated_at.isoformat() if n.updated_at else None
            }
            for n in narratives
        ]
    }


@app.get("/statistics/sentiment")
async def get_sentiment_stats(db: Session = Depends(get_db)):
    """Get sentiment statistics"""
    stats = db_operations.get_sentiment_statistics(db)
    return stats


@app.delete("/saved-narratives/{narrative_id}")
async def delete_narrative(narrative_id: int, db: Session = Depends(get_db)):
    """Delete a saved narrative"""
    success = db_operations.delete_saved_narrative(db, narrative_id)
    if success:
        return {"message": "Narrative deleted successfully", "deleted_id": narrative_id}
    return {"error": "Narrative not found"}


@app.put("/saved-narratives/{narrative_id}")
async def update_narrative(narrative_id: int, title: str = None, tags: str = None, db: Session = Depends(get_db)):
    """Update a saved narrative"""
    narrative = db_operations.update_saved_narrative(db, narrative_id, title=title, tags=tags)
    if narrative:
        return narrative
    return {"error": "Narrative not found"}


@app.post("/retrain-lda")
async def retrain_lda_model(payload: TrainLDAPayload):
    """
    Retrain the LDA model using external training data from a directory.
    
    Args:
        directory_path: Path to directory containing TXT files
        n_topics: Number of topics (default: 10)
        max_features: Max vocabulary size (default: 5000)
        max_files: Max files to load (default: None = all files)
    """
    global lda_model, lda_vectorizer
    
    try:
        print(f"[OK] Starting LDA retraining from: {payload.directory_path}")
        
        lda_vectorizer, lda_model = create_lda_model_from_external_data(
            directory_path=payload.directory_path,
            n_topics=payload.n_topics,
            max_features=payload.max_features,
            max_files=payload.max_files
        )
        
        try:
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump(lda_vectorizer, str(LDA_VECTORIZER_PATH))
            joblib.dump(lda_model, str(LDA_MODEL_PATH))
            print("[OK] Models saved to disk")
        except Exception as e:
            print(f"[WARN] Could not save models to disk: {e}")
        
        return {
            "success": True,
            "message": "LDA model retrained successfully",
            "n_topics": payload.n_topics,
            "max_features": payload.max_features,
            "directory": payload.directory_path
        }
    except FileNotFoundError as e:
        return {"success": False, "error": f"Directory not found: {e}"}
    except ValueError as e:
        return {"success": False, "error": f"No training data found: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to retrain model: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
