from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel

from app.core.preprocessing import TextPreprocessor, extract_text_from_file
from app.models.sentiment_model import sentiment_analyzer
from app.models.sentiment_model_bert import bert_sentiment_analyzer
from app.models.summarization_model import text_summarizer  
from app.models.topic_model import topic_predictor


app = FastAPI(
    title="DyNarrative",
    description="The Dynamic Text Analysis Platform",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = ['.txt']


preprocessor = TextPreprocessor()


def recommended_sentiment_route(ext: str) -> str:
    if ext == '.txt':
        return "/sentiment"
    elif ext == '.csv':
        return "/sentiment-bert"
    return ""


@app.get("/")
def root():
    return {
        "message": "DyNarrative API is running!",
        "status": "ok",
        "version": "1.0.0",
        "supported_formats": ALLOWED_EXTENSIONS,
        "available_endpoints": [
            "GET / - Server status",
            "GET /topic-info - Available Topics",
            "POST /upload - Upload a file",
            "POST /preprocess - Data Preprocessing",
            "POST /sentiment - Sentiment Analysis",
            "POST /summarize - Summarization",
            "POST /infer-topics - Topic Modelling",  
        ]
    }

from fastapi import HTTPException
import os


@app.get("/topic-info")
async def get_topic_info():
    """Get all topics info."""
    if not topic_predictor.model:
        raise HTTPException(
            status_code=400,
            detail="Topic model not loaded. Run train_bertopic.py first"
        )

    result = topic_predictor.get_topic_info()

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(500, detail=f"Error saving file: {str(e)}")
    
    try:
        text = extract_text_from_file(file_path, file_ext)
        word_count = len(text.split())
        char_count = len(text)
        preview = text[:300] + "..." if len(text) > 300 else text
    except Exception as e:
        raise HTTPException(400, detail=f"Error processing file: {str(e)}")
    
    if char_count < 50:
        raise HTTPException(400, detail="File content too short. Need at least 50 characters.")
    
    rec_endpoint = recommended_sentiment_route(file_ext)
    
    return {
        "message": "File uploaded and processed successfully!",
        "filename": file.filename,
        "file_path": file_path,
        "file_size_bytes": len(content),
        "file_type": file_ext,
        "character_count": char_count,
        "word_count": word_count,
        "preview": preview,
        "recommended_sentiment_endpoint": rec_endpoint,
        "instructions": f"Use {rec_endpoint} POST with the file_path for sentiment analysis"
    }


@app.post("/preprocess")
async def preprocess_text(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    text = extract_text_from_file(file_path, file_ext)
    result = preprocessor.preprocess(text)
    
    return {
        "message": "Text preprocessed successfully!",
        "file_path": file_path,
        "preprocessing_result": result
    }


@app.post("/sentiment")
async def basic_sentiment(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.txt':
        return {
            "warning": "This endpoint is optimized for text files. Consider using '/sentiment-bert' for CSV files.",
            "file_path": file_path
        }
    
    text = extract_text_from_file(file_path, ext)
    preprocessed = preprocessor.preprocess(text)['processed_text']
    result = sentiment_analyzer.analyze_text(preprocessed)
    
    return {
        "message": "Sentiment analysis completed on preprocessed text!",
        "file_path": file_path,
        "sentiment_result": result
    }


@app.post("/sentiment-bert")
async def bert_sentiment(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.csv':
        return {
            "warning": "This endpoint is optimized for CSV files. Consider using '/sentiment' for text files.",
            "file_path": file_path
        }
    
    text = extract_text_from_file(file_path, ext)
    result = bert_sentiment_analyzer.analyze_text(text)
    
    return {
        "message": "BERT sentiment analysis completed",
        "file_path": file_path,
        "sentiment_result": result
    }

class FilePathRequest(BaseModel):
    file_path: str


@app.post("/summarize")
async def summarize_text(payload: FilePathRequest):
    file_path = payload.file_path

    if not os.path.exists(file_path):
        raise HTTPException(404, detail="File not found")

    ext = os.path.splitext(file_path)[1].lower()
    text = extract_text_from_file(file_path, ext)

    preprocessed = preprocessor.preprocess(text)
    cleaned_text = preprocessed["processed_text"]

    if len(cleaned_text.strip()) < 50:
        raise HTTPException(400, detail="Text too short for summarization. Need at least 50 characters.")

    result = text_summarizer.summarize(cleaned_text)

    if result["status"] != "success":
        raise HTTPException(500, detail=f"Summarization failed: {result.get('error', 'unknown error')}")

    return {
        "message": "Summarization completed",
        "file_path": file_path,
        "cleaned_text_preview": cleaned_text[:300],
        "summary": result["summary"],
    }


@app.post("/infer-topics")
async def infer_topics(text: str):
    """Infer topic for a single text."""
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short")

    if not topic_predictor.model:
        raise HTTPException(
            status_code=400,
            detail="Topic model not loaded. Run train_bertopic.py first"
        )

    result = topic_predictor.predict_single(text)

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))

    return {
        "message": "Topic inferred successfully",
        **result
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
