"""
FastAPI Backend for AI Narrative Nexus
Advanced Text Analysis with Transformers + BERTopic + Groq LLM Integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import os
from datetime import datetime
from dotenv import load_dotenv

from backend.models import AnalysisResponse, HealthResponse
from backend.groq_analyzer import GroqAnalyzer
from backend.utils import allowed_file, extract_text_from_file
from backend.preprocessing import TextPreprocessor

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

app = FastAPI(
    title="AI Narrative Nexus API",
    description="Advanced Text Analysis with Transformers + BERTopic + Groq LLM",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = GroqAnalyzer(api_key=GROQ_API_KEY)
preprocessor = TextPreprocessor()

os.makedirs("data/uploads", exist_ok=True)


@app.get("/")
async def root():
    return {
        "message": "AI Narrative Nexus API v3.0",
        "version": "3.0.0",
        "description": "Advanced Text Analysis with Transformers + BERTopic + Groq LLM",
        "features": [
            "Advanced Transformer Sentiment Analysis (RoBERTa)",
            "Advanced Topic Modeling (BERTopic)",
            "Groq LLM Integration",
            "Automatic Fallback Systems"
        ],
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze/files",
            "sentiment_advanced": "/api/sentiment/advanced", 
            "topics_advanced": "/api/topics/advanced"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        groq_configured=GROQ_API_KEY is not None,
        version="3.0.0"
    )


@app.post("/api/analyze/files")
async def analyze_files(files: List[UploadFile] = File(...)):
    """
    Enhanced Multi-File Analysis with Advanced Methods:
    - Advanced Sentiment Analysis (Transformers + Groq)
    - Advanced Topic Modeling (BERTopic + Groq)
    - Automatic fallback to Groq if advanced methods fail
    """
    try:
        all_texts = []
        file_names = []
        
        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed: {file.filename}"
                )
            
            content = await file.read()
            texts = extract_text_from_file(file.filename, content)
            all_texts.extend(texts)
            file_names.append(file.filename)
        
        if not all_texts:
            raise HTTPException(
                status_code=400,
                detail="No text content found in uploaded files"
            )
        
        # Clean texts
        cleaned_texts = [preprocessor.clean_text(t) for t in all_texts if len(t.strip()) > 10]
        
        if not cleaned_texts:
            raise HTTPException(
                status_code=400,
                detail="No valid text content after preprocessing"
            )
        
        # Use comprehensive advanced analysis
        analysis_result = analyzer.analyze_comprehensive(cleaned_texts)
        
        result = {
            "status": "success",
            "num_files": len(files),
            "num_documents": len(cleaned_texts),
            "file_names": file_names,
            "sentiment": analysis_result.get("sentiment"),
            "summary": analysis_result.get("summary"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add topics if available
        if "topics" in analysis_result:
            result["topics"] = analysis_result["topics"]
            sentiment_model = analysis_result["sentiment"].get("model", "Unknown")
            topics_model = analysis_result["topics"].get("model", "Unknown")
            result["analysis_type"] = f"Advanced ({sentiment_model} + {topics_model})"
        else:
            sentiment_model = analysis_result["sentiment"].get("model", "Unknown")
            result["analysis_type"] = f"Enhanced Sentiment ({sentiment_model})"
            result["note"] = "Upload 2+ documents to enable topic modeling"
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/sentiment/advanced")
async def analyze_sentiment_advanced(text: str):
    """Advanced sentiment analysis endpoint using transformers"""
    if not text or len(text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text must be at least 5 characters")
    
    try:
        result = analyzer.analyze_sentiment_advanced(text)
        return JSONResponse(content={
            "status": "success",
            "sentiment_analysis": result,
            "analysis_type": f"Advanced Sentiment ({result.get('model', 'Unknown')})"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced sentiment analysis failed: {str(e)}")


@app.post("/api/topics/advanced")
async def extract_topics_advanced(files: List[UploadFile] = File(...)):
    """Advanced topic extraction endpoint using BERTopic"""
    try:
        all_texts = []
        file_names = []
        
        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(status_code=400, detail=f"File type not allowed: {file.filename}")
            
            content = await file.read()
            texts = extract_text_from_file(file.filename, content)
            all_texts.extend(texts)
            file_names.append(file.filename)
        
        if not all_texts:
            raise HTTPException(status_code=400, detail="No text content found")
        
        cleaned_texts = [preprocessor.clean_text(t) for t in all_texts if len(t.strip()) > 20]
        
        if len(cleaned_texts) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 documents for topic modeling")
        
        topics_result = analyzer.extract_topics_advanced(cleaned_texts)
        
        return JSONResponse(content={
            "status": "success",
            "num_files": len(files),
            "num_documents": len(cleaned_texts),
            "file_names": file_names,
            "topics": topics_result,
            "analysis_type": f"Advanced Topics ({topics_result.get('model', 'Unknown')})"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced topic extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
