# FastAPI backend for the text analysis app
# Handles requests from the frontend and runs the actual analysis

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

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Need GROQ_API_KEY in .env file")

# Set up the app
app = FastAPI(
    title="AI Narrative Nexus API",
    description="Advanced text analysis using RoBERTa, BERTopic, and Groq LLM",
    version="3.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analyzer
analyzer = GroqAnalyzer(api_key=GROQ_API_KEY)
preprocessor = TextPreprocessor()

# Make sure upload folder exists
os.makedirs("data/uploads", exist_ok=True)


@app.get("/")
async def root():
    """Just a quick endpoint to see what this API does"""
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
    """Check if everything is running properly"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        groq_configured=GROQ_API_KEY is not None,
        version="3.0.0"
    )


@app.post("/api/analyze/files")
async def analyze_files(files: List[UploadFile] = File(...)):
    """
    Main analysis endpoint
    Takes files, extracts text, and runs sentiment + topic analysis
    This is the workhorse endpoint - handles everything
    """
    try:
        all_texts = []
        file_names = []
        
        print(f"Received {len(files)} file(s) for analysis")
        
        # Extract text from all uploaded files
        for file in files:
            print(f"Processing file: {file.filename}")
            
            if not allowed_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Can't process {file.filename} - try TXT, CSV, or DOCX"
                )
            
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            
            texts = extract_text_from_file(file.filename, content)
            
            # Make sure we got something
            if isinstance(texts, list):
                all_texts.extend(texts)
                print(f"Extracted {len(texts)} text chunks from {file.filename}")
            else:
                all_texts.append(texts)
                print(f"Extracted text from {file.filename}")
            
            file_names.append(file.filename)
        
        print(f"Total text chunks extracted: {len(all_texts)}")
        
        if not all_texts:
            raise HTTPException(
                status_code=400,
                detail="No text found in your files"
            )
        
        # Join all texts together
        combined_text = " ".join([str(t) for t in all_texts if t])
        print(f"Combined text length: {len(combined_text)} characters")
        
        # Only do light cleaning - don't remove content
        if combined_text.strip():
            cleaned_text = preprocessor.clean_text(combined_text)
        else:
            cleaned_text = combined_text
        
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            print(f"ERROR: Text too short after cleaning. Length: {len(cleaned_text)}")
            print(f"Cleaned text preview: '{cleaned_text[:100]}'")
            raise HTTPException(
                status_code=400,
                detail=f"After cleaning, no usable text left. Original: {len(combined_text)} chars, After cleaning: {len(cleaned_text)} chars"
            )
        
        # For topics, we need the original texts separately
        cleaned_texts = [
            preprocessor.clean_text(str(t)) 
            for t in all_texts 
            if t and len(str(t).strip()) > 5
        ]
        
        print(f"Cleaned texts ready for analysis: {len(cleaned_texts)}")
        
        # Run the full analysis
        analysis_result = analyzer.analyze_comprehensive(
            cleaned_texts if cleaned_texts else [cleaned_text]
        )
        
        result = {
            "status": "success",
            "num_files": len(files),
            "num_documents": len(cleaned_texts) if cleaned_texts else 1,
            "file_names": file_names,
            "sentiment": analysis_result.get("sentiment"),
            "summary": analysis_result.get("summary"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add topics if we found any
        if "topics" in analysis_result:
            result["topics"] = analysis_result["topics"]
            sentiment_model = analysis_result["sentiment"].get("model", "Unknown")
            topics_model = analysis_result["topics"].get("model", "Unknown")
            result["analysis_type"] = f"Advanced ({sentiment_model} + {topics_model})"
        else:
            sentiment_model = analysis_result["sentiment"].get("model", "Unknown")
            result["analysis_type"] = f"Enhanced Sentiment ({sentiment_model})"
            result["note"] = "Need 2+ documents to discover topics"
        
        print("Analysis complete - returning results")
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in analyze_files: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Something broke: {str(e)}")


@app.post("/api/sentiment/advanced")
async def analyze_sentiment_advanced(text: str):
    """
    Analyze just the sentiment of some text
    Good for quick sentiment checks on individual texts
    """
    if not text or len(text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text is too short")
    
    try:
        print(f"Sentiment analysis for text: {len(text)} characters")
        result = analyzer.analyze_sentiment_advanced(text)
        
        return JSONResponse(content={
            "status": "success",
            "sentiment_analysis": result,
            "analysis_type": f"Advanced Sentiment ({result.get('model', 'Unknown')})"
        })
    except Exception as e:
        print(f"ERROR in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.post("/api/topics/advanced")
async def extract_topics_advanced(files: List[UploadFile] = File(...)):
    """
    Extract topics from multiple documents
    Requires at least 2 documents to find meaningful topics
    """
    try:
        all_texts = []
        file_names = []
        
        print(f"Topic extraction for {len(files)} file(s)")
        
        # Extract text
        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(status_code=400, detail=f"Can't process {file.filename}")
            
            content = await file.read()
            texts = extract_text_from_file(file.filename, content)
            
            if isinstance(texts, list):
                all_texts.extend(texts)
            else:
                all_texts.append(texts)
            
            file_names.append(file.filename)
        
        if not all_texts:
            raise HTTPException(status_code=400, detail="No text found")
        
        # Clean texts
        cleaned_texts = [
            preprocessor.clean_text(str(t)) 
            for t in all_texts 
            if t and len(str(t).strip()) > 20
        ]
        
        print(f"Cleaned texts for topic modeling: {len(cleaned_texts)}")
        
        if len(cleaned_texts) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 documents for topic modeling")
        
        # Extract topics
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
        print(f"ERROR in topic extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Narrative Nexus API v3.0")
    print(f"Groq API configured: {bool(GROQ_API_KEY)}")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
