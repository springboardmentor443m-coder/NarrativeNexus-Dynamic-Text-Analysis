from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

from database import init_db, get_db, TextDocument, AnalysisResult, TopicModel
from text_processor import TextProcessor
from sentiment_analyzer import SentimentAnalyzer
from lda_modeler import LDAModeler
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Text Analysis Platform",
    description="Intelligent platform for processing, analyzing, and summarizing text data with LDA topic modeling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
text_processor = TextProcessor()
sentiment_analyzer = SentimentAnalyzer()
lda_modeler = LDAModeler(
    num_topics=settings.num_topics,
    passes=settings.passes,
    alpha=settings.alpha,
    beta=settings.beta
)

# Initialize database
init_db()

# Pydantic models
class TextInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=settings.max_text_length)
    source: Optional[str] = "unknown"


class BatchTextInput(BaseModel):
    texts: List[TextInput] = Field(..., min_length=1)


class TrainLDAInput(BaseModel):
    model_name: str = "default"
    num_topics: Optional[int] = None
    document_ids: Optional[List[int]] = None


class AnalysisResponse(BaseModel):
    document_id: int
    sentiment_score: float
    sentiment_label: str
    summary: str
    topics: List[dict]
    themes: List[str]
    keywords: List[dict]
    created_at: datetime


class TopicModelResponse(BaseModel):
    model_id: int
    model_name: str
    num_topics: int
    topics: List[dict]
    coherence_score: Optional[float]
    created_at: datetime


# WebSocket connection manager
class ConnectionManager:
    def _init_(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


def analyze_text(text: str, document_id: int, db: Session):
    """Background task to analyze text"""
    try:
        # Update document status to processing
        document = db.query(TextDocument).filter(TextDocument.id == document_id).first()
        if document:
            document.processed = 1
            db.commit()
        
        # Perform analysis
        sentiment_result = sentiment_analyzer.analyze(text)
        summary = text_processor.summarize(text)
        keywords = text_processor.extract_keywords(text)
        themes = text_processor.detect_themes(text)
        
        # Predict topics using LDA model
        topics = []
        try:
            # Get the latest topic model
            topic_model_record = db.query(TopicModel).order_by(TopicModel.updated_at.desc()).first()
            if topic_model_record and topic_model_record.model_data:
                # Recreate LDA modeler with stored model data
                model_data = topic_model_record.model_data
                temp_modeler = LDAModeler(
                    num_topics=model_data.get("num_topics", settings.num_topics),
                    passes=model_data.get("passes", settings.passes)
                )
                temp_modeler.load_model(model_data)
                # For prediction, we'd need the full model - simplified for now
                # In production, you'd store and load the full Gensim model
                topics = [{"topic_id": i, "probability": 0.0} for i in range(model_data.get("num_topics", settings.num_topics))]
        except Exception as e:
            print(f"Error predicting topics: {e}")
        
        # Save analysis result
        analysis = AnalysisResult(
            document_id=document_id,
            sentiment_score=sentiment_result["sentiment_score"],
            sentiment_label=sentiment_result["sentiment_label"],
            summary=summary,
            topics=topics,
            themes=themes,
            keywords=keywords
        )
        db.add(analysis)
        
        # Update document status to completed
        if document:
            document.processed = 2
        db.commit()
        
    except Exception as e:
        print(f"Error analyzing text: {e}")
        if document:
            document.processed = 0  # Reset to pending on error
            db.commit()


@app.on_event("startup")
async def startup_event():
    print("Starting Dynamic Text Analysis Platform...")


@app.get("/")
async def root():
    """Root endpoint - serves frontend if available, otherwise returns API info"""
    html_file = Path("frontend_example.html")
    if html_file.exists():
        # Serve frontend at root for convenience
        return FileResponse(html_file)
    return {
        "message": "Dynamic Text Analysis Platform API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "batch_analyze": "/api/analyze/batch",
            "documents": "/api/documents",
            "analysis": "/api/analysis/{document_id}",
            "train_lda": "/api/lda/train",
            "topics": "/api/lda/topics",
            "websocket": "/ws",
            "docs": "/docs",
            "frontend": "/app"
        },
        "frontend": "Visit /app for the web interface"
    }


@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    html_file = Path("frontend_example.html")
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Frontend file not found. Make sure frontend_example.html exists.")


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Dynamic Text Analysis Platform API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "batch_analyze": "/api/analyze/batch",
            "documents": "/api/documents",
            "analysis": "/api/analysis/{document_id}",
            "train_lda": "/api/lda/train",
            "topics": "/api/lda/topics",
            "websocket": "/ws",
            "frontend": "/app"
        }
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_text_endpoint(
    text_input: TextInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Analyze a single text document"""
    # Save document
    document = TextDocument(
        content=text_input.content,
        source=text_input.source
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Start background analysis
    background_tasks.add_task(analyze_text, text_input.content, document.id, db)
    
    # Return initial response (analysis will be completed in background)
    return AnalysisResponse(
        document_id=document.id,
        sentiment_score=0.0,
        sentiment_label="processing",
        summary="Analysis in progress...",
        topics=[],
        themes=[],
        keywords=[],
        created_at=document.created_at
    )


@app.post("/api/analyze/batch")
async def batch_analyze_texts(
    batch_input: BatchTextInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Analyze multiple text documents"""
    document_ids = []
    
    for text_input in batch_input.texts:
        document = TextDocument(
            content=text_input.content,
            source=text_input.source
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        document_ids.append(document.id)
        
        # Start background analysis
        background_tasks.add_task(analyze_text, text_input.content, document.id, db)
    
    return {
        "message": f"Processing {len(document_ids)} documents",
        "document_ids": document_ids,
        "status": "processing"
    }


@app.get("/api/documents")
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    source: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all documents with optional filtering"""
    query = db.query(TextDocument)
    
    if source:
        query = query.filter(TextDocument.source == source)
    
    documents = query.offset(skip).limit(limit).all()
    
    return {
        "total": len(documents),
        "documents": [
            {
                "id": doc.id,
                "source": doc.source,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "processed": doc.processed,
                "created_at": doc.created_at
            }
            for doc in documents
        ]
    }


@app.get("/api/analysis/{document_id}", response_model=AnalysisResponse)
async def get_analysis(document_id: int, db: Session = Depends(get_db)):
    """Get analysis results for a document"""
    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.document_id == document_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    document = db.query(TextDocument).filter(TextDocument.id == document_id).first()
    
    return AnalysisResponse(
        document_id=document_id,
        sentiment_score=analysis.sentiment_score,
        sentiment_label=analysis.sentiment_label,
        summary=analysis.summary,
        topics=analysis.topics or [],
        themes=analysis.themes or [],
        keywords=analysis.keywords or [],
        created_at=analysis.created_at
    )


@app.post("/api/lda/train")
async def train_lda_model(
    train_input: TrainLDAInput = TrainLDAInput(),
    db: Session = Depends(get_db)
):
    """Train LDA topic model on documents"""
    # Get documents to train on
    if train_input.document_ids:
        documents = db.query(TextDocument).filter(
            TextDocument.id.in_(train_input.document_ids),
            TextDocument.processed == 2
        ).all()
    else:
        documents = db.query(TextDocument).filter(
            TextDocument.processed == 2
        ).all()
    
    if len(documents) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 processed documents to train LDA model"
        )
    
    texts = [doc.content for doc in documents]
    
    # Train model
    num_topics_to_use = train_input.num_topics or settings.num_topics
    modeler = LDAModeler(
        num_topics=num_topics_to_use,
        passes=settings.passes,
        alpha=settings.alpha,
        beta=settings.beta
    )
    
    try:
        result = modeler.train(texts)
        topics = modeler.get_topics()
        
        # Save model
        model_data = modeler.serialize_model()
        
        # Check if model with this name exists
        existing_model = db.query(TopicModel).filter(
            TopicModel.model_name == train_input.model_name
        ).first()
        
        if existing_model:
            existing_model.num_topics = num_topics_to_use
            existing_model.model_data = model_data
            existing_model.updated_at = datetime.utcnow()
            db.commit()
            model_id = existing_model.id
        else:
            new_model = TopicModel(
                model_name=train_input.model_name,
                num_topics=num_topics_to_use,
                model_data=model_data
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)
            model_id = new_model.id
        
        return {
            "message": "LDA model trained successfully",
            "model_id": model_id,
            "model_name": train_input.model_name,
            "num_topics": num_topics_to_use,
            "coherence_score": result["coherence_score"],
            "topics": topics,
            "documents_used": len(texts)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/api/lda/topics")
async def get_lda_topics(
    model_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get topics from the latest or specified LDA model"""
    if model_name:
        topic_model = db.query(TopicModel).filter(
            TopicModel.model_name == model_name
        ).first()
    else:
        topic_model = db.query(TopicModel).order_by(
            TopicModel.updated_at.desc()
        ).first()
    
    if not topic_model:
        raise HTTPException(status_code=404, detail="No LDA model found")
    
    topics = topic_model.model_data.get("topics", [])
    
    return TopicModelResponse(
        model_id=topic_model.id,
        model_name=topic_model.model_name,
        num_topics=topic_model.num_topics,
        topics=topics,
        coherence_score=None,  # Would need to store this separately
        created_at=topic_model.created_at
    )


@app.get("/api/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get platform statistics"""
    total_documents = db.query(TextDocument).count()
    processed_documents = db.query(TextDocument).filter(
        TextDocument.processed == 2
    ).count()
    total_analyses = db.query(AnalysisResult).count()
    total_models = db.query(TopicModel).count()
    
    # Average sentiment
    avg_sentiment = db.query(
        db.func.avg(AnalysisResult.sentiment_score)
    ).scalar() or 0.0
    
    return {
        "total_documents": total_documents,
        "processed_documents": processed_documents,
        "pending_documents": total_documents - processed_documents,
        "total_analyses": total_analyses,
        "total_models": total_models,
        "average_sentiment": float(avg_sentiment)
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process message
            await manager.send_personal_message(
                f"Message received: {data}",
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
