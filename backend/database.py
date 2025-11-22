from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TextDocument(Base):
    _tablename_ = "text_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)  # 0: pending, 1: processing, 2: completed


class AnalysisResult(Base):
    _tablename_ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)  # positive, negative, neutral
    summary = Column(Text)
    topics = Column(JSON)  # List of topics with weights
    themes = Column(JSON)  # List of detected themes
    keywords = Column(JSON)  # List of important keywords
    created_at = Column(DateTime, default=datetime.utcnow)


class TopicModel(Base):
    _tablename_ = "topic_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True)
    num_topics = Column(Integer)
    model_data = Column(JSON)  # Serialized LDA model data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
