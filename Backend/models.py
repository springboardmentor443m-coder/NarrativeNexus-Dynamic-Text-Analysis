from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Analysis(Base):
    """Store text analysis results"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer)
    char_count = Column(Integer)
    reading_time_minutes = Column(Float)
    sentiment = Column(String(50))
    summary = Column(Text)
    narrative = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Topic(Base):
    """Store extracted topics"""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, nullable=False, index=True)
    topic_number = Column(Integer)
    words = Column(JSON)
    words_string = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class SavedNarrative(Base):
    """Store complete analysis narratives"""
    __tablename__ = "saved_narratives"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer)
    sentiment = Column(String(50))
    topics_json = Column(JSON)
    summary = Column(Text)
    narrative = Column(Text)
    tags = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
