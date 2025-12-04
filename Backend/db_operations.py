from sqlalchemy.orm import Session
from typing import List, Optional
from models import Analysis, Topic, SavedNarrative
from schemas import SavedNarrativeCreate, SavedNarrativeResponse
from datetime import datetime


# ===== Analyses =====

def create_analysis(
    db: Session,
    title: str,
    content: str,
    word_count: int,
    char_count: int,
    reading_time_minutes: float,
    sentiment: str,
    summary: str,
    narrative: str
) -> Analysis:
    """Save analysis to database"""
    db_analysis = Analysis(
        title=title,
        content=content,
        word_count=word_count,
        char_count=char_count,
        reading_time_minutes=reading_time_minutes,
        sentiment=sentiment,
        summary=summary,
        narrative=narrative
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis


def save_topics_for_analysis(db: Session, analysis_id: int, topics_list: List[List[str]]) -> List[Topic]:
    """Save extracted topics for an analysis"""
    saved_topics = []
    for topic_number, words in enumerate(topics_list):
        db_topic = Topic(
            analysis_id=analysis_id,
            topic_number=topic_number,
            words=words,
            words_string=" / ".join(words[:3])
        )
        db.add(db_topic)
        saved_topics.append(db_topic)
    
    db.commit()
    return saved_topics


def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[Analysis]:
    """Get analysis by ID"""
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


def get_all_analyses(db: Session, skip: int = 0, limit: int = 50) -> List[Analysis]:
    """Get all analyses with pagination"""
    return db.query(Analysis).offset(skip).limit(limit).all()


def get_analyses_by_sentiment(db: Session, sentiment: str) -> List[Analysis]:
    """Get analyses filtered by sentiment"""
    return db.query(Analysis).filter(Analysis.sentiment == sentiment).all()


def delete_analysis(db: Session, analysis_id: int) -> bool:
    """Delete analysis and associated topics"""
    db_analysis = get_analysis_by_id(db, analysis_id)
    if db_analysis:
        db.query(Topic).filter(Topic.analysis_id == analysis_id).delete()
        db.delete(db_analysis)
        db.commit()
        return True
    return False


# ===== Saved Narratives =====

def create_saved_narrative(db: Session, narrative: SavedNarrativeCreate) -> SavedNarrative:
    """Save complete narrative"""
    db_narrative = SavedNarrative(
        title=narrative.title,
        content=narrative.content,
        word_count=narrative.word_count,
        sentiment=narrative.sentiment,
        topics_json=narrative.topics_json,
        summary=narrative.summary,
        narrative=narrative.narrative,
        tags=narrative.tags
    )
    db.add(db_narrative)
    db.commit()
    db.refresh(db_narrative)
    return db_narrative


def get_saved_narrative_by_id(db: Session, narrative_id: int) -> Optional[SavedNarrative]:
    """Get saved narrative by ID"""
    return db.query(SavedNarrative).filter(SavedNarrative.id == narrative_id).first()


def get_all_saved_narratives(db: Session, skip: int = 0, limit: int = 50) -> List[SavedNarrative]:
    """Get all saved narratives"""
    return db.query(SavedNarrative).offset(skip).limit(limit).all()


def get_saved_narratives_count(db: Session) -> int:
    """Count total saved narratives"""
    return db.query(SavedNarrative).count()


def search_saved_narratives(db: Session, query: str) -> List[SavedNarrative]:
    """Search narratives by title or content"""
    return db.query(SavedNarrative).filter(
        (SavedNarrative.title.ilike(f"%{query}%")) |
        (SavedNarrative.content.ilike(f"%{query}%"))
    ).all()


def filter_saved_narratives_by_sentiment(db: Session, sentiment: str) -> List[SavedNarrative]:
    """Filter narratives by sentiment"""
    return db.query(SavedNarrative).filter(SavedNarrative.sentiment == sentiment).all()


def update_saved_narrative(
    db: Session,
    narrative_id: int,
    title: Optional[str] = None,
    tags: Optional[str] = None
) -> Optional[SavedNarrative]:
    """Update saved narrative"""
    db_narrative = get_saved_narrative_by_id(db, narrative_id)
    if db_narrative:
        if title:
            db_narrative.title = title
        if tags:
            db_narrative.tags = tags
        db_narrative.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_narrative)
    return db_narrative


def delete_saved_narrative(db: Session, narrative_id: int) -> bool:
    """Delete saved narrative"""
    db_narrative = get_saved_narrative_by_id(db, narrative_id)
    if db_narrative:
        db.delete(db_narrative)
        db.commit()
        return True
    return False


def get_sentiment_statistics(db: Session) -> dict:
    """Get statistics about saved narratives by sentiment"""
    analyses = db.query(SavedNarrative).all()
    stats = {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "total": len(analyses)
    }
    
    for analysis in analyses:
        sentiment = analysis.sentiment.lower()
        if sentiment in stats:
            stats[sentiment] += 1
    
    return stats
