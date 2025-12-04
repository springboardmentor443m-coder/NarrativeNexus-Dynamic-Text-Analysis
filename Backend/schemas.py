from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TopicBase(BaseModel):
    topic_number: Optional[int] = None
    words: List[str]


class TopicCreate(TopicBase):
    analysis_id: int


class Topic(TopicBase):
    id: int
    analysis_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class AnalysisBase(BaseModel):
    title: str
    content: str


class AnalysisCreate(AnalysisBase):
    word_count: int
    char_count: int
    reading_time_minutes: float
    sentiment: str
    summary: str
    narrative: str


class Analysis(AnalysisCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    topics: List[Topic] = []
    
    class Config:
        from_attributes = True


class SavedNarrativeBase(BaseModel):
    title: str
    content: str
    sentiment: str
    summary: str
    narrative: str
    tags: Optional[str] = None


class SavedNarrativeCreate(SavedNarrativeBase):
    word_count: int
    topics_json: List[List[str]]


class SavedNarrative(SavedNarrativeCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SavedNarrativeResponse(BaseModel):
    id: int
    title: str
    word_count: int
    sentiment: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SavedNarrativeDetail(SavedNarrative):
    pass


class AllSavedNarratives(BaseModel):
    total: int
    narratives: List[SavedNarrativeResponse]


class DeleteResponse(BaseModel):
    message: str
    deleted_id: int
