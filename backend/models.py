# backend/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AnalysisResponse(BaseModel):
    status: str
    num_files: int
    num_documents: int
    file_names: List[str]
    analysis_type: Optional[str] = None
    sentiment: Dict[str, Any]
    summary: str
    keywords: List[str]
    topics: Dict[str, Any]
    timestamp: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    groq_configured: bool
    version: str

class AnalysisHistory(BaseModel):
    id: Optional[int] = None
    file_names: List[str]
    sentiment: Dict[str, Any]
    summary: str
    keywords: List[str]
    topics: Dict[str, Any]
    num_files: int
    num_documents: int
    timestamp: str
    analysis_type: Optional[str] = None
