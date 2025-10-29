# Data models for the API responses
# Using Pydantic to validate the data types

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class AnalysisResponse(BaseModel):
    """Response object after analyzing files"""
    status: str
    num_files: int
    num_documents: int
    file_names: List[str]
    analysis_type: str
    sentiment: Dict[str, Any]
    summary: str
    topics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    note: Optional[str] = None


class HealthResponse(BaseModel):
    """Quick check to make sure backend is alive"""
    status: str
    timestamp: str
    groq_configured: bool
    version: str
