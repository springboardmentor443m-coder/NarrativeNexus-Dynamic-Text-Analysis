"""
Pydantic Models for API Request/Response
Enhanced with Advanced Sentiment & Topic Support
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union


class AnalysisResponse(BaseModel):
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
    status: str
    timestamp: str
    groq_configured: bool
    version: str
