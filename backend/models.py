from typing import List, Optional, Any, Dict
from pydantic import BaseModel

# --- Data Structures ---

class TopicData(BaseModel):
    """Explicit schema for topic classification results."""
    topic_id: int
    topic_name: str
    similarity: float

class SentimentData(BaseModel):
    """Explicit schema for sentiment analysis results."""
    sentiment: str
    confidence: float

class PerFileResult(BaseModel):
    """
    Encapsulates all AI analysis for a single document.
    Now includes topic_info nested correctly per file.
    """
    file_name: str
    summary: str
    sentiment: SentimentData
    assigned_topic: TopicData 

# --- API Responses ---

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    groq_configured: bool
    version: str

class AnalysisResponse(BaseModel):
    """
    The main payload returned to the frontend.
    """
    status: str
    timestamp: str
    num_files: int
    num_documents: int
    file_names: List[str]
    per_file: List[PerFileResult]
    history_id: Optional[int] = None

# --- Database / History Models ---

class HistoryListItem(BaseModel):
    """Lightweight model for list views (sidebar/history page)."""
    id: int
    timestamp: str
    file_names: List[str]
    num_files: int
    num_documents: int

class HistoryItem(HistoryListItem):
    """Detailed model for retrieving a full historical record."""
    per_file: List[Dict[str, Any]]  # Flexible dict to handle stored JSON structure