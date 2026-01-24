"""Data models and schemas for DataBrain AI Agent."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""
    name: str
    schema: Dict[str, Any] = Field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Result of a data analysis operation."""
    query: str
    result: Any
    tool_used: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Response from the agent."""
    message: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    dataset_context: Optional[DatasetInfo] = None
