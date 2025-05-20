from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class QueryInput(BaseModel):
    """Input for processing a user query."""
    query: str = Field(..., description="The user's query")
    context: Dict[str, Any] = Field(default_factory=dict, description="Optional context information")
    run_id: Optional[str] = Field(default=None, description="Optional run ID for tracking")
    
    @validator("query")
    def query_not_empty(cls, v: str) -> str:
        """Validate that the query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v


class AgentRunOptions(BaseModel):
    """Options for running an agent."""
    use_cache: bool = Field(default=False, description="Whether to use cached results")
    streaming: bool = Field(default=False, description="Whether to stream results")
    timeout_seconds: int = Field(default=60, description="Timeout in seconds")
    
    # Agent-specific settings
    temperature: Optional[float] = Field(default=None, description="Temperature for non-deterministic agents")
    top_p: Optional[float] = Field(default=None, description="Top-p for non-deterministic agents")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")


class FileInput(BaseModel):
    """Input for processing a file."""
    file_path: str = Field(..., description="Path to the file")
    content_type: Optional[str] = Field(default=None, description="Content type of the file")
    
    @validator("file_path")
    def file_path_not_empty(cls, v: str) -> str:
        """Validate that the file path is not empty."""
        if not v.strip():
            raise ValueError("File path cannot be empty")
        return v