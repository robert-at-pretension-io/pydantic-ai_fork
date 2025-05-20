from __future__ import annotations

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Output from the search agent."""
    query: str
    results: List[str]
    source_urls: Optional[List[str]] = None


class CalculationResult(BaseModel):
    """Output from the calculation agent."""
    input: str
    result: float
    steps: List[str] = Field(description="Step-by-step explanation of the calculation")


class CodeResult(BaseModel):
    """Output from the code completion agent."""
    task: str
    code: str
    explanation: Optional[str] = None
    tests: Optional[str] = None


class BashResult(BaseModel):
    """Output from the bash execution agent."""
    command: str
    output: str
    exit_code: int
    success: bool = Field(description="Whether the command executed successfully")
    explanation: Optional[str] = None


class AggregatedResponse(BaseModel):
    """Combined output from multiple agents."""
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    code_results: Optional[CodeResult] = None
    bash_results: Optional[BashResult] = None
    final_answer: str = Field(description="Synthesized response incorporating all agent outputs")