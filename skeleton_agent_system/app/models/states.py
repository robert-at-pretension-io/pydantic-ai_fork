from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.models.outputs import (
    SearchResult,
    CalculationResult,
    CodeResult,
    BashResult
)


@dataclass
class BaseState:
    """Base state class for all graph states."""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    agent_messages: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class MainState(BaseState):
    """Main state for the root graph."""
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    code_results: Optional[CodeResult] = None
    bash_results: Optional[BashResult] = None
    final_response: Optional[str] = None
    

@dataclass
class SearchState(BaseState):
    """State for the search sub-graph."""
    formulated_query: Optional[str] = None
    search_results: Optional[List[Dict[str, str]]] = None
    filtered_results: Optional[SearchResult] = None


@dataclass
class CodingState(BaseState):
    """State for the coding sub-graph."""
    task_analysis: Optional[Dict[str, Any]] = None
    generated_code: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    final_code: Optional[CodeResult] = None


@dataclass
class OperationsState(BaseState):
    """State for the operations sub-graph."""
    command: Optional[str] = None
    command_output: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    final_result: Optional[BashResult] = None