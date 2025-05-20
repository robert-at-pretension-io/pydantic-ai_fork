from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

# Define the output type for the classifier agent
class QueryClassification(BaseModel):
    """Classification of a user query."""
    category: Literal["SEARCH", "CODING", "OPERATIONS", "PARALLEL", "SIMPLE"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# Create the classifier agent
# Use type casting to avoid mypy checking model name against literal list
model_name = cast(str, "openai:gpt-4o")

classifier_agent = Agent(
    model=model_name,
    output_type=QueryClassification,
    system_prompt="""You are a specialized query classification agent.
Your task is to analyze user queries and determine which specialized processing system should handle them.

Categories:
- SEARCH: Queries about facts, information retrieval, or questions that require external knowledge
- CODING: Queries about writing, debugging, or analyzing code
- OPERATIONS: Queries that involve running commands, system operations, or file manipulations
- PARALLEL: Complex queries that would benefit from both search and coding/operations
- SIMPLE: Simple queries that can be answered directly without specialized processing

Analyze each query carefully and provide your classification with confidence level and reasoning.
"""
)