from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, TypeVar, Awaitable, cast
from datetime import timedelta

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge, IsInstance
from pydantic_evals.generation import generate_dataset

from app.models.inputs import QueryInput
from app.models.outputs import AggregatedResponse
from app.graphs.root import root_graph
from app.models.states import MainState


T = TypeVar("T")
U = TypeVar("U")


class QueryMetadata(BaseModel):
    """Metadata for query evaluation."""
    domain: str
    difficulty: str
    expected_subgraphs: List[str]


class GraphPerformanceEvaluator(Evaluator[QueryInput, AggregatedResponse]):
    """
    Evaluates the performance of the graph system
    based on execution time and correct routing.
    """
    
    async def evaluate(
        self, ctx: EvaluatorContext[QueryInput, AggregatedResponse]
    ) -> Dict[str, float]:
        # Get the metadata
        metadata = cast(Optional[QueryMetadata], ctx.metadata)
        if not metadata:
            return {"performance": 0.0, "routing": 0.0}
        
        # Evaluate performance based on execution time
        # The duration is a timedelta object
        if isinstance(ctx.duration, timedelta):
            execution_time = ctx.duration.total_seconds()
        else:
            execution_time = 0.0
        performance_score = 1.0 if execution_time < 3.0 else 0.5
        
        # Check if the correct subgraphs were used
        # This would require enhancing the system to track which subgraphs were used
        routing_score = 1.0  # Default to 1.0 for now
        
        # Use OpenTelemetry trace analysis if available
        span_tree = ctx.span_tree
        if span_tree:
            # Find all subgraph spans
            subgraph_spans = span_tree.find(lambda node: "subgraph" in node.name)
            used_subgraphs = [span.name.split(":")[1] if ":" in span.name else span.name 
                             for span in subgraph_spans]
            
            # Check if all expected subgraphs were used
            expected = set(metadata.expected_subgraphs)
            actual = set(used_subgraphs)
            
            if expected.issubset(actual):
                routing_score = 1.0
            else:
                # Partial score based on overlap
                overlap = len(expected.intersection(actual))
                routing_score = overlap / len(expected) if expected else 0.0
        
        return {
            "performance": performance_score,
            "routing": routing_score
        }


async def process_query(query_input: QueryInput) -> AggregatedResponse:
    """
    Process a query through the root graph.
    
    Args:
        query_input: The query input
        
    Returns:
        The processed result
    """
    # Create initial state
    state = MainState(
        query=query_input.query,
        context=query_input.context or {}
    )
    
    # Run the graph with its start node
    from app.graphs.root import QueryClassification
    result = await root_graph.run(
        start_node=QueryClassification(),
        state=state
    )
    
    # Return the output
    return cast(AggregatedResponse, result.output)


async def generate_eval_dataset(n_examples: int = 10) -> Dataset[QueryInput, AggregatedResponse, QueryMetadata]:
    """
    Generate a dataset for evaluating the agent system.
    
    Args:
        n_examples: Number of examples to generate
        
    Returns:
        A dataset with test cases
    """
    # Generate a dataset
    return await generate_dataset(
        dataset_type=Dataset[QueryInput, AggregatedResponse, QueryMetadata],
        n_examples=n_examples,
        extra_instructions="""
        Generate diverse query examples that would test different aspects of the agent system:
        1. Some queries should be about search (e.g., "What is the population of Japan?")
        2. Some queries should be about coding (e.g., "Write a Python function to calculate Fibonacci numbers")
        3. Some queries should be about operations (e.g., "How do I check disk usage on Linux?")
        4. Some queries should require multiple domains (e.g., "Write a Python script to fetch and analyze weather data")
        
        For each query, specify the expected domains that should be used in the metadata.
        """
    )


async def evaluate_agent_system() -> None:
    """
    Evaluate the agent system using generated test cases.
    """
    # Check if we have an existing dataset
    dataset_path = Path("eval_dataset.yaml")
    
    if dataset_path.exists():
        # Load existing dataset
        dataset = Dataset[QueryInput, AggregatedResponse, QueryMetadata].from_file(dataset_path)
    else:
        # Generate a new dataset
        dataset = await generate_eval_dataset(10)
        # Save to file for future use
        dataset.to_file(dataset_path)
    
    # Add evaluators
    dataset.add_evaluator(IsInstance(type_name="AggregatedResponse"))
    dataset.add_evaluator(GraphPerformanceEvaluator())
    dataset.add_evaluator(LLMJudge(
        rubric="Response should directly answer the query and be well-structured",
        include_input=True
    ))
    
    # Run evaluation
    report = await dataset.evaluate(process_query, max_concurrency=2)
    
    # Print report
    report.print(include_input=True, include_output=True)
    
    # You can also save the report
    # report.to_json_file("eval_results.json")


# To run the evaluation
if __name__ == "__main__":
    asyncio.run(evaluate_agent_system())