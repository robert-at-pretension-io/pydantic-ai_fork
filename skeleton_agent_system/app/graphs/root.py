from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, cast

from pydantic_ai import Agent, format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.core.base_agent import agent_registry
from app.models.outputs import AggregatedResponse
from app.models.states import MainState, SearchState, CodingState, OperationsState

# Import domain-specific sub-graphs
from app.graphs.domains.search.graph import search_graph, QueryFormulation
from app.graphs.domains.coding.graph import coding_graph, CodeTaskAnalysis
from app.graphs.domains.operations.graph import operations_graph, BashCommandFormulation


@dataclass
class QueryClassification(BaseNode[MainState]):
    """
    Analyzes the user query and determines which specialized 
    sub-graph should handle it.
    """
    
    async def run(
        self, 
        ctx: GraphRunContext[MainState]
    ) -> Union[
        SearchSubgraphNode, 
        CodingSubgraphNode, 
        OperationsSubgraphNode, 
        ParallelExecution,
        ResultAggregation
    ]:
        # Get the classifier agent from registry
        classifier_agent = agent_registry.get_agent("query_classifier")
        
        # Classify the query
        result = await classifier_agent.run(
            f"Classify query: {ctx.state.query}\n\nRespond with exactly one of: SEARCH, CODING, OPERATIONS, PARALLEL, or SIMPLE",
            message_history=ctx.state.agent_messages.get("classification", [])
        )
        
        # Store the classification result in state
        ctx.state.agent_messages["classification"] = result.new_messages()
        
        # Route to the appropriate node based on classification
        classification = result.output.strip().upper()
        if "SEARCH" in classification:
            return SearchSubgraphNode()
        elif "CODING" in classification:
            return CodingSubgraphNode()
        elif "OPERATIONS" in classification:
            return OperationsSubgraphNode()
        elif "PARALLEL" in classification:
            return ParallelExecution()
        else:
            # Simple queries that don't need specialized processing
            return ResultAggregation()


@dataclass
class SearchSubgraphNode(BaseNode[MainState]):
    """Executes the search sub-graph."""
    
    async def run(self, ctx: GraphRunContext[MainState]) -> ResultAggregation:
        # Create a search-specific state
        search_state = SearchState(
            query=ctx.state.query,
            context=ctx.state.context
        )
        
        # Run the search sub-graph with its start node
        result = await search_graph.run(QueryFormulation(), state=search_state)
        
        # Update the main state with search results
        ctx.state.search_results = result.output
        
        # Continue to aggregation
        return ResultAggregation()


@dataclass
class CodingSubgraphNode(BaseNode[MainState]):
    """Executes the coding sub-graph."""
    
    async def run(self, ctx: GraphRunContext[MainState]) -> ResultAggregation:
        # Create a coding-specific state
        coding_state = CodingState(
            query=ctx.state.query,
            context=ctx.state.context
        )
        
        # Run the coding sub-graph with its start node
        result = await coding_graph.run(CodeTaskAnalysis(), state=coding_state)
        
        # Update the main state with coding results
        ctx.state.code_results = result.output
        
        # Continue to aggregation
        return ResultAggregation()


@dataclass
class OperationsSubgraphNode(BaseNode[MainState]):
    """Executes the operations sub-graph."""
    
    async def run(self, ctx: GraphRunContext[MainState]) -> ResultAggregation:
        # Create an operations-specific state
        operations_state = OperationsState(
            query=ctx.state.query,
            context=ctx.state.context
        )
        
        # Run the operations sub-graph with its start node
        result = await operations_graph.run(BashCommandFormulation(), state=operations_state)
        
        # Update the main state with operations results
        ctx.state.bash_results = result.output
        
        # Continue to aggregation
        return ResultAggregation()


@dataclass
class ParallelExecution(BaseNode[MainState]):
    """Executes multiple sub-graphs in parallel."""
    
    async def run(self, ctx: GraphRunContext[MainState]) -> ResultAggregation:
        # Create states for each domain
        search_state = SearchState(query=ctx.state.query, context=ctx.state.context)
        coding_state = CodingState(query=ctx.state.query, context=ctx.state.context)
        operations_state = OperationsState(query=ctx.state.query, context=ctx.state.context)
        
        # Run sub-graphs in parallel with their start nodes
        search_task = search_graph.run(QueryFormulation(), state=search_state)
        coding_task = coding_graph.run(CodeTaskAnalysis(), state=coding_state)
        operations_task = operations_graph.run(BashCommandFormulation(), state=operations_state)
        
        # Wait for all tasks to complete
        search_result, coding_result, operations_result = await asyncio.gather(
            search_task, coding_task, operations_task
        )
        
        # Update the main state with all results
        ctx.state.search_results = search_result.output
        ctx.state.code_results = coding_result.output
        ctx.state.bash_results = operations_result.output
        
        # Continue to aggregation
        return ResultAggregation()


@dataclass
class ResultAggregation(BaseNode[MainState, None, AggregatedResponse]):
    """Aggregates results from all sub-graphs and produces a final response."""
    
    async def run(self, ctx: GraphRunContext[MainState]) -> End[AggregatedResponse]:
        # Get the synthesizer agent from registry
        synthesizer_agent = agent_registry.get_agent("result_synthesizer")
        
        # Create a dictionary with all available results
        results_dict = {
            "query": ctx.state.query,
            "search_results": ctx.state.search_results,
            "calculation_results": ctx.state.calculation_results,
            "code_results": ctx.state.code_results,
            "bash_results": ctx.state.bash_results
        }
        
        # Run the synthesizer agent to aggregate results
        result = await synthesizer_agent.run(
            f"Synthesize results into a coherent response:\n{format_as_xml(results_dict)}",
            message_history=ctx.state.agent_messages.get("aggregation", [])
        )
        
        # Update the main state with synthesized results
        ctx.state.agent_messages["aggregation"] = result.new_messages()
        
        # If result.output has a final_answer attribute, use it
        if hasattr(result.output, "final_answer"):
            ctx.state.final_response = result.output.final_answer
        
        # End the graph with the aggregated response
        return End(cast(AggregatedResponse, result.output))


# Create the root graph
root_graph = Graph(
    nodes=[
        QueryClassification,
        SearchSubgraphNode,
        CodingSubgraphNode,
        OperationsSubgraphNode,
        ParallelExecution,
        ResultAggregation
    ],
    state_type=MainState
)