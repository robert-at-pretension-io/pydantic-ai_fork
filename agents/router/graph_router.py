"""
Graph-based implementation of the router agent.

This module implements a router agent using pydantic-graph for state management and flow control.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from .router_agent import (
    AggregatedResponse,
    CalculationResult,
    RouterDeps,
    SearchResult,
    TextAnalysisResult,
    calculation_agent,
    search_agent,
    text_analysis_agent,
)


@dataclass
class RouterState:
    """State for the router graph."""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    text_analysis_results: Optional[TextAnalysisResult] = None
    history: List[ModelMessage] = field(default_factory=list)
    agent_messages: Dict[str, List[ModelMessage]] = field(default_factory=dict)
    
    # Timing and monitoring
    start_time: float = field(default_factory=time.time)
    node_timings: Dict[str, float] = field(default_factory=dict)
    executed_nodes: List[str] = field(default_factory=list)
    
    # Track node entry and exit for timing
    def record_node_entry(self, node_name: str) -> None:
        """Record when processing enters a node."""
        self.node_timings[f"{node_name}_start"] = time.time()
        
    def record_node_exit(self, node_name: str) -> None:
        """Record when processing exits a node."""
        if f"{node_name}_start" in self.node_timings:
            elapsed = time.time() - self.node_timings[f"{node_name}_start"]
            self.node_timings[f"{node_name}_elapsed"] = elapsed
            self.executed_nodes.append(node_name)
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since state initialization."""
        return time.time() - self.start_time
    

@dataclass
class QueryClassification(BaseNode[RouterState]):
    """Analyzes the query to determine which agents to invoke."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> Union["SearchNode", "CalculationNode", "TextAnalysisNode", "AggregateResults"]:
        analysis_agent = Agent(
            'openai:gpt-4o',
            output_type=str,
            system_prompt="Analyze this query and respond with exactly one of: SEARCH, CALCULATION, TEXT_ANALYSIS, or MULTIPLE"
        )
        
        result = await analysis_agent.run(
            f"Analyze this query: {ctx.state.query}\n\nRespond with the primary category: SEARCH, CALCULATION, TEXT_ANALYSIS, or MULTIPLE",
            message_history=ctx.state.agent_messages.get("analysis", [])
        )
        
        ctx.state.agent_messages["analysis"] = result.all_messages()
        
        response = result.output.strip().upper()
        
        if "SEARCH" in response:
            return SearchNode()
        elif "CALCULATION" in response:
            return CalculationNode()
        elif "TEXT_ANALYSIS" in response:
            return TextAnalysisNode()
        else:  # MULTIPLE or any other response
            return MultipleQueriesNode()


@dataclass
class SearchNode(BaseNode[RouterState]):
    """Performs a search using the search agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "AggregateResults":
        # Record entry into this node
        ctx.state.record_node_entry("SearchNode")
        
        # Add usage tracking to context if not present
        ctx.state.context.setdefault("usage", {"tokens": 0, "cost": 0})
        
        # Display progress if console is available
        console = ctx.state.context.get("console")
        if console:
            console.print(f"[dim]➡️ search: {ctx.state.query}[/dim]")
        
        # Execute the search agent
        result = await search_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("search", [])
        )
        
        # Update the state with results
        ctx.state.agent_messages["search"] = result.all_messages()
        ctx.state.search_results = result.output
        
        # Track token usage if available
        if hasattr(result, "usage") and hasattr(result.usage, "total_tokens"):
            ctx.state.context["usage"]["tokens"] = ctx.state.context["usage"].get("tokens", 0) + result.usage.total_tokens
            # Approximate cost calculation - could be refined based on model
            ctx.state.context["usage"]["cost"] = ctx.state.context["usage"].get("cost", 0) + (result.usage.total_tokens * 0.000004)
        
        # Record exit from this node
        ctx.state.record_node_exit("SearchNode")
        
        # Display completion if console is available
        if console:
            elapsed = ctx.state.node_timings.get("SearchNode_elapsed", 0)
            console.print(f"[green]✓ search done in {elapsed:.2f}s ({len(result.output.results)} results)[/green]")
        
        return AggregateResults()


@dataclass
class CalculationNode(BaseNode[RouterState]):
    """Performs calculations using the calculation agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "AggregateResults":
        result = await calculation_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("calculation", [])
        )
        
        ctx.state.agent_messages["calculation"] = result.all_messages()
        ctx.state.calculation_results = result.output
        
        return AggregateResults()


@dataclass
class TextAnalysisNode(BaseNode[RouterState]):
    """Analyzes text using the text analysis agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "AggregateResults":
        result = await text_analysis_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("text_analysis", [])
        )
        
        ctx.state.agent_messages["text_analysis"] = result.all_messages()
        ctx.state.text_analysis_results = result.output
        
        return AggregateResults()


@dataclass
class MultipleQueriesNode(BaseNode[RouterState]):
    """Handles queries that require multiple agents."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "ParallelExecutionNode":
        # Determine which agents to invoke based on query analysis
        analysis_agent = Agent(
            'openai:gpt-4o',
            output_type=str,
            system_prompt="Analyze this query and respond with a comma-separated list of needed agents: SEARCH, CALCULATION, TEXT_ANALYSIS"
        )
        
        result = await analysis_agent.run(
            f"Analyze this query: {ctx.state.query}\n\nRespond with a comma-separated list of needed agents: SEARCH, CALCULATION, TEXT_ANALYSIS",
            message_history=ctx.state.agent_messages.get("multi_analysis", [])
        )
        
        ctx.state.agent_messages["multi_analysis"] = result.all_messages()
        
        return ParallelExecutionNode(agents=result.output.strip().upper().split(","))


@dataclass
class ParallelExecutionNode(BaseNode[RouterState]):
    """Executes multiple agents in parallel."""
    agents: List[str]
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "AggregateResults":
        import asyncio
        
        tasks = []
        
        if "SEARCH" in self.agents:
            tasks.append(self._run_search(ctx))
        
        if "CALCULATION" in self.agents:
            tasks.append(self._run_calculation(ctx))
        
        if "TEXT_ANALYSIS" in self.agents:
            tasks.append(self._run_text_analysis(ctx))
        
        # Run all tasks in parallel
        await asyncio.gather(*tasks)
        
        return AggregateResults()
    
    async def _run_search(self, ctx: GraphRunContext[RouterState]):
        result = await search_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("search", [])
        )
        ctx.state.agent_messages["search"] = result.all_messages()
        ctx.state.search_results = result.output
    
    async def _run_calculation(self, ctx: GraphRunContext[RouterState]):
        result = await calculation_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("calculation", [])
        )
        ctx.state.agent_messages["calculation"] = result.all_messages()
        ctx.state.calculation_results = result.output
    
    async def _run_text_analysis(self, ctx: GraphRunContext[RouterState]):
        result = await text_analysis_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("text_analysis", [])
        )
        ctx.state.agent_messages["text_analysis"] = result.all_messages()
        ctx.state.text_analysis_results = result.output


@dataclass
class AggregateResults(BaseNode[RouterState, None, AggregatedResponse]):
    """Aggregates results from all agents and produces a final response."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[AggregatedResponse]:
        aggregation_agent = Agent(
            'openai:gpt-4o',
            output_type=AggregatedResponse,
            system_prompt="Synthesize outputs from multiple agents into one coherent response."
        )
        
        # Build an XML representation of all available results
        state_dict = {
            "query": ctx.state.query,
            "search_results": ctx.state.search_results,
            "calculation_results": ctx.state.calculation_results,
            "text_analysis": ctx.state.text_analysis_results
        }
        
        result = await aggregation_agent.run(
            f"Synthesize these outputs into one coherent response:\n{format_as_xml(state_dict)}",
            message_history=ctx.state.agent_messages.get("aggregation", [])
        )
        
        ctx.state.agent_messages["aggregation"] = result.all_messages()
        
        # Combine all agent messages for history
        all_messages = []
        for messages in ctx.state.agent_messages.values():
            all_messages.extend(messages)
        
        # Update state with message history
        ctx.state.history = all_messages
        
        return End(result.output)


# Create the router graph
router_graph = Graph(
    nodes=[
        QueryClassification, 
        SearchNode, 
        CalculationNode, 
        TextAnalysisNode, 
        MultipleQueriesNode,
        ParallelExecutionNode,
        AggregateResults
    ],
    state_type=RouterState
)


async def process_query_with_graph(
    query: str,
    context: Dict = None,
    message_history: List[ModelMessage] = None,
    console = None,
    verbose: bool = False
) -> Tuple[AggregatedResponse, List[ModelMessage]]:
    """
    Process a query using the graph-based router.
    
    Args:
        query: The user's query
        context: Optional context information
        message_history: Optional message history from previous interactions
        console: Optional rich console for displaying progress
        verbose: Whether to show verbose output
        
    Returns:
        A tuple containing the aggregated response and the updated message history
    """
    # Initialize context with usage tracking and console
    context_dict = context or {}
    context_dict.setdefault("usage", {"tokens": 0, "cost": 0})
    if console:
        context_dict["console"] = console
    
    # Initialize state with provided query
    state = RouterState(
        query=query,
        context=context_dict,
    )
    
    # Initialize agent messages if we have previous message history
    if message_history:
        # Add message history to the agent messages dictionary
        state.agent_messages = {"history": message_history}
        state.history = message_history
    
    if console and verbose:
        console.print(f"[blue]Processing query using graph-based router: {query}[/blue]")
    
    # Run the graph starting with query classification
    result = await router_graph.run(QueryClassification(), state=state)
    
    # Display timing and usage information if console is available
    if console:
        # Get usage information
        usage = state.context.get("usage", {})
        tokens = usage.get("tokens", 0)
        cost = usage.get("cost", 0)
        
        # Get execution path
        path = " → ".join(state.executed_nodes) if state.executed_nodes else "router"
        
        # Display timing and usage
        elapsed = state.get_elapsed_time()
        console.print(f"[dim]⏱️ Graph execution finished in {elapsed:.2f}s via {path}[/dim]")
        console.print(f"[dim]({tokens/1000:.1f}k tokens, ${cost:.4f})[/dim]")
    
    # Return the output and updated message history
    return result.output, state.history


# Visualization function for the graph
def visualize_graph() -> str:
    """Generate Mermaid code to visualize the router graph."""
    return router_graph.mermaid_code(start_node=QueryClassification)