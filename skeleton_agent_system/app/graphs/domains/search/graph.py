from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, cast, TYPE_CHECKING, Sequence

from pydantic_ai import format_as_xml, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.core.base_agent import agent_registry
from app.core.context import context_builder, ContextBuildRequest
from app.models.outputs import SearchResult
from app.models.states import SearchState


@dataclass
class QueryFormulation(BaseNode[SearchState]):
    """Formulates an effective search query from the user's input."""
    
    async def run(self, ctx: GraphRunContext[SearchState]) -> WebSearch:
        # Get search agent from registry
        search_agent = agent_registry.get_agent("search")
        
        # Build optimized context
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="search",
                task=f"Formulate an effective search query: {ctx.state.query}",
                deps=ctx.state.context,
                max_tokens=1024
            )
        )
        
        # Run the search agent to formulate the query
        # Convert GraphRunContext to RunContext
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                            ctx.state.agent_messages.get("formulation", []))
        
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await search_agent.run(  # type: ignore
            context_pkg.user_prompt,
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the formulated query in state
        ctx.state.agent_messages["formulation"] = result.new_messages()
        ctx.state.formulated_query = result.output
        
        # Continue to the web search
        return WebSearch()


@dataclass
class WebSearch(BaseNode[SearchState]):
    """Executes a web search with the formulated query."""
    
    async def run(self, ctx: GraphRunContext[SearchState]) -> ResultFiltering:
        # Here, you'd typically execute a web search using a tool
        # For this skeleton, we'll simulate results
        
        query = ctx.state.formulated_query or ctx.state.query
        
        # Simulated search results
        ctx.state.search_results = [
            {
                "title": f"Result for {query} #1",
                "snippet": f"This is the first search result for {query}.",
                "url": "https://example.com/1"
            },
            {
                "title": f"Result for {query} #2",
                "snippet": f"This is the second search result for {query}.",
                "url": "https://example.com/2"
            },
            {
                "title": f"Result for {query} #3",
                "snippet": f"This is the third search result for {query}.",
                "url": "https://example.com/3"
            }
        ]
        
        # Continue to result filtering
        return ResultFiltering()


@dataclass
class ResultFiltering(BaseNode[SearchState, None, SearchResult]):
    """Filters and processes search results into a structured response."""
    
    async def run(self, ctx: GraphRunContext[SearchState]) -> End[SearchResult]:
        # Get search agent from registry
        search_agent = agent_registry.get_agent("search")
        
        # Format the search results as XML
        formatted_results = format_as_xml(ctx.state.search_results)
        
        # Build context for filtering
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="search",
                task=f"Filter search results for query: {ctx.state.query}",
                deps={"results": ctx.state.search_results, **ctx.state.context},
                max_tokens=1024
            )
        )
        
        # Run the search agent to filter results
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                            ctx.state.agent_messages.get("filtering", []))
        
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await search_agent.run(  # type: ignore
            f"Filter and process these search results:\n{formatted_results}",
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the filtering results in state
        ctx.state.agent_messages["filtering"] = result.new_messages()
        ctx.state.filtered_results = cast(SearchResult, result.output)
        
        # End the graph with the filtered search results
        return End(ctx.state.filtered_results)


# Create the search graph
search_graph = Graph(
    nodes=[QueryFormulation, WebSearch, ResultFiltering],
    state_type=SearchState
)