"""Router agent for dispatching requests to specialized agents."""

from .router_agent import (
    RouterDeps,
    SearchResult,
    CalculationResult,
    TextAnalysisResult,
    AggregatedResponse,
    router_agent,
    process_query,
)

# Import the clai-compatible agent
from .clai_compatible import cli_router_agent, agent

# Import the graph-based router
from .graph_router import (
    router_graph,
    process_query_with_graph,
    RouterState,
    visualize_graph,
)

# Import visualization tools
from .visualizer import generate_html, open_in_browser, save_to_file

__all__ = [
    "RouterDeps",
    "SearchResult",
    "CalculationResult",
    "TextAnalysisResult", 
    "AggregatedResponse",
    "router_agent",
    "process_query",
    "cli_router_agent",
    "agent",
    # Graph-based router exports
    "router_graph",
    "process_query_with_graph",
    "RouterState",
    "visualize_graph",
    # Visualization tools
    "generate_html",
    "open_in_browser",
    "save_to_file",
]