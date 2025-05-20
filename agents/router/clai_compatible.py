"""
CLI-compatible version of the router agent that works with clai.

This module provides a version of the router agent that works with the clai CLI tool.
"""

from pydantic_ai import Agent, RunContext

from .router_agent import process_query, AggregatedResponse, RouterDeps

# Create a CLI-compatible router agent that returns string output
cli_router_agent = Agent(
    'openai:gpt-4o',
    deps_type=RouterDeps,
    output_type=str,  # Use string output for CLI compatibility
    system_prompt="""
    You are a routing agent responsible for:
    1. Analyzing user queries to determine which specialized agents to invoke
    2. Sending appropriate sub-tasks to specialized agents via tools
    3. Aggregating results from all agents
    4. Synthesizing a final response
    
    Only invoke agents that are necessary for the query. Multiple agents can be used for complex queries.
    """
)


# Cache for conversation history
_message_history_cache = {}

@cli_router_agent.tool
async def route_query(ctx: RunContext[RouterDeps], query: str) -> str:
    """
    Route the query to specialized agents and return their aggregated response.
    This tool wraps the main router agent and formats its response as plain text.
    """
    # Get a unique identifier for this conversation (using the usage context)
    if ctx.usage:
        # Use the usage object as a key to track conversation context
        conversation_id = id(ctx.usage)
        
        # Retrieve existing history for this conversation if available
        current_history = _message_history_cache.get(conversation_id, [])
    else:
        # Fallback when usage is not available
        conversation_id = None
        current_history = []
    
    deps = ctx.deps or RouterDeps()
    response, updated_history = await process_query(query, deps.context, current_history)
    
    # Store updated history if we have a conversation ID
    if conversation_id is not None:
        _message_history_cache[conversation_id] = updated_history
    
    # Format the response as a string
    output_parts = []
    
    # Add final answer
    output_parts.append(f"# {response.final_answer}")
    
    # Add search results if available
    if response.search_results:
        output_parts.append("\n## Search Results")
        output_parts.append(f"**Query:** {response.search_results.query}")
        for result in response.search_results.results:
            output_parts.append(f"- {result}")
    
    # Add calculation results if available
    if response.calculation_results:
        output_parts.append("\n## Calculation Results")
        output_parts.append(f"**Input:** {response.calculation_results.input}")
        output_parts.append(f"**Result:** {response.calculation_results.result}")
        output_parts.append("**Steps:**")
        for i, step in enumerate(response.calculation_results.steps, 1):
            output_parts.append(f"{i}. {step}")
    
    # Add text analysis results if available
    if response.text_analysis:
        output_parts.append("\n## Text Analysis Results")
        output_parts.append(f"**Sentiment:** {response.text_analysis.sentiment}")
        output_parts.append(f"**Summary:** {response.text_analysis.summary}")
        output_parts.append("**Key Points:**")
        for point in response.text_analysis.key_points:
            output_parts.append(f"- {point}")
    
    # Indicate if conversation context is being maintained
    if len(updated_history) > 2:  # More than just this exchange
        output_parts.append("\n\n_Conversation context is being maintained_")
    
    return "\n".join(output_parts)


# Create a runnable agent
agent = cli_router_agent