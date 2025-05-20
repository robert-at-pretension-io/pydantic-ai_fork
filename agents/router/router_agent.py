"""
Router agent implementation that dispatches requests to specialized agents.

This module implements a router agent that can analyze user queries,
delegate to specialized agents, and aggregate their results.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import Usage
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage


@dataclass
class RouterDeps:
    """Dependencies for the router agent."""
    context: Dict[str, Any] = field(default_factory=dict)
    console: Any = None  # Will be a rich.console.Console instance


class SearchResult(BaseModel):
    """Result from the search agent."""
    query: str
    results: List[str]


class CalculationResult(BaseModel):
    """Result from the calculation agent."""
    input: str
    result: float
    steps: List[str]


class TextAnalysisResult(BaseModel):
    """Result from the text analysis agent."""
    sentiment: str
    summary: str
    key_points: List[str]


class AggregatedResponse(BaseModel):
    """Aggregated response from all specialized agents."""
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    text_analysis: Optional[TextAnalysisResult] = None
    final_answer: str = Field(description="Synthesized response incorporating all agent outputs")


# Define model names with fallbacks
SEARCH_MODEL = os.environ.get("ROUTER_SEARCH_MODEL", "openai:gpt-4o")
CALCULATION_MODEL = os.environ.get("ROUTER_CALCULATION_MODEL", "openai:gpt-4o")  # Fallback to OpenAI
TEXT_ANALYSIS_MODEL = os.environ.get("ROUTER_TEXT_ANALYSIS_MODEL", "openai:gpt-4o")  # Fallback to OpenAI

# Create specialized agents with fallbacks
# Create search agent
search_agent = Agent(
    SEARCH_MODEL,
    output_type=SearchResult,
    system_prompt="You are a search specialist. Extract relevant information from context."
)

# Create calculation agent
calculation_agent = Agent(
    CALCULATION_MODEL,
    output_type=CalculationResult,
    system_prompt="You are a calculation specialist. Solve mathematical problems step by step."
)

# Create text analysis agent
text_analysis_agent = Agent(
    TEXT_ANALYSIS_MODEL,
    output_type=TextAnalysisResult,
    system_prompt="You are a text analysis specialist. Analyze text for sentiment, summary, and key points."
)

# Create the router agent
router_agent = Agent(
    'openai:gpt-4o',
    deps_type=RouterDeps,
    output_type=AggregatedResponse,
    system_prompt="""
    You are a routing agent responsible for:
    1. Analyzing user queries to determine which specialized agents to invoke
    2. Sending appropriate sub-tasks to specialized agents via tools
    3. Aggregating results from all agents
    4. Synthesizing a final response
    
    Only invoke agents that are necessary for the query. Multiple agents can be used for complex queries.
    """
)


@router_agent.tool
async def search_information(ctx: RunContext[RouterDeps], query: str) -> SearchResult:
    """
    Use the search agent to find information related to the query.
    Only use this when factual information needs to be gathered.
    """
    import time
    start_time = time.time()
    
    if ctx.deps.console:
        ctx.deps.console.print(f"[dim]➡️ search: {query}[/dim]")
    
    result = await search_agent.run(query, usage=ctx.usage)
    
    elapsed = time.time() - start_time
    if ctx.deps.console:
        ctx.deps.console.print(f"[green]✓ search done in {elapsed:.2f}s ({len(result.output.results)} results)[/green]")
    
    # Track token usage in context
    if ctx.usage and hasattr(ctx.usage, "total_tokens"):
        ctx.deps.context.setdefault("usage", {"tokens": 0, "cost": 0})
        ctx.deps.context["usage"]["tokens"] = ctx.deps.context["usage"].get("tokens", 0) + ctx.usage.total_tokens
        # Approximate cost calculation - could be refined based on model
        ctx.deps.context["usage"]["cost"] = ctx.deps.context["usage"].get("cost", 0) + (ctx.usage.total_tokens * 0.000004)
    
    return result.output


@router_agent.tool
async def perform_calculation(ctx: RunContext[RouterDeps], problem: str) -> CalculationResult:
    """
    Use the calculation agent to solve mathematical problems.
    Only use this for queries that involve numerical calculations.
    """
    import time
    start_time = time.time()
    
    if ctx.deps.console:
        ctx.deps.console.print(f"[dim]➡️ calculation: {problem}[/dim]")
    
    result = await calculation_agent.run(problem, usage=ctx.usage)
    
    elapsed = time.time() - start_time
    if ctx.deps.console:
        ctx.deps.console.print(f"[green]✓ calculation done in {elapsed:.2f}s (result: {result.output.result})[/green]")
    
    # Track token usage in context
    if ctx.usage and hasattr(ctx.usage, "total_tokens"):
        ctx.deps.context.setdefault("usage", {"tokens": 0, "cost": 0})
        ctx.deps.context["usage"]["tokens"] = ctx.deps.context["usage"].get("tokens", 0) + ctx.usage.total_tokens
        # Approximate cost calculation - could be refined based on model
        ctx.deps.context["usage"]["cost"] = ctx.deps.context["usage"].get("cost", 0) + (ctx.usage.total_tokens * 0.000004)
    
    return result.output


@router_agent.tool
async def analyze_text(ctx: RunContext[RouterDeps], text: str) -> TextAnalysisResult:
    """
    Use the text analysis agent to perform sentiment analysis, summarization, and extract key points.
    Only use this for queries that require text understanding.
    """
    import time
    start_time = time.time()
    
    if ctx.deps.console:
        ctx.deps.console.print(f"[dim]➡️ text analysis: {text[:50]}{'...' if len(text) > 50 else ''}[/dim]")
    
    result = await text_analysis_agent.run(text, usage=ctx.usage)
    
    elapsed = time.time() - start_time
    if ctx.deps.console:
        ctx.deps.console.print(f"[green]✓ text analysis done in {elapsed:.2f}s (sentiment: {result.output.sentiment})[/green]")
    
    # Track token usage in context
    if ctx.usage and hasattr(ctx.usage, "total_tokens"):
        ctx.deps.context.setdefault("usage", {"tokens": 0, "cost": 0})
        ctx.deps.context["usage"]["tokens"] = ctx.deps.context["usage"].get("tokens", 0) + ctx.usage.total_tokens
        # Approximate cost calculation - could be refined based on model
        ctx.deps.context["usage"]["cost"] = ctx.deps.context["usage"].get("cost", 0) + (ctx.usage.total_tokens * 0.000004)
    
    return result.output


async def process_query(
    query: str, 
    context: Dict = None, 
    message_history: List[ModelMessage] = None,
    console = None,
    verbose: bool = False
) -> Tuple[AggregatedResponse, List[ModelMessage]]:
    """
    Process a user query through the routing agent, which will delegate to specialized agents as needed.
    
    Args:
        query: The user's query
        context: Optional context information
        message_history: Optional message history from previous interactions
        console: Optional rich console for displaying progress
        verbose: Whether to show verbose output
        
    Returns:
        A tuple containing the aggregated response and the updated message history
    """
    import time
    start_time = time.time()
    
    # Initialize context with empty usage tracking
    context_dict = context or {}
    context_dict.setdefault("usage", {"tokens": 0, "cost": 0})
    
    # Create dependencies with console
    deps = RouterDeps(context=context_dict, console=console)
    
    if console and verbose:
        console.print(f"[blue]Processing query: {query}[/blue]")
        
    # Run the router agent
    result = await router_agent.run(query, deps=deps, message_history=message_history)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Display usage information if available
    usage = deps.context.get("usage", {})
    tokens = usage.get("tokens", 0)
    cost = usage.get("cost", 0)
    
    if console:
        # Display result summary based on which agents were used
        domains_used = []
        if result.output.search_results:
            domains_used.append("SEARCH")
            if verbose:
                console.print(f"[dim]Search found {len(result.output.search_results.results)} results[/dim]")
        
        if result.output.calculation_results:
            domains_used.append("CALCULATION")
            if verbose:
                console.print(f"[dim]Calculation result: {result.output.calculation_results.result}[/dim]")
        
        if result.output.text_analysis:
            domains_used.append("TEXT_ANALYSIS")
            if verbose:
                console.print(f"[dim]Text analysis sentiment: {result.output.text_analysis.sentiment}[/dim]")
        
        # Show usage and timing information
        console.print(f"[dim]⏱️ Finished in {elapsed:.2f}s via {' → '.join(domains_used) if domains_used else 'router'}[/dim]")
        console.print(f"[dim]({tokens/1000:.1f}k tokens, ${cost:.4f})[/dim]")
    
    # Return both the response and the updated message history
    return result.output, result.all_messages()


if __name__ == "__main__":
    import asyncio
    import sys
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Router Agent CLI")
    parser.add_argument("query", nargs="?", default="What is the population of Paris, and has it grown in the last decade?", 
                       help="The query to process")
    parser.add_argument("-q", "--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--no-console", action="store_true", help="Disable rich console output")
    args = parser.parse_args()
    
    # Set verbose mode as the default (unless quiet flag is used)
    args.verbose = not args.quiet
    
    # Initialize rich console (unless no-console flag is used)
    console = None if args.no_console else Console(highlight=True)
    
    async def process_and_display(query, message_history=None, context=None):
        """Process a query and display the results."""
        response, messages = await process_query(
            query,
            console=console,
            verbose=args.verbose,
            message_history=message_history,
            context=context
        )
        
        # Display the final aggregated response
        if console:
            console.print()
            console.print(Panel(Markdown(response.final_answer), title="Final Answer", border_style="green"))
            
            # Display detailed results if verbose is enabled
            if args.verbose:
                if response.search_results:
                    console.print(Panel(
                        "\n".join([f"{i}. {result}" for i, result in enumerate(response.search_results.results, 1)]),
                        title=f"Search Results for: {response.search_results.query}",
                        border_style="blue"
                    ))
                        
                if response.calculation_results:
                    console.print(Panel(
                        f"Input: {response.calculation_results.input}\n" +
                        f"Result: {response.calculation_results.result}\n\n" +
                        "Steps:\n" +
                        "\n".join([f"{i}. {step}" for i, step in enumerate(response.calculation_results.steps, 1)]),
                        title="Calculation Results",
                        border_style="cyan"
                    ))
                        
                if response.text_analysis:
                    console.print(Panel(
                        f"Sentiment: {response.text_analysis.sentiment}\n" +
                        f"Summary: {response.text_analysis.summary}\n\n" +
                        "Key Points:\n" +
                        "\n".join([f"{i}. {point}" for i, point in enumerate(response.text_analysis.key_points, 1)]),
                        title="Text Analysis Results",
                        border_style="magenta"
                    ))
        else:
            # Plain text output for when console is disabled
            print("\nFinal Answer:")
            print(response.final_answer)
            
            # Display detailed results if verbose is enabled
            if args.verbose:
                if response.search_results:
                    print("\nSearch Results:")
                    print(f"Query: {response.search_results.query}")
                    for i, result in enumerate(response.search_results.results, 1):
                        print(f"{i}. {result}")
                        
                if response.calculation_results:
                    print("\nCalculation Results:")
                    print(f"Input: {response.calculation_results.input}")
                    print(f"Result: {response.calculation_results.result}")
                    print("Steps:")
                    for i, step in enumerate(response.calculation_results.steps, 1):
                        print(f"{i}. {step}")
                        
                if response.text_analysis:
                    print("\nText Analysis Results:")
                    print(f"Sentiment: {response.text_analysis.sentiment}")
                    print(f"Summary: {response.text_analysis.summary}")
                    print("Key Points:")
                    for i, point in enumerate(response.text_analysis.key_points, 1):
                        print(f"{i}. {point}")
        
        return response, messages
    
    async def interactive_mode():
        """Run the router in interactive mode with conversation memory."""
        import os
        
        # Print welcome message
        if console:
            from rich.prompt import Prompt
            console.print(Panel(
                "[bold]Router Agent Interactive Mode[/bold]\n\n"
                "Type your questions and the router will delegate to specialized agents.\n"
                "Commands:\n"
                "/exit - Exit the session\n"
                "/help - Show this help message\n"
                "/clear - Clear the screen\n"
                "/reset - Reset the conversation history",
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("\nRouter Agent Interactive Mode\n")
            print("Type your questions and the router will delegate to specialized agents.")
            print("Commands:")
            print("/exit - Exit the session")
            print("/help - Show this help message")
            print("/clear - Clear the screen")
            print("/reset - Reset the conversation history")
        
        # Initialize conversation memory
        message_history = []
        context = {"session_id": os.urandom(8).hex()}
        
        while True:
            try:
                # Get user input
                if console:
                    from rich.prompt import Prompt
                    query = Prompt.ask("\n[bold blue]Router[/bold blue]")
                else:
                    query = input("\nRouter > ")
                
                # Handle special commands
                if query.lower() == "/exit":
                    if console:
                        console.print("[yellow]Exiting interactive mode.[/yellow]")
                    else:
                        print("Exiting interactive mode.")
                    break
                elif query.lower() == "/help":
                    if console:
                        console.print(Panel(
                            "Commands:\n"
                            "/exit - Exit the session\n"
                            "/help - Show this help message\n"
                            "/clear - Clear the screen\n"
                            "/reset - Reset the conversation history",
                            title="Help",
                            border_style="green"
                        ))
                    else:
                        print("\nCommands:")
                        print("/exit - Exit the session")
                        print("/help - Show this help message")
                        print("/clear - Clear the screen")
                        print("/reset - Reset the conversation history")
                    continue
                elif query.lower() == "/clear":
                    if console:
                        console.clear()
                    else:
                        # Print a bunch of newlines for "clearing" in regular terminal
                        print("\n" * 50)
                    continue
                elif query.lower() == "/reset":
                    message_history = []
                    context = {"session_id": os.urandom(8).hex()}
                    if console:
                        console.print("[yellow]Conversation history reset.[/yellow]")
                    else:
                        print("Conversation history reset.")
                    continue
                
                # Process the query
                response, message_history = await process_and_display(
                    query, 
                    message_history=message_history,
                    context=context
                )
                
                # Indicate that context is being maintained
                if len(message_history) > 2:
                    if console:
                        console.print("[dim](Conversation context is being maintained)[/dim]")
                    else:
                        print("(Conversation context is being maintained)")
                
            except KeyboardInterrupt:
                if console:
                    console.print("\n[yellow]Exiting interactive mode.[/yellow]")
                else:
                    print("\nExiting interactive mode.")
                break
            except Exception as e:
                if console:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                else:
                    print(f"Error: {str(e)}")
    
    async def main():
        """Main entry point for the CLI."""
        if args.interactive or not args.query:
            # Run in interactive mode
            await interactive_mode()
        else:
            # Process a single query
            await process_and_display(args.query)
    
    # Run the main function
    asyncio.run(main())