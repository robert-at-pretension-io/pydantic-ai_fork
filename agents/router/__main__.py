"""
Router Agent CLI application.

This module allows running the router agent from the command line.
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import print as rich_print

from .router_agent import process_query, router_agent, AggregatedResponse
from pydantic_ai.messages import ModelMessage


HISTORY_DIR = Path.home() / ".pydantic-ai" / "router-agent"


def format_response(response: AggregatedResponse) -> str:
    """Format the agent response as markdown for pretty display."""
    sections = []
    
    # Final answer
    sections.append(f"## Final Answer\n\n{response.final_answer}")
    
    # Search results
    if response.search_results:
        results = "\n".join([f"- {result}" for result in response.search_results.results])
        sections.append(f"## Search Results\n\nQuery: {response.search_results.query}\n\n{results}")
    
    # Calculation results
    if response.calculation_results:
        steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(response.calculation_results.steps)])
        sections.append(
            f"## Calculation Results\n\n"
            f"Input: {response.calculation_results.input}\n\n"
            f"Result: {response.calculation_results.result}\n\n"
            f"Steps:\n{steps}"
        )
    
    # Text analysis results
    if response.text_analysis:
        key_points = "\n".join([f"- {point}" for point in response.text_analysis.key_points])
        sections.append(
            f"## Text Analysis Results\n\n"
            f"Sentiment: {response.text_analysis.sentiment}\n\n"
            f"Summary: {response.text_analysis.summary}\n\n"
            f"Key Points:\n{key_points}"
        )
    
    return "\n\n".join(sections)


async def run_router_cli(
    query: str, 
    context: Optional[Dict[str, Any]] = None, 
    message_history: Optional[List[ModelMessage]] = None,
    theme: str = "dark"
) -> Tuple[AggregatedResponse, List[ModelMessage]]:
    """
    Run the router agent with the given query and context, displaying results with rich formatting.
    
    Args:
        query: The user query to process
        context: Optional context information
        message_history: Optional message history from previous interactions
        theme: Theme for code syntax highlighting
    
    Returns:
        A tuple containing the response and updated message history
    """
    console = Console()
    
    with console.status("[bold green]Routing query to specialized agents..."):
        response, updated_history = await process_query(query, context, message_history)
    
    markdown_content = format_response(response)
    console.print(Markdown(markdown_content, code_theme=theme))
    return response, updated_history


def save_chat_history(query: str, response: AggregatedResponse) -> None:
    """Save the chat interaction to history file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_file = HISTORY_DIR / "history.jsonl"
    
    # Create record
    record = {
        "timestamp": str(asyncio.get_event_loop().time()),
        "query": query,
        "response": response.model_dump(),
    }
    
    # Append to history file
    with open(history_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def view_history(limit: int = 5, theme: str = "dark") -> None:
    """View recent chat history."""
    console = Console()
    history_file = HISTORY_DIR / "history.jsonl"
    
    if not history_file.exists():
        console.print("[yellow]No chat history found.[/yellow]")
        return
        
    # Read history entries
    entries = []
    with open(history_file, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    # Display most recent entries first
    entries = sorted(entries, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    if not entries:
        console.print("[yellow]No chat history found.[/yellow]")
        return
        
    for i, entry in enumerate(entries, 1):
        query = entry["query"]
        response = AggregatedResponse.model_validate(entry["response"])
        
        console.print(Panel(f"[bold blue]Query {i}:[/bold blue] {query}"))
        console.print(Markdown(format_response(response), code_theme=theme))
        console.print("---")


async def interactive_mode(theme: str = "dark") -> None:
    """Run the router agent in interactive mode."""
    console = Console()
    console.print("[bold green]Router Agent Interactive Mode[/bold green]")
    console.print("Type '/exit' to quit, '/help' for help, or '/history' to view chat history.")
    
    # Initialize conversation history
    message_history: List[ModelMessage] = []
    
    while True:
        try:
            query = console.input("\n[bold blue]Router >[/bold blue] ")
            query = query.strip()
            
            if not query:
                continue
                
            if query.lower() == "/exit":
                break
            elif query.lower() == "/help":
                console.print(Markdown("""
                # Router Agent Help
                
                ## Commands
                - `/exit`: Exit interactive mode
                - `/help`: Show this help message
                - `/history [n]`: View last n chat history entries (default: 5)
                - `/clear`: Clear the screen
                - `/version`: Show version information
                - `/reset`: Reset conversation history
                
                ## Examples
                - Simple query: `What is the capital of France?`
                - Mathematical query: `Calculate 15% of 250`
                - Text analysis: `Analyze this text: The economy is showing signs of recovery.`
                - Follow-up query: `What's its population?` (refers to previous context)
                """, code_theme=theme))
            elif query.lower().startswith("/history"):
                parts = query.split()
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                view_history(limit, theme)
            elif query.lower() == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
            elif query.lower() == "/version":
                console.print("[green]Router Agent v0.1.0[/green]")
            elif query.lower() == "/reset":
                message_history = []
                console.print("[yellow]Conversation history has been reset.[/yellow]")
            else:
                # Process the query with the router agent
                with console.status("[bold green]Routing query to specialized agents..."):
                    response, message_history = await process_query(query, message_history=message_history)
                
                # Save to history file
                save_chat_history(query, response)
                
                # Display formatted response
                markdown_content = format_response(response)
                console.print(Markdown(markdown_content, code_theme=theme))
                
                # Indicate that conversation has history
                if len(message_history) > 2:  # More than just this exchange
                    console.print("[dim](Conversation context is being maintained)[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type '/exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def main():
    """Parse command line arguments and run the router agent."""
    parser = argparse.ArgumentParser(
        description="Router Agent CLI - Route queries to specialized AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Special commands in interactive mode:
  /exit            Exit the application
  /help            Show help information
  /history [n]     View last n chat history entries
  /clear           Clear the screen
  /version         Show version information
""",
    )
    parser.add_argument("query", nargs="*", help="The query to process")
    parser.add_argument("--context", type=str, help="Optional context as a JSON string")
    parser.add_argument(
        "-t", "--theme", 
        choices=["dark", "light", "monokai", "github-dark", "gruvbox-dark"], 
        default="dark",
        help="Theme for syntax highlighting (default: dark)"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Start in interactive mode")
    parser.add_argument("--history", "-H", action="store_true", help="View chat history")
    parser.add_argument("--history-limit", type=int, default=5, help="Number of history entries to show (default: 5)")
    parser.add_argument("--version", "-v", action="store_true", help="Show version information")
    
    args = parser.parse_args()
    
    if args.version:
        rich_print("[green]Router Agent v0.1.0[/green]")
        return
        
    if args.history:
        view_history(args.history_limit, args.theme)
        return
    
    # Parse context if provided
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            rich_print("[bold red]Error:[/bold red] Context must be a valid JSON string")
            return 1
    
    # Run in appropriate mode
    if args.interactive or not args.query:
        asyncio.run(interactive_mode(args.theme))
    else:
        # Join query words into a single string
        query = " ".join(args.query)
        response, _ = asyncio.run(run_router_cli(query, context, None, args.theme))
        save_chat_history(query, response)


def run_direct_cli():
    """Launch the CLI directly from the router_agent module."""
    router_agent.to_cli_sync()


if __name__ == "__main__":
    main()