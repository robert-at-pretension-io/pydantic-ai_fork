from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import uuid
from typing import Dict, Any, Optional

from pydantic_ai.usage import Usage
from pydantic_graph.persistence.in_mem import FullStatePersistence

from app.graphs.root import root_graph
from app.models.states import MainState
from app.core.telemetry import telemetry
from app.core.observability import observability


async def process_query(
    query: str, 
    context: Optional[Dict[str, Any]] = None, 
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a user query through the root graph.
    
    Args:
        query: The user's query
        context: Optional context information
        run_id: Optional run ID for tracking
        
    Returns:
        The processed result
    """
    # Generate a run ID if not provided
    if not run_id:
        run_id = str(uuid.uuid4())
    
    # Start telemetry and observability tracking
    telemetry.start_run(run_id)
    observability.start_trace(run_id, attributes={"query": query})
    
    try:
        # Create initial state
        state = MainState(
            query=query,
            context=context or {}
        )
        
        # Create persistence
        persistence = FullStatePersistence()
        
        # Track usage
        usage = Usage()
        
        # Run the graph with QueryClassification as the start node
        from app.graphs.root import QueryClassification
        result = await root_graph.run(
            start_node=QueryClassification(),
            state=state,
            persistence=persistence
        )
        
        # Get the result
        output = result.output
        
        # Update telemetry and observability
        telemetry.end_run(run_id, usage=usage)
        
        # Format the result as a dictionary
        result_dict = {
            "run_id": run_id,
            "result": output.model_dump() if hasattr(output, "model_dump") else output,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        }
        
        # End observability tracing
        observability.end_trace(run_id, result=result_dict)
        
        return result_dict
    
    except Exception as e:
        # Record the error
        error_msg = str(e)
        telemetry.record_error(run_id, error_msg)
        observability.record_error(run_id, error_msg, details={"exception": repr(e)})
        telemetry.end_run(run_id)
        
        # Re-raise the exception
        raise


async def main() -> None:
    """Main entry point for the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the agent system.")
    parser.add_argument("query", help="The query to process")
    parser.add_argument("--context", help="JSON context information", default="{}")
    parser.add_argument("--run-id", help="Run ID for tracking", default=None)
    parser.add_argument("--debug", help="Enable debug logging", action="store_true")
    parser.add_argument("--no-logfire", help="Disable Logfire integration", action="store_true")
    parser.add_argument("--env", help="Environment (development, staging, production)", default="development")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize observability
    observability.initialize(
        project_name="skeleton-agent-system",
        environment=args.env,
        send_to_logfire=not args.no_logfire,
        capture_http=args.debug  # Only capture HTTP in debug mode
    )
    
    # Parse context
    try:
        context = json.loads(args.context)
    except json.JSONDecodeError:
        context = {}
    
    # Process the query
    try:
        result = await process_query(args.query, context, args.run_id)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())