import argparse
import asyncio
import logging
import os
from pathlib import Path

from app.core.observability import observability
from app.evals.agent_eval import evaluate_agent_system, generate_eval_dataset


async def main() -> None:
    """CLI entry point for evaluation tools."""
    parser = argparse.ArgumentParser(description="Agent System Evaluation Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate dataset command
    gen_parser = subparsers.add_parser("generate", help="Generate evaluation dataset")
    gen_parser.add_argument(
        "--count", type=int, default=10, 
        help="Number of test cases to generate"
    )
    gen_parser.add_argument(
        "--output", type=str, default="eval_dataset.yaml",
        help="Output file path"
    )
    
    # Run evaluation command
    eval_parser = subparsers.add_parser("run", help="Run evaluation")
    eval_parser.add_argument(
        "--dataset", type=str, default="eval_dataset.yaml",
        help="Dataset file path"
    )
    eval_parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Output results file path"
    )
    eval_parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Maximum concurrent evaluations"
    )
    
    # Common arguments
    for subparser in [gen_parser, eval_parser]:
        subparser.add_argument(
            "--debug", action="store_true",
            help="Enable debug logging"
        )
        subparser.add_argument(
            "--no-logfire", action="store_true",
            help="Disable sending data to Logfire"
        )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize observability
    observability.initialize(
        project_name="agent-system-evals",
        environment="evaluation",
        send_to_logfire=not getattr(args, "no_logfire", False),
        capture_http=args.debug
    )
    
    if args.command == "generate":
        # Generate dataset
        dataset = await generate_eval_dataset(args.count)
        output_path = Path(args.output)
        dataset.to_file(output_path)
        print(f"Generated dataset with {len(dataset.cases)} test cases")
        print(f"Saved to {output_path.absolute()}")
        
    elif args.command == "run":
        # Run evaluation
        await evaluate_agent_system()
        print("Evaluation complete")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())