# Monitoring and Observability

The skeleton agent system includes comprehensive monitoring and observability features that help you understand what's happening inside the system during execution.

## Overview

The system uses the following components for observability:

1. **Telemetry Manager**: Basic metrics tracking via the `TelemetryManager` class
2. **Logfire Integration**: Advanced tracing and logging via Logfire and OpenTelemetry
3. **Pydantic Evals**: Systematic evaluation of agent performance
4. **Decorators**: Helper functions to add tracing to custom code

## Getting Started with Observability

### Installation

First, install the required dependencies:

```bash
# Install basic dependencies
pip install -e .

# Install observability tools
pip install -e ".[observability]"
```

### Setting Up Logfire

1. Sign up for a [Logfire account](https://logfire.pydantic.dev/)
2. Authenticate your local environment:

```bash
py-cli logfire auth
```

3. Configure a project:

```bash
py-cli logfire projects new my-agent-system
```

### CLI Arguments

The main application supports several CLI arguments for observability:

```bash
# Run with Logfire enabled (dev environment)
python -m app "What is the capital of France?" --env development

# Run with debug logging and HTTP captures
python -m app "Write a Python function to calculate Fibonacci numbers" --debug

# Run without Logfire
python -m app "How do I check disk usage on Linux?" --no-logfire
```

## Running Evaluations

The system includes a framework for evaluating agent performance using Pydantic Evals:

```bash
# Generate a new evaluation dataset
python -m app.evals.cli generate --count 10 --output eval_dataset.yaml

# Run evaluation on the dataset
python -m app.evals.cli run --dataset eval_dataset.yaml --concurrency 2
```

## Adding Observability to Custom Components

### Traced Graph Nodes

Use the `TracedNode` class to add automatic tracing to your graph nodes:

```python
from app.core.tracing import TracedNode
from pydantic_graph import GraphRunContext

@dataclass
class MyCustomNode(TracedNode[MyState]):
    """A custom node with built-in tracing."""
    
    async def process(self, ctx: GraphRunContext[MyState]) -> NextNode:
        # Your node logic here
        return NextNode()
```

### Function Tracing Decorators

Use the tracing decorators for regular and async functions:

```python
from app.core.decorators import trace_function, trace_async_function

@trace_function()
def my_regular_function(arg1, arg2):
    # Function logic
    
@trace_async_function()
async def my_async_function(arg1, arg2):
    # Async function logic
```

## Interpreting Traces in Logfire

When viewing traces in Logfire, you'll see several types of spans:

1. **agent_run**: The top-level span for a complete run
2. **graph_node**: Spans for each node execution in the graph
3. **agent_call**: Spans for agent calls to model providers
4. **tool_call**: Spans for tool function executions
5. **function**: Spans for traced regular functions
6. **async_function**: Spans for traced async functions

The spans include attributes such as:

- **query**: The user's original query
- **node_type**: The type of graph node
- **state_type**: The type of state being processed
- **model**: The model being used (for agent calls)
- **duration**: Execution time
- **errors**: Any errors that occurred

## Example: Monitoring Query Classification

This example shows how to track query classification accuracy over time:

1. Add a custom span attribute for classification:

```python
# In app/graphs/root.py
@dataclass
class QueryClassification(TracedNode[MainState]):
    async def process(self, ctx: GraphRunContext[MainState]) -> Union[...]:
        # Get the classifier agent from registry
        classifier_agent = agent_registry.get_agent("query_classifier")
        
        # Classify the query
        with logfire.span("query_classification", query=ctx.state.query):
            result = await classifier_agent.run(...)
            classification = result.output.strip().upper()
            
            # Log the classification result
            logfire.info(
                f"Query classified as: {classification}",
                classification=classification,
                confidence=result.output.confidence if hasattr(result.output, "confidence") else 1.0
            )
        
        # Return the appropriate node
        if "SEARCH" in classification:
            return SearchSubgraphNode()
        # ...
```

2. Query the data in Logfire to track classification patterns over time.

## Troubleshooting

### Common Issues

1. **No data appearing in Logfire**
   - Check if you have authenticated with `py-cli logfire auth`
   - Ensure you're not using `--no-logfire` flag
   - Verify your project settings

2. **Missing spans in traces**
   - Ensure all relevant functions are decorated with tracing decorators
   - Check that nodes extend `TracedNode` or explicitly add spans

3. **Errors in OTel initialization**
   - Verify that all observability dependencies are installed
   - Check if another OTel provider is already initialized

### Contact Support

For issues with the observability setup, please reach out to the maintainers:

- GitHub issues: [pydantic/pydantic-ai/issues](https://github.com/pydantic/pydantic-ai/issues)
- Logfire Support: [logfire.pydantic.dev/support](https://logfire.pydantic.dev/support)