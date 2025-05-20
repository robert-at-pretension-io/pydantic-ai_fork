# Router Agent

A multi-agent router implementation built with Pydantic-AI.

## Overview

This module implements a router agent that can:

1. Analyze user queries
2. Dispatch requests to specialized agents based on query content
3. Aggregate results from multiple agents
4. Synthesize a comprehensive response
5. Maintain conversation context between interactions
6. Visualize routing logic as a graph

The router can handle various types of queries, from factual information requests to mathematical calculations and text analysis. It intelligently determines which specialized agents are needed for each query and can process complex queries that require multiple agents working together.

## Architecture

The router agent is implemented using three complementary approaches:

1. **Agent Delegation**: A primary "router" agent delegates to specialized agents through tools and aggregates their responses. This is the approach used in the default implementation.

2. **Graph-Based Structure**: A more advanced implementation using Pydantic-Graph, where the routing flow is represented as a state machine with explicit nodes for different stages of processing. This approach provides better visualization and a clearer path to building more complex routing logic.

3. **Iterative Router (Gemini)**: A Google Gemini-based implementation that uses iterative planning and execution to solve complex tasks using a loop of specialized agents. This implementation is available in `router_agent2.py` and represents the first step toward the task delegation loop architecture outlined in our Future Plans.

### Router Graph Structure

The graph-based router uses the following node structure:

- **QueryClassification**: Analyzes the query to determine which specialized agents to invoke
- **SearchNode**: Handles information retrieval and fact-finding
- **CalculationNode**: Performs mathematical calculations
- **TextAnalysisNode**: Analyzes text for sentiment, summarization, and key points
- **MultipleQueriesNode**: Handles complex queries requiring multiple agents
- **ParallelExecutionNode**: Runs multiple agents in parallel when needed
- **AggregateResults**: Combines all results and synthesizes a final response

This explicit state machine approach allows for:
- Clear visualization of the routing logic
- Easy extension with new specialized agent nodes
- Robust state management throughout the process
- Parallel execution of multiple agents when appropriate

### Graph Visualization

You can visualize the graph-based router structure using:

```bash
# Open graph visualization in browser
python -m agents.router.visualizer

# Save visualization to an HTML file
python -m agents.router.visualizer --output router_graph.html

# Display raw Mermaid code
python -m agents.router.visualizer --raw
```

## Specialized Agents

### Standard Router

The standard router currently delegates to three specialized agents:

1. **Search Agent**: Finds and extracts relevant information
2. **Calculation Agent**: Performs mathematical calculations and provides step-by-step solutions
3. **Text Analysis Agent**: Analyzes text for sentiment, summarizations, and key points

### Iterative Router (Gemini)

The Gemini-based iterative router has access to a wider range of specialized tools/agents:

1. **Search Information**: Retrieves factual information
2. **Perform Calculation**: Executes mathematical calculations
3. **Analyze Text**: Analyzes text content comprehensively
4. **Create Plan**: Breaks down complex tasks into subtasks
5. **Perform Reasoning**: Handles logical deduction problems
6. **Execute Code**: Writes or modifies code snippets
7. **Analyze Data**: Processes structured data
8. **Verify Information**: Checks correctness of statements
9. **Summarize Content**: Creates focused summaries
10. **Make Decision**: Evaluates options against criteria
11. **Knowledge Base**: Stores and retrieves facts within a session
12. **Media Analysis**: Processes images, audio, video, and documents

The iterative router represents a powerful step toward our vision of a full task delegation system outlined in the Future Plans.

## Setup

### API Keys

The router agent requires API keys for the LLM providers it uses. By default, it falls back to OpenAI for all agents, but you can configure different providers:

```bash
# Required for OpenAI models
export OPENAI_API_KEY=your_openai_api_key

# Optional for using other models
export ANTHROPIC_API_KEY=your_anthropic_api_key
export MISTRAL_API_KEY=your_mistral_api_key

# Required for the Iterative Router (Gemini)
export GEMINI_API_KEY=your_gemini_api_key
```

### Model Configuration

You can configure which models to use for each specialized agent:

```bash
export ROUTER_SEARCH_MODEL="openai:gpt-4o"
export ROUTER_CALCULATION_MODEL="anthropic:claude-3-5-sonnet"
export ROUTER_TEXT_ANALYSIS_MODEL="mistral:mistral-large"
```

## Usage

### As a Module

#### Using Tool-Based Router Agent

```python
from agents.router import process_query

async def example():
    # Simple query
    response, messages = await process_query("What is the population of Paris?")
    print(response.final_answer)
    
    # With context
    context = {"user_history": ["Previous queries about Paris"]}
    response, messages = await process_query("What is the population of Paris?", context=context)
    
    # With conversation history for follow-up questions
    response2, messages = await process_query("What's its average temperature?", message_history=messages)
    print(f"Follow-up answer: {response2.final_answer}")
```

#### Using Graph-Based Router Agent

```python
from agents.router import process_query_with_graph

async def example_graph():
    # Simple query with the graph-based router
    response, messages = await process_query_with_graph("What is the population of Paris?")
    print(response.final_answer)
    
    # With conversation history for follow-up questions
    response2, messages = await process_query_with_graph("What's its average temperature?", message_history=messages)
    print(f"Follow-up answer: {response2.final_answer}")
    
    # Visualize the graph structure (programmatically)
    from agents.router import visualize_graph, save_to_file, generate_html
    mermaid_code = visualize_graph()
    html_content = generate_html(mermaid_code)
    save_to_file(html_content, "router_graph.html")
```

#### Using the Iterative Router (Gemini)

```python
from agents.router.router_agent2 import process_query_iteratively

async def example_iterative():
    # Process a complex query using the iterative router
    response, messages = await process_query_iteratively(
        "What is the capital of France? Then search for its current population."
    )
    print(response.final_answer)
    
    # With context and conversation history
    initial_context = {"shared_context": {"language": "en-US"}}
    response2, messages = await process_query_iteratively(
        "What are the top 3 tourist attractions there?", 
        initial_context=initial_context,
        initial_message_history=messages,
        verbose=True  # Enable detailed logging
    )
    print(f"Follow-up answer: {response2.final_answer}")
```

To use the iterative router, you need to set the environment variable for Google Gemini API access:

```bash
export GEMINI_API_KEY=your_gemini_api_key
```

### Command Line Interface

The router agent includes a rich command-line interface with several features:

#### Basic Usage

```bash
# From project root
python -m agents.router "What is the population of Paris?"

# With context
python -m agents.router "What is the population of Paris?" --context '{"user_history": ["Previous queries about Paris"]}'
```

#### Interactive Mode with Conversation Memory

```bash
# Start interactive mode
python -m agents.router --interactive  # or -i for short

# Interactive mode is also the default when no query is provided
python -m agents.router
```

The interactive mode automatically maintains conversation context between your queries, allowing for natural follow-up questions and references to previous information.

**Example conversation:**
```
Router > What is the capital of France?
# Paris is the capital of France.

Router > What's its population?
# The population of Paris is approximately 2.1 million in the city proper.
# (About 12 million in the metropolitan area)
(Conversation context is being maintained)
```

In interactive mode, you can use special commands:
- `/exit`: Exit the session
- `/help`: Show help information
- `/history [n]`: View last n chat history entries (default: 5)
- `/clear`: Clear the screen
- `/version`: Show version information
- `/reset`: Reset the conversation history and start fresh

#### Chat History

The CLI automatically saves your interactions and lets you view your history:

```bash
# View recent chat history
python -m agents.router --history  # or -H for short

# View a specific number of history entries
python -m agents.router --history --history-limit 10
```

#### Theme Options

You can customize the appearance of the CLI:

```bash
# Set a theme for syntax highlighting
python -m agents.router --theme light  # Options: dark, light, monokai, github-dark, gruvbox-dark
```

#### Running the Iterative Router (CLI)

The Gemini-based iterative router comes with its own CLI interface:

```bash
# Run with a direct query
python -m agents.router.router_agent2 "What is the capital of France? Then search for its population."

# Run in interactive mode
python -m agents.router.router_agent2 --interactive

# Run with less verbose output
python -m agents.router.router_agent2 --quiet "What's the tallest building in Europe?"

# Run without rich console formatting
python -m agents.router.router_agent2 --no-console "Calculate 34 * 56 + 12"
```

In interactive mode, you can use these commands:
- `/exit`: Exit the session
- `/help`: Show help information
- `/clear`: Clear the screen
- `/verbose`: Enable detailed logging
- `/quiet`: Disable detailed logging

#### Using with UV and CLAI

For integration with `uv` and the PydanticAI `clai` tool:

```bash
# Install clai with uv
uv pip install clai

# Run with uvx and the clai-compatible agent
uvx clai --agent agents.router.clai_compatible:agent "What is the capital of Brazil?"

# Or for a one-line setup with API key
OPENAI_API_KEY=your_api_key_here uvx clai --agent agents.router.clai_compatible:agent
```

#### Running with UV

You can use `uv` to run the router agent efficiently:

```bash
# Run directly with uv
uv run -m agents.router "What is the population of Tokyo?"

# Run interactive mode
uv run -m agents.router -i

# Install as a local package
uv pip install -e ./agents/router
router-agent "What is the capital of France?"
```

#### Help

```bash
# Show all available options
python -m agents.router --help

# Or with uv
uv run -m agents.router --help
```

## Extending

### Adding New Specialized Agents

#### For Tool-Based Router

1. Create a new model type in `router_agent.py`
2. Create a new agent instance
3. Add a new tool that delegates to this agent
4. Update the `AggregatedResponse` model to include the new results

#### For Graph-Based Router

1. Create a new model type in `router_agent.py` (shared with tool-based router)
2. Create a new agent instance (shared with tool-based router)
3. Add a new node class to `graph_router.py` that uses this agent
4. Update the router graph node list to include your new node
5. Update the `QueryClassification` node to handle routing to your new node
6. Update the `RouterState` class to store the new agent's results

## Best Practices

- Only invoke relevant agents for a given query
- Pass usage tracking to delegated agents
- Use structured outputs for type safety
- Implement proper error handling
- Parallelize agent calls when possible

## Future Plans

We have ambitious plans to enhance the router agent with a task delegation loop architecture that can handle complex multi-step tasks through iterative delegation to specialized agents.

See [FUTURE_PLANS.md](./FUTURE_PLANS.md) for detailed information about our vision for:

- Task decomposition and planning
- Iterative problem-solving through specialized agents
- State management across multiple agent interactions
- Advanced orchestration capabilities

The `router_agent2.py` implementation represents our first step toward this vision, using Google's Gemini models with a planner-driven approach that can iteratively solve problems through multiple steps.