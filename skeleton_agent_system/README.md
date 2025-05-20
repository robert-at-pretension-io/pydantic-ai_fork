# Hierarchical Multi-Agent System

A framework for building hierarchical multi-agent systems using Pydantic-AI and Pydantic-Graph.

## Overview

This skeleton provides a foundation for building complex, multi-agent systems with:

- Hierarchical graph-based control flow
- Specialized deterministic agents
- Context optimization
- State management
- Robust telemetry

## Architecture

The system is structured around a few key concepts:

1. **Root Router Graph**: Classifies queries and dispatches to domain-specific subgraphs
2. **Domain Subgraphs**: Handle specialized workflows (search, coding, operations)
3. **Leaf Agents**: Perform specific tasks with deterministic behavior
4. **Context Builder**: Optimizes prompts for each specialized agent

## Key Components

- **Agent Registry**: Configuration-driven agent loading and management
- **Context Builder**: Optimizes prompts for specialized agents
- **Graph Hierarchy**: Structured workflow execution with well-defined state transitions
- **Persistence Layer**: State management for long-running tasks
- **Telemetry**: Comprehensive monitoring and logging

## Getting Started

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Create an agent:
   
   See the [guidelines.md](guidelines.md) file for detailed instructions on extending the system.

## Running the System

```python
from app.graphs.root import root_graph
from app.models.states import MainState

# Create initial state
state = MainState(query="What is the weather in New York?", context={})

# Run the graph
result = await root_graph.run(state=state)

# Access the result
print(result.output.final_answer)
```

## Features

- **Declarative Configuration**: Add new agents by updating YAML files
- **Dynamic Loading**: Load agents and graphs at runtime
- **Hierarchical Workflows**: Compose complex workflows from simpler subgraphs
- **Parallel Execution**: Run independent subgraphs concurrently
- **Deterministic Foundation**: Build on deterministic leaf agents for reliability
- **Optimized Context**: Avoid context window overflows with targeted context building

## License

MIT