# Technical Document: Building a Multi-Agent Router with Pydantic-AI

## Overview

This document outlines how to implement a routing agent that can dispatch requests to specialized agents and aggregate their responses. This pattern enables complex problem-solving by decomposing tasks into subtasks handled by specialized agents, while maintaining a unified interface for the caller.

## Architectural Approaches

Pydantic-AI offers four primary approaches for implementing a routing agent system:

1. **Agent Delegation**: A primary agent routes requests to specialized agents via tool calls
2. **Programmatic Agent Hand-off**: Sequential execution of agents controlled by application code
3. **Graph-Based Control Flow**: State machine approach for complex routing logic
4. **A2A Protocol**: Distributed agent communication across network boundaries

Each approach has different complexity and use cases. Let's begin with the most direct implementation.

## Implementation: Agent Delegation Router

This approach uses a primary "router" agent that delegates to specialized agents through tools and aggregates their responses.

```python
from dataclasses import dataclass
from typing import List, Union, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import Usage

# Define our router dependencies
@dataclass
class RouterDeps:
    """Dependencies for the router agent."""
    context: dict[str, Any] = field(default_factory=dict)
    
# Define specialized agent output types
class SearchResult(BaseModel):
    query: str
    results: List[str]
    
class CalculationResult(BaseModel):
    input: str
    result: float
    steps: List[str]

class TextAnalysisResult(BaseModel):
    sentiment: str
    summary: str
    key_points: List[str]

# Define the aggregated response type
class AggregatedResponse(BaseModel):
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    text_analysis: Optional[TextAnalysisResult] = None
    final_answer: str = Field(description="Synthesized response incorporating all agent outputs")

# Create specialized agents
search_agent = Agent(
    'openai:gpt-4o',
    output_type=SearchResult,
    system_prompt="You are a search specialist. Extract relevant information from context."
)

calculation_agent = Agent(
    'anthropic:claude-3-5-sonnet',
    output_type=CalculationResult,
    system_prompt="You are a calculation specialist. Solve mathematical problems step by step."
)

text_analysis_agent = Agent(
    'mistral:mistral-large',
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

# Define tools for the router to access specialized agents
@router_agent.tool
async def search_information(ctx: RunContext[RouterDeps], query: str) -> SearchResult:
    """
    Use the search agent to find information related to the query.
    Only use this when factual information needs to be gathered.
    """
    result = await search_agent.run(query, usage=ctx.usage)
    return result.output

@router_agent.tool
async def perform_calculation(ctx: RunContext[RouterDeps], problem: str) -> CalculationResult:
    """
    Use the calculation agent to solve mathematical problems.
    Only use this for queries that involve numerical calculations.
    """
    result = await calculation_agent.run(problem, usage=ctx.usage)
    return result.output

@router_agent.tool
async def analyze_text(ctx: RunContext[RouterDeps], text: str) -> TextAnalysisResult:
    """
    Use the text analysis agent to perform sentiment analysis, summarization, and extract key points.
    Only use this for queries that require text understanding.
    """
    result = await text_analysis_agent.run(text, usage=ctx.usage)
    return result.output

# Main function to run the router
async def process_query(query: str, context: dict = None) -> AggregatedResponse:
    """
    Process a user query through the routing agent, which will delegate to specialized agents as needed.
    
    Args:
        query: The user's query
        context: Optional context information
        
    Returns:
        An aggregated response from all relevant specialized agents
    """
    deps = RouterDeps(context=context or {})
    result = await router_agent.run(query, deps=deps)
    
    # Access individual agent results if needed
    if result.output.search_results:
        print(f"Search found {len(result.output.search_results.results)} results")
    
    return result.output
```

## Advanced Implementation: Graph-Based Router

For more complex routing logic, we can implement a graph-based approach that explicitly models state transitions between agents:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent, format_as_xml
from pydantic_graph import BaseNode, Graph, GraphRunContext, End

# Define our specialized agents (as before)
# ...

# State shared between nodes in the graph
@dataclass
class RouterState:
    """State for the router graph."""
    query: str
    context: Dict[str, Any]
    search_results: Optional[SearchResult] = None
    calculation_results: Optional[CalculationResult] = None
    text_analysis: Optional[TextAnalysisResult] = None
    agent_messages: Dict[str, List] = field(default_factory=dict)

# Graph nodes representing processing steps
@dataclass
class QueryAnalysis(BaseNode[RouterState]):
    """Analyzes the query to determine which agents to invoke."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> Union[SearchNode, CalculationNode, TextAnalysisNode, AggregateResults]:
        analysis_agent = Agent(
            'openai:gpt-4o',
            output_type=str,
            system_prompt="Analyze this query and respond with exactly one of: SEARCH, CALCULATION, TEXT_ANALYSIS, or AGGREGATE"
        )
        
        result = await analysis_agent.run(
            f"Analyze this query: {ctx.state.query}\n\nRespond with the primary category: SEARCH, CALCULATION, TEXT_ANALYSIS, or AGGREGATE",
            message_history=ctx.state.agent_messages.get("analysis", [])
        )
        
        ctx.state.agent_messages["analysis"] = result.new_messages()
        
        if "SEARCH" in result.output:
            return SearchNode()
        elif "CALCULATION" in result.output:
            return CalculationNode()
        elif "TEXT_ANALYSIS" in result.output:
            return TextAnalysisNode()
        else:
            return AggregateResults()

@dataclass
class SearchNode(BaseNode[RouterState]):
    """Performs a search using the search agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> AggregateResults:
        result = await search_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("search", [])
        )
        
        ctx.state.agent_messages["search"] = result.new_messages()
        ctx.state.search_results = result.output
        
        return AggregateResults()

@dataclass
class CalculationNode(BaseNode[RouterState]):
    """Performs calculations using the calculation agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> AggregateResults:
        result = await calculation_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("calculation", [])
        )
        
        ctx.state.agent_messages["calculation"] = result.new_messages()
        ctx.state.calculation_results = result.output
        
        return AggregateResults()

@dataclass
class TextAnalysisNode(BaseNode[RouterState]):
    """Analyzes text using the text analysis agent."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> AggregateResults:
        result = await text_analysis_agent.run(
            ctx.state.query,
            message_history=ctx.state.agent_messages.get("text_analysis", [])
        )
        
        ctx.state.agent_messages["text_analysis"] = result.new_messages()
        ctx.state.text_analysis = result.output
        
        return AggregateResults()

@dataclass
class AggregateResults(BaseNode[RouterState, None, AggregatedResponse]):
    """Aggregates results from all agents and produces a final response."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[AggregatedResponse]:
        aggregation_agent = Agent(
            'openai:gpt-4o',
            output_type=AggregatedResponse,
            system_prompt="Synthesize outputs from multiple agents into one coherent response."
        )
        
        # Build an XML representation of all available results
        state_dict = {
            "query": ctx.state.query,
            "search_results": ctx.state.search_results,
            "calculation_results": ctx.state.calculation_results,
            "text_analysis": ctx.state.text_analysis
        }
        
        result = await aggregation_agent.run(
            f"Synthesize these outputs into one coherent response:\n{format_as_xml(state_dict)}",
            message_history=ctx.state.agent_messages.get("aggregation", [])
        )
        
        ctx.state.agent_messages["aggregation"] = result.new_messages()
        
        return End(result.output)

# Create the router graph
router_graph = Graph(
    nodes=[QueryAnalysis, SearchNode, CalculationNode, TextAnalysisNode, AggregateResults],
    state_type=RouterState
)

# Main function to use the graph-based router
async def process_query_with_graph(query: str, context: dict = None) -> AggregatedResponse:
    """Process a query using the graph-based router."""
    state = RouterState(query=query, context=context or {})
    result = await router_graph.run(QueryAnalysis(), state=state)
    return result.output
```

## Distributed Agent Router with A2A Protocol

For cases where agents need to run in separate processes or even separate machines, you can use the A2A protocol:

```python
from pydantic_ai import Agent
from fasta2a import FastA2A, InMemoryStorage, InMemoryBroker
from fasta2a.client import A2AClient
from fasta2a.schema import AgentCard, Provider, Skill

# Convert specialized agents to A2A servers
search_server = search_agent.to_a2a(
    name="SearchAgent",
    url="http://localhost:8001",
    description="Specialized agent for search tasks"
)

calculation_server = calculation_agent.to_a2a(
    name="CalculationAgent",
    url="http://localhost:8002",
    description="Specialized agent for calculation tasks"
)

text_analysis_server = text_analysis_agent.to_a2a(
    name="TextAnalysisAgent",
    url="http://localhost:8003",
    description="Specialized agent for text analysis tasks"
)

# Create a router agent that communicates with other agents via A2A
@router_agent.tool
async def call_search_agent(ctx: RunContext[RouterDeps], query: str) -> SearchResult:
    """Call the search agent via A2A protocol."""
    client = A2AClient("http://localhost:8001")
    response = await client.run(query)
    return SearchResult.model_validate_json(response.result)

@router_agent.tool
async def call_calculation_agent(ctx: RunContext[RouterDeps], problem: str) -> CalculationResult:
    """Call the calculation agent via A2A protocol."""
    client = A2AClient("http://localhost:8002")
    response = await client.run(problem)
    return CalculationResult.model_validate_json(response.result)

@router_agent.tool
async def call_text_analysis_agent(ctx: RunContext[RouterDeps], text: str) -> TextAnalysisResult:
    """Call the text analysis agent via A2A protocol."""
    client = A2AClient("http://localhost:8003")
    response = await client.run(text)
    return TextAnalysisResult.model_validate_json(response.result)
```

## Best Practices

1. **Dynamic Agent Selection**: 
   - Router should analyze queries to determine which specialized agents to call
   - Only invoke agents that are relevant to the query

2. **Usage Tracking**:
   - Always pass usage tracking (`ctx.usage`) to delegated agents
   - This ensures accurate token counting and cost tracking

3. **Structured Outputs**:
   - Use Pydantic models for specialized agent outputs
   - This ensures type safety and validation

4. **Error Handling**:
   - Router should gracefully handle failures from individual specialized agents
   - Implement retry logic or alternate agents for resilience

5. **Parallelization**:
   - When possible, invoke multiple specialized agents in parallel using `asyncio.gather`
   - This reduces total latency when multiple agents need to be consulted

## Example Implementation for Specific Use Cases

### Content Creation Router

```python
# Define content stages
class ContentOutline(BaseModel):
    title: str
    sections: List[Dict[str, str]]

class ContentDraft(BaseModel):
    title: str
    content: str

class ContentEdited(BaseModel):
    title: str
    content: str
    improvements: List[str]

# Specialized agents
outline_agent = Agent(
    'openai:gpt-4o',
    output_type=ContentOutline,
    system_prompt="Create detailed content outlines with sections and key points."
)

draft_agent = Agent(
    'anthropic:claude-3-opus',
    output_type=ContentDraft,
    system_prompt="Write detailed first drafts based on content outlines."
)

edit_agent = Agent(
    'gemini:gemini-1.5-pro',
    output_type=ContentEdited,
    system_prompt="Edit and improve content drafts, focus on clarity and engagement."
)

# Router agent tools
@router_agent.tool
async def create_outline(ctx: RunContext[RouterDeps], topic: str) -> ContentOutline:
    """Create a content outline for the given topic."""
    result = await outline_agent.run(f"Create an outline for: {topic}", usage=ctx.usage)
    return result.output

@router_agent.tool
async def write_draft(ctx: RunContext[RouterDeps], outline: ContentOutline) -> ContentDraft:
    """Write a draft based on the outline."""
    result = await draft_agent.run(format_as_xml(outline), usage=ctx.usage)
    return result.output

@router_agent.tool
async def edit_content(ctx: RunContext[RouterDeps], draft: ContentDraft) -> ContentEdited:
    """Edit and improve the draft."""
    result = await edit_agent.run(format_as_xml(draft), usage=ctx.usage)
    return result.output
```

## Conclusion

Pydantic-AI provides multiple approaches for implementing routing agents. The choice of approach depends on the complexity of your routing logic:

1. **Agent Delegation**: Best for straightforward routing needs
2. **Programmatic Hand-off**: Best for sequential agent workflows
3. **Graph-Based Control Flow**: Best for complex routing logic with explicit state transitions
4. **A2A Protocol**: Best for distributed multi-agent systems

By following these patterns, you can build robust routing agents that decompose complex tasks into manageable subtasks handled by specialized agents, while providing a unified interface for the caller.

## Hierarchical State Machines with Deterministic Agents

For complex projects requiring hierarchical state machines and deterministic agent behavior, this section provides guidance on structuring such a system using Pydantic-AI and Pydantic-Graph.

### Building Deterministic Agents

While LLMs are inherently non-deterministic, you can make agent behavior more predictable through:

1. **Structured Tools and Outputs**
   - Use function tools to implement deterministic business logic
   - Define strict output models with validation
   - Implement fallbacks for unpredictable outputs

2. **Model Configuration**
   - Set low temperature (e.g., 0.0) in model settings
   - Use the `seed` parameter (supported by OpenAI, Groq, Cohere, Mistral)
   ```python
   result = agent.run_sync(
       "Perform analysis",
       model_settings={"seed": 42, "temperature": 0.0}
   )
   ```

3. **Input Standardization**
   - Pre-process all inputs with deterministic rules
   - Provide explicit context in XML format
   - Use `format_as_xml` to ensure consistent input structure

### Implementing Hierarchical State Machines

For complex workflows with nested logic, implement a hierarchical structure:

1. **Parent-Child Graph Structure**
   ```python
   @dataclass
   class ParentNode(BaseNode[ParentState]):
       async def run(self, ctx: GraphRunContext[ParentState]) -> ChildGraphNode | EndNode:
           # Determine if we should move to child graph or end
           if condition:
               return ChildGraphNode()
           else:
               return EndNode()

   @dataclass
   class ChildGraphNode(BaseNode[ParentState, None, ChildResult]):
       async def run(self, ctx: GraphRunContext[ParentState]) -> NextParentNode:
           # Run entire child graph
           child_state = ChildState()
           result = await child_graph.run(ChildStartNode(), state=child_state)
           # Store results in parent state
           ctx.state.child_results = result.output
           return NextParentNode()
   ```

2. **State Composition**
   - Define clear state boundaries between parent and child graphs
   - Transfer only necessary state data between graph levels
   - Use dependency injection for shared services

### Project Organization

For large projects with hierarchical state machines, use this file organization structure:

```
myproject/
├── agents/                     # Agent definitions
│   ├── __init__.py
│   ├── base.py                 # Base agent configurations and utilities
│   ├── deterministic/          # Deterministic agent implementations
│   │   ├── __init__.py
│   │   ├── search_agent.py     # Deterministic search agent
│   │   └── calculator_agent.py # Deterministic calculation agent
│   └── llm/                    # LLM-based agent implementations
│       ├── __init__.py
│       ├── router_agent.py     # Main router agent
│       └── synthesis_agent.py  # Results synthesis agent
├── models/                     # Data models
│   ├── __init__.py
│   ├── inputs.py               # Input model definitions
│   ├── outputs.py              # Output model definitions
│   └── states/                 # State definitions for graphs
│       ├── __init__.py
│       ├── parent_state.py     # State for main workflow
│       └── child_states.py     # States for subworkflows
├── graphs/                     # Graph definitions
│   ├── __init__.py
│   ├── main_graph.py           # Top-level workflow graph
│   ├── utils.py                # Graph utilities and shared functions
│   └── sub_graphs/             # Component subgraphs
│       ├── __init__.py
│       ├── search_flow.py      # Search subgraph
│       ├── calculation_flow.py # Calculation subgraph
│       └── analysis_flow.py    # Analysis subgraph
├── persistence/                # Persistence implementations
│   ├── __init__.py
│   ├── base.py                 # Base persistence classes
│   └── db_persistence.py       # Database persistence implementation
├── services/                   # External service integrations
│   ├── __init__.py
│   ├── search_service.py       # External search API
│   └── database_service.py     # Database connection service
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── xml_formatter.py        # Utilities for XML formatting
│   └── deterministic_tools.py  # Deterministic tool implementations
├── config.py                   # Configuration settings
└── main.py                     # Application entry point
```

### Example: Nested Graph Implementation

Here's how to implement a child graph that's called from a parent graph node:

```python
# In graphs/sub_graphs/search_flow.py
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class SearchState:
    query: str
    results: list[str] = field(default_factory=list)

@dataclass
class QueryFormulation(BaseNode[SearchState]):
    async def run(self, ctx: GraphRunContext[SearchState]) -> ExecuteSearch:
        # Format query for search
        return ExecuteSearch()

@dataclass
class ExecuteSearch(BaseNode[SearchState]):
    async def run(self, ctx: GraphRunContext[SearchState]) -> FilterResults:
        # Perform search
        ctx.state.results = ["result1", "result2"]
        return FilterResults()

@dataclass
class FilterResults(BaseNode[SearchState, None, list[str]]):
    async def run(self, ctx: GraphRunContext[SearchState]) -> End[list[str]]:
        # Filter and clean results
        return End(ctx.state.results)

search_graph = Graph(
    nodes=[QueryFormulation, ExecuteSearch, FilterResults],
    state_type=SearchState
)
```

```python
# In graphs/main_graph.py
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from .sub_graphs.search_flow import search_graph, QueryFormulation, SearchState

@dataclass
class MainState:
    query: str
    search_results: list[str] = field(default_factory=list)
    # Other state fields...

@dataclass
class AnalyzeQuery(BaseNode[MainState]):
    async def run(self, ctx: GraphRunContext[MainState]) -> SearchNode | CalculateNode:
        # Analyze query to determine next step
        return SearchNode()

@dataclass
class SearchNode(BaseNode[MainState]):
    async def run(self, ctx: GraphRunContext[MainState]) -> SynthesizeResults:
        # Run the entire search subgraph
        search_state = SearchState(query=ctx.state.query)
        result = await search_graph.run(QueryFormulation(), state=search_state)
        ctx.state.search_results = result.output
        return SynthesizeResults()

@dataclass
class SynthesizeResults(BaseNode[MainState, None, str]):
    async def run(self, ctx: GraphRunContext[MainState]) -> End[str]:
        # Combine all results
        return End(f"Found: {', '.join(ctx.state.search_results)}")

main_graph = Graph(
    nodes=[AnalyzeQuery, SearchNode, SynthesizeResults],
    state_type=MainState
)
```

### State Persistence for Complex Hierarchical Systems

For large hierarchical systems, implement a database-backed persistence mechanism:

```python
# In persistence/db_persistence.py
from dataclasses import dataclass
from typing import Any, Optional, Generic, TypeVar
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_graph.persistence import BaseStatePersistence, NodeSnapshot, EndSnapshot

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")

class DatabaseStatePersistence(BaseStatePersistence[StateT, RunEndT], Generic[StateT, RunEndT]):
    def __init__(self, db_session: AsyncSession, run_id: str):
        self.db_session = db_session
        self.run_id = run_id
        self._state_type = None
        self._run_end_type = None

    async def save_node(self, snapshot: NodeSnapshot[StateT]) -> None:
        # Store the node snapshot in database
        # ...

    async def save_end(self, snapshot: EndSnapshot[RunEndT]) -> None:
        # Store the end snapshot in database
        # ...

    async def load_next(self) -> Optional[NodeSnapshot[StateT]]:
        # Load the next node from database
        # ...

    async def load_all(self) -> list[NodeSnapshot[StateT] | EndSnapshot[RunEndT]]:
        # Load all snapshots from database
        # ...
```

This organization and implementation approach allows for building complex, deterministic, hierarchical state machines that can handle sophisticated workflows while maintaining clear boundaries between components.