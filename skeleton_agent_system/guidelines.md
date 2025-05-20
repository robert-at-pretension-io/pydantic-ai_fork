# Guidelines for Extending the Agent System

This document provides guidelines for extending the hierarchical multi-agent system with new agents, tools, and subgraphs.

## Adding a New Agent

The agent system uses a configuration-driven approach to make adding new agents easy. Follow these steps:

1. **Create Agent Module**

   Create a new Python file in the appropriate directory:
   
   - `app/agents/deterministic/` for deterministic agents (temperature=0)
   - `app/agents/router/` for non-deterministic routing agents

   Example for a deterministic agent:

   ```python
   # app/agents/deterministic/sentiment_agent.py
   from pydantic_ai import Agent
   from pydantic import BaseModel, Field
   
   class SentimentAnalysis(BaseModel):
       """Output for sentiment analysis."""
       text: str
       sentiment: str = Field(description="Positive, negative, or neutral")
       confidence: float = Field(ge=0.0, le=1.0)
       key_phrases: list[str]
   
   sentiment_agent = Agent(
       "openai:gpt-4o",
       output_type=SentimentAnalysis,
       system_prompt="Analyze the sentiment of the given text.",
       # Deterministic settings
       temperature=0,
       top_p=1,
       seed=42,
   )
   ```

2. **Register in Configuration**

   Add the agent to `config/agents.yaml`:

   ```yaml
   sentiment:
     module: app.agents.deterministic.sentiment_agent
     class: sentiment_agent
     deterministic: true
     model: "openai:gpt-4o"
     tools: []
     description: "Specialized agent for sentiment analysis"
   ```

3. **Define Agent Output Model**

   If your agent has a unique output structure, add its model to `app/models/outputs.py`:

   ```python
   class SentimentResult(BaseModel):
       """Output from the sentiment analysis agent."""
       text: str
       sentiment: str = Field(description="Positive, negative, or neutral")
       confidence: float = Field(ge=0.0, le=1.0)
       key_phrases: list[str]
   ```

## Creating a New Subgraph

For complex workflows, create a new domain-specific subgraph:

1. **Create Graph Module**

   Create a new directory under `app/graphs/domains/` for your domain:

   ```
   mkdir -p app/graphs/domains/sentiment/
   touch app/graphs/domains/sentiment/__init__.py
   touch app/graphs/domains/sentiment/graph.py
   ```

2. **Define Graph State**

   Add a state class to `app/models/states.py`:

   ```python
   @dataclass
   class SentimentState(BaseState):
       """State for the sentiment analysis sub-graph."""
       preprocessed_text: Optional[str] = None
       sentiment_results: Optional[SentimentResult] = None
   ```

3. **Implement Graph Nodes**

   In `app/graphs/domains/sentiment/graph.py`:

   ```python
   from dataclasses import dataclass
   from typing import cast
   
   from pydantic_ai import format_as_xml
   from pydantic_graph import BaseNode, End, Graph, GraphRunContext
   
   from app.core.base_agent import agent_registry
   from app.core.context import context_builder, ContextBuildRequest
   from app.models.outputs import SentimentResult
   from app.models.states import SentimentState
   
   
   @dataclass
   class TextPreprocessing(BaseNode[SentimentState]):
       """Preprocesses text for sentiment analysis."""
       
       async def run(self, ctx: GraphRunContext[SentimentState]) -> SentimentAnalysis:
           # Preprocessing logic
           ctx.state.preprocessed_text = ctx.state.query.strip()
           return SentimentAnalysis()
   
   
   @dataclass
   class SentimentAnalysis(BaseNode[SentimentState, None, SentimentResult]):
       """Analyzes sentiment of preprocessed text."""
       
       async def run(self, ctx: GraphRunContext[SentimentState]) -> End[SentimentResult]:
           # Get sentiment agent from registry
           sentiment_agent = agent_registry.get_agent("sentiment")
           
           # Build context
           context_pkg = await context_builder.build_context(
               ctx,
               ContextBuildRequest(
                   agent_name="sentiment",
                   task=f"Analyze sentiment: {ctx.state.preprocessed_text}",
                   deps=ctx.state.context,
                   max_tokens=1024
               )
           )
           
           # Run the sentiment agent
           result = await sentiment_agent.run(
               ctx.state.preprocessed_text,
               system_prompt=context_pkg.system_prompt,
               message_history=ctx.state.agent_messages.get("sentiment", [])
           )
           
           # Store results in state
           ctx.state.agent_messages["sentiment"] = result.new_messages()
           ctx.state.sentiment_results = cast(SentimentResult, result.output)
           
           # End the graph with sentiment results
           return End(ctx.state.sentiment_results)
   
   
   # Create the sentiment graph
   sentiment_graph = Graph(
       nodes=[TextPreprocessing, SentimentAnalysis],
       state_type=SentimentState
   )
   ```

4. **Update Root Graph**

   To make the root graph aware of your new domain, update `app/graphs/root.py`:

   ```python
   # Add import
   from app.graphs.domains.sentiment.graph import sentiment_graph
   
   # Add a new node class
   @dataclass
   class SentimentSubgraphNode(BaseNode[MainState]):
       """Executes the sentiment analysis sub-graph."""
       
       async def run(self, ctx: GraphRunContext[MainState]) -> ResultAggregation:
           # Create a sentiment-specific state
           sentiment_state = SentimentState(
               query=ctx.state.query,
               context=ctx.state.context
           )
           
           # Run the sentiment sub-graph
           result = await sentiment_graph.run(sentiment_state)
           
           # Update the main state with sentiment results
           ctx.state.sentiment_results = result.output
           
           # Continue to aggregation
           return ResultAggregation()
   
   # Update the QueryClassification node
   ```

5. **Register in Configuration**

   Add the graph to `config/graphs.yaml`:

   ```yaml
   sentiment:
     module: app.graphs.domains.sentiment.graph
     class: sentiment_graph
     nodes:
       - TextPreprocessing
       - SentimentAnalysis
     start_node: TextPreprocessing
   ```

## Adding a New Tool

For external integrations or helper functions:

1. **Create Tool Module**

   Create a new Python file in `app/tools/`:

   ```python
   # app/tools/sentiment_api.py
   from typing import Dict, Any
   
   async def analyze_sentiment(text: str) -> Dict[str, Any]:
       """
       Analyzes sentiment using an external API.
       
       Args:
           text: The text to analyze
           
       Returns:
           A dictionary with sentiment analysis results
       """
       # Implementation that calls an external service
       # ...
       
       return {
           "sentiment": "positive",
           "confidence": 0.95,
           "key_phrases": ["excellent", "amazing"]
       }
   ```

2. **Use the Tool in an Agent**

   Register the tool with an agent:

   ```python
   from app.tools.sentiment_api import analyze_sentiment
   
   @sentiment_agent.tool
   async def external_sentiment_analysis(ctx: RunContext[Any], text: str) -> Dict[str, Any]:
       """Use an external API to analyze sentiment."""
       return await analyze_sentiment(text)
   ```

## Best Practices

1. **Deterministic Leaf Agents**
   - Set `temperature=0` for predictable behavior
   - Use `seed` parameter for even more consistency
   - Define strict output schemas with validation

2. **Context Optimization**
   - Always use the context builder for specialized agents
   - Tailor prompts to each specific task
   - Include only relevant information to avoid token waste

3. **State Management**
   - Define clear state boundaries between domains
   - Only pass necessary information between subgraphs
   - Use strong typing for all state classes

4. **Configuration-Driven Development**
   - Define new components in YAML first
   - Use dynamic loading for extensibility
   - Make components configurable without code changes

5. **Testing**
   - Write snapshot tests for deterministic agents
   - Test subgraphs in isolation
   - Use mocks for external dependencies

## Example: End-to-End Implementation

Here's a complete example of adding a weather agent and subgraph:

1. **Output and State Models**

   ```python
   # app/models/outputs.py
   class WeatherResult(BaseModel):
       location: str
       temperature: float
       conditions: str
       forecast: List[Dict[str, Any]]
   
   # app/models/states.py
   @dataclass
   class WeatherState(BaseState):
       location: Optional[str] = None
       weather_data: Optional[Dict[str, Any]] = None
       weather_results: Optional[WeatherResult] = None
   ```

2. **Agent Implementation**

   ```python
   # app/agents/deterministic/weather.py
   from pydantic_ai import Agent
   
   from app.models.outputs import WeatherResult
   
   weather_agent = Agent(
       "anthropic:claude-3-sonnet",
       output_type=WeatherResult,
       system_prompt="Provide accurate weather information.",
       temperature=0,
       seed=42,
   )
   ```

3. **Tool Implementation**

   ```python
   # app/tools/weather_api.py
   async def get_weather(location: str) -> Dict[str, Any]:
       # Call external weather API
       # ...
       return weather_data
   ```

4. **Graph Implementation**

   ```python
   # app/graphs/domains/weather/graph.py
   @dataclass
   class LocationExtraction(BaseNode[WeatherState]):
       async def run(self, ctx: GraphRunContext[WeatherState]) -> WeatherFetch:
           # Extract location logic
           return WeatherFetch()
           
   @dataclass
   class WeatherFetch(BaseNode[WeatherState]):
       async def run(self, ctx: GraphRunContext[WeatherState]) -> WeatherFormatting:
           # Fetch weather data
           return WeatherFormatting()
           
   @dataclass
   class WeatherFormatting(BaseNode[WeatherState, None, WeatherResult]):
       async def run(self, ctx: GraphRunContext[WeatherState]) -> End[WeatherResult]:
           # Format results
           return End(weather_result)
   
   weather_graph = Graph(
       nodes=[LocationExtraction, WeatherFetch, WeatherFormatting],
       state_type=WeatherState
   )
   ```

5. **Configuration**

   ```yaml
   # config/agents.yaml
   weather:
     module: app.agents.deterministic.weather
     class: weather_agent
     deterministic: true
     
   # config/graphs.yaml
   weather:
     module: app.graphs.domains.weather.graph
     class: weather_graph
     nodes:
       - LocationExtraction
       - WeatherFetch
       - WeatherFormatting
   ```

By following these guidelines, you can easily extend the system with new capabilities while maintaining its hierarchical structure and clean architecture.