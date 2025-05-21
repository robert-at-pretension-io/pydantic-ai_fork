# Pipeline Agent Implementation Plan

## 1. Architecture Overview

The Pipeline Agent system implements a modular, agent-based code review process for Merge Requests. It follows a structured pipeline approach where specialized agents operate on shared state, each contributing unique insights toward a unified code review response.

### Key Design Principles

1. **Agent Modularity**: Each agent has a well-defined role and interface
2. **Shared State**: A common state object tracks progress across the pipeline
3. **Loopable Agents**: Complex reasoning tasks can iterate until completion
4. **Parallel Processing**: Independent agents can run concurrently
5. **Clear Interfaces**: Structured inputs and outputs between agents
6. **Graceful Recovery**: Each stage can handle failures or timeouts

### Data Flow

```
Repository + Jira Ticket → [Pipeline] → CI Pass/Fail + Review Feedback
```

Where the pipeline consists of:

```
Ticket Identifier → Codebase Explorer → Implementation Planners (x3) → Diff Reviewer → Verifier → Synthesizer → Final Gate
```

## 2. Core Components

### 2.1 Pipeline State Management

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
import uuid
from datetime import datetime

class CodebaseContext(BaseModel):
    architecture_notes: List[str] = Field(default_factory=list)
    key_code_snippets: Dict[str, str] = Field(default_factory=dict)
    observed_conventions: List[str] = Field(default_factory=list)

class ImplementationPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan_name: str
    summary: str
    steps: List[str]
    files_touched: List[str]
    pros: List[str]
    cons: List[str]
    estimated_effort: str

class DiffReview(BaseModel):
    summary: str
    architectural_feedback: List[str]
    completion_feedback: List[str]
    test_coverage_feedback: List[str]
    style_feedback: List[str]
    concerns: List[str] = Field(default_factory=list)

class VerifierAnalysis(BaseModel):
    summary: str
    is_valid_solution: bool
    strengths: List[str]
    improvement_areas: List[str]
    alternative_suggestions: Optional[List[str]] = None

class PipelineVerdict(Literal["PASS", "SOFT_FAIL", "HARD_FAIL"])

class JiraTicket(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: Optional[List[str]] = None
    comments: Optional[List[Dict[str, Any]]] = None

class PipelineState(BaseModel):
    """Central state object shared across all pipeline agents."""
    session_id: str = Field(default_factory=lambda: f"pipeline_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Repository metadata
    repo_path: str
    current_branch: str
    
    # Pipeline inputs
    jira_ticket: Optional[JiraTicket] = None
    mr_title: Optional[str] = None
    mr_description: Optional[str] = None
    
    # Pipeline agent outputs
    codebase_context: Optional[CodebaseContext] = None
    implementation_plans: List[ImplementationPlan] = Field(default_factory=list)
    diff_review: Optional[DiffReview] = None
    verifier_analysis: Optional[VerifierAnalysis] = None
    
    # Pipeline output
    synthesized_feedback: Optional[str] = None
    verdict: Optional[PipelineVerdict] = None
    
    # Message history across agent interactions
    message_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Execution metadata
    errors: Dict[str, str] = Field(default_factory=dict)
    stage_durations: Dict[str, float] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    instrumentation_data: Dict[str, Any] = Field(default_factory=dict)
```

### 2.2 Base Pipeline Agent

```python
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Generic, TypeVar, List
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage
from pydantic_ai.result import AgentRunResult
import time
import opentelemetry.trace
import inspect

T = TypeVar('T')  # Output type

class PipelineAgent(Generic[T], ABC):
    """Base class for all pipeline agents."""
    
    def __init__(
        self, 
        name: str,
        model: str = "openai:gpt-4o",
        max_iterations: int = 5,
        model_settings: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        retries: int = 1,
        instrument: bool = True,
    ):
        self.name = name
        self.model = model
        self.max_iterations = max_iterations
        self.model_settings = model_settings or {}
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.retries = retries
        self.instrument = instrument
        
        # Create tracer for this agent
        self.tracer = opentelemetry.trace.get_tracer(f"pipeline_agent.{name}")
        
    @abstractmethod
    async def process(self, state: PipelineState) -> T:
        """Process the pipeline state and return agent output."""
        pass
        
    async def run_with_timing(self, state: PipelineState) -> T:
        """Run agent with timing and error tracking."""
        with self.tracer.start_as_current_span(f"{self.name}_process") as span:
            start_time = time.time()
            try:
                # Get calling frame for better error context
                frame = inspect.currentframe()
                caller_info = inspect.getframeinfo(frame.f_back) if frame else None
                span.set_attribute("caller_info", str(caller_info) if caller_info else "unknown")
                
                # Process state
                result = await self.process(state)
                
                # Track timing
                duration = time.time() - start_time
                state.stage_durations[self.name] = duration
                span.set_attribute("duration_seconds", duration)
                
                return result
            except Exception as e:
                state.errors[self.name] = str(e)
                state.stage_durations[self.name] = time.time() - start_time
                span.record_exception(e)
                span.set_status(opentelemetry.trace.Status(opentelemetry.trace.StatusCode.ERROR))
                raise
    
    def _message_to_dict(self, message: ModelMessage) -> Dict[str, Any]:
        """Convert a ModelMessage to a serializable dictionary."""
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"type": message.__class__.__name__, "content": str(message)}
    
    def track_agent_usage(self, state: PipelineState, result: AgentRunResult[Any]) -> None:
        """Track token usage from agent run result."""
        if not result.usage:
            return
            
        # Track token usage
        tokens = {
            "total": result.usage.total_tokens or 0,
            "prompt": result.usage.request_tokens or 0,
            "completion": result.usage.response_tokens or 0,
        }
        
        # Store in state token usage
        for key, value in tokens.items():
            state.token_usage.setdefault(self.name, {})
            state.token_usage[self.name].setdefault(key, 0)
            state.token_usage[self.name][key] += value
            
        # Store messages for potential reuse
        if result.messages:
            state.message_history.setdefault(self.name, [])
            state.message_history[self.name].extend([self._message_to_dict(msg) for msg in result.messages])
            
    def create_agent(self, output_type: Any = str) -> Agent:
        """Create a configured agent instance."""
        return Agent(
            model=self.model,
            output_type=output_type,
            instructions=self.instructions,
            system_prompt=self.system_prompt,
            name=f"{self.name}_agent",
            model_settings=self.model_settings,
            retries=self.retries,
            instrument=self.instrument,
        )
```

### 2.3 Loopable Agent Pattern

```python
from pydantic_ai.messages import ModelMessage
import hashlib
import json

class LoopablePipelineAgent(PipelineAgent[T]):
    """Pipeline agent that can loop until completion or max iterations."""
    
    def __init__(
        self,
        name: str,
        model: str = "openai:gpt-4o",
        max_iterations: int = 5,
        model_settings: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        retries: int = 1,
        instrument: bool = True,
        stall_detection: bool = True,
    ):
        super().__init__(
            name=name, 
            model=model, 
            max_iterations=max_iterations,
            model_settings=model_settings,
            system_prompt=system_prompt,
            instructions=instructions,
            retries=retries,
            instrument=instrument,
        )
        
        # Create decider agent model
        self.decider_agent = self.create_agent(output_type=bool)
        self.stall_detection = stall_detection
        
    async def should_continue(
        self, 
        state: PipelineState, 
        iteration: int, 
        context: Dict[str, Any],
        previous_state_hash: Optional[str] = None
    ) -> bool:
        """Determine if the agent should continue processing."""
        # Check iteration limit
        if iteration >= self.max_iterations:
            return False
            
        # Check for stalled state if enabled
        if self.stall_detection and previous_state_hash is not None:
            current_hash = self._calculate_state_hash(context)
            if current_hash == previous_state_hash:
                # State hasn't changed, we're stalled
                return False
                
        # Create a prompt for the decider agent
        prompt = f"""
        You are deciding whether to continue processing in the {self.name} agent.
        
        Current iteration: {iteration}/{self.max_iterations}
        
        Context gathered so far:
        {json.dumps(context, indent=2)}
        
        Based on this information, should the agent continue processing (True) or has it 
        gathered sufficient information to stop (False)?
        
        Return True to continue, False to stop.
        """
        
        result = await self.decider_agent.run(prompt)
        self.track_agent_usage(state, result)
        return result.output
        
    def _calculate_state_hash(self, context: Dict[str, Any]) -> str:
        """Calculate a hash of the current context for stall detection."""
        # Encode and hash the context dict to detect if state has changed
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode('utf-8')).hexdigest()
        
    async def process_one_iteration(
        self, 
        state: PipelineState, 
        iteration: int, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process one iteration of the loop."""
        # To be implemented by subclasses
        pass
        
    async def process(self, state: PipelineState) -> T:
        """Process with looping until completion."""
        with self.tracer.start_as_current_span(f"{self.name}_loop") as span:
            context = {}
            iteration = 0
            previous_hash = None
            
            span.set_attribute("max_iterations", self.max_iterations)
            
            # Main loop
            while await self.should_continue(state, iteration, context, previous_hash):
                with self.tracer.start_as_current_span(f"{self.name}_iteration_{iteration}") as iter_span:
                    # Calculate state hash for stall detection
                    previous_hash = self._calculate_state_hash(context)
                    iter_span.set_attribute("iteration", iteration)
                    iter_span.set_attribute("context_hash", previous_hash)
                    
                    # Process iteration
                    iteration_result = await self.process_one_iteration(state, iteration, context)
                    
                    # Update context
                    context.update(iteration_result)
                    iteration += 1
                    
                    # Record state
                    span.set_attribute("completed_iterations", iteration)
            
            # Finalize results
            return self.finalize(state, context)
        
    def finalize(self, state: PipelineState, context: Dict[str, Any]) -> T:
        """Finalize the output after looping completes."""
        # To be implemented by subclasses
        pass
```

## 3. Repository Tools Implementation

### 3.1 Tool Interfaces

```python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Match:
    file: str
    line: int
    content: str

@dataclass
class DiffHunk:
    file: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str

class RepoTools:
    """Tools for interacting with the repository."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
    async def grep_repo(self, pattern: str) -> List[Match]:
        """Search repo for pattern and return matches."""
        # Implementation using subprocess to run grep
        
    async def read_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
        """Read file content, optionally specifying line range."""
        # Implementation to read file content
        
    async def list_files(self, path: Optional[str] = None) -> List[str]:
        """List files in the repo, optionally in a specific path."""
        # Implementation to list files
        
    async def get_diff(self) -> List[DiffHunk]:
        """Get diff hunks from the merge request."""
        # Implementation to get diff
```

### 3.2 Tool Integration with Agents

```python
from pydantic_ai import Agent
from pydantic_ai.tools import Tool

@dataclass
class PipelineAgentDeps:
    """Dependencies for pipeline agents."""
    repo_tools: RepoTools
    state: PipelineState

# Example tools for codebase explorer agent
class CodebaseExplorerAgent(LoopablePipelineAgent[CodebaseContext]):
    
    def __init__(
        self,
        name: str = "codebase_explorer",
        model: str = "openai:gpt-4o",
        max_iterations: int = 5,
    ):
        super().__init__(name, model, max_iterations)
        
        # Create agent with tools
        self.explorer_agent = Agent(model)
        
        # Add tools to agent
        @self.explorer_agent.tool
        async def grep_repo(ctx: PipelineAgentDeps, pattern: str) -> List[Match]:
            """Search the repository for a pattern."""
            return await ctx.repo_tools.grep_repo(pattern)
            
        @self.explorer_agent.tool
        async def read_file(ctx: PipelineAgentDeps, path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
            """Read a file from the repository."""
            return await ctx.repo_tools.read_file(path, start, end)
            
        @self.explorer_agent.tool
        async def list_files(ctx: PipelineAgentDeps, path: Optional[str] = None) -> List[str]:
            """List files in the repository."""
            return await ctx.repo_tools.list_files(path)
```

## 4. Pipeline Module Implementations

### 4.1 Ticket Identifier Agent

```python
class TicketIdentifierAgent(PipelineAgent[JiraTicket]):
    """Extract Jira ticket ID from MR metadata using LLM and fetch ticket data via API."""
    
    def __init__(
        self, 
        name: str = "ticket_identifier",
        model: str = "openai:gpt-4o",
        model_settings: Optional[Dict[str, Any]] = None,
        jira_api_url: Optional[str] = None,
        jira_api_token: Optional[str] = None,
    ):
        # Set up appropriate system prompt for ticket ID extraction
        system_prompt = """You are a Ticket Identifier Agent tasked with extracting Jira ticket IDs from merge request metadata.
        Your task is to identify Jira ticket IDs (like PROJECT-123) from the MR title, description, or branch name.
        
        Focus on accuracy. Only return a valid Jira ticket ID if you're confident it exists."""
        
        instructions = """Analyze the MR metadata to extract any Jira ticket references.
        Look for patterns like PROJECT-123 in the title, description, branch name, or commit messages.
        If multiple ticket IDs are found, prioritize the one most prominently featured.
        Return only the ticket ID without any additional text."""
        
        super().__init__(
            name=name,
            model=model,
            model_settings=model_settings,
            system_prompt=system_prompt,
            instructions=instructions,
        )
        
        # Create identifier agent for extracting ticket ID
        self.identifier_agent = self.create_agent(output_type=str)
        
        # Store Jira API configuration
        self.jira_api_url = jira_api_url
        self.jira_api_token = jira_api_token
    
    async def process(self, state: PipelineState) -> JiraTicket:
        # Extract Jira ticket ID from MR metadata using LLM
        ticket_id = await self._extract_ticket_id(state)
        
        # Fetch ticket data using Jira API
        ticket_data = await self._fetch_jira_ticket(ticket_id)
        
        # Create and return JiraTicket object
        ticket = JiraTicket(
            id=ticket_id,
            title=ticket_data.get("title", "Unknown Title"),
            description=ticket_data.get("description", "No description available"),
            acceptance_criteria=ticket_data.get("acceptance_criteria"),
            comments=ticket_data.get("comments")
        )
        
        return ticket
    
    async def _extract_ticket_id(self, state: PipelineState) -> str:
        """Extract Jira ticket ID from MR metadata using LLM."""
        # Create prompt for ticket ID extraction
        prompt = f"""
        # Merge Request Metadata
        
        ## Title
        {state.mr_title or 'No title provided'}
        
        ## Description
        {state.mr_description or 'No description provided'}
        
        ## Branch
        {state.current_branch or 'No branch information'}
        
        Extract the Jira ticket ID from this merge request information.
        Look for patterns like PROJECT-123 in the title, description, or branch name.
        Return only the ticket ID without any additional text.
        If no Jira ticket ID is found, return "UNKNOWN".
        """
        
        # Run identifier agent to extract ticket ID
        result = await self.identifier_agent.run(prompt)
        self.track_agent_usage(state, result)
        
        # Extract and normalize the ticket ID
        ticket_id = result.output.strip()
        
        # Fallback if no ticket ID found
        if ticket_id == "UNKNOWN" or not ticket_id:
            ticket_id = "UNKNOWN"
            state.errors[f"{self.name}_extract_id"] = "No Jira ticket ID found in MR metadata"
            
        return ticket_id
        
    async def _fetch_jira_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """Fetch Jira ticket data via API."""
        if ticket_id == "UNKNOWN":
            # Return empty data if no ticket ID found
            return {}
            
        try:
            # Implementation using Jira API client or direct HTTP requests
            # This would be replaced with actual API call in a real implementation
            
            # Example with httpx:
            # async with httpx.AsyncClient() as client:
            #     response = await client.get(
            #         f"{self.jira_api_url}/rest/api/2/issue/{ticket_id}",
            #         headers={"Authorization": f"Bearer {self.jira_api_token}"},
            #     )
            #     response.raise_for_status()
            #     data = response.json()
            
            # For now, just return mock data for demonstration
            # In the real implementation, this would parse the API response
            return {
                "title": f"Sample ticket {ticket_id}",
                "description": "This is a sample ticket description",
                "acceptance_criteria": ["Criteria 1", "Criteria 2"],
                "comments": [
                    {"author": "user1", "content": "Comment 1"},
                    {"author": "user2", "content": "Comment 2"}
                ]
            }
            
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"Failed to fetch Jira data for {ticket_id}: {str(e)}"
            logging.error(error_message)
            return {"error": error_message}

```

### 4.2 Codebase Explorer Agent

```python
class CodebaseExplorerAgent(LoopablePipelineAgent[CodebaseContext]):
    """Explore codebase to understand architecture relevant to ticket."""
    
    def __init__(
        self, 
        name: str = "codebase_explorer",
        model: str = "openai:gpt-4o",
        max_iterations: int = 5,
        model_settings: Optional[Dict[str, Any]] = None,
        stall_detection: bool = True,
    ):
        # Set up appropriate system prompt for codebase exploration
        system_prompt = """You are a Codebase Explorer Agent tasked with understanding the architecture 
        and conventions of a codebase as they relate to a specific Jira ticket. 
        
        Your task is to:
        1. Explore the repository to understand its structure
        2. Identify files and components relevant to the ticket
        3. Document key architectural patterns and design decisions
        4. Find examples of project conventions that should be followed
        
        Use the available tools to explore systematically and build a comprehensive understanding."""
        
        instructions = """Focus on understanding the overall architecture first before diving into 
        specific implementations. Look for patterns in the code, identify key abstractions, and pay 
        close attention to how components interact. Be thorough but prioritize information most relevant 
        to implementing the requirements in the Jira ticket."""
        
        super().__init__(
            name=name,
            model=model,
            max_iterations=max_iterations,
            model_settings=model_settings,
            system_prompt=system_prompt,
            instructions=instructions,
            stall_detection=stall_detection,
        )
        
        # Create explorer agent with tools for repo exploration
        self.explorer_agent = self.create_agent()
        
        # Register tools
        @self.explorer_agent.tool
        async def grep_repo(ctx: PipelineAgentDeps, pattern: str) -> List[Match]:
            """Search the repository for a pattern."""
            return await ctx.repo_tools.grep_repo(pattern)
            
        @self.explorer_agent.tool
        async def read_file(ctx: PipelineAgentDeps, path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
            """Read a file from the repository."""
            return await ctx.repo_tools.read_file(path, start, end)
            
        @self.explorer_agent.tool
        async def list_files(ctx: PipelineAgentDeps, path: Optional[str] = None) -> List[str]:
            """List files in the repository."""
            return await ctx.repo_tools.list_files(path)
            
        @self.explorer_agent.tool
        async def add_architecture_note(ctx: PipelineAgentDeps, note: str) -> bool:
            """Add an important architectural note to the context."""
            if "architecture_notes" not in ctx.state.shared_context:
                ctx.state.shared_context["architecture_notes"] = []
            ctx.state.shared_context["architecture_notes"].append(note)
            return True
            
        @self.explorer_agent.tool
        async def add_code_snippet(ctx: PipelineAgentDeps, file_path: str, snippet: str, description: str) -> bool:
            """Store an important code snippet with description."""
            if "key_code_snippets" not in ctx.state.shared_context:
                ctx.state.shared_context["key_code_snippets"] = {}
            ctx.state.shared_context["key_code_snippets"][file_path] = {
                "code": snippet,
                "description": description
            }
            return True
            
        @self.explorer_agent.tool  
        async def add_convention(ctx: PipelineAgentDeps, convention: str, example: str) -> bool:
            """Record a coding convention with example."""
            if "observed_conventions" not in ctx.state.shared_context:
                ctx.state.shared_context["observed_conventions"] = []
            ctx.state.shared_context["observed_conventions"].append({
                "convention": convention,
                "example": example
            })
            return True
    
    def _create_exploration_prompt(self, state: PipelineState, context: Dict[str, Any], iteration: int) -> str:
        """Create exploration prompt based on current context and state."""
        # Initialize prompt with ticket information
        prompt = f"""
        # Exploration Task

        ## Jira Ticket Information
        Ticket ID: {state.jira_ticket.id if state.jira_ticket else 'Not available'}
        Title: {state.jira_ticket.title if state.jira_ticket else state.mr_title or 'Not available'}
        
        Description:
        {state.jira_ticket.description if state.jira_ticket else state.mr_description or 'Not available'}
        
        ## Current Understanding
        Iteration: {iteration + 1}/{self.max_iterations}
        
        Architecture Notes: {len(context.get('architecture_notes', []))} notes collected
        Code Snippets: {len(context.get('key_code_snippets', {}))} snippets stored
        Conventions: {len(context.get('observed_conventions', []))} conventions identified
        """
        
        # Add guidance based on iteration number
        if iteration == 0:
            prompt += """
            ## First Iteration Focus:
            Start by understanding the overall repository structure. Use list_files to browse directories
            and identify key areas. Look for configuration files, README files, and entry points that can
            help understand the project organization.
            """
        elif iteration == 1:
            prompt += """
            ## Second Iteration Focus:
            Now that you have a general understanding of the repository, identify files and components
            most likely related to the ticket. Search for key terms from the ticket using grep_repo.
            """
        elif iteration == 2:
            prompt += """
            ## Third Iteration Focus:
            Examine specific implementations in detail. Look at relevant class definitions, interfaces,
            and existing patterns that would be important for implementing the ticket requirements.
            """
        else:
            prompt += """
            ## Further Exploration:
            Fill any remaining gaps in your understanding. Explore any aspects not yet covered that would
            be important for implementing the ticket. Focus on identifying conventions and patterns that
            should be followed.
            """
            
        # Add specific direction based on what we've discovered so far
        prompt += """
        ## Investigation Plan
        
        Based on what you've learned so far, what specific aspects of the codebase should you explore next?
        Use the tools available to continue your investigation systematically.
        
        You should:
        1. Formulate specific questions to answer in this iteration
        2. Use grep_repo, read_file, and list_files tools to find answers
        3. Document your findings using add_architecture_note, add_code_snippet, and add_convention tools
        """
        
        return prompt
    
    async def process_one_iteration(
        self, 
        state: PipelineState,
        iteration: int, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Create prompt based on gathered context and iteration
        prompt = self._create_exploration_prompt(state, context, iteration)
        
        # Prepare state context
        if "architecture_notes" not in context:
            context["architecture_notes"] = []
        if "key_code_snippets" not in context:
            context["key_code_snippets"] = {}
        if "observed_conventions" not in context:
            context["observed_conventions"] = []
            
        # Create shared context for dependencies
        shared_context = {
            "architecture_notes": context["architecture_notes"],
            "key_code_snippets": context["key_code_snippets"],
            "observed_conventions": context["observed_conventions"],
        }
        
        # Run agent with tools
        deps = PipelineAgentDeps(
            repo_tools=RepoTools(state.repo_path),
            state=type('SharedContext', (), {'shared_context': shared_context})
        )
        
        # Run exploration with tools
        exploration_result = await self.explorer_agent.run(prompt, deps=deps)
        self.track_agent_usage(state, exploration_result)
        
        # Collect new information from shared context
        new_context = {
            "architecture_notes": shared_context["architecture_notes"],
            "key_code_snippets": shared_context["key_code_snippets"],
            "observed_conventions": shared_context["observed_conventions"],
        }
        
        return new_context
        
    def finalize(self, state: PipelineState, context: Dict[str, Any]) -> CodebaseContext:
        """Create structured CodebaseContext from exploration results."""
        # Convert the key_code_snippets dict format to the expected format
        key_snippets = {}
        for file_path, details in context.get("key_code_snippets", {}).items():
            key_snippets[file_path] = details["code"]
        
        # Extract conventions as simple strings
        conventions = []
        for conv_item in context.get("observed_conventions", []):
            conventions.append(f"{conv_item['convention']} - Example: {conv_item['example']}")
            
        # Create and return final context
        return CodebaseContext(
            architecture_notes=context.get("architecture_notes", []),
            key_code_snippets=key_snippets,
            observed_conventions=conventions
        )
```

### 4.3 Implementation Plan Generator Agents

```python
import asyncio

class ImplementationPlanGenerator(PipelineAgent[ImplementationPlan]):
    """Generate an implementation plan for the ticket."""
    
    def __init__(
        self,
        name: str,
        model: str = "openai:gpt-4o",
        plan_name: str = "Default Plan",
    ):
        super().__init__(name, model)
        self.plan_name = plan_name
        self.planner_agent = Agent(model, output_type=Dict[str, Any])
        
    async def process(self, state: PipelineState) -> ImplementationPlan:
        # Create prompt for planner
        prompt = self._create_planner_prompt(state)
        
        # Run planner agent
        plan_result = await self.planner_agent.run(prompt)
        plan_data = plan_result.output
        
        # Create implementation plan
        plan = ImplementationPlan(
            plan_name=self.plan_name,
            summary=plan_data["summary"],
            steps=plan_data["steps"],
            files_touched=plan_data["files_touched"],
            pros=plan_data["pros"],
            cons=plan_data["cons"],
            estimated_effort=plan_data["estimated_effort"]
        )
        
        return plan

class ParallelPlanGenerators(PipelineAgent[List[ImplementationPlan]]):
    """Run multiple plan generators in parallel."""
    
    async def process(self, state: PipelineState) -> List[ImplementationPlan]:
        # Create three plan generators with different perspectives
        conservative_planner = ImplementationPlanGenerator(
            "conservative_planner",
            plan_name="Conservative Approach"
        )
        
        innovative_planner = ImplementationPlanGenerator(
            "innovative_planner",
            plan_name="Innovative Approach"
        )
        
        pragmatic_planner = ImplementationPlanGenerator(
            "pragmatic_planner",
            plan_name="Pragmatic Approach"
        )
        
        # Run planners in parallel
        plans = await asyncio.gather(
            conservative_planner.process(state),
            innovative_planner.process(state),
            pragmatic_planner.process(state)
        )
        
        return plans
```

### 4.4 Diff Reviewer Agent

```python
class DiffReviewerAgent(LoopablePipelineAgent[DiffReview]):
    """Review diff against implementation plans and codebase context."""
    
    async def process_one_iteration(
        self,
        state: PipelineState,
        iteration: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Get diff if not already in context
        if "diff_hunks" not in context:
            repo_tools = RepoTools(state.repo_path)
            diff_hunks = await repo_tools.get_diff()
            context["diff_hunks"] = diff_hunks
            
        # Create review prompt for current iteration focus
        if iteration == 0:
            # First iteration: Focus on architectural soundness
            prompt = self._create_architecture_review_prompt(state, context)
            result_key = "architectural_feedback"
        elif iteration == 1:
            # Second iteration: Focus on completion
            prompt = self._create_completion_review_prompt(state, context)
            result_key = "completion_feedback"
        elif iteration == 2:
            # Third iteration: Focus on test coverage
            prompt = self._create_test_coverage_review_prompt(state, context)
            result_key = "test_coverage_feedback"
        else:
            # Final iteration: Focus on style
            prompt = self._create_style_review_prompt(state, context)
            result_key = "style_feedback"
            
        # Run reviewer agent
        reviewer_result = await self.reviewer_agent.run(prompt)
        
        # Extract feedback
        feedback = self._extract_feedback(reviewer_result.messages)
        return {result_key: feedback}
        
    def finalize(self, state: PipelineState, context: Dict[str, Any]) -> DiffReview:
        return DiffReview(
            summary=context.get("summary", "No summary provided"),
            architectural_feedback=context.get("architectural_feedback", []),
            completion_feedback=context.get("completion_feedback", []),
            test_coverage_feedback=context.get("test_coverage_feedback", []),
            style_feedback=context.get("style_feedback", []),
            concerns=context.get("concerns", [])
        )
```

### 4.5 Verifier Agent

```python
class VerifierAgent(PipelineAgent[VerifierAnalysis]):
    """Independently verify MR as a standalone solution."""
    
    async def process(self, state: PipelineState) -> VerifierAnalysis:
        # Create verifier agent
        verifier_agent = Agent(self.model, output_type=Dict[str, Any])
        
        # Add tools for repository inspection
        @verifier_agent.tool
        async def grep_repo(pattern: str) -> List[Match]:
            """Search the repository for a pattern."""
            repo_tools = RepoTools(state.repo_path)
            return await repo_tools.grep_repo(pattern)
            
        @verifier_agent.tool
        async def read_file(path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
            """Read a file from the repository."""
            repo_tools = RepoTools(state.repo_path)
            return await repo_tools.read_file(path, start, end)
            
        @verifier_agent.tool
        async def get_diff() -> List[DiffHunk]:
            """Get the diff of the merge request."""
            repo_tools = RepoTools(state.repo_path)
            return await repo_tools.get_diff()
        
        # Create verifier prompt with just the Jira ticket and no plan context
        prompt = self._create_verifier_prompt(state)
        
        # Run verifier agent
        verifier_result = await verifier_agent.run(prompt)
        analysis = verifier_result.output
        
        return VerifierAnalysis(
            summary=analysis["summary"],
            is_valid_solution=analysis["is_valid_solution"],
            strengths=analysis["strengths"],
            improvement_areas=analysis["improvement_areas"],
            alternative_suggestions=analysis.get("alternative_suggestions")
        )
```

### 4.6 Synthesizer Agent

```python
class SynthesizerAgent(PipelineAgent[str]):
    """Synthesize all prior agent outputs into a coherent report."""
    
    async def process(self, state: PipelineState) -> str:
        # Create synthesizer agent
        synthesizer_agent = Agent(self.model, output_type=str)
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(state)
        
        # Run synthesizer agent
        result = await synthesizer_agent.run(prompt)
        
        return result.output
```

### 4.7 Final Gate Agent

```python
class FinalGateAgent(PipelineAgent[PipelineVerdict]):
    """Determine final verdict on the merge request."""
    
    async def process(self, state: PipelineState) -> PipelineVerdict:
        # Create final gate agent
        gate_agent = Agent(self.model, output_type=str)
        
        # Create verdict prompt
        prompt = f"""
        As the Final Gate Agent, your job is to determine the final verdict for this merge request.
        
        Based on the synthesized feedback:
        
        {state.synthesized_feedback}
        
        Determine if this MR should:
        - PASS: The implementation is acceptable to merge
        - SOFT_FAIL: The implementation is mergeable but has suggestions for improvement
        - HARD_FAIL: The implementation has architectural or logic problems that must be fixed
        
        Return only one of: "PASS", "SOFT_FAIL", or "HARD_FAIL"
        """
        
        # Run gate agent
        result = await gate_agent.run(prompt)
        verdict = result.output.strip()
        
        # Validate verdict
        if verdict not in ("PASS", "SOFT_FAIL", "HARD_FAIL"):
            raise ValueError(f"Invalid verdict: {verdict}")
            
        return verdict
```

## 5. Pipeline Orchestration

### 5.1 Main Pipeline Orchestrator

```python
import os

class PipelineOrchestrator:
    """Orchestrates the entire pipeline."""
    
    async def run_pipeline(
        self,
        repo_path: str,
        current_branch: str,
        mr_title: str,
        mr_description: str,
    ) -> Dict[str, Any]:
        # Initialize pipeline state
        state = PipelineState(
            repo_path=repo_path,
            current_branch=current_branch,
            mr_title=mr_title,
            mr_description=mr_description,
        )
        
        # Run ticket identifier to extract Jira ID using LLM and fetch details via API
        ticket_identifier = TicketIdentifierAgent(
            jira_api_url=os.environ.get("JIRA_API_URL"),
            jira_api_token=os.environ.get("JIRA_API_TOKEN")
        )
        state.jira_ticket = await ticket_identifier.run_with_timing(state)
        
        # Run codebase explorer
        explorer = CodebaseExplorerAgent("codebase_explorer")
        state.codebase_context = await explorer.run_with_timing(state)
        
        # Run implementation planners in parallel
        plan_generators = ParallelPlanGenerators("implementation_planners")
        state.implementation_plans = await plan_generators.run_with_timing(state)
        
        # Run diff reviewer
        diff_reviewer = DiffReviewerAgent("diff_reviewer")
        state.diff_review = await diff_reviewer.run_with_timing(state)
        
        # Run verifier
        verifier = VerifierAgent("verifier")
        state.verifier_analysis = await verifier.run_with_timing(state)
        
        # Run synthesizer
        synthesizer = SynthesizerAgent("synthesizer")
        state.synthesized_feedback = await synthesizer.run_with_timing(state)
        
        # Run final gate
        gate = FinalGateAgent("final_gate")
        state.verdict = await gate.run_with_timing(state)
        
        # Return final state
        return {
            "session_id": state.session_id,
            "verdict": state.verdict,
            "feedback": state.synthesized_feedback,
            "metadata": {
                "durations": state.stage_durations,
                "token_usage": state.token_usage,
                "errors": state.errors,
            }
        }
```

### 5.2 CLI Interface

```python
import asyncio
import argparse
import json
import os
import sys

async def main(args):
    orchestrator = PipelineOrchestrator()
    
    try:
        result = await orchestrator.run_pipeline(
            repo_path=args.repo_path,
            current_branch=args.branch,
            mr_title=args.mr_title,
            mr_description=args.mr_description or "",
        )
        
        # Print result
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Verdict: {result['verdict']}")
            print("\nFeedback:")
            print(result['feedback'])
            
        # Set exit code based on verdict
        if result['verdict'] == "HARD_FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Agent for code review")
    parser.add_argument("--repo-path", required=True, help="Path to the repository")
    parser.add_argument("--branch", required=True, help="Current branch name")
    parser.add_argument("--mr-title", required=True, help="Merge request title")
    parser.add_argument("--mr-description", help="Merge request description")
    parser.add_argument("--jira-api-url", help="Jira API URL (can also use JIRA_API_URL env var)")
    parser.add_argument("--jira-api-token", help="Jira API token (can also use JIRA_API_TOKEN env var)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Set environment variables from CLI arguments if provided
    if args.jira_api_url:
        os.environ["JIRA_API_URL"] = args.jira_api_url
    if args.jira_api_token:
        os.environ["JIRA_API_TOKEN"] = args.jira_api_token
    
    asyncio.run(main(args))
```

## 6. Configuration and CI Integration

### 6.1 Docker and Kubernetes Configuration

```yaml
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git

# Copy application code
COPY . /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Set default command
ENTRYPOINT ["python", "-m", "agents.pipeline_agent"]
```

```yaml
# kubernetes.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pipeline-agent-job
spec:
  template:
    spec:
      containers:
      - name: pipeline-agent
        image: pipeline-agent:latest
        args:
          - "--repo-path=/repo"
          - "--branch=$(BRANCH)"
          - "--mr-title=$(MR_TITLE)"
          - "--mr-description=$(MR_DESCRIPTION)"
          - "--json"
        env:
          - name: BRANCH
            value: "{{.Branch}}"
          - name: MR_TITLE
            value: "{{.MergeRequest.Title}}"
          - name: MR_DESCRIPTION
            value: "{{.MergeRequest.Description}}"
        volumeMounts:
          - name: repo-volume
            mountPath: /repo
      volumes:
        - name: repo-volume
          emptyDir: {}
      initContainers:
        - name: git-clone
          image: alpine/git
          command:
            - git
            - clone
            - "{{.Repository.CloneURL}}"
            - /repo
          volumeMounts:
            - name: repo-volume
              mountPath: /repo
      restartPolicy: Never
  backoffLimit: 1
```

### 6.2 GitLab CI Integration

```yaml
# .gitlab-ci.yml
stages:
  - test
  - code-review

code-review:
  stage: code-review
  image: 
    name: pipeline-agent:latest
    entrypoint: [""]
  script:
    - git clone $CI_REPOSITORY_URL repo
    - cd repo
    - git checkout $CI_COMMIT_SHA
    - python -m agents.pipeline_agent --repo-path=. --branch=$CI_COMMIT_REF_NAME --mr-title="$CI_MERGE_REQUEST_TITLE" --mr-description="$CI_MERGE_REQUEST_DESCRIPTION" --json > review.json
    - cat review.json | jq -r '.feedback' > review.md
    - echo "Pipeline verdict: $(cat review.json | jq -r '.verdict')"
  artifacts:
    paths:
      - review.md
      - review.json
    reports:
      junit: review.xml
  only:
    - merge_requests
```

## 7. Implementation Roadmap

### Phase 1: Core Framework (Week 1)
- [ ] Set up project structure
- [ ] Implement PipelineState and base PipelineAgent classes
- [ ] Implement repository tools
- [ ] Create basic CLI interface for testing
- [ ] Test on simple repository examples

### Phase 2: MVP Implementation (Week 2)
- [ ] Implement TicketIdentifierAgent
- [ ] Implement CodebaseExplorerAgent with basic loop capability
- [ ] Implement simple ImplementationPlanGenerator
- [ ] Create basic DiffReviewerAgent
- [ ] Implement simple SynthesizerAgent
- [ ] End-to-end testing with minimal pipeline

### Phase 3: Enhanced Functionality (Week 3)
- [ ] Implement full loopable pattern for explorer and reviewer
- [ ] Add parallel plan generators
- [ ] Implement VerifierAgent
- [ ] Enhance SynthesizerAgent with structured output
- [ ] Implement FinalGateAgent
- [ ] Add token tracking and budget controls

### Phase 4: Integration and Optimization (Week 4)
- [ ] Create Docker container
- [ ] Set up Kubernetes configuration
- [ ] Implement CI pipeline integration
- [ ] Add timeout handling and error recovery
- [ ] Optimize performance and token usage
- [ ] Final system testing

## 8. Testing Strategy

### 8.1 Unit Testing
- Test each agent independently with mocked inputs and tools
- Test agent looping with deterministic outputs
- Verify state object validation and serialization

### 8.2 Integration Testing
- Test interactions between adjacent agents
- Verify state updates through pipeline stages
- Test parallel execution of plan generators

### 8.3 System Testing
- End-to-end tests on real repositories
- Test with various types of merge requests
- Test error recovery and timeout handling

### 8.4 Performance Testing
- Measure token usage and timing information
- Optimize prompt design for efficiency
- Test with various repository sizes

## 9. Prompt Engineering Guidelines

For each agent, prompts should follow these guidelines:

1. **Clear Role Definition**: Start with explicit agent role
2. **Context Sharing**: Include relevant context from pipeline state
3. **Specific Instructions**: Detail exactly what the agent should analyze
4. **Output Structure**: Define expected output format
5. **Reasoning Process**: Encourage step-by-step reasoning
6. **Tool Usage**: Explain available tools and when to use them
7. **Stopping Criteria**: Define when the agent should stop iteration

## 10. Limitations and Future Enhancements

### Current Limitations
- No support for building or running tests
- Limited to static code analysis
- Requires structured Jira ticket data
- May generate false positives for architectural concerns

### Future Enhancements
- Cache exploration results between MRs on the same codebase
- Support for .agentignore file to skip irrelevant directories
- More configurable verdict criteria based on project needs
- Support for linting and style guide enforcement
- Integration with other CI systems beyond GitLab

## 11. Advanced Pydantic AI Features

### 11.1 Instrumentation and Observability

The implementation leverages Pydantic AI's built-in observability features:

```python
# Enable instrumentation for all agents
Agent.instrument_all(True)

# Configure instrumented model
from pydantic_ai.models.instrumented import InstrumentationSettings

instrumentation_settings = InstrumentationSettings(
    include_prompt=True,
    include_system_prompt=True,
    include_messages=True,
    span_prefix="pipeline_agent.",
)

# Create agent with instrumentation
agent = Agent(
    "openai:gpt-4o",
    instrument=instrumentation_settings,
)
```

### 11.2 Using Message History

The implementation uses Pydantic AI's message history capabilities to maintain conversation context:

```python
# Run agent with previous message history
previous_messages = state.message_history.get(agent_name, [])
result = await agent.run(
    prompt,
    message_history=[ModelMessage.model_validate(msg) for msg in previous_messages],
)

# Update message history with new messages
state.message_history[agent_name] = [
    self._message_to_dict(msg) for msg in result.messages
]
```

### 11.3 Stream Processing

For agents where real-time feedback is important, we can use Pydantic AI's streaming capabilities:

```python
async def process_with_streaming(self, state: PipelineState, prompt: str) -> str:
    """Process with streaming for real-time feedback."""
    async with self.agent.run_stream(prompt) as stream:
        async for chunk in stream:
            # Process chunk in real-time if needed
            if hasattr(chunk, "delta"):
                # This is a content chunk
                print(f"Received chunk: {chunk.delta}")
            
    # Return final result
    return stream.result.output
```

### 11.4 Agent Iteration with Context

Using Agent's `iter` method for finer-grained control over agent execution:

```python
async def process_with_iteration(self, state: PipelineState, prompt: str) -> Any:
    """Process with iteration for maximum control over execution flow."""
    async with self.agent.iter(prompt) as agent_run:
        # Iterate through each node execution
        async for node in agent_run:
            # We can inspect each node as it's processed
            if isinstance(node, ModelRequestNode):
                # Log or process the model request
                logger.debug(f"Model request: {node.request}")
            elif isinstance(node, CallToolsNode):
                # Process tool calls
                logger.debug(f"Model response: {node.model_response}")
                
        # Get the final result
        return agent_run.result.output
```

### 11.5 Capturing Run Messages

When detailed inspection of agent execution is needed:

```python
from pydantic_ai import capture_run_messages

async def process_with_message_capture(self, state: PipelineState, prompt: str) -> Tuple[Any, List[ModelMessage]]:
    async with capture_run_messages() as messages:
        result = await self.agent.run(prompt)
        
    # Messages contains all the interaction details
    return result.output, messages
```

### 11.6 Agent Output Validation

We can implement output validators to ensure agent responses meet quality criteria:

```python
from pydantic_ai import Agent, ModelRetry

class CodeReviewAgent(LoopablePipelineAgent):
    """Agent for reviewing code with validation."""
    
    def __init__(self, name: str = "code_reviewer", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        
        # Create base agent
        self.review_agent = self.create_agent()
        
        # Add output validator to ensure thorough reviews
        @self.review_agent.output_validator
        def validate_review_thoroughness(review: DiffReview) -> DiffReview:
            """Ensure the review is thorough and actionable."""
            if len(review.architectural_feedback) < 2:
                raise ModelRetry("Review needs more architectural feedback")
                
            if not any(feedback for feedback in review.style_feedback):
                raise ModelRetry("Review should include style feedback")
                
            return review
```

### 11.7 Agent Overrides for Testing

We can use Agent's override functionality to simplify testing:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_model():
    """Mock model that returns predefined responses for testing."""
    model_mock = AsyncMock()
    model_mock.model_name = "test-model"
    model_mock.request.return_value = MockModelResponse(...)
    return model_mock

def test_pipeline_agent(mock_model):
    """Test pipeline agent with mocked model."""
    agent = CodeReviewAgent()
    
    # Override the model for testing
    with agent.review_agent.override(model=mock_model):
        # Run the agent - it will use the mock model
        result = agent.run_sync("Test prompt")
        
    # Assertions about the result
    assert result.output.summary == "Expected summary"
```

### 11.8 Rich System Prompt Handling

Leverage Pydantic AI's dynamic system prompt capabilities:

```python
class PipelineAgent(Generic[T], ABC):
    """Base class for all pipeline agents."""
    
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        
        # Register dynamic system prompt that changes based on context
        @self.agent.system_prompt(dynamic=True)
        async def build_system_prompt(ctx: RunContext) -> str:
            """Build dynamic system prompt based on the current pipeline state."""
            pipeline_state = ctx.deps
            
            # Build different prompts based on context
            if pipeline_state.codebase_context and pipeline_state.jira_ticket:
                # Full context available
                return f"""You are a specialized agent working on ticket {pipeline_state.jira_ticket.id}.
                Your task is to {self.task_description}.
                
                Codebase architecture notes:
                {'\n'.join(pipeline_state.codebase_context.architecture_notes)}
                
                Observed conventions:
                {'\n'.join(pipeline_state.codebase_context.observed_conventions)}
                """
            else:
                # Limited context
                return f"""You are a specialized agent with the role of {self.name}.
                Your task is to {self.task_description}.
                """
```

## 12. Reliability and Error Handling

### 12.1 Request Rate Limiting

Implementing rate limiting to prevent overloading the API:

```python
import asyncio
import time
from datetime import datetime, timedelta

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire a request slot, waiting if necessary."""
        async with self.lock:
            now = datetime.now()
            # Remove expired timestamps
            self.request_times = [t for t in self.request_times if t > now - timedelta(minutes=1)]
            
            if len(self.request_times) >= self.max_requests:
                # Calculate wait time
                oldest = min(self.request_times)
                wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
                await asyncio.sleep(wait_time)
                
            # Add current timestamp
            self.request_times.append(now)
            
class PipelineAgentWithRateLimit(PipelineAgent[T]):
    """Pipeline agent with rate limiting."""
    
    _rate_limiter = RateLimiter(max_requests_per_minute=60)
    
    async def _make_model_request(self, *args, **kwargs):
        """Make model request with rate limiting."""
        await self._rate_limiter.acquire()
        return await super()._make_model_request(*args, **kwargs)
```

### 12.2 Graceful Error Recovery

Enhanced error handling with recovery strategies:

```python
class PipelineWithErrorRecovery:
    """Pipeline with error recovery strategies."""
    
    async def run_with_recovery(self, repo_path: str, current_branch: str, mr_title: str, mr_description: str) -> Dict[str, Any]:
        """Run pipeline with error recovery."""
        state = PipelineState(
            repo_path=repo_path,
            current_branch=current_branch,
            mr_title=mr_title,
            mr_description=mr_description,
        )
        
        # Build the pipeline stage sequence
        stages = [
            ("ticket_identifier", self._run_ticket_identifier),
            ("codebase_explorer", self._run_codebase_explorer),
            ("implementation_planners", self._run_implementation_planners),
            ("diff_reviewer", self._run_diff_reviewer),
            ("verifier", self._run_verifier),
            ("synthesizer", self._run_synthesizer),
            ("final_gate", self._run_final_gate),
        ]
        
        # Run stages with recovery
        for stage_name, stage_func in stages:
            try:
                await stage_func(state)
            except Exception as e:
                # Log error
                state.errors[stage_name] = str(e)
                
                # Apply recovery strategy
                success = await self._apply_recovery_strategy(stage_name, state, e)
                if not success and self._is_critical_stage(stage_name):
                    # Critical stage failed, end pipeline with default judgment
                    state.verdict = "HARD_FAIL"
                    state.synthesized_feedback = f"Critical error in {stage_name}: {str(e)}"
                    break
                    
        return self._build_final_result(state)
        
    async def _apply_recovery_strategy(self, stage_name: str, state: PipelineState, error: Exception) -> bool:
        """Apply recovery strategy for a failed stage."""
        # Different recovery strategies based on the stage
        if stage_name == "ticket_identifier":
            # Use MR title/description instead
            if state.mr_title and state.mr_description:
                state.jira_ticket = JiraTicket(
                    id="UNKNOWN",
                    title=state.mr_title,
                    description=state.mr_description,
                    acceptance_criteria=None,
                    comments=None
                )
                return True
        elif stage_name == "codebase_explorer":
            # Use minimal exploration results
            state.codebase_context = CodebaseContext()
            return True
        elif stage_name == "implementation_planners":
            # Skip planning stage
            state.implementation_plans = []
            return True
            
        # Default strategy: retry once with timeout
        try:
            recovery_agent = Agent(
                "openai:gpt-4o", 
                model_settings={"temperature": 0.5, "timeout": 60},
                retries=1,
            )
            # Simaplified execution of the failed stage
            # Implementation details depend on the specific stage
            return True
        except Exception:
            return False
```

### 12.3 Circuit Breaker Pattern

Implementing a circuit breaker to prevent cascading failures:

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is open, fail fast
    HALF_OPEN = "half_open"  # Testing if service is back to normal

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0
        
    async def execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit is open, fast fail")
                
        try:
            result = await func(*args, **kwargs)
            
            # Success, reset if in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            # Record failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                
            raise e
```

## 13. Conclusion

This implementation plan provides a comprehensive approach to building the Pipeline Agent system. By following a modular, agent-based architecture with shared state and loopable processing, we can create a robust code review system that provides valuable, architecture-aware feedback on merge requests.

The plan leverages the pydantic-ai SDK's Agent framework while adding specialized patterns like the loopable agent and parallel processing that are tailored to the code review domain. By utilizing advanced Pydantic AI features like instrumentation, message history management, streaming, and message capture, we ensure the system is both powerful and observable.

The implementation also includes robust reliability features such as rate limiting, circuit breakers, and graceful error recovery strategies to handle real-world production scenarios. This ensures that the system can operate reliably even when faced with API rate limits, timeouts, or temporary service disruptions.

The result will be a CI-integrated system that can significantly improve code review quality and consistency while reducing manual review burden, with rich insights into the reasoning process of each specialized agent component.