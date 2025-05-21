"""
pipeline_agent_fleshy.py

A more fleshed-out implementation of the Static Agentic Code-Review Pipeline.
This builds on the skeleton to create a more functional implementation with:
- Actual repository tools using subprocess
- LLM-based agents using pydantic-ai
- Basic error handling and recovery
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Tuple, Union

# Pydantic core
from pydantic import BaseModel, Field

# Pydantic AI framework
try:
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import AgentRunResult
except ImportError:
    logging.warning("pydantic_ai not installed - LLM features will be unavailable")
    # Create stub classes for testing without pydantic_ai
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
        
        def tool(self, func):
            return func
    
    class ModelMessage:
        pass
    
    class AgentRunResult:
        pass

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

class JiraTicket(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: Optional[List[str]] = None
    comments: Optional[List[Dict[str, Any]]] = None

class CodebaseContext(BaseModel):
    architecture_notes: List[str] = Field(default_factory=list)
    key_code_snippets: Dict[str, str] = Field(default_factory=dict)
    observed_conventions: List[str] = Field(default_factory=list)

class ImplementationPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    plan_name: str
    summary: str
    steps: List[str] = Field(default_factory=list)
    files_touched: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    estimated_effort: str = "Medium"

class DiffReview(BaseModel):
    summary: str
    architectural_feedback: List[str] = Field(default_factory=list)
    completion_feedback: List[str] = Field(default_factory=list)
    test_coverage_feedback: List[str] = Field(default_factory=list)
    style_feedback: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)

class VerifierAnalysis(BaseModel):
    summary: str
    is_valid_solution: bool
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    alternative_suggestions: Optional[List[str]] = None

class PipelineVerdict(str, Enum):
    """Pipeline verdict: "PASS", "SOFT_FAIL", or "HARD_FAIL"."""
    PASS = "PASS"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"

class PipelineState(BaseModel):
    """Central state object shared across the pipeline."""
    # Repository metadata
    repo_path: str
    current_branch: str
    mr_title: str
    mr_description: str

    # Session metadata
    session_id: str = Field(default_factory=lambda: f"pipeline_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Outputs from each pipeline step (initially None / empty)
    jira_ticket: Optional[JiraTicket] = None
    codebase_context: Optional[CodebaseContext] = None
    implementation_plans: List[ImplementationPlan] = Field(default_factory=list)
    diff_review: Optional[DiffReview] = None
    verifier_analysis: Optional[VerifierAnalysis] = None
    synthesized_feedback: Optional[str] = None
    verdict: Optional[PipelineVerdict] = None
    
    # Message history and execution metadata
    message_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    stage_durations: Dict[str, float] = Field(default_factory=dict)
    token_usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)

# --------------------------------------------------------------------------- #
# Repository tools implementation
# --------------------------------------------------------------------------- #

@dataclass
class Match:
    """Represents a grep/ripgrep match."""
    file: str
    line: int
    content: str

@dataclass
class DiffHunk:
    """Represents a diff hunk."""
    file: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str

class RepoTools:
    """Real implementations of repository access tools via subprocess."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._logger = logging.getLogger("RepoTools")

    async def grep_repo(self, pattern: str) -> List[Match]:
        """Search repo for pattern and return matches using ripgrep."""
        cmd = ["rg", "--line-number", "--no-heading", pattern, self.repo_path]
        
        try:
            # Run ripgrep asynchronously
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode not in (0, 1):  # 1 is "no matches" for ripgrep
                self._logger.warning(f"grep command failed: {stderr.decode()}")
                return []
                
            matches = []
            for line in stdout.decode().splitlines():
                # Parse ripgrep output format: file:line:content
                if not line.strip():
                    continue
                    
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts
                    # Make path relative to repo_path
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    matches.append(Match(
                        file=rel_path,
                        line=int(line_num),
                        content=content.strip()
                    ))
            
            return matches
            
        except Exception as e:
            self._logger.error(f"Error in grep_repo: {str(e)}")
            return []

    async def list_files(self, path: str | None = None) -> List[str]:
        """List files in the repo, optionally in a specific path."""
        search_path = os.path.join(self.repo_path, path or "")
        
        try:
            # Use find command to list files
            cmd = ["find", search_path, "-type", "f", "-not", "-path", "*/\\.*"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                self._logger.warning(f"list_files command failed: {stderr.decode()}")
                return []
                
            # Convert absolute paths to repo-relative paths
            files = []
            for file_path in stdout.decode().splitlines():
                if file_path.strip():
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    files.append(rel_path)
            
            return files
            
        except Exception as e:
            self._logger.error(f"Error in list_files: {str(e)}")
            return []

    async def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read file content, optionally specifying line range."""
        file_path = os.path.join(self.repo_path, path)
        
        try:
            if not os.path.exists(file_path):
                self._logger.warning(f"File not found: {file_path}")
                return ""
                
            # If line range is specified, use sed to extract specific lines
            if start is not None:
                cmd = ["sed", "-n"]
                
                if end is not None:
                    # Range of lines
                    cmd.append(f"{start},{end}p")
                else:
                    # Single line
                    cmd.append(f"{start}p")
                    
                cmd.append(file_path)
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if proc.returncode != 0:
                    self._logger.warning(f"read_file command failed: {stderr.decode()}")
                    return ""
                    
                return stdout.decode()
            else:
                # Read the entire file
                async with asyncio.open_file(file_path, mode="r") as file:
                    content = await file.read()
                    return content
                    
        except Exception as e:
            self._logger.error(f"Error in read_file: {str(e)}")
            return ""

    async def get_diff(self) -> List[DiffHunk]:
        """Get diff hunks from the merge request."""
        try:
            # Assuming we want to diff against the main branch
            # In real scenario, the diff would likely be accessed from the MR API
            cmd = ["git", "-C", self.repo_path, "diff", "main...HEAD"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                self._logger.warning(f"git diff command failed: {stderr.decode()}")
                return []
                
            # Parse the diff output into hunks
            diff_text = stdout.decode()
            return self._parse_diff_hunks(diff_text)
            
        except Exception as e:
            self._logger.error(f"Error in get_diff: {str(e)}")
            return []
            
    def _parse_diff_hunks(self, diff_text: str) -> List[DiffHunk]:
        """Parse git diff output into DiffHunk objects."""
        hunks = []
        current_file = None
        hunk_content = []
        old_start = new_start = old_lines = new_lines = 0
        
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                # Start of a new file diff
                if current_file and hunk_content:
                    # Save previous hunk
                    hunks.append(DiffHunk(
                        file=current_file,
                        old_start=old_start,
                        old_lines=old_lines,
                        new_start=new_start,
                        new_lines=new_lines,
                        content="\n".join(hunk_content)
                    ))
                    hunk_content = []
                
                # Extract filename from diff --git a/path/to/file b/path/to/file
                parts = line.split()
                if len(parts) >= 3:
                    current_file = parts[2][2:]  # Remove 'b/' prefix
                    
            elif line.startswith("@@"):
                # Start of a hunk, format: @@ -old_start,old_lines +new_start,new_lines @@
                if current_file and hunk_content:
                    # Save previous hunk
                    hunks.append(DiffHunk(
                        file=current_file,
                        old_start=old_start,
                        old_lines=old_lines,
                        new_start=new_start,
                        new_lines=new_lines,
                        content="\n".join(hunk_content)
                    ))
                    hunk_content = []
                
                # Parse hunk header
                try:
                    # Extract ranges like "-1,5 +2,6"
                    ranges = line.split("@@")[1].strip()
                    old_range, new_range = ranges.split(" ")
                    
                    # Parse old range "-1,5" or "-1"
                    if "," in old_range:
                        old_start, old_lines = map(int, old_range[1:].split(","))
                    else:
                        old_start = int(old_range[1:])
                        old_lines = 1
                        
                    # Parse new range "+2,6" or "+2"
                    if "," in new_range:
                        new_start, new_lines = map(int, new_range[1:].split(","))
                    else:
                        new_start = int(new_range[1:])
                        new_lines = 1
                        
                except Exception as e:
                    self._logger.warning(f"Failed to parse hunk header '{line}': {e}")
                    old_start = new_start = 1
                    old_lines = new_lines = 0
                
                hunk_content.append(line)
            elif current_file:
                # Add content line to current hunk
                hunk_content.append(line)
        
        # Add the last hunk if there is one
        if current_file and hunk_content:
            hunks.append(DiffHunk(
                file=current_file,
                old_start=old_start,
                old_lines=old_lines,
                new_start=new_start,
                new_lines=new_lines,
                content="\n".join(hunk_content)
            ))
            
        return hunks

# --------------------------------------------------------------------------- #
# Base agent classes
# --------------------------------------------------------------------------- #

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
        self.logger = logging.getLogger(name)
        
    @abstractmethod
    async def process(self, state: PipelineState) -> T:
        """Process the pipeline state and return agent output."""
        pass
        
    async def run(self, state: PipelineState) -> T:
        """Run agent with timing and error tracking."""
        start_time = time.time()
        try:
            # Process state
            result = await self.process(state)
            
            # Track timing
            duration = time.time() - start_time
            state.stage_durations[self.name] = duration
            self.logger.info(f"Agent {self.name} completed in {duration:.2f}s")
            
            return result
        except Exception as e:
            state.errors[self.name] = str(e)
            state.stage_durations[self.name] = time.time() - start_time
            self.logger.error(f"Error in agent {self.name}: {str(e)}")
            raise
    
    def _message_to_dict(self, message: ModelMessage) -> Dict[str, Any]:
        """Convert a ModelMessage to a serializable dictionary."""
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"type": message.__class__.__name__, "content": str(message)}
    
    def track_agent_usage(self, state: PipelineState, result: AgentRunResult[Any]) -> None:
        """Track token usage from agent run result."""
        if not hasattr(result, "usage") or not result.usage:
            return
            
        # Track token usage
        tokens = {
            "total": getattr(result.usage, "total_tokens", 0) or 0,
            "prompt": getattr(result.usage, "request_tokens", 0) or 0,
            "completion": getattr(result.usage, "response_tokens", 0) or 0,
        }
        
        # Store in state token usage
        for key, value in tokens.items():
            state.token_usage.setdefault(self.name, {})
            state.token_usage[self.name].setdefault(key, 0)
            state.token_usage[self.name][key] += value
            
        # Store messages for potential reuse
        if hasattr(result, "messages") and result.messages:
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
                self.logger.info(f"Stalled state detected after {iteration} iterations")
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
        
        try:
            result = await self.decider_agent.run(prompt)
            self.track_agent_usage(state, result)
            return result.output
        except Exception as e:
            self.logger.warning(f"Decider agent failed: {str(e)}, defaulting to stop")
            return False
        
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
        raise NotImplementedError
        
    async def process(self, state: PipelineState) -> T:
        """Process with looping until completion."""
        context = {}
        iteration = 0
        previous_hash = None
        
        self.logger.info(f"Starting loopable agent {self.name}, max iterations: {self.max_iterations}")
        
        # Main loop
        while await self.should_continue(state, iteration, context, previous_hash):
            self.logger.info(f"{self.name} iteration {iteration+1}/{self.max_iterations}")
            
            # Calculate state hash for stall detection
            previous_hash = self._calculate_state_hash(context)
            
            # Process iteration
            iteration_result = await self.process_one_iteration(state, iteration, context)
            
            # Update context
            context.update(iteration_result)
            iteration += 1
        
        self.logger.info(f"{self.name} completed after {iteration} iterations")
        
        # Finalize results
        return self.finalize(state, context)
        
    def finalize(self, state: PipelineState, context: Dict[str, Any]) -> T:
        """Finalize the output after looping completes."""
        # To be implemented by subclasses
        raise NotImplementedError

# --------------------------------------------------------------------------- #
# Agent dependencies
# --------------------------------------------------------------------------- #

@dataclass
class PipelineAgentDeps:
    """Dependencies for pipeline agents."""
    repo_tools: RepoTools
    state: Any  # Shared context

# --------------------------------------------------------------------------- #
# Agent implementations
# --------------------------------------------------------------------------- #

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
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"Failed to extract ticket ID: {str(e)}")
            state.errors[f"{self.name}_extract_id"] = str(e)
            return "UNKNOWN"
        
    async def _fetch_jira_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """Fetch Jira ticket data via API."""
        if ticket_id == "UNKNOWN":
            # Return empty data if no ticket ID found
            return {}
            
        try:
            # TODO: Implement real Jira API client (using httpx or aiohttp)
            # For now, return mock data
            self.logger.info(f"Would fetch ticket data for {ticket_id} from Jira API")
            
            # Mock data for demonstration
            return {
                "title": f"Implement feature for {ticket_id}",
                "description": "This is a sample ticket description for implementing a new feature",
                "acceptance_criteria": [
                    "Feature should handle edge cases correctly",
                    "Unit tests should cover at least 80% of the new code",
                    "Documentation should be updated"
                ],
                "comments": [
                    {"author": "user1", "content": "This is high priority"},
                    {"author": "user2", "content": "Related to the Q2 roadmap"}
                ]
            }
            
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"Failed to fetch Jira data for {ticket_id}: {str(e)}"
            self.logger.error(error_message)
            state.errors[f"{self.name}_fetch_ticket"] = error_message
            return {"error": error_message}

class CodebaseExplorerAgent(LoopablePipelineAgent[CodebaseContext]):
    """Explore codebase to understand architecture relevant to ticket."""
    
    def __init__(
        self, 
        name: str = "codebase_explorer",
        model: str = "openai:gpt-4o",
        max_iterations: int = 3,  # Reduced from 5 for demonstration
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
        
    async def process_one_iteration(
        self, 
        state: PipelineState,
        iteration: int, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Initialize context structures if needed
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
        
        # Create a prompt for exploration based on iteration
        prompt = self._create_exploration_prompt(state, context, iteration)
        
        # Create repo tools
        repo_tools = RepoTools(state.repo_path)
        
        # Register tools for the agent
        @self.explorer_agent.tool
        async def grep_repo(pattern: str) -> List[Match]:
            """Search the repository for a pattern."""
            return await repo_tools.grep_repo(pattern)
            
        @self.explorer_agent.tool
        async def read_file(path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
            """Read a file from the repository."""
            return await repo_tools.read_file(path, start, end)
            
        @self.explorer_agent.tool
        async def list_files(path: Optional[str] = None) -> List[str]:
            """List files in the repository."""
            return await repo_tools.list_files(path)
            
        @self.explorer_agent.tool
        async def add_architecture_note(note: str) -> bool:
            """Add an important architectural note to the context."""
            shared_context["architecture_notes"].append(note)
            return True
            
        @self.explorer_agent.tool
        async def add_code_snippet(file_path: str, snippet: str, description: str) -> bool:
            """Store an important code snippet with description."""
            shared_context["key_code_snippets"][file_path] = {
                "code": snippet,
                "description": description
            }
            return True
            
        @self.explorer_agent.tool  
        async def add_convention(convention: str, example: str) -> bool:
            """Record a coding convention with example."""
            shared_context["observed_conventions"].append({
                "convention": convention,
                "example": example
            })
            return True
            
        try:
            # Run exploration with tools
            exploration_result = await self.explorer_agent.run(prompt)
            self.track_agent_usage(state, exploration_result)
            
            # Return updated context
            return shared_context
            
        except Exception as e:
            self.logger.error(f"Error in exploration iteration {iteration}: {str(e)}")
            state.errors[f"{self.name}_iteration_{iteration}"] = str(e)
            
            # Return original context to avoid losing progress
            return context
    
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
        else:
            prompt += """
            ## Further Exploration:
            Examine specific implementations in detail. Look at relevant class definitions, interfaces,
            and existing patterns that would be important for implementing the ticket requirements.
            """
            
        # Add direction for tools usage
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
        
    def finalize(self, state: PipelineState, context: Dict[str, Any]) -> CodebaseContext:
        """Create structured CodebaseContext from exploration results."""
        # Convert the key_code_snippets dict format to the expected format
        key_snippets = {}
        for file_path, details in context.get("key_code_snippets", {}).items():
            if isinstance(details, dict) and "code" in details:
                key_snippets[file_path] = details["code"]
            else:
                key_snippets[file_path] = str(details)
        
        # Extract conventions as simple strings
        conventions = []
        for conv_item in context.get("observed_conventions", []):
            if isinstance(conv_item, dict):
                conventions.append(f"{conv_item.get('convention', '')} - Example: {conv_item.get('example', '')}")
            else:
                conventions.append(str(conv_item))
            
        # Create and return final context
        return CodebaseContext(
            architecture_notes=context.get("architecture_notes", []),
            key_code_snippets=key_snippets,
            observed_conventions=conventions
        )

class ImplementationPlanGenerator(PipelineAgent[ImplementationPlan]):
    """Generate an implementation plan for the ticket."""
    
    def __init__(
        self,
        name: str,
        model: str = "openai:gpt-4o",
        perspective: str = "Default",
        plan_name: str = "Default Plan",
    ):
        # Set up appropriate system prompt
        system_prompt = f"""You are an Implementation Plan Generator with a {perspective} perspective.
        Your task is to analyze a Jira ticket and codebase context to generate a practical
        implementation plan that follows existing patterns and conventions."""
        
        super().__init__(name, model, system_prompt=system_prompt)
        self.perspective = perspective
        self.plan_name = plan_name
        
        # Define instructions based on perspective
        if perspective.lower() == "conservative":
            self.instructions = """Focus on minimal changes, reliability, and backward compatibility.
            Prefer well-established patterns, avoid introducing new dependencies, and prioritize maintainability."""
        elif perspective.lower() == "innovative":
            self.instructions = """Think creatively about novel approaches that could improve the codebase.
            Consider opportunities for refactoring, new design patterns, or alternative libraries
            that could make the solution more elegant or efficient."""
        elif perspective.lower() == "pragmatic":
            self.instructions = """Balance practical considerations with code quality.
            Aim for the most efficient implementation that meets requirements while maintaining
            reasonable quality standards. Consider trade-offs between time, complexity, and maintainability."""
        else:
            self.instructions = """Generate a balanced implementation plan that meets requirements
            while respecting existing codebase patterns and practices."""
            
        # Create planner agent
        self.planner_agent = self.create_agent(output_type=ImplementationPlan)
    
    async def process(self, state: PipelineState) -> ImplementationPlan:
        # Create prompt for planner
        prompt = self._create_planner_prompt(state)
        
        try:
            # Run planner agent
            plan_result = await self.planner_agent.run(prompt)
            self.track_agent_usage(state, plan_result)
            
            # Ensure plan has the correct name
            plan = plan_result.output
            plan.plan_name = self.plan_name
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error generating implementation plan: {str(e)}")
            state.errors[f"{self.name}_plan_generation"] = str(e)
            
            # Return a minimal plan if there's an error
            return ImplementationPlan(
                plan_name=self.plan_name,
                summary=f"Error generating {self.perspective} plan: {str(e)}",
                steps=["Error occurred during plan generation"],
                pros=["N/A"],
                cons=["Failed to generate complete plan"],
                estimated_effort="Unknown"
            )
    
    def _create_planner_prompt(self, state: PipelineState) -> str:
        """Create a prompt for the implementation planner."""
        # Extract codebase context
        architecture_notes = []
        observed_conventions = []
        
        if state.codebase_context:
            architecture_notes = state.codebase_context.architecture_notes
            observed_conventions = state.codebase_context.observed_conventions
        
        # Build prompt
        prompt = f"""
        # Implementation Planning Task
        
        ## Jira Ticket Information
        Ticket ID: {state.jira_ticket.id if state.jira_ticket else 'Not available'}
        Title: {state.jira_ticket.title if state.jira_ticket else state.mr_title or 'Not available'}
        
        Description:
        {state.jira_ticket.description if state.jira_ticket else state.mr_description or 'Not available'}
        
        {"## Acceptance Criteria\\n" + "\\n".join([f"- {criterion}" for criterion in state.jira_ticket.acceptance_criteria]) if state.jira_ticket and state.jira_ticket.acceptance_criteria else ""}
        
        ## Codebase Context
        
        ### Architecture Notes
        {("\\n".join([f"- {note}" for note in architecture_notes])) if architecture_notes else "No architecture notes available"}
        
        ### Observed Conventions
        {("\\n".join([f"- {conv}" for conv in observed_conventions])) if observed_conventions else "No conventions observed"}
        
        ## Your Task
        
        From a {self.perspective} perspective, create an implementation plan for this ticket that includes:
        
        1. A concise summary of the approach
        2. Specific steps for implementation
        3. Which files you'd expect to change
        4. Pros and cons of this approach
        5. Estimated effort (Low, Medium, High)
        
        Your output should be a valid ImplementationPlan object with fields for all of the above.
        """
        
        return prompt

class ParallelPlanGenerators(PipelineAgent[List[ImplementationPlan]]):
    """Run multiple plan generators in parallel."""
    
    def __init__(
        self, 
        name: str = "implementation_planners",
        model: str = "openai:gpt-4o",
    ):
        super().__init__(name, model)
        
    async def process(self, state: PipelineState) -> List[ImplementationPlan]:
        # Create three plan generators with different perspectives
        conservative_planner = ImplementationPlanGenerator(
            "conservative_planner",
            perspective="Conservative",
            plan_name="Conservative Approach"
        )
        
        innovative_planner = ImplementationPlanGenerator(
            "innovative_planner",
            perspective="Innovative",
            plan_name="Innovative Approach"
        )
        
        pragmatic_planner = ImplementationPlanGenerator(
            "pragmatic_planner",
            perspective="Pragmatic",
            plan_name="Pragmatic Approach"
        )
        
        self.logger.info("Running parallel plan generators")
        
        try:
            # Run planners in parallel
            plans = await asyncio.gather(
                conservative_planner.process(state),
                innovative_planner.process(state),
                pragmatic_planner.process(state)
            )
            
            return plans
            
        except Exception as e:
            self.logger.error(f"Error in parallel plan generation: {str(e)}")
            state.errors[f"{self.name}_parallel"] = str(e)
            
            # Return at least one plan if possible
            plans = []
            for planner_name, planner in [
                ("conservative", conservative_planner),
                ("innovative", innovative_planner),
                ("pragmatic", pragmatic_planner)
            ]:
                try:
                    plan = await planner.process(state)
                    plans.append(plan)
                except Exception as e:
                    self.logger.error(f"Error with {planner_name} planner: {str(e)}")
                    state.errors[f"{self.name}_{planner_name}"] = str(e)
            
            if not plans:
                # If all planners failed, return a fallback plan
                plans = [ImplementationPlan(
                    plan_name="Fallback Plan",
                    summary="This is a fallback plan due to errors in plan generation",
                    steps=["Review the code", "Make minimal necessary changes"],
                    pros=["Minimizes risk"],
                    cons=["May not be optimal"],
                    estimated_effort="Medium"
                )]
                
            return plans

class DiffReviewerAgent(PipelineAgent[DiffReview]):
    """Review diff against implementation plans and codebase context."""
    
    def __init__(
        self, 
        name: str = "diff_reviewer",
        model: str = "openai:gpt-4o",
    ):
        # Set up appropriate system prompt
        system_prompt = """You are a Diff Reviewer Agent tasked with analyzing code changes in a merge request.
        Your job is to provide constructive feedback on the changes, focusing on architectural soundness,
        completeness, test coverage, and coding style."""
        
        super().__init__(name, model, system_prompt=system_prompt)
        
        # Create reviewer agent
        self.reviewer_agent = self.create_agent(output_type=DiffReview)
    
    async def process(self, state: PipelineState) -> DiffReview:
        # Get diff from repository
        repo_tools = RepoTools(state.repo_path)
        diff_hunks = await repo_tools.get_diff()
        
        # Create review prompt
        prompt = self._create_review_prompt(state, diff_hunks)
        
        try:
            # Run reviewer agent
            review_result = await self.reviewer_agent.run(prompt)
            self.track_agent_usage(state, review_result)
            
            return review_result.output
            
        except Exception as e:
            self.logger.error(f"Error in diff review: {str(e)}")
            state.errors[f"{self.name}_review"] = str(e)
            
            # Return a minimal review if there's an error
            return DiffReview(
                summary=f"Error during diff review: {str(e)}",
                architectural_feedback=["Unable to complete architectural review"],
                completion_feedback=["Unable to complete implementation review"],
                test_coverage_feedback=["Unable to complete test coverage review"],
                style_feedback=["Unable to complete style review"],
                concerns=["Review failed due to an error"]
            )
    
    def _create_review_prompt(self, state: PipelineState, diff_hunks: List[DiffHunk]) -> str:
        """Create a prompt for diff review."""
        # Prepare diff content
        diff_content = ""
        for i, hunk in enumerate(diff_hunks[:10]):  # Limit to 10 hunks to avoid token limits
            diff_content += f"\n--- {hunk.file} ---\n{hunk.content}\n"
            
        if len(diff_hunks) > 10:
            diff_content += f"\n... and {len(diff_hunks) - 10} more hunks not shown ...\n"
            
        # Extract implementation plans
        plans_summary = ""
        for i, plan in enumerate(state.implementation_plans):
            plans_summary += f"""
            ## Plan {i+1}: {plan.plan_name}
            Summary: {plan.summary}
            Key steps: {', '.join(plan.steps[:3])}{"..." if len(plan.steps) > 3 else ""}
            """
        
        # Build prompt
        prompt = f"""
        # Diff Review Task
        
        ## Jira Ticket Information
        Ticket ID: {state.jira_ticket.id if state.jira_ticket else 'Not available'}
        Title: {state.jira_ticket.title if state.jira_ticket else state.mr_title or 'Not available'}
        
        Description:
        {state.jira_ticket.description if state.jira_ticket else state.mr_description or 'Not available'}
        
        ## Implementation Plans
        {plans_summary if plans_summary else "No implementation plans available"}
        
        ## Code Changes (Diff)
        {diff_content if diff_content else "No changes found in the diff"}
        
        ## Your Task
        
        Review the changes in the diff against the implementation plans and provide feedback in these areas:
        
        1. Architectural Feedback: Do the changes align with good architectural principles and the codebase patterns?
        2. Completion Feedback: Do the changes fully address the requirements in the ticket?
        3. Test Coverage Feedback: Are there sufficient tests for the changes?
        4. Style Feedback: Do the changes follow coding conventions and style guidelines?
        
        Also note any concerns or potential issues with the implementation.
        
        Provide your review as a DiffReview object with fields for each feedback category.
        """
        
        return prompt

class VerifierAgent(PipelineAgent[VerifierAnalysis]):
    """Independently verify MR as a standalone solution."""
    
    def __init__(
        self, 
        name: str = "verifier",
        model: str = "openai:gpt-4o",
    ):
        # Set up appropriate system prompt
        system_prompt = """You are a Verification Agent tasked with independently assessing whether 
        a merge request properly addresses its requirements. You should view the changes objectively, 
        looking at whether they solve the stated problem correctly and completely."""
        
        super().__init__(name, model, system_prompt=system_prompt)
        
        # Create verifier agent
        self.verifier_agent = self.create_agent(output_type=VerifierAnalysis)
    
    async def process(self, state: PipelineState) -> VerifierAnalysis:
        # Get diff from repository
        repo_tools = RepoTools(state.repo_path)
        diff_hunks = await repo_tools.get_diff()
        
        # Create verification prompt - don't include implementation plans to keep it independent
        prompt = self._create_verifier_prompt(state, diff_hunks)
        
        try:
            # Run verifier agent
            verifier_result = await self.verifier_agent.run(prompt)
            self.track_agent_usage(state, verifier_result)
            
            return verifier_result.output
            
        except Exception as e:
            self.logger.error(f"Error in verification: {str(e)}")
            state.errors[f"{self.name}_verification"] = str(e)
            
            # Return a minimal verification if there's an error
            return VerifierAnalysis(
                summary=f"Error during verification: {str(e)}",
                is_valid_solution=False,
                strengths=["Unable to determine strengths due to error"],
                improvement_areas=["Unable to complete verification"]
            )
    
    def _create_verifier_prompt(self, state: PipelineState, diff_hunks: List[DiffHunk]) -> str:
        """Create a prompt for verification."""
        # Prepare diff content
        diff_content = ""
        for i, hunk in enumerate(diff_hunks[:10]):  # Limit to 10 hunks to avoid token limits
            diff_content += f"\n--- {hunk.file} ---\n{hunk.content}\n"
            
        if len(diff_hunks) > 10:
            diff_content += f"\n... and {len(diff_hunks) - 10} more hunks not shown ...\n"
            
        # Build prompt - intentionally exclude implementation plans to keep analysis independent
        prompt = f"""
        # Verification Task
        
        ## Jira Ticket Information
        Ticket ID: {state.jira_ticket.id if state.jira_ticket else 'Not available'}
        Title: {state.jira_ticket.title if state.jira_ticket else state.mr_title or 'Not available'}
        
        Description:
        {state.jira_ticket.description if state.jira_ticket else state.mr_description or 'Not available'}
        
        {"## Acceptance Criteria\\n" + "\\n".join([f"- {criterion}" for criterion in state.jira_ticket.acceptance_criteria]) if state.jira_ticket and state.jira_ticket.acceptance_criteria else ""}
        
        ## Code Changes (Diff)
        {diff_content if diff_content else "No changes found in the diff"}
        
        ## Your Task
        
        Independently verify whether the changes in the diff properly address the requirements.
        Focus only on the diff and ticket information, without considering any previous analyses.
        
        Provide your verification as a VerifierAnalysis object that includes:
        
        1. A summary of your assessment
        2. Whether this appears to be a valid solution (true/false)
        3. Strengths of the implementation
        4. Areas that could be improved
        5. Alternative approaches that might be worth considering (optional)
        """
        
        return prompt

class SynthesizerAgent(PipelineAgent[str]):
    """Synthesize all prior agent outputs into a coherent report."""
    
    def __init__(
        self, 
        name: str = "synthesizer",
        model: str = "openai:gpt-4o",
    ):
        # Set up appropriate system prompt
        system_prompt = """You are a Synthesis Agent tasked with combining multiple analyses 
        into a cohesive, actionable review. Your goal is to provide clear, balanced feedback 
        that acknowledges strengths while tactfully highlighting areas for improvement."""
        
        super().__init__(name, model, system_prompt=system_prompt)
        
        # Create synthesizer agent
        self.synthesizer_agent = self.create_agent(output_type=str)
    
    async def process(self, state: PipelineState) -> str:
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(state)
        
        try:
            # Run synthesizer agent
            result = await self.synthesizer_agent.run(prompt)
            self.track_agent_usage(state, result)
            
            return result.output
            
        except Exception as e:
            self.logger.error(f"Error in synthesis: {str(e)}")
            state.errors[f"{self.name}_synthesis"] = str(e)
            
            # Return a minimal synthesis if there's an error
            return f"""
            # Code Review Synthesis (Error)
            
            An error occurred during synthesis: {str(e)}
            
            ## Summary
            
            The review process was unable to complete successfully.
            
            ## Recommendation
            
            Please check the logs for more information about the error.
            """
    
    def _create_synthesis_prompt(self, state: PipelineState) -> str:
        """Create a prompt for synthesizing all agent outputs."""
        # Format implementation plans
        plans_summary = ""
        for i, plan in enumerate(state.implementation_plans):
            plans_summary += f"""
            ## Plan {i+1}: {plan.plan_name}
            
            Summary: {plan.summary}
            
            Steps:
            {chr(10).join([f"- {step}" for step in plan.steps])}
            
            Pros:
            {chr(10).join([f"- {pro}" for pro in plan.pros])}
            
            Cons:
            {chr(10).join([f"- {con}" for con in plan.cons])}
            
            Estimated effort: {plan.estimated_effort}
            """
            
        # Format diff review
        diff_review = ""
        if state.diff_review:
            diff_review = f"""
            ## Diff Review
            
            Summary: {state.diff_review.summary}
            
            Architectural feedback:
            {chr(10).join([f"- {feedback}" for feedback in state.diff_review.architectural_feedback])}
            
            Completion feedback:
            {chr(10).join([f"- {feedback}" for feedback in state.diff_review.completion_feedback])}
            
            Test coverage feedback:
            {chr(10).join([f"- {feedback}" for feedback in state.diff_review.test_coverage_feedback])}
            
            Style feedback:
            {chr(10).join([f"- {feedback}" for feedback in state.diff_review.style_feedback])}
            
            Concerns:
            {chr(10).join([f"- {concern}" for concern in state.diff_review.concerns])}
            """
            
        # Format verifier analysis
        verifier_analysis = ""
        if state.verifier_analysis:
            verifier_analysis = f"""
            ## Independent Verification
            
            Summary: {state.verifier_analysis.summary}
            
            Valid solution: {"Yes" if state.verifier_analysis.is_valid_solution else "No"}
            
            Strengths:
            {chr(10).join([f"- {strength}" for strength in state.verifier_analysis.strengths])}
            
            Improvement areas:
            {chr(10).join([f"- {area}" for area in state.verifier_analysis.improvement_areas])}
            
            {f"Alternative suggestions:\\n{chr(10).join([f'- {suggestion}' for suggestion in state.verifier_analysis.alternative_suggestions])}" if state.verifier_analysis.alternative_suggestions else ""}
            """
            
        # Build prompt
        prompt = f"""
        # Synthesis Task
        
        ## Jira Ticket Information
        Ticket ID: {state.jira_ticket.id if state.jira_ticket else 'Not available'}
        Title: {state.jira_ticket.title if state.jira_ticket else state.mr_title or 'Not available'}
        
        Description:
        {state.jira_ticket.description if state.jira_ticket else state.mr_description or 'Not available'}
        
        ## Implementation Plans
        {plans_summary if plans_summary else "No implementation plans available"}
        
        ## Review Feedback
        {diff_review if diff_review else "No diff review available"}
        
        ## Verification
        {verifier_analysis if verifier_analysis else "No verification analysis available"}
        
        ## Your Task
        
        Synthesize the above information into a cohesive code review report in Markdown format.
        
        Your synthesis should:
        1. Provide a clear executive summary
        2. Highlight which implementation plan (if any) the submitted code most closely follows
        3. Present the most important feedback points, organized by category (architecture, functionality, etc.)
        4. Note where the verification agrees or disagrees with the review
        5. Include actionable recommendations
        
        Be balanced, diplomatic, and constructive in your feedback.
        """
        
        return prompt

class FinalGateAgent(PipelineAgent[PipelineVerdict]):
    """Determine final verdict on the merge request."""
    
    def __init__(
        self, 
        name: str = "final_gate",
        model: str = "openai:gpt-4o",
    ):
        # Set up appropriate system prompt
        system_prompt = """You are a Final Gate Agent tasked with determining whether a merge request 
        should be approved, approved with suggestions, or rejected based on the code review feedback."""
        
        super().__init__(name, model, system_prompt=system_prompt)
        
        # Create gate agent
        self.gate_agent = self.create_agent(output_type=str)
    
    async def process(self, state: PipelineState) -> PipelineVerdict:
        # Check if we have synthesized feedback
        if not state.synthesized_feedback:
            self.logger.warning("No synthesized feedback available for final verdict")
            state.errors[f"{self.name}_no_feedback"] = "Missing synthesized feedback"
            return PipelineVerdict.SOFT_FAIL
            
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
        
        try:
            # Run gate agent
            result = await self.gate_agent.run(prompt)
            self.track_agent_usage(state, result)
            
            # Extract and validate verdict
            verdict = result.output.strip()
            
            if verdict not in (PipelineVerdict.PASS, PipelineVerdict.SOFT_FAIL, PipelineVerdict.HARD_FAIL):
                self.logger.warning(f"Invalid verdict '{verdict}', defaulting to SOFT_FAIL")
                state.errors[f"{self.name}_invalid_verdict"] = f"Invalid verdict: {verdict}"
                return PipelineVerdict.SOFT_FAIL
                
            return verdict
            
        except Exception as e:
            self.logger.error(f"Error in final gate: {str(e)}")
            state.errors[f"{self.name}_error"] = str(e)
            
            # Default to SOFT_FAIL in case of error
            return PipelineVerdict.SOFT_FAIL

# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class PipelineOrchestrator:
    """Orchestrates the entire pipeline with error handling and parallelization."""
    
    def __init__(self, jira_api_url: Optional[str] = None, jira_api_token: Optional[str] = None) -> None:
        self.logger = logging.getLogger("PipelineOrchestrator")
        
        # Initialize agents
        self.ticket_identifier = TicketIdentifierAgent(
            jira_api_url=jira_api_url,
            jira_api_token=jira_api_token
        )
        self.explorer = CodebaseExplorerAgent()
        self.plan_generators = ParallelPlanGenerators()
        self.diff_reviewer = DiffReviewerAgent()
        self.verifier = VerifierAgent()
        self.synthesizer = SynthesizerAgent()
        self.gate = FinalGateAgent()
        
    async def run_pipeline(self, state: PipelineState) -> PipelineState:
        """Run the complete pipeline with error handling."""
        self.logger.info(f"Starting pipeline for MR '{state.mr_title}'")
        
        try:
            # Run pipeline stages sequentially, with error handling for each
            self.logger.info("Identifying ticket...")
            state.jira_ticket = await self.ticket_identifier.run(state)
            
            self.logger.info("Exploring codebase...")
            state.codebase_context = await self.explorer.run(state)
            
            self.logger.info("Generating implementation plans...")
            state.implementation_plans = await self.plan_generators.run(state)
            
            self.logger.info("Reviewing diff...")
            state.diff_review = await self.diff_reviewer.run(state)
            
            self.logger.info("Performing verification...")
            state.verifier_analysis = await self.verifier.run(state)
            
            self.logger.info("Synthesizing feedback...")
            state.synthesized_feedback = await self.synthesizer.run(state)
            
            self.logger.info("Determining final verdict...")
            state.verdict = await self.gate.run(state)
            
            self.logger.info(f"Pipeline completed with verdict: {state.verdict}")
            return state
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            
            # Ensure we have at least a verdict and feedback
            if not state.verdict:
                state.verdict = PipelineVerdict.HARD_FAIL
                
            if not state.synthesized_feedback:
                state.synthesized_feedback = f"""
                # Pipeline Error
                
                The code review pipeline encountered an error: {str(e)}
                
                ## Recommendation
                
                Please fix any issues with the pipeline and try again.
                """
                
            return state
            
    def get_usage_summary(self, state: PipelineState) -> Dict[str, Any]:
        """Get a summary of token usage and durations."""
        total_tokens = 0
        for agent, usage in state.token_usage.items():
            total_tokens += usage.get("total", 0)
            
        return {
            "total_tokens": total_tokens,
            "agent_tokens": {agent: usage.get("total", 0) for agent, usage in state.token_usage.items()},
            "durations": state.stage_durations,
            "errors": state.errors
        }

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

async def _main() -> None:
    import argparse
    import json
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pipeline Agent for code reviews")
    parser.add_argument("--repo-path", required=True, help="Path to the repository")
    parser.add_argument("--branch", required=True, help="Current branch name")
    parser.add_argument("--mr-title", required=True, help="Merge request title")
    parser.add_argument("--mr-description", default="", help="Merge request description")
    parser.add_argument("--jira-api-url", help="Jira API URL (can also use JIRA_API_URL env var)")
    parser.add_argument("--jira-api-token", help="Jira API token (can also use JIRA_API_TOKEN env var)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set environment variables from CLI arguments if provided
    if args.jira_api_url:
        os.environ["JIRA_API_URL"] = args.jira_api_url
    if args.jira_api_token:
        os.environ["JIRA_API_TOKEN"] = args.jira_api_token
    
    # Initialize state
    state = PipelineState(
        repo_path=args.repo_path,
        current_branch=args.branch,
        mr_title=args.mr_title,
        mr_description=args.mr_description,
    )
    
    # Initialize and run orchestrator
    orchestrator = PipelineOrchestrator(
        jira_api_url=os.environ.get("JIRA_API_URL"),
        jira_api_token=os.environ.get("JIRA_API_TOKEN")
    )
    
    try:
        final_state = await orchestrator.run_pipeline(state)
        
        # Get usage summary
        usage = orchestrator.get_usage_summary(final_state)
        
        if args.json:
            # Output JSON with full state
            output = {
                "verdict": final_state.verdict,
                "feedback": final_state.synthesized_feedback,
                "session_id": final_state.session_id,
                "usage": usage
            }
            print(json.dumps(output, indent=2))
        else:
            # Output human-readable format
            print(f"Verdict: {final_state.verdict}")
            print("\nFeedback:")
            print(final_state.synthesized_feedback or "")
            print(f"\nTotal tokens used: {usage['total_tokens']}")
            
        # Set exit code based on verdict
        if final_state.verdict == PipelineVerdict.HARD_FAIL:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(_main())