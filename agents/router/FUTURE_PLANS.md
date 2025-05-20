# Router Agent: Future Development Plans



**Biggest-bang-for-buck, in order of ROI**

| Rank  | Idea                                                                  | Why it unlocks outsized value right now                                                                                                                                                                                                                                     |
| ----- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Central planning loop (single "Decider" + explicit `RouterState`)** | Turns the router from a dumb dispatcher into a *reasoning engine*: lets you chain agents, stop when goal met, parallelise independent work, and reuse the same logic for every new domain. Everything else (interrupts, metrics, new agents) plugs into this state machine. |
| **2** | **Shared session-context + semantic compression**                     | Eliminates the current re-fetch/re-compute tax, keeps token bills predictable, and gives every downstream agent richer input. Without it, the planner quickly blows past context limits.                                                                                    |
| **3** | **Graceful user-interrupt handling**                                  | Makes the system feel interactive and saves wasted tokens/work when the user changes direction. Cheap to add once you have the planner loop; high UX win.                                                                                                                   |
| **4** | **Multi-modal media handling**                                        | Enables richer inputs and outputs beyond text, supporting images, audio, video, and documents throughout the routing system. Leverages modern LLMs' multi-modal capabilities for more powerful applications.                                                                |
| 5     | Parallel execution of independent subtasks                            | Simple concurrency wrapper once the planner exists; cuts wall-clock latency for most real-world composite questions.                                                                                                                                                        |
| 6     | Verification agent                                                    | Directly improves answer correctness and trust; implement as a post-hook the planner can call when `confidence < x`.                                                                                                                                                        |
| 7     | Knowledge-base agent (lightweight)                                    | Start by stashing key/value facts in the shared context; full retrieval-augmented memory can come later.                                                                                                                                                                    |
| 8     | Summarisation agent                                                   | Mostly a utility function the planner can call; value scales with context size.                                                                                                                                                                                             |
| 9     | Reasoning / Decision agents                                           | Useful but overlap with the planner's own LLM calls—do later unless you need domain-specific logic.                                                                                                                                                                         |
| 10    | Formal loop-control heuristics (stall detection, budget caps)         | Important, but you'll get 80 % of the protection by hard-capping iterations and issuing a warning.                                                                                                                                                                          |
| 11    | Fancy visualiser & debugging UI                                       | Nice for demos; adds little direct end-user value compared with the capabilities above.                                                                                                                                                                                     |

**TL;DR:**
Ship the **planner loop + shared context** first—they're the backbone that lets every other fancy agent or UX feature plug in cleanly.






## Task Delegation Loop

The current router agent implementation provides a foundation for routing queries to specialized agents. Our future development will focus on transforming this router into a full-fledged orchestrator that can break down complex tasks and iteratively work toward solutions through a delegation loop.

## Core Architecture Vision

Instead of just routing a query once, the enhanced router will:

1. **Analyze** the user's task
2. **Decompose** it into subtasks
3. **Plan** the execution order
4. **Delegate** each subtask to appropriate specialized agents
5. **Evaluate** intermediate results
6. **Iterate** until the complete task is solved

This approach will enable handling complex multi-step tasks while maintaining a coherent context across multiple agent interactions.

## Specialized Agents to Implement

The following specialized agents would form the ecosystem of our router's delegation network:

### Text-Based Agents

| Agent Type | Primary Function | Examples |
|------------|-----------------|----------|
| **Planning Agent** | Break down complex tasks into specific subtasks | "Create a 3-step plan to analyze this dataset" |
| **Research Agent** | Find and extract information from various sources | "What are the latest studies on renewable energy?" |
| **Reasoning Agent** | Analyze problems and create logical arguments | "Evaluate the pros and cons of this approach" |
| **Coding Agent** | Write and modify code for specific programming tasks | "Create a function to parse this JSON format" |
| **Data Analysis Agent** | Process and analyze numerical data | "Calculate the statistical significance of these results" |
| **Verification Agent** | Check correctness of solutions | "Is this SQL query optimized?" |
| **Summarization Agent** | Condense information from multiple sources | "Summarize the key findings from these reports" |
| **Decision Agent** | Make recommendations based on multiple inputs | "Which database technology should I use for this case?" |
| **Knowledge Base Agent** | Maintain context information throughout task-solving | "Remember and apply information from earlier steps" |
| **User Interaction Agent** | Ask clarifying questions when needed | "What specific format would you like the output in?" |

### Multi-Modal Agents

| Agent Type | Primary Function | Examples |
|------------|-----------------|----------|
| **Image Analysis Agent** | Process and understand visual information | "Identify objects in this image", "Describe what's shown in this diagram" |
| **Audio Processing Agent** | Analyze audio files, transcribe speech | "Transcribe this recording", "Identify speakers in this conversation" |
| **Video Processing Agent** | Process video content, extract key frames | "Summarize what happens in this video", "Extract key moments" |
| **Document Analysis Agent** | Extract information from PDFs, spreadsheets | "Extract data tables from this PDF", "Summarize this research paper" |
| **Media Transformation Agent** | Convert between media formats | "Convert this video to annotated image frames", "Create audio narration from text" |
| **Multi-Modal Synthesis Agent** | Combine insights from multiple media types | "Analyze this image and related audio to explain what's happening" |

## Iterative Planner Approach

Our enhanced router architecture will adopt an iterative planner approach, treating the router as a dynamic orchestrator rather than a one-shot dispatcher:

### Key Components of the Iterative Planner

1. **Single "Decider" Step After Every Subgraph Call**
   - After any specialized agent finishes, its structured output (plus the evolving shared context) is fed back through the router's "planning" function
   - This planning step decides whether:
     - The overall goal is met → exit with final answer
     - Another domain agent should run next → enqueue it
     - The router needs to ask the user a clarifying question → surface it
   - This keeps all branching logic in one place and avoids hard-wiring agent sequences

2. **Explicit State Object**
   - The state maintains:
     - **Goal**: The user's original request
     - **Context**: Facts and intermediate results shared across turns
     - **Plan**: List of remaining subtasks (which can grow or shrink dynamically)
     - **History**: Messages exchanged so far
   - Each loop iteration updates this state, making every agent invocation reproducible and debuggable

3. **Loop Control & Safety Rails**
   - **Max iterations / token budget**: Stop if the plan cycles or cost grows beyond thresholds
   - **Dependence check**: Only enqueue an agent if its input requirements are satisfied
   - **Progress heuristic**: If two consecutive iterations leave the plan unchanged, assume you're stuck and exit gracefully

4. **Uniform Contract for Agent Outputs**
   - Every specialized agent returns `{status, data, suggestions}`
   - `Suggestions` is an ordered list of next actions the agent thinks would be useful
   - The router isn't obligated to follow them, but can merge these hints into its planning step

5. **Planning Implementation Options**
   - **LLM-prompted**: Pass the current state (compressed) to a "planning" LLM that emits the next action
   - **Rule-based first, LLM fallback**: Simple routing for common cases, LLM for high ambiguity
   - **Graph node**: In our Pydantic-Graph, a planning node decides the next node and loops back until reaching End

6. **Parallel vs. Serial Refinement**
   - For independent subtasks, enqueue multiple agents in parallel
   - For dependent chains (e.g., search → calculation), keep them serial so each result feeds the next

7. **Stop-criteria & Final Synthesis**
   - The planner declares completion when plan is empty and context contains an answer
   - A dedicated "synthesis" agent creates the final user-visible answer

8. **Graceful User Interrupts**
   - Users might need to refine or redirect the task mid-loop
   - While executing agents, the system listens for incoming user messages
   - If a user message is received, it raises a `UserInterrupt` signal
   - The planning node catches this interrupt and re-evaluates with the new input
   - The planner can then decide to:
     - Discard current work and start a new plan
     - Incorporate the new information into the existing plan
     - Ask for clarification about how to proceed
   - This creates a more responsive, conversational experience

### Enhanced Router State With Planning Capabilities

```python
@dataclass
class RouterState:
    """Enhanced router state with planning capabilities."""
    # Core state
    query: str  # Original user query
    goal: str  # Interpreted goal (may be refined during processing)
    
    # Planning
    tasks: List[Task] = field(default_factory=list)  # Evolving plan
    completed_tasks: List[Task] = field(default_factory=list)
    
    # Traditional tracking
    domain_states: Dict[Domain, WorkflowState] = field(default_factory=dict)
    domain_results: Dict[Domain, Any] = field(default_factory=dict)
    
    # Shared context (for cross-agent knowledge sharing)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Loop control
    iteration_count: int = 0
    max_iterations: int = 10
    last_plan_hash: Optional[str] = None  # For detecting stalled plans
    user_interrupt: Optional[str] = None  # Stores any user interrupt message
    
    # Media handling
    media_items: Dict[str, Union[
        ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent
    ]] = field(default_factory=dict)
    media_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Final result
    final_response: Optional[str] = None
    
    # Utility methods
    def is_stalled(self) -> bool:
        """Check if the plan hasn't changed in the last iteration."""
        if self.last_plan_hash is None:
            return False
        
        # Hash the current task list to compare with previous iteration
        current_hash = self._hash_tasks()
        return current_hash == self.last_plan_hash
    
    def update_plan_hash(self) -> None:
        """Update the hash of the current plan for stall detection."""
        self.last_plan_hash = self._hash_tasks()
    
    def _hash_tasks(self) -> str:
        """Create a simple hash of the current tasks."""
        # In practice, use a more sophisticated hashing method
        return ",".join(str(task.id) for task in self.tasks)
    
    def reached_iteration_limit(self) -> bool:
        """Check if we've reached the maximum allowed iterations."""
        return self.iteration_count >= self.max_iterations
    
    def record_iteration(self) -> None:
        """Record that an iteration has completed."""
        self.iteration_count += 1
        
    def set_user_interrupt(self, message: str) -> None:
        """Record a user interrupt message."""
        self.user_interrupt = message
        
    def has_user_interrupt(self) -> bool:
        """Check if a user interrupt has been received."""
        return self.user_interrupt is not None
    
    def clear_user_interrupt(self) -> None:
        """Clear the user interrupt after handling it."""
        self.user_interrupt = None
        
    # Media handling methods
    def add_media(self, 
                 media_id: str, 
                 media: Union[ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent],
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a media item to the state with optional metadata.
        
        Args:
            media_id: Unique identifier for the media
            media: The media item object
            metadata: Optional metadata about the media
        """
        self.media_items[media_id] = media
        if metadata:
            self.media_metadata[media_id] = metadata
            
    def get_media(self, media_id: str) -> Optional[Union[ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent]]:
        """Retrieve a media item from the state."""
        return self.media_items.get(media_id)
        
    def get_media_metadata(self, media_id: str) -> Dict[str, Any]:
        """Get metadata for a media item."""
        return self.media_metadata.get(media_id, {})
        
    def update_media_metadata(self, media_id: str, key: str, value: Any) -> None:
        """Update metadata for a media item."""
        if media_id not in self.media_metadata:
            self.media_metadata[media_id] = {}
        self.media_metadata[media_id][key] = value
        
    def get_media_by_type(self, media_type: str) -> List[Tuple[str, Any]]:
        """Get all media items of a specific type."""
        return [
            (media_id, media) for media_id, media in self.media_items.items()
            if (
                (media_type == "image" and hasattr(media, "is_image") and media.is_image) or
                (media_type == "audio" and hasattr(media, "is_audio") and media.is_audio) or
                (media_type == "video" and hasattr(media, "is_video") and media.is_video) or
                (media_type == "document" and hasattr(media, "is_document") and media.is_document)
            )
        ]
```

### Planning Node With Interrupt Handling

```python
@dataclass
class PlanningNode(BaseNode[RouterState]):
    """Central planning node that decides next steps after each agent execution."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> Union[
        "DomainNode", "UserClarificationNode", "FinalizeResponseNode", "HandleInterruptNode"
    ]:
        # 1. Check for user interrupts first
        if ctx.state.has_user_interrupt():
            return HandleInterruptNode()
            
        # 2. Increment iteration counter
        ctx.state.record_iteration()
        
        # 3. Check if we've reached limits
        if ctx.state.reached_iteration_limit():
            # We've hit the maximum iterations, exit with what we have
            return FinalizeResponseNode(reason="max_iterations_reached")
        
        if ctx.state.is_stalled():
            # Plan hasn't changed, we're stuck in a loop
            return FinalizeResponseNode(reason="plan_stalled")
        
        # 4. Update plan hash for next iteration
        ctx.state.update_plan_hash()
        
        # 5. Determine next steps with LLM
        planning_agent = Agent(
            "openai:gpt-4o",
            output_type=Dict[str, Any],
            system_prompt="""You are a planning agent that decides next steps in a workflow.
            Based on the current state, decide whether the task is complete, requires further processing,
            or needs user clarification."""
        )
        
        # Compress state to avoid token limits
        compressed_state = {
            "query": ctx.state.query,
            "goal": ctx.state.goal,
            "remaining_tasks": [t.description for t in ctx.state.tasks],
            "completed_tasks": [t.description for t in ctx.state.completed_tasks],
            "context": summarize_context(ctx.state.context),
            "latest_result": get_latest_result(ctx.state),
            "iteration": ctx.state.iteration_count
        }
        
        # Get planning decision
        result = await planning_agent.run(
            f"""
            Current state:
            {format_as_xml(compressed_state)}
            
            Decide the next step:
            1. If the goal appears to be met, return {{"action": "complete", "reason": "goal_met"}}
            2. If a clarification is needed from the user, return {{"action": "clarify", "question": "..."}}
            3. If further processing is needed, return one of:
               {{"action": "execute", "domain": "search", "input": "..."}}
               {{"action": "execute", "domain": "calculation", "input": "..."}}
               {{"action": "execute", "domain": "text_analysis", "input": "..."}}
            
            Your decision should be based on what would most efficiently advance toward the goal.
            """
        )
        
        # 6. Process the planning decision
        decision = result.output
        
        if decision.get("action") == "complete":
            # Task is complete, move to final response
            return FinalizeResponseNode(reason=decision.get("reason", "goal_met"))
        
        elif decision.get("action") == "clarify":
            # Need user clarification
            return UserClarificationNode(question=decision.get("question", "Can you provide more information?"))
        
        elif decision.get("action") == "execute":
            # Execute a domain agent
            try:
                domain = Domain(decision.get("domain", "search"))
                return DomainNode(domain=domain, input=decision.get("input", ctx.state.query))
            except ValueError:
                # Invalid domain, default to search
                return DomainNode(domain=Domain.SEARCH, input=ctx.state.query)
        
        # Default fallback
        return FinalizeResponseNode(reason="planning_error")


@dataclass
class HandleInterruptNode(BaseNode[RouterState]):
    """Handles a user interrupt by re-evaluating the plan."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "PlanningNode":
        # Get the interrupt message
        interrupt_message = ctx.state.user_interrupt
        
        # Create an agent to evaluate the interrupt
        interrupt_agent = Agent(
            "openai:gpt-4o",
            output_type=Dict[str, Any],
            system_prompt="""You evaluate user interrupts during an ongoing task
            and decide how to incorporate the new information."""
        )
        
        # Prepare context about current state
        compressed_state = {
            "original_query": ctx.state.query,
            "current_goal": ctx.state.goal,
            "progress_so_far": summarize_progress(ctx.state),
            "interrupt_message": interrupt_message
        }
        
        # Get evaluation of interrupt
        result = await interrupt_agent.run(
            f"""
            A user has sent a new message while you were working on a task:
            {format_as_xml(compressed_state)}
            
            Decide how to handle this interrupt:
            1. If it's a completely new task, return {{"action": "restart", "new_goal": "..."}}
            2. If it refines or adds information to the current task, return {{"action": "refine", "updated_goal": "...", "keep_context": true}}
            3. If it's a simple status request, return {{"action": "status_update"}}
            4. If it's unclear how it relates to the current task, return {{"action": "clarify", "question": "..."}}
            """
        )
        
        decision = result.output
        action = decision.get("action", "clarify")
        
        # Handle based on the decision
        if action == "restart":
            # Clear existing state and start with new goal
            ctx.state.tasks.clear()
            ctx.state.completed_tasks.clear()
            ctx.state.query = interrupt_message
            ctx.state.goal = decision.get("new_goal", interrupt_message)
            ctx.state.iteration_count = 0
            
            # Optionally clear context if it's a completely new topic
            if not decision.get("keep_context", False):
                ctx.state.context.clear()
                
        elif action == "refine":
            # Update goal but keep progress
            ctx.state.goal = decision.get("updated_goal", ctx.state.goal)
            
            # Optionally modify current tasks based on refinement
            if "updated_tasks" in decision:
                # This would have logic to update the task list without losing progress
                pass
                
        elif action == "status_update":
            # No changes to state, just add a status message to results
            status_message = generate_status_update(ctx.state)
            ctx.state.domain_results["status_update"] = status_message
            
        # Clear the interrupt now that we've handled it
        ctx.state.clear_user_interrupt()
        
        # Always return to the planning node to re-evaluate
        return PlanningNode()


# In the execution coordinator that runs domain nodes
async def execute_domain_with_interrupt_check(
    domain: Domain, 
    state: RouterState, 
    user_message_queue: Queue
) -> Any:
    """Executes a domain while checking for user interrupts."""
    # Create a task for the domain execution
    domain_task = asyncio.create_task(execute_domain(domain, state))
    
    # Create a task that monitors for user messages
    async def monitor_user_messages():
        while True:
            if not user_message_queue.empty():
                message = await user_message_queue.get()
                state.set_user_interrupt(message)
                return True
            await asyncio.sleep(0.1)  # Check every 100ms
    
    interrupt_task = asyncio.create_task(monitor_user_messages())
    
    # Wait for either domain completion or user interrupt
    done, pending = await asyncio.wait(
        [domain_task, interrupt_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # Cancel the pending task
    for task in pending:
        task.cancel()
    
    # If domain completed normally, return its result
    if domain_task in done:
        return await domain_task
    
    # If interrupted, return a special indicator
    return None  # Domain execution was interrupted
```

## Task Delegation Loop Architecture

The enhanced router will maintain:

1. **Task State**: Current progress toward the goal, information gathered, decisions made
2. **Execution Plan**: Sequence of subtasks to be completed
3. **Success Criteria**: How to determine when the task is complete

### Loop Algorithm Pseudocode

```
1. Receive user task
2. Initialize task state and plan
3. LOOP:
   a. Check for user interrupts (process immediately if present)
   b. Check for exit conditions (max iterations, plan stalled, etc.)
   c. Execute planning step to evaluate current state
   d. If goal is met → finalize and return result
   e. If user clarification needed → request clarification and wait
   f. If further processing needed:
      i. Select appropriate specialized agent
      ii. Execute agent with context from current state (with interrupt monitoring)
      iii. If interrupted during execution → handle interrupt
      iv. Update state with results and suggestions
   g. Loop back to 3.a
```

## Implementation Approach

For the task delegation loop, we've chosen to build upon our existing **graph-based implementation** using Pydantic-Graph. This approach provides several advantages:

- **Explicit state representation**: The state machine clearly models the flow of processing
- **Visualizable architecture**: The graph structure can be visualized using Mermaid diagrams
- **Extensible nodes**: New specialized agents can be added as nodes in the graph
- **Clear transitions**: The flow between different processing stages is explicit
- **State inspection**: The state at any point in the flow can be inspected and debugged

Our current implementation already uses Pydantic-Graph to model the router as a state machine where:
- Each node represents a stage in the task-solving process (classification, search, calculation, etc.)
- Edges represent transitions based on intermediate results
- Node handlers correspond to specialized agent calls

This foundation leverages the existing graph capabilities of the Pydantic ecosystem and provides a clear path for enhancing the router with more complex task delegation capabilities.

### Supporting Approaches

While our primary implementation uses Pydantic-Graph, we'll complement it with these approaches:

#### Tool-Driven Integration

Within our graph nodes, specialized agents will be accessed through a tool-driven interface:

```python
@dataclass
class PlanningNode(BaseNode[RouterState]):
    """Planning node that breaks down complex tasks."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> NextNode:
        # Execute the planning agent
        result = await planning_agent.run(ctx.state.query, message_history=ctx.state.history)
        ctx.state.plan = result.output
        return NextNode()
```

#### Meta-Prompting

For specialized or dynamic cases, the router will generate prompts for specialized LLM agents, programming them on-the-fly for specific subtasks.

### Intermediary Context-Builder Layer

A critical enhancement will be adding a shared context that enables more effective communication between specialized agents:

| Goal | Why it matters | Key considerations |
| ---- | -------------- | ------------------ |
| **1. Introduce a shared session context** | Lets every specialist agent reuse facts, numbers, and intermediate results instead of recomputing. | • Define what kinds of data belong here (facts, parsed numbers, embeddings, partial calcs).  <br>• Keep the schema explicit—even if it's just a documented dict—so agents know what to expect. |
| **2. Use a dedicated "context-builder" agent as the first hop** | Centralizes extraction of entities, normalization of numbers, generation of search keys, etc. | • Feed it the raw user query *plus* any existing context.  <br>• Ask it to **augment**, not overwrite, the context.  <br>• Keep its output light and structured. |
| **3. Make every downstream tool receive that context** | Ensures search, calculation, and analysis agents start from a richer baseline. | • Pass the context object through your run-context / dependency mechanism.  <br>• Encourage tools to consult the context first (e.g., check for cached calculations). |
| **4. Persist conversation state safely** | You already track history; make it durable and process-safe. | • Use a small local store (SQLite / TinyDB / plain JSONL) instead of an in-memory dict.  <br>• Index by conversation/session ID.  <br>• Provide a purge/expiry policy to avoid unbounded growth. |
| **5. Separate orchestration from I/O** | Cleaner architecture and easier testing. | • Keep the core router pure: given *query + context + history* → returns *answer + new context + new messages*.  <br>• Wrap CLI/GUI/HTTP layers around that pure function. |
| **6. Guard against loop & bloat** | A context that grows unchecked can slow or derail the system. | • Set size limits (e.g., only keep N most recent facts).  <br>• Deduplicate equivalent entries.  <br>• Have the router decide when to stop delegating if progress stalls. |
| **7. Document the contract** | Future contributors—and future you—need clarity. | • Briefly state what each agent may read/write in the shared context.  <br>• Describe expected lifetimes (per request, per session, persistent). |
| **8. Implement semantic context compression** | Raw truncation loses meaning; preserve semantic content while reducing tokens | • Add an "auto-abstract" step that summarizes large text blobs into bullet points before entering shared context<br>• Apply progressive compression levels (light, medium, heavy) based on content size<br>• Use content-aware compression strategies for different data types (code, text, data)<br>• Maintain compression metadata for retrievability |

#### Context-Builder Implementation Pipeline

1. **Pipeline**

   1. Receive user input.
   2. Run *context-builder* → enrich shared context.
   3. Pass query + context into router.
   4. Router decides which specialist(s) to call, providing the enriched context each time.
   5. Router aggregates outputs, may update context further, returns answer.

2. **Persistence**

   * At the end of each turn, serialize both message history and the latest context.
   * Restore them on the next turn so the user can ask follow-ups seamlessly.

3. **Testing & iteration**

   * Start with a minimal context schema (e.g., `facts`, `numbers`).
   * Grow it only when a new use-case clearly benefits.
   * Monitor token usage; prune or compress context as needed.

#### Semantic Context Compression

To effectively manage token usage while preserving meaning, we'll implement intelligent semantic compression:

1. **Auto-Abstract Generation**
   - Summarize large text blocks into bullet points before storing in shared context
   - Apply semantic prioritization to retain the most important elements
   - LLM-powered summarization that preserves core meaning with fewer tokens

2. **Progressive Compression Levels**
   - Level 1: Lightweight compression (remove redundancies, standardize formatting)
   - Level 2: Medium compression (convert paragraphs to bullet points, extract key facts)
   - Level 3: Heavy compression (distill to core facts and relationships only)

3. **Content-Aware Compression**
   - Code: Preserve function signatures and critical logic, summarize implementation details
   - Text: Keep key points and conclusions, compress examples and explanations
   - Data: Maintain statistical significance, compress raw examples

4. **Retrievability**
   - Track what information was compressed and how
   - Maintain pointers to original content for retrieval if needed
   - Implement expansion capabilities when full details are required

```python
class ContextCompressor:
    """Intelligently compresses context while preserving semantic meaning."""
    
    def compress(self, content: Any, compression_level: int = 1) -> Tuple[Any, Dict[str, Any]]:
        """
        Compress content at the specified level while preserving key meaning.
        
        Args:
            content: The content to compress (text, dict, list, etc.)
            compression_level: Level of compression (1-3)
            
        Returns:
            Tuple of (compressed_content, metadata)
        """
        content_type = self._detect_content_type(content)
        compressor = self._get_compressor_for_type(content_type, compression_level)
        
        return compressor.compress(content)

# Integration with RouterState
@dataclass
class RouterState:
    # ... existing fields ...
    
    # Context compression
    context_compressor: ContextCompressor = field(default_factory=ContextCompressor)
    max_context_size_tokens: int = 2000
    context_compression_level: int = 1
    context_compression_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_to_context(self, key: str, value: Any) -> None:
        """Add content to context with semantic compression if needed."""
        current_size = self._estimate_context_tokens()
        
        if current_size + self._estimate_tokens(value) > self.max_context_size_tokens:
            # Apply progressive compression based on size
            compression_level = min(3, self.context_compression_level + 1)
            
            # Compress the value
            compressed_value, metadata = self.context_compressor.compress(
                value, compression_level=compression_level
            )
            
            # Store the compressed value and metadata
            self.context[key] = compressed_value
            self.context_compression_metadata[key] = metadata
        else:
            # Store uncompressed
            self.context[key] = value
```

This approach ensures that context is intelligently managed without losing critical semantic meaning, which is far superior to raw truncation methods.

### Multi-Modal Router Implementation

A key enhancement will be extending the router to fully support multi-modal interactions across all specialized agents:

1. **Media Type Support**
   - Extend the `RouterState` to track media assets throughout the processing pipeline
   - Implement seamless passing of media between specialized agents (images, audio, video, documents)
   - Support both URL-based media and binary content representations

2. **Specialized Media-Aware Agents**
   - **Audio Processing Agent**: Analyze audio files, transcribe speech, identify speakers
   - **Image Analysis Agent**: Analyze images, detect objects, generate descriptions
   - **Video Processing Agent**: Extract frames, analyze content, summarize
   - **Document Analysis Agent**: Parse documents, extract structured information, summarize content

3. **Media Type Conversion**
   - Support conversion between media formats when needed (e.g., video to image frames)
   - Implement media transformation tools for agents (transcription, captioning, etc.)
   - Handle compression and optimization for efficient processing

4. **Multi-Modal Context Management**
   - Track media references in shared context
   - Apply specialized compression strategies for different media types
   - Implement content-aware semantic summarization for non-text media

5. **Implementation Architecture**
   ```python
   @dataclass
   class RouterMediaState:
       """Tracks media assets through the routing process."""
       media_items: Dict[str, Union[ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent]] = field(default_factory=dict)
       media_annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
       
       def add_media(self, media_id: str, media: Any) -> None:
           """Add a media item to the state."""
           self.media_items[media_id] = media
           
       def get_media(self, media_id: str) -> Optional[Any]:
           """Retrieve a media item from the state."""
           return self.media_items.get(media_id)
           
       def add_annotation(self, media_id: str, annotation_type: str, content: Any) -> None:
           """Add an annotation (e.g., description, transcription) to a media item."""
           if media_id not in self.media_annotations:
               self.media_annotations[media_id] = {}
           self.media_annotations[media_id][annotation_type] = content
   ```

6. **Media-Aware Tool Definitions**
   ```python
   @router_agent.tool
   async def analyze_image(ctx: RunContext[RouterDeps], image: Union[ImageUrl, BinaryContent]) -> ImageAnalysisResult:
       """
       Analyze an image using the image analysis agent.
       Only use this when processing image content.
       """
       # Add to media state for tracking
       media_id = f"img_{uuid.uuid4()}"
       ctx.state.media_state.add_media(media_id, image)
       
       # Process with specialized agent
       result = await image_analysis_agent.run(image, usage=ctx.usage)
       
       # Store annotations
       ctx.state.media_state.add_annotation(
           media_id, "analysis", result.output.description
       )
       
       return result.output
   ```

## Key Technical Challenges

1. **Loop Prevention**: Mechanisms to avoid infinite loops if a task can't be completed
2. **State Management**: Efficiently tracking progress across multiple iterations
3. **Knowledge Transfer**: Ensuring information gathered by one agent is available to others
4. **Balance**: Not getting stuck on one approach when another would work better
5. **Performance Optimization**: Minimizing token usage while maintaining context
6. **Domain Boundaries**: Defining clear boundaries between specialized agents while allowing flexible execution paths
7. **Interrupt Handling**: Gracefully handling user interrupts without losing important context or progress
8. **Semantic Compression**: Intelligently compressing context while preserving meaning and retrievability
9. **Multi-Modal Media Handling**: Supporting various media types (images, audio, video, documents) across the entire routing system

## Implementation Roadmap

1. **Phase 1**: Implement basic task decomposition and simple looping with 2-3 agents
2. **Phase 2**: Add state management and context persistence
3. **Phase 3**: Develop centralized planning node with decision capabilities
4. **Phase 4**: Implement safety mechanisms (iteration limits, stall detection, progress metrics)
5. **Phase 5**: Introduce a wider range of specialized agents
6. **Phase 6**: Add parallel execution for independent subtasks
7. **Phase 7**: Develop advanced user interaction for clarifications and feedback
8. **Phase 8**: Implement graceful interrupt handling for mid-execution refinements
9. **Phase 9**: Implement semantic context compression for efficient token management
10. **Phase 10**: Add multi-modal media handling capabilities across the router system
11. **Phase 11**: Create visualization and debugging tools for workflow inspection

## Example Use Cases

These complex workflows would benefit from the task delegation loop:

1. **Research Report Generation**
   - Research a topic across multiple sources
   - Analyze and synthesize findings
   - Generate a structured report with citations

2. **Data Analysis Pipeline**
   - Clean and prepare data
   - Run multiple analyses
   - Visualize results
   - Generate insights

3. **Software Development Assistant**
   - Design system architecture
   - Generate code for components
   - Write tests
   - Document the implementation

4. **Interactive Learning Assistant**
   - Assess user's knowledge level
   - Present appropriate learning materials
   - Test understanding
   - Adapt to user's progress

5. **Multi-Modal Content Analysis**
   - Analyze uploaded media (images, audio, video, documents)
   - Extract key information from different modalities
   - Integrate insights across modalities
   - Provide unified understanding of complex multi-modal content

6. **Media Production Assistant**
   - Take a concept and create complementary media assets
   - Generate text descriptions based on visual content
   - Create audio narratives from text content
   - Transform between different media formats as needed
   - Combine multiple media types into cohesive presentations

## Contribution Guidelines

If you're interested in contributing to these future developments:

1. Focus on implementing one specialized agent at a time
2. Ensure strong typing and validation for all inputs and outputs
3. Write comprehensive tests for your agent
4. Document expected behavior and limitations

By implementing this iterative planning architecture, our router agent will evolve into a powerful orchestrator capable of solving complex problems through dynamic planning, delegation, and refinement.


That's an incredibly comprehensive and well-thought-out plan for the Router Agent! The level of detail, especially regarding the iterative planner, state management, context handling (including semantic compression), and multi-modal capabilities, is excellent. You've covered most of the critical aspects for building a robust and intelligent orchestration system.

When thinking about an *abstract flow for tackling problems* that can apply generally but also fit your initial programming workflow use case, the stages you've implicitly and explicitly outlined are very strong. Let's try to distill them into a slightly more generalized abstract flow and see if anything feels underrepresented or could be highlighted.

Your current "Core Architecture Vision" (Analyze, Decompose, Plan, Delegate, Evaluate, Iterate) is a solid foundation. We can expand slightly on this for a general problem-solving framework:

**Proposed Abstract Problem-Solving Stages (and how your plan maps to them):**

1.  **Problem Intake & Understanding (Clarification)**
    * **Description:** Receiving the initial problem statement from the user and ensuring the system has a clear grasp of the *actual* need, not just the literal words. This might involve initial clarifying questions even before deep planning.
    * **Your Plan:**
        * "Receive user task"
        * Initial part of "Analyze the user's task"
        * `RouterState.query` and `RouterState.goal` (with refinement)
        * `UserInteractionAgent` ("Ask clarifying questions when needed") and `PlanningNode` logic for `{"action": "clarify"}`.
    * **Consideration:** While your plan covers clarification *during* the loop, explicitly emphasizing an *initial* understanding/clarification phase can be beneficial, especially for ambiguous requests. This could be a dedicated first pass by a lightweight "Understanding Agent" or a specific mode of the Planning Agent.

2.  **Goal Definition & Scoping**
    * **Description:** Translating the understood problem into a well-defined, achievable goal with clear success criteria. This includes understanding constraints and the desired output format.
    * **Your Plan:**
        * `RouterState.goal` (interpreted and refined)
        * "Success Criteria: How to determine when the task is complete"
        * "Stop-criteria & Final Synthesis"
    * **Consideration:** This is well covered. Perhaps making the "Success Criteria" more explicit and potentially user-confirmable at the start for complex tasks could be an addition.

3.  **Strategy Formulation & Decomposition (Planning)**
    * **Description:** Developing a high-level strategy to achieve the goal. This involves breaking the main goal into manageable sub-tasks, identifying dependencies, and selecting potential tools/approaches (or agents). For a programming task, this would be like outlining modules, functions, and data flows.
    * **Your Plan:**
        * "Decompose it into subtasks"
        * "Plan the execution order"
        * `Planning Agent`
        * `RouterState.tasks`
        * "Central planning loop (single "Decider" + explicit `RouterState`)"
    * **Consideration:** This is the core of your planner and is very well detailed.

4.  **Resource & Knowledge Acquisition (Gathering)**
    * **Description:** Identifying and gathering necessary information, data, tools, or pre-existing knowledge components required for the sub-tasks. For programming, this could be finding relevant libraries, APIs, code snippets, or documentation.
    * **Your Plan:**
        * `Research Agent`
        * `Knowledge-base agent` (even lightweight)
        * "Shared session-context + semantic compression" (crucial for making knowledge available)
        * `Context-Builder Implementation Pipeline`
    * **Consideration:** This is well covered, especially with the focus on shared context.

5.  **Execution & Monitoring (Doing & Tracking)**
    * **Description:** Executing the planned sub-tasks using the appropriate agents/tools. This stage includes monitoring the progress of each sub-task and managing any runtime issues.
    * **Your Plan:**
        * "Delegate each subtask to appropriate specialized agents"
        * "Execute agent with context from current state (with interrupt monitoring)"
        * `DomainNode` execution
        * `RouterState.iteration_count`, `completed_tasks`
        * "Loop Control & Safety Rails" (Max iterations, stall detection)
        * "Parallel execution of independent subtasks"
    * **Consideration:** Thoroughly covered. The interrupt handling and parallel execution are excellent additions here.

6.  **Intermediate Evaluation & Refinement (Checking & Adjusting)**
    * **Description:** After each significant sub-task or a set of sub-tasks, evaluating the intermediate results against the plan and the overall goal. This is where the system decides if the current path is fruitful, if the plan needs adjustment, if more clarification is needed, or if a sub-goal has been met. This is the core of the iterative loop.
    * **Your Plan:**
        * "Evaluate intermediate results"
        * "Single "Decider" Step After Every Subgraph Call"
        * `PlanningNode` logic (complete, clarify, execute)
        * `Verification Agent` (can be used here)
        * "Graceful user-interrupt handling" (allows user-driven refinement)
    * **Consideration:** This is a central strength of your design.

7.  **Solution Synthesis & Presentation (Compiling & Delivering)**
    * **Description:** Once the overall goal is deemed met, compiling all intermediate results, analyses, and outputs into a coherent, final solution that directly addresses the user's defined goal.
    * **Your Plan:**
        * "Dedicated 'synthesis' agent creates the final user-visible answer"
        * `FinalizeResponseNode`
        * `RouterState.final_response`
        * `Summarisation Agent` (can play a role here)
    * **Consideration:** Well covered.

8.  **(Optional but valuable) Post-Solution Reflection & Learning**
    * **Description:** After delivering the solution, the system (or its developers) could analyze the problem-solving process itself. What worked well? What was inefficient? How can the planning, agent selection, or knowledge utilization be improved for similar future tasks? For user-facing systems, this could also involve soliciting user feedback on the solution's quality.
    * **Your Plan:** This is more of a meta-level stage. While not explicitly a runtime stage for solving a *single* problem, the data collected (`RouterState`, history, metrics) could feed into a longer-term improvement loop for the agent system itself. "Fancy visualiser & debugging UI" could help in this offline analysis.
    * **Consideration:** For the *immediate* problem-solving loop, this is out of scope, but it's a natural extension for evolving the system's intelligence over time.

**Are any stages missing?**

Based on the abstract flow above, your plan is exceptionally comprehensive. The stages you've designed cover the problem-solving lifecycle in great detail. The areas I've highlighted are more about emphasis or slight re-framing rather than outright missing pieces:

* **Explicit Initial Understanding/Scoping:** While your `PlanningNode` can ask for clarification, ensuring a very solid understanding and agreed-upon scope *before* extensive decomposition could prevent wasted cycles. This might just be an early phase of your existing "Analyze" step, perhaps with a higher tendency to use the `UserInteractionAgent` if ambiguity is high.
* **Constraint Management:** You mention "budget caps." Explicitly considering other constraints (e.g., user-specified preferences for tools/libraries in a programming task, desired output formats, ethical boundaries if applicable) as part of Goal Definition or Strategy Formulation could be beneficial. Your flexible `RouterState` and context could accommodate this.

**Mapping to Programming Workflow:**

Your system is well-suited for programming:

1.  **Problem Intake & Understanding:** User: "Write a Python script to scrape a website and save data to CSV." System might ask: "Which website? What specific data points? Any login required?"
2.  **Goal Definition & Scoping:** Goal: Script takes URL, extracts specific fields (e.g., product name, price), handles pagination, saves to `output.csv`. Constraints: Use `requests` and `BeautifulSoup`, handle network errors.
3.  **Strategy Formulation & Decomposition:**
    * Task 1: Fetch HTML (Coding Agent with `requests`).
    * Task 2: Parse HTML to find data (Coding Agent with `BeautifulSoup`).
    * Task 3: Handle pagination logic (Coding Agent).
    * Task 4: Write to CSV (Coding Agent).
    * Task 5: Main script structure, error handling (Coding Agent).
    * Task 6: Test individual functions (Verification Agent/Coding Agent).
4.  **Resource & Knowledge Acquisition:** Identify `requests`, `BeautifulSoup` docs. (Research Agent if needed for complex APIs).
5.  **Execution & Monitoring:** Agents generate code snippets. System tracks which parts are done.
6.  **Intermediate Evaluation & Refinement:** Snippet for fetching HTML works. Parsing snippet fails on some pages – refine selectors (user interrupt or planner re-evaluates). Tests pass/fail.
7.  **Solution Synthesis & Presentation:** Combine all code, add comments, provide a `README` on how to run it.
8.  **(Post-Solution):** User reports the script is slow. Future "Reflection & Learning" might prioritize more efficient scraping techniques or caching strategies for similar tasks.

**Conclusion:**

Your plan is not missing any major *fundamental* stages for a sophisticated problem-solving agent. The architecture is robust and forward-thinking. The slight suggestions above are more about potential refinements in emphasis or breaking out sub-phases within your existing components for clarity in an abstract model. The key is that your iterative loop with the "Decider" (`PlanningNode`) inherently allows for revisiting any of these conceptual stages as needed.

The focus on the "Central planning loop" and "Shared session-context" is spot on, as these are the enablers for truly dynamic and intelligent problem-solving.