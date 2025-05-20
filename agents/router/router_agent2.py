"""
Router agent implementation that dispatches requests to specialized agents
iteratively to solve complex tasks using Google Gemini models.
"""

import os
import time
from dataclasses import dataclass, field as dataclass_field # aliased to avoid pydantic field
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, get_type_hints
import warnings
import uuid
import hashlib # For stall detection hash

from pydantic import BaseModel, Field as PydanticField, create_model
from typing_extensions import Annotated
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import Usage
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, 
    SystemPromptPart, UserPromptPart, TextPart
)

# Import Google specific classes
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

# Rich imports will be handled locally where console output is generated.


# --- Tool Argument Models ---
class SearchArgs(BaseModel):
    query: str

class CalculationArgs(BaseModel):
    problem: str

class TextAnalysisArgs(BaseModel):
    text: str
    
class PlanningArgs(BaseModel):
    query_for_planning: str
    
class ReasoningArgs(BaseModel):
    problem_statement: str
    
class CodeTaskArgs(BaseModel):
    task_description: str
    language: str = "python"
    
class DataAnalysisArgs(BaseModel):
    data_description: str
    analysis_query: str
    
class VerificationArgs(BaseModel):
    item_to_verify: str
    criteria: Optional[str] = None
    
class SummarizationArgs(BaseModel):
    content_to_summarize: str
    length_constraint: Optional[str] = "concise"
    
class DecisionArgs(BaseModel):
    decision_problem: str
    options: List[str]
    criteria: Optional[List[str]] = None
    
class KnowledgeBaseArgs(BaseModel):
    kb_query: str
    action: str = "retrieve"
    
class ImageAnalysisArgs(BaseModel):
    image_identifier: str
    prompt: Optional[str] = None
    
class AudioProcessingArgs(BaseModel):
    audio_identifier: str
    task: str = "transcribe"
    
class VideoProcessingArgs(BaseModel):
    video_identifier: str
    task: str = "summarize"
    
class DocumentAnalysisArgs(BaseModel):
    document_identifier: str
    query: Optional[str] = None
    
class MediaTransformationArgs(BaseModel):
    input_media_id: str
    target_format: str
    transformation_prompt: Optional[str] = None
    
class MultiModalSynthesisArgs(BaseModel):
    media_ids: List[str]
    synthesis_goal: str

# --- Planner Decision Models (Discriminated Union) ---
class CallTool(BaseModel):
    thought: str = PydanticField(description="Your reasoning for the decision. Explain why this next step is the best one.")
    action: Literal["call_tool"] = "call_tool"
    tool_name: str = PydanticField(description="The name of the tool to call. Must be one of the available tools.")
    task_description: str = PydanticField(description="A concise description of this specific sub-task.")
    updated_goal: Optional[str] = PydanticField(None, description="If the overall goal needs refinement, provide the new goal statement.")
    args: Union[
        SearchArgs, CalculationArgs, TextAnalysisArgs, PlanningArgs, ReasoningArgs,
        CodeTaskArgs, DataAnalysisArgs, VerificationArgs, SummarizationArgs,
        DecisionArgs, KnowledgeBaseArgs, ImageAnalysisArgs, AudioProcessingArgs,
        VideoProcessingArgs, DocumentAnalysisArgs, MediaTransformationArgs,
        MultiModalSynthesisArgs
    ] = PydanticField(description="The appropriate arguments object for the chosen tool.")

class SynthesizeFinalAnswer(BaseModel):
    thought: str = PydanticField(description="Your reasoning for the decision. Explain why this next step is the best one.")
    action: Literal["synthesize_final_answer"] = "synthesize_final_answer"
    reason_for_synthesis_or_stop: str = PydanticField(description="Explain why you're synthesizing the final answer now.")
    updated_goal: Optional[str] = PydanticField(None, description="If the overall goal needs refinement, provide the new goal statement.")

class StopProcessing(BaseModel):
    thought: str = PydanticField(description="Your reasoning for the decision. Explain why this next step is the best one.")
    action: Literal["stop_processing"] = "stop_processing"
    reason_for_synthesis_or_stop: str = PydanticField(description="Explain why you're stopping the processing.")
    updated_goal: Optional[str] = PydanticField(None, description="If the overall goal needs refinement, provide the new goal statement.")

# Discriminated union for planner decisions
PlannerDecision = Annotated[
    Union[CallTool, SynthesizeFinalAnswer, StopProcessing],
    PydanticField(discriminator="action")
]

# --- Task and State Models for the Loop ---
class Task(BaseModel):
    """Represents a single task in the plan."""
    task_id: str = PydanticField(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    description: str
    agent_tool_name: Optional[str] = None # e.g., "search_information"
    tool_input: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result_summary: Optional[str] = None # Short summary of the result for the planner
    result_data: Optional[Any] = None # The actual data from the tool

# Define prefix for keys to be excluded from shared_context when passing to planner
SHARED_CONTEXT_INTERNAL_KEY_PREFIX = "__loop_internal_"

class RouterLoopState(BaseModel):
    """Manages the state of the problem-solving process across iterations."""
    session_id: str = PydanticField(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    original_query: str
    current_goal: str # Can be refined by the planner
    
    plan: List[Task] = PydanticField(default_factory=list)
    task_results: Dict[str, Any] = PydanticField(default_factory=dict)
    shared_context: Dict[str, Any] = PydanticField(default_factory=dict)
    
    iteration_count: int = 0 # Number of iterations *completed*
    max_iterations: int = 7
    
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None
    
    previous_state_hash: Optional[str] = None

    def add_task_result(self, task: Task, result: Any):
        """Adds a task result and updates the task status."""
        task.status = "completed"
        task.result_data = result
        if isinstance(result, BaseModel):
            task.result_summary = f"Completed. Result: {str(result.model_dump(exclude_none=True))[:200]}..."
        else:
            task.result_summary = f"Completed. Result: {str(result)[:200]}..."
        self.task_results[task.task_id] = result

    def get_summary_for_planner(self) -> str:
        """Creates a concise summary of the current state for the planner LLM."""
        # Iteration number for display is 1-indexed
        display_iteration = self.iteration_count + 1
        summary = f"Original Query: {self.original_query}\nCurrent Goal: {self.current_goal}\n"
        summary += f"Current Iteration: {display_iteration}/{self.max_iterations}\n\n"

        summary += "Context from previous steps (if any):\n"
        if not self.task_results:
            summary += "No tasks completed yet.\n"
        else:
            for task_id, task_obj_result in self.task_results.items():
                original_task = next((t for t in self.plan if t.task_id == task_id), None)
                task_desc = original_task.description if original_task else "Unknown task"
                
                result_summary = "Result not summarized"
                if original_task and hasattr(original_task, 'result_summary') and original_task.result_summary:
                    result_summary = original_task.result_summary
                elif isinstance(task_obj_result, BaseModel):
                    result_summary = f"Result: {str(task_obj_result.model_dump(exclude_none=True))[:150]}..."
                else:
                    result_summary = f"Result: {str(task_obj_result)[:150]}..."
                summary += f"- Task '{task_desc}' (ID: {task_id}): {result_summary}\n"
        
        if self.shared_context:
            planner_friendly_shared_context = {
                k: v for k, v in self.shared_context.items() 
                if not k.startswith(SHARED_CONTEXT_INTERNAL_KEY_PREFIX)
            }
            if planner_friendly_shared_context:
                 summary += f"\nShared Context: {str(planner_friendly_shared_context)[:300]}...\n"
            
        summary += "\nWhat is the next single, most critical step or tool to use to achieve the current goal? Or should we synthesize the final answer?"
        return summary

    def _calculate_state_hash(self) -> str:
        """Calculates a hash of the current relevant state for stall detection."""
        data_to_hash = self.current_goal
        for task_id in sorted(self.task_results.keys()): # Sort for consistent hash
            # Attempt to get a stable string representation of results
            res_str = ""
            if isinstance(self.task_results[task_id], BaseModel):
                try:
                    res_str = self.task_results[task_id].model_dump_json()
                except: # Fallback for complex objects
                    res_str = str(self.task_results[task_id])

            else:
                res_str = str(self.task_results[task_id])
            data_to_hash += f"{task_id}:{res_str}"
        
        planner_friendly_shared_context = {
            k: v for k, v in self.shared_context.items() 
            if not k.startswith(SHARED_CONTEXT_INTERNAL_KEY_PREFIX)
        }
        # Sort shared context by key for consistent hashing
        for key in sorted(planner_friendly_shared_context.keys()):
            data_to_hash += f"{key}:{str(planner_friendly_shared_context[key])}"

        return hashlib.md5(data_to_hash.encode('utf-8', 'ignore')).hexdigest()



# --- Agent Result Models (Unchanged) ---
class SearchResult(BaseModel): query: str; results: List[str]
class CalculationResult(BaseModel): input: str; result: float; steps: List[str]
class TextAnalysisResult(BaseModel): sentiment: str; summary: str; key_points: List[str]
class PlanningAgentResult(BaseModel): original_query: str; plan: List[str]; subtasks: List[Dict[str, Any]]
class ReasoningAgentResult(BaseModel): problem: str; reasoning_steps: List[str]; conclusion: str
class CodingAgentResult(BaseModel): task_description: str; language: Optional[str] = "python"; code_snippet: str; explanation: Optional[str] = None; dependencies: Optional[List[str]] = None
class DataAnalysisResult(BaseModel): dataset_description: Optional[str] = None; analysis_type: str; results: Dict[str, Any]; insights: List[str]
class VerificationResult(BaseModel): item_to_verify: str; is_correct: bool; confidence: float; explanation: Optional[str] = None
class SummarizationAgentResult(BaseModel): original_content_snippet: str; summary: str; length_constraint: Optional[str] = None
class DecisionAgentResult(BaseModel): inputs: List[Dict[str, Any]]; decision: str; justification: str; alternatives_considered: Optional[List[str]] = None
class KnowledgeBaseResult(BaseModel): query: str; retrieved_facts: List[Dict[str, Any]]; action_taken: Optional[str] = None
class MediaInput(BaseModel): media_id: str; media_type: str; url: Optional[str] = None; content_description: Optional[str] = None
class ImageAnalysisResult(BaseModel): image_description: str; objects_detected: Optional[List[str]] = None; text_in_image: Optional[str] = None
class AudioProcessingResult(BaseModel): transcription: Optional[str] = None; speaker_diarization: Optional[List[Dict[str, Any]]] = None; sound_events: Optional[List[str]] = None
class VideoProcessingResult(BaseModel): video_summary: str; key_frames_descriptions: Optional[List[str]] = None; scene_detection: Optional[List[Dict[str, Any]]] = None
class DocumentAnalysisResult(BaseModel): document_type: str; extracted_text_summary: str; tables_found: Optional[List[Dict[str, Any]]] = None; key_information: Optional[Dict[str, Any]] = None
class MediaTransformationResult(BaseModel): input_media: MediaInput; output_media_type: str; output_description: str
class MultiModalSynthesisResult(BaseModel): inputs_summary: List[MediaInput]; combined_insights: str; generated_content_description: Optional[str] = None

class AggregatedResponse(BaseModel):
    """Final response after iterative processing."""
    final_answer: str = PydanticField(description="Synthesized response from the loop.")
    iterations_attempted: int # Renamed for clarity
    total_tokens_consumed: Optional[int] = None
    total_cost_incurred: Optional[float] = None
    session_id: str
    stop_reason: Optional[str] = None

@dataclass
class RouterDeps:
    context: Dict[str, Any] = dataclass_field(default_factory=dict) # Correct use of default_factory
    console: Any = None

# --- Google Gemini Model Setup ---
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    warnings.warn(
        "GEMINI_API_KEY environment variable not set. "
        "Google Gemini models may not function without an API key or other configured authentication.",
        UserWarning
    )
google_provider = GoogleGLAProvider(api_key=gemini_api_key)

try:
    shared_gemini_model = GeminiModel(GEMINI_MODEL_NAME, provider=google_provider)
except Exception as e:
    warnings.warn(f"Failed to initialize shared GeminiModel '{GEMINI_MODEL_NAME}': {e}. Agents may not function.", UserWarning)
    shared_gemini_model = None

# --- Planner/Orchestrator and Synthesis Agents ---
planner_orchestrator_agent = Agent(
    shared_gemini_model if shared_gemini_model else "placeholder_for_failed_gemini_init",
    output_type=Union[CallTool, SynthesizeFinalAnswer, StopProcessing],
    system_prompt="""
You are a master orchestrator and planner. Your goal is to achieve the user's overall objective by breaking it down if necessary and intelligently calling available tools one step at a time.

For each step, you need to decide whether to call a tool, synthesize a final answer, or stop processing.

Available tools (with required arguments):

1. search_information:
   - args: SearchArgs with field `query: str`
   - Use to find factual information

2. perform_calculation:
   - args: CalculationArgs with field `problem: str`
   - Use for numerical calculations

3. analyze_text_comprehensively:
   - args: TextAnalysisArgs with field `text: str`
   - Use for general text understanding

4. create_plan:
   - args: PlanningArgs with field `query_for_planning: str`
   - Use to break down a complex task
   - Rarely needed by you

5. perform_reasoning:
   - args: ReasoningArgs with field `problem_statement: str`
   - Use for logical deduction

6. execute_code_task:
   - args: CodeTaskArgs with fields `task_description: str, language: str = "python"`
   - Use to write/modify code

7. analyze_data:
   - args: DataAnalysisArgs with fields `data_description: str, analysis_query: str`
   - Use to analyze data

8. verify_information:
   - args: VerificationArgs with fields `item_to_verify: str, criteria: Optional[str] = None`
   - Use to check correctness

9. summarize_content:
   - args: SummarizationArgs with fields `content_to_summarize: str, length_constraint: Optional[str] = "concise"`
   - Use for focused summarization

10. make_decision:
    - args: DecisionArgs with fields `decision_problem: str, options: List[str], criteria: Optional[List[str]] = None`
    - Use to evaluate options

11. interact_with_knowledge_base:
    - args: KnowledgeBaseArgs with fields `kb_query: str, action: str = "retrieve"`
    - Use to store/retrieve facts

12. analyze_image:
    - args: ImageAnalysisArgs with fields `image_identifier: str, prompt: Optional[str] = None`
    - Use to understand images

13. process_audio:
    - args: AudioProcessingArgs with fields `audio_identifier: str, task: str = "transcribe"`
    - Use to analyze audio

14. process_video:
    - args: VideoProcessingArgs with fields `video_identifier: str, task: str = "summarize"`
    - Use to analyze video

15. analyze_document:
    - args: DocumentAnalysisArgs with fields `document_identifier: str, query: Optional[str] = None`
    - Use to extract info from docs

16. transform_media:
    - args: MediaTransformationArgs with fields `input_media_id: str, target_format: str, transformation_prompt: Optional[str] = None`
    - Use to convert media

17. synthesize_from_multi_modal_inputs:
    - args: MultiModalSynthesisArgs with fields `media_ids: List[str], synthesis_goal: str`
    - Use to combine insights

Based on the current state summary, decide the single best next action:

1. To call a tool:
   - Return CallTool with:
     - action: "call_tool"
     - tool_name: the name of the tool (e.g., "search_information")
     - task_description: a concise description of this specific sub-task
     - args: fill the appropriate args object for the chosen tool with all required fields

2. To finalize the answer:
   - Return SynthesizeFinalAnswer with:
     - action: "synthesize_final_answer"
     - reason_for_synthesis_or_stop: explanation for why you're synthesizing now

3. To stop processing:
   - Return StopProcessing with:
     - action: "stop_processing"
     - reason_for_synthesis_or_stop: explanation for why you're stopping

In all cases, include:
- thought: your reasoning for choosing this action
- updated_goal: if the goal needs refinement (optional)

Focus on making progress towards the `current_goal`.
    """
)

synthesis_agent = Agent(
    shared_gemini_model if shared_gemini_model else "placeholder_for_failed_gemini_init",
    output_type=str,
    system_prompt="""
You are a synthesis expert. Generate a comprehensive, coherent, and user-friendly final answer based on the original query, achieved goal, and relevant information gathered from various processing steps.
Combine key findings and results into a single, well-structured response. Address the user's original query directly.
    """
)

agent_model_for_tools = shared_gemini_model if shared_gemini_model else "placeholder_for_failed_gemini_init"

search_agent = Agent(agent_model_for_tools, output_type=SearchResult)
calculation_agent = Agent(agent_model_for_tools, output_type=CalculationResult)
text_analysis_agent = Agent(agent_model_for_tools, output_type=TextAnalysisResult)
planning_agent_instance = Agent(agent_model_for_tools, output_type=PlanningAgentResult)
reasoning_agent_instance = Agent(agent_model_for_tools, output_type=ReasoningAgentResult)
coding_agent_instance = Agent(agent_model_for_tools, output_type=CodingAgentResult)
data_analysis_agent_instance = Agent(agent_model_for_tools, output_type=DataAnalysisResult)
verification_agent_instance = Agent(agent_model_for_tools, output_type=VerificationResult)
summarization_agent_instance = Agent(agent_model_for_tools, output_type=SummarizationAgentResult)
decision_agent_instance = Agent(agent_model_for_tools, output_type=DecisionAgentResult)
knowledge_base_agent_instance = Agent(agent_model_for_tools, output_type=KnowledgeBaseResult)
image_analysis_agent_instance = Agent(agent_model_for_tools, output_type=ImageAnalysisResult)
audio_processing_agent_instance = Agent(agent_model_for_tools, output_type=AudioProcessingResult)
video_processing_agent_instance = Agent(agent_model_for_tools, output_type=VideoProcessingResult)
document_analysis_agent_instance = Agent(agent_model_for_tools, output_type=DocumentAnalysisResult)
media_transformation_agent_instance = Agent(agent_model_for_tools, output_type=MediaTransformationResult)
multi_modal_synthesis_agent_instance = Agent(agent_model_for_tools, output_type=MultiModalSynthesisResult)

AVAILABLE_TOOLS_SPEC = {
    "search_information": {"agent": search_agent, "output_type": SearchResult, "primary_input_key": "query"},
    "perform_calculation": {"agent": calculation_agent, "output_type": CalculationResult, "primary_input_key": "problem"},
    "analyze_text_comprehensively": {"agent": text_analysis_agent, "output_type": TextAnalysisResult, "primary_input_key": "text"},
    "create_plan": {"agent": planning_agent_instance, "output_type": PlanningAgentResult, "primary_input_key": "query_for_planning"},
    "perform_reasoning": {"agent": reasoning_agent_instance, "output_type": ReasoningAgentResult, "primary_input_key": "problem_statement"},
    "execute_code_task": {"agent": coding_agent_instance, "output_type": CodingAgentResult, "primary_input_key": "task_description"}, # other args: language
    "analyze_data": {"agent": data_analysis_agent_instance, "output_type": DataAnalysisResult, "primary_input_key": "data_description"}, # other args: analysis_query
    "verify_information": {"agent": verification_agent_instance, "output_type": VerificationResult, "primary_input_key": "item_to_verify"}, # other args: criteria
    "summarize_content": {"agent": summarization_agent_instance, "output_type": SummarizationAgentResult, "primary_input_key": "content_to_summarize"}, # other args: length_constraint
    "make_decision": {"agent": decision_agent_instance, "output_type": DecisionAgentResult, "primary_input_key": "decision_problem"}, # other args: options, criteria
    "interact_with_knowledge_base": {"agent": knowledge_base_agent_instance, "output_type": KnowledgeBaseResult, "primary_input_key": "kb_query"}, # other args: action
    "analyze_image": {"agent": image_analysis_agent_instance, "output_type": ImageAnalysisResult, "primary_input_key": "image_identifier"}, # other args: prompt
    "process_audio": {"agent": audio_processing_agent_instance, "output_type": AudioProcessingResult, "primary_input_key": "audio_identifier"}, # other args: task
    "process_video": {"agent": video_processing_agent_instance, "output_type": VideoProcessingResult, "primary_input_key": "video_identifier"}, # other args: task
    "analyze_document": {"agent": document_analysis_agent_instance, "output_type": DocumentAnalysisResult, "primary_input_key": "document_identifier"}, # other args: query
    "transform_media": {"agent": media_transformation_agent_instance, "output_type": MediaTransformationResult, "primary_input_key": "input_media_id"}, # other args: target_format, transformation_prompt
    "synthesize_from_multi_modal_inputs": {"agent": multi_modal_synthesis_agent_instance, "output_type": MultiModalSynthesisResult, "primary_input_key": "media_ids"}, # other args: synthesis_goal
}

def _track_usage(loop_state: RouterLoopState, usage_data: Optional[Usage], console: Optional[Any]):
    """Tracks token usage and cost in the loop state."""
    if usage_data and getattr(usage_data, "total_tokens", None) is not None: # Fix #4
        loop_state.shared_context.setdefault(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_tokens_consumed_loop", 0)
        loop_state.shared_context.setdefault(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_cost_incurred_loop", 0.0)
        
        loop_state.shared_context[f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_tokens_consumed_loop"] += usage_data.total_tokens
        cost_per_token = 0.000000525 
        current_cost = usage_data.total_tokens * cost_per_token
        loop_state.shared_context[f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_cost_incurred_loop"] += current_cost
        if console:
            console.print(f"[dim](Usage: {usage_data.total_tokens} tokens, ~${current_cost:.6f})[/dim]")
            
async def plan_next_step(
    loop_state: RouterLoopState, 
    console: Optional[Any] = None,
    verbose: bool = False
) -> Union[CallTool, SynthesizeFinalAnswer, StopProcessing]:
    """
    Runs the planner agent to determine the next step in the problem-solving process.
    
    Args:
        loop_state: The current state of the routing loop
        console: Optional rich console for output
        verbose: Whether to show verbose debug output
        
    Returns:
        A planner decision (CallTool, SynthesizeFinalAnswer, or StopProcessing)
        
    Raises:
        Exception: If the planner agent fails
    """
    # Get the state summary for the planner
    state_summary_for_planner = loop_state.get_summary_for_planner()
    
    # Display the planner input if verbose mode
    if console and verbose:
        console.print("[bold dim]Planner Input Summary:[/bold dim]")
        try:
            from rich.panel import Panel as RichPanel
            console.print(RichPanel(state_summary_for_planner, title="State for Planner", expand=False, border_style="dim"))
        except ImportError:
            console.print(state_summary_for_planner)

    # Prepare dependencies for the planner agent
    planner_deps = RouterDeps(context=loop_state.shared_context.copy(), console=console)
    
    # Run the planner agent
    planner_start_time = time.time()
    try:
        planner_result_obj = await planner_orchestrator_agent.run(state_summary_for_planner, deps=planner_deps)
        planner_decision = planner_result_obj.output
        
        # Debug output if in verbose mode
        if console and verbose:
            console.print(f"[dim]Debug - Planner decision type: {type(planner_decision)}, content: {planner_decision}[/dim]")
            
        # Track token usage
        _track_usage(loop_state, planner_result_obj.usage, console)
        
        # Log the planner decision
        planner_elapsed = time.time() - planner_start_time
        if console:
            console.print(f"[dim]Planner decision in {planner_elapsed:.2f}s: Action='{planner_decision.action}'[/dim]")
            if verbose and planner_decision.thought:
                try:
                    from rich.panel import Panel as RichPanel
                    console.print(RichPanel(planner_decision.thought, title="Planner's Thought", border_style="yellow", expand=False))
                except ImportError:
                    console.print(f"Planner Thought: {planner_decision.thought}")
        
        # Update the goal if the planner has refined it
        if planner_decision.updated_goal:
            if console: 
                console.print(f"[cyan]Goal refined by planner to: '{planner_decision.updated_goal}'[/cyan]")
            loop_state.current_goal = planner_decision.updated_goal
            
        return planner_decision
        
    except Exception as e:
        if console: 
            console.print(f"[bold red]Planner Error: {e}[/bold red]")
        raise


async def execute_tool_call(
    call_tool_decision: CallTool,
    loop_state: RouterLoopState,
    console: Optional[Any] = None,
    verbose: bool = False
) -> bool:
    """
    Executes a tool call based on the planner's decision.
    
    Args:
        call_tool_decision: The CallTool decision from the planner
        loop_state: The current state of the routing loop
        console: Optional rich console for output
        verbose: Whether to show verbose debug output
        
    Returns:
        True if execution succeeded, False otherwise
    """
    tool_name = call_tool_decision.tool_name
    task_description = call_tool_decision.task_description
    
    # Get tool input from the args field
    tool_input_dict = call_tool_decision.args.model_dump() if call_tool_decision.args else {}

    # Validate tool_name
    if not tool_name or tool_name not in AVAILABLE_TOOLS_SPEC:
        if console: 
            console.print(f"[bold red]Planner Error: Invalid or missing tool_name '{tool_name}'. Stopping.[/bold red]")
        loop_state.stop_reason = f"Planner chose invalid tool: {tool_name}"
        return False

    # Create task and add to plan
    current_task = Task(description=task_description, agent_tool_name=tool_name, tool_input=tool_input_dict)
    loop_state.plan.append(current_task)

    # Log tool execution if console is provided
    if console:
        console.print(f"[cyan]Executing Tool:[/cyan] '{tool_name}' for task '{current_task.description}'")
        if verbose: 
            console.print(f"[dim]Tool Input: {tool_input_dict}[/dim]")
    
    # Get tool specification and agent
    tool_spec = AVAILABLE_TOOLS_SPEC[tool_name]
    tool_agent = tool_spec["agent"]
    
    # Prepare arguments for tool_agent.run()
    prompt_value_for_agent: Union[str, List[ModelMessage], BaseModel]
    kwargs_for_agent: Dict[str, Any] = {}

    # Extract primary input for the tool
    temp_tool_input_copy = tool_input_dict.copy()
    primary_input_key = tool_spec.get("primary_input_key")

    # Determine the prompt value based on available inputs
    if primary_input_key and primary_input_key in temp_tool_input_copy:
        prompt_value_for_agent = temp_tool_input_copy[primary_input_key]
        temp_copy = temp_tool_input_copy.copy()
        temp_copy.pop(primary_input_key)
        kwargs_for_agent = temp_copy
    elif "prompt" in temp_tool_input_copy:  # General fallback if planner uses "prompt"
        prompt_value_for_agent = temp_tool_input_copy["prompt"]
        temp_copy = temp_tool_input_copy.copy()
        temp_copy.pop("prompt")
        kwargs_for_agent = temp_copy
    elif temp_tool_input_copy:  # If specific keys but no primary_input_key matched
        if len(temp_tool_input_copy) == 1:
            prompt_value_for_agent = list(temp_tool_input_copy.values())[0]
        else:  # Multiple args, no clear primary
            prompt_value_for_agent = current_task.description
            kwargs_for_agent = temp_tool_input_copy
    else:  # No tool_input provided by planner
        prompt_value_for_agent = current_task.description
    
    # Add router dependencies to kwargs
    kwargs_for_agent['deps'] = RouterDeps(context=loop_state.shared_context.copy(), console=console)

    # Execute the tool
    tool_start_time = time.time()
    try:
        # Special case for knowledge base storage
        if tool_name == "interact_with_knowledge_base" and tool_input_dict.get("action") == "store":
            kb_query_to_store = tool_input_dict.get("kb_query", "Missing kb_query")
            loop_state.shared_context.setdefault("knowledge_base", {})  # Ensure KB dict exists
            fact_id = f"fact_{uuid.uuid4().hex[:8]}"
            loop_state.shared_context["knowledge_base"][fact_id] = kb_query_to_store
            tool_result_data = KnowledgeBaseResult(
                query=kb_query_to_store, 
                retrieved_facts=[{"id": fact_id, "data": kb_query_to_store}], 
                action_taken="stored"
            )
            if console: 
                console.print(f"[dim]Stored in mock KB via tool: '{kb_query_to_store}' as {fact_id}[/dim]")
            # Track usage
            usage_obj = Usage()
            usage_obj.total_tokens = 20
            _track_usage(loop_state, usage_obj, console)
        else:
            # Standard tool execution
            tool_result_obj = await tool_agent.run(prompt_value_for_agent, **kwargs_for_agent)
            tool_result_data = tool_result_obj.output
            _track_usage(loop_state, tool_result_obj.usage, console)

        # Store the result
        loop_state.add_task_result(current_task, tool_result_data)
        
        # Report success if console is provided
        if console:
            console.print(f"[green]✓ Tool '{tool_name}' completed in {time.time() - tool_start_time:.2f}s.[/green]")
            if verbose:
                try:
                    from rich.panel import Panel as RichPanel
                    console.print(
                        RichPanel(
                            str(tool_result_data)[:500] + ('...' if len(str(tool_result_data)) > 500 else ''), 
                            title=f"Result from {tool_name}", 
                            expand=False, 
                            border_style="dim"
                        )
                    )
                except ImportError:
                    console.print(f"Result from {tool_name}: {str(tool_result_data)[:500]}")
        
        return True
        
    except Exception as e:
        # Handle tool execution failure
        if console: 
            console.print(f"[bold red]Tool Error ({tool_name}): {e}[/bold red]")
        current_task.status = "failed"
        current_task.result_summary = f"Error: {e}"
        loop_state.stop_reason = f"Tool {tool_name} failed: {e}"
        return False


async def handle_terminal_decision(
    decision: Union[SynthesizeFinalAnswer, StopProcessing],
    loop_state: RouterLoopState,
    console: Optional[Any] = None,
    verbose: bool = False
) -> None:
    """
    Handles terminal decisions (synthesis or stop) from the planner.
    
    Args:
        decision: Either a SynthesizeFinalAnswer or StopProcessing decision
        loop_state: The current state of the routing loop 
        console: Optional rich console for output
        verbose: Whether to show verbose debug output
    """
    # Handle SynthesizeFinalAnswer
    if isinstance(decision, SynthesizeFinalAnswer):
        reason = decision.reason_for_synthesis_or_stop
        if console:
            console.print(f"[bold green]Planner decided to synthesize final answer.[/bold green] Reason: {reason}")
        
        # Create synthesis prompt
        synthesis_prompt = f"Original Query: {loop_state.original_query}\nAchieved Goal: {loop_state.current_goal}\n\nKey information gathered:\n"
        if not loop_state.task_results:
            synthesis_prompt += "No specific information was gathered by tools. Synthesize based on the query and goal directly if possible, or state that no further information could be obtained."
        else:
            for task_id, result_data in loop_state.task_results.items():
                original_task = next((t for t in loop_state.plan if t.task_id == task_id), None)
                task_desc_for_synth = original_task.description if original_task else f"Task {task_id}"
                synthesis_prompt += f"- Result from '{task_desc_for_synth}': {str(result_data)[:300]}...\n"
        
        # Add shared context if available
        planner_friendly_shared_context = {
            k: v for k, v in loop_state.shared_context.items() 
            if not k.startswith(SHARED_CONTEXT_INTERNAL_KEY_PREFIX)
        }
        if planner_friendly_shared_context:
             synthesis_prompt += f"\nRelevant Shared Context: {str(planner_friendly_shared_context)[:300]}...\n"

        # Run synthesis 
        synthesis_start_time = time.time()
        try:
            synthesis_result_obj = await synthesis_agent.run(synthesis_prompt)
            loop_state.final_answer = synthesis_result_obj.output
            _track_usage(loop_state, synthesis_result_obj.usage, console)
        except Exception as e:
            if console: 
                console.print(f"[bold red]Synthesis Error: {e}[/bold red]")
            loop_state.final_answer = f"Error during final synthesis: {e}."
            loop_state.stop_reason = f"Synthesis agent failed: {e}"
        
        if console: 
            console.print(f"[green]✓ Synthesis completed in {time.time() - synthesis_start_time:.2f}s.[/green]")
    
    # Handle StopProcessing 
    elif isinstance(decision, StopProcessing):
        loop_state.stop_reason = decision.reason_for_synthesis_or_stop
        if console:
            console.print(f"[bold yellow]Planner decided to stop processing.[/bold yellow] Reason: {loop_state.stop_reason}")


async def handle_max_iterations_reached(
    loop_state: RouterLoopState,
    console: Optional[Any] = None
) -> None:
    """
    Handles the case when maximum iterations are reached.
    
    Args:
        loop_state: The current state of the routing loop
        console: Optional rich console for output
    """
    loop_state.stop_reason = "Maximum iterations reached."
    if console:
        console.print(f"[bold yellow]Maximum iterations ({loop_state.max_iterations}) reached. Stopping.[/bold yellow]")
    
    # Only attempt synthesis if we don't already have a final answer
    if not loop_state.final_answer:  
        if console:
            console.print("[dim]Attempting final synthesis before exiting...[/dim]")
            
        # Create a simpler synthesis prompt for max iterations case
        synthesis_prompt = f"Original Query: {loop_state.original_query}\nGoal: {loop_state.current_goal}\n"
        synthesis_prompt += "Max iterations reached. Synthesize the best possible answer with information gathered so far:\n"
        
        # Add task results if available
        if loop_state.task_results:
            for task_id, result_data in loop_state.task_results.items():
                original_task = next((t for t in loop_state.plan if t.task_id == task_id), None) 
                task_desc = original_task.description if original_task else f"Task {task_id}"
                synthesis_prompt += f"- {task_desc}: {str(result_data)[:150]}...\n"
        
        # Run synthesis
        try:
            synthesis_result_obj = await synthesis_agent.run(synthesis_prompt)
            loop_state.final_answer = synthesis_result_obj.output
            _track_usage(loop_state, synthesis_result_obj.usage, console)
        except Exception as e:
            loop_state.final_answer = f"Max iterations reached. Synthesis failed: {e}"


def update_iteration_state(
    loop_state: RouterLoopState,
    console: Optional[Any] = None
) -> bool:
    """
    Updates the iteration state and performs stall detection.
    
    Args:
        loop_state: The current state of the routing loop
        console: Optional rich console for output
        
    Returns:
        True if the iteration should continue, False if we should stop (stalled)
    """
    # Calculate state hash for stall detection
    current_iteration_end_hash = loop_state._calculate_state_hash()
    
    # Check for stall if we have a previous hash
    if loop_state.previous_state_hash is not None and current_iteration_end_hash == loop_state.previous_state_hash:
        loop_state.stop_reason = "Processing stalled; state has not changed since last iteration."
        current_iteration_display_num = loop_state.iteration_count + 1
        if console:
            console.print(f"[bold yellow]Processing stalled after iteration {current_iteration_display_num}. Stopping.[/bold yellow]")
        loop_state.iteration_count += 1  # Count the stalled iteration
        return False
    
    # Store current hash for next iteration comparison
    loop_state.previous_state_hash = current_iteration_end_hash
    
    # Increment iteration counter
    loop_state.iteration_count += 1
    
    # Check if we've reached max iterations
    if loop_state.iteration_count >= loop_state.max_iterations:
        return False
        
    return True


async def process_query_iteratively(
    query: str,
    initial_context: Optional[Dict] = None,
    initial_message_history: Optional[List[ModelMessage]] = None,
    console: Any = None,
    verbose: bool = False
) -> Tuple[AggregatedResponse, List[ModelMessage]]:
    """Process a user query iteratively using a planner-driven loop."""
    start_time_overall = time.time()

    # Check if model is initialized correctly
    if not shared_gemini_model:
        error_message = "Core Gemini model initialization failed. Cannot process query."
        if console: 
            console.print(f"[bold red]{error_message}[/bold red]")
        agg_response = AggregatedResponse(
            final_answer=error_message, 
            iterations_attempted=0, 
            session_id="error_session",
            stop_reason="Model initialization failure"
        )
        return agg_response, initial_message_history or [UserMessage(content=query), AssistantMessage(content=error_message)]

    # Initialize router loop state
    loop_state = RouterLoopState(
        original_query=query,
        current_goal=query,
        shared_context=initial_context.get("shared_context", {}) if initial_context else {}
    )
    
    # Initialize internal tracking keys
    loop_state.shared_context.setdefault(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_tokens_consumed_loop", 0)
    loop_state.shared_context.setdefault(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_cost_incurred_loop", 0.0)

    # Prepare message history
    final_message_history: List[ModelMessage] = list(initial_message_history) if initial_message_history else []
    
    # Add user query to history if not already present
    user_query_in_history = False
    for msg in final_message_history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and part.content == query:
                    user_query_in_history = True
                    break
    if not user_query_in_history:
        final_message_history.append(ModelRequest(parts=[UserPromptPart(content=query)]))

    # Display initial info
    if console:
        console.print(f"[bold blue]Starting iterative processing (model: {GEMINI_MODEL_NAME}):[/bold blue] '{query}'")
        console.print(f"[dim]Session ID: {loop_state.session_id}[/dim]")

    # Main loop
    while loop_state.iteration_count < loop_state.max_iterations:
        # Display iteration heading
        current_iteration_display_num = loop_state.iteration_count + 1  # For logging 1-indexed
        if console:
            console.print(f"\n[bold magenta]--- Iteration {current_iteration_display_num} ---[/bold magenta]")

        # Get next step from planner
        try:
            planner_decision = await plan_next_step(loop_state, console, verbose)
        except Exception as e:
            loop_state.stop_reason = f"Planner agent failed: {e}"
            break
            
        # Process the planner decision
        if isinstance(planner_decision, CallTool):
            # Execute the tool call
            success = await execute_tool_call(planner_decision, loop_state, console, verbose)
            if not success:
                break  # Tool execution failed
                
        elif isinstance(planner_decision, SynthesizeFinalAnswer) or isinstance(planner_decision, StopProcessing):
            # Handle terminal decisions (synthesis or stop)
            await handle_terminal_decision(planner_decision, loop_state, console, verbose)
            # Terminal decisions increment iteration_count internally and break the loop
            break
            
        else:
            # Unknown decision type (shouldn't happen with discriminated union)
            loop_state.stop_reason = "Planner returned unknown action type"
            if console:
                console.print(f"[bold red]Error: Unknown planner action type. Stopping.[/bold red]")
            loop_state.iteration_count += 1  # Count the error iteration
            break
            
        # Update iteration state and check for stalls/max iterations
        should_continue = update_iteration_state(loop_state, console)
        if not should_continue:
            # If we hit max iterations, handle that specifically
            if loop_state.iteration_count >= loop_state.max_iterations:
                await handle_max_iterations_reached(loop_state, console)
            break

    # Post-Loop processing
    # Ensure iteration count is at least 1 even if loop didn't run but model was ok
    final_iterations_attempted = loop_state.iteration_count
    if final_iterations_attempted == 0 and shared_gemini_model:
        final_iterations_attempted = 1

    # Set default final answer if none was generated
    if not loop_state.final_answer:
        loop_state.final_answer = f"Processing stopped. Reason: {loop_state.stop_reason or 'Unknown reason after loop.'}"
        if console:
            console.print(f"[yellow]No final answer explicitly synthesized. Defaulting stop message.[/yellow]")

    # Create the aggregated response
    aggregated_response = AggregatedResponse(
        final_answer=loop_state.final_answer,
        iterations_attempted=final_iterations_attempted,
        session_id=loop_state.session_id,
        stop_reason=loop_state.stop_reason,
        total_tokens_consumed=loop_state.shared_context.get(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_tokens_consumed_loop", 0),
        total_cost_incurred=loop_state.shared_context.get(f"{SHARED_CONTEXT_INTERNAL_KEY_PREFIX}total_cost_incurred_loop", 0.0)
    )
    
    # Add the final answer to the message history
    from datetime import datetime
    final_message_history.append(
        ModelResponse(
            parts=[TextPart(content=aggregated_response.final_answer)],
            model_name=GEMINI_MODEL_NAME,
            usage=Usage(),
            timestamp=datetime.now(),
        )
    )

    # Display final statistics
    if console:
        console.print(f"\n[bold green]--- Iterative Processing Complete ---[/bold green]")
        console.print(f"[dim]Total time: {time.time() - start_time_overall:.2f}s[/dim]")
        console.print(f"[dim]Iterations Attempted: {aggregated_response.iterations_attempted}[/dim]")
        console.print(f"[dim]Total Tokens (Loop): {aggregated_response.total_tokens_consumed}[/dim]")
        console.print(f"[dim]Estimated Cost (Loop): ${aggregated_response.total_cost_incurred:.6f}[/dim]")
        if aggregated_response.stop_reason:
             console.print(f"[yellow]Stop Reason: {aggregated_response.stop_reason}[/yellow]")

    return aggregated_response, final_message_history

if __name__ == "__main__":
    import asyncio
    import sys
    import argparse
    from rich.console import Console
    # Panel and Markdown are imported globally now for use in process_query_iteratively's verbose output
    # from rich.panel import Panel 
    # from rich.markdown import Markdown
    
    parser = argparse.ArgumentParser(description="Iterative Router Agent CLI (Gemini)")
    parser.add_argument("query", nargs="?", default="What is the capital of France? Then search for its current population.",
                       help="The query to process")
    parser.add_argument("-q", "--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--no-console", action="store_true", help="Disable rich console output")
    args = parser.parse_args()
    
    is_cli_verbose = not args.quiet
    console_instance = None if args.no_console else Console(highlight=False, log_time=False, log_path=False)
    
    async def display_results_cli(response: AggregatedResponse, console_cli: Optional[Console]):
        # Import Rich components here if only used for CLI display and console_cli is active
        if console_cli:
            from rich.panel import Panel as RichPanelCli # Use alias to avoid conflict if global Panel exists
            from rich.markdown import Markdown as RichMarkdownCli
            console_cli.print("\n" * 2)
            try:
                console_cli.print(RichPanelCli(
                    RichMarkdownCli(response.final_answer), 
                    title=f"Final Answer (Session: {response.session_id})", 
                    border_style="bold green",
                    subtitle=f"Completed in {response.iterations_attempted} iterations." + (f" Stopped: {response.stop_reason}" if response.stop_reason else "")
                ))
            except Exception: # Fallback if Rich components fail for any reason
                console_cli.print("--- Final Answer ---")
                console_cli.print(response.final_answer)

            console_cli.print(f"[dim]Overall Tokens: {response.total_tokens_consumed}, Cost: ~${response.total_cost_incurred:.6f}[/dim]")
        else:
            print("\n--- Final Answer ---")
            print(f"Session ID: {response.session_id}")
            print(f"Iterations: {response.iterations_attempted}")
            if response.stop_reason: print(f"Stop Reason: {response.stop_reason}")
            print(response.final_answer)
            print(f"Overall Tokens: {response.total_tokens_consumed}, Cost: ~${response.total_cost_incurred:.6f}")

    async def single_query_mode_cli(query_text):
        response_obj, _ = await process_query_iteratively(
            query_text,
            console=console_instance,
            verbose=is_cli_verbose
        )
        await display_results_cli(response_obj, console_instance)

    async def interactive_mode_session_cli():
        from rich.prompt import Prompt # Keep local to interactive mode
        from rich.panel import Panel as RichPanelCli # Keep local to interactive mode

        if console_instance:
            console_instance.print(RichPanelCli(
                "[bold]Iterative Router Agent (Gemini) - Interactive Mode[/bold]\n"
                "Commands: /exit, /help, /clear, /verbose, /quiet",
                title="Welcome", border_style="blue"
            ))
        else: print("\nIterative Router Agent (Gemini) - Interactive Mode\nCommands: /exit, /help, /clear, /verbose, /quiet")
        
        current_verbose_setting = is_cli_verbose

        while True:
            try:
                if console_instance: user_query = Prompt.ask("\n[bold blue]Your Query[/bold blue]")
                else: user_query = input("\nYour Query > ")
                
                if not user_query.strip(): continue

                if user_query.lower() == "/exit": break
                elif user_query.lower() == "/help":
                    help_text = "Commands:\n/exit - Exit\n/help - Show this help\n/clear - Clear screen\n/verbose - Enable detailed logs\n/quiet - Disable detailed logs"
                    if console_instance: console_instance.print(RichPanelCli(help_text, title="Help"))
                    else: print(help_text)
                    continue
                elif user_query.lower() == "/clear":
                    if console_instance: console_instance.clear()
                    else: print("\n" * 50)
                    continue
                elif user_query.lower() == "/verbose":
                    current_verbose_setting = True
                    if console_instance: console_instance.print("[yellow]Verbose logging enabled.[/yellow]")
                    else: print("Verbose logging enabled.")
                    continue
                elif user_query.lower() == "/quiet":
                    current_verbose_setting = False
                    if console_instance: console_instance.print("[yellow]Quiet logging enabled.[/yellow]")
                    else: print("Quiet logging enabled.")
                    continue
                
                response_obj, _ = await process_query_iteratively(
                    user_query,
                    initial_context={"shared_context": {}}, 
                    console=console_instance,
                    verbose=current_verbose_setting
                )
                await display_results_cli(response_obj, console_instance)
                
            except KeyboardInterrupt: break
            except Exception as e:
                if console_instance: console_instance.print(f"[bold red]CLI Error: {str(e)}[/bold red]")
                else: print(f"CLI Error: {str(e)}")
        
        if console_instance: console_instance.print("[yellow]Exiting interactive mode.[/yellow]")
        else: print("Exiting interactive mode.")

    async def main_cli_runner():
        if args.interactive or not (args.query and args.query.strip()):
            await interactive_mode_session_cli()
        else:
            await single_query_mode_cli(args.query)
    
    asyncio.run(main_cli_runner())