from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, cast, List, Optional, Sequence, TYPE_CHECKING

from pydantic_ai import format_as_xml, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.core.base_agent import agent_registry
from app.core.context import context_builder, ContextBuildRequest
from app.models.outputs import BashResult
from app.models.states import OperationsState
from app.tools.bash import bash_executor


@dataclass
class BashCommandFormulation(BaseNode[OperationsState]):
    """Formulates a bash command from the user's query."""
    
    async def run(self, ctx: GraphRunContext[OperationsState]) -> CommandExecution:
        # Get bash agent from registry
        bash_agent = agent_registry.get_agent("bash_executor")
        
        # Build optimized context for command formulation
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="bash_executor",
                task=f"Formulate bash command for: {ctx.state.query}",
                deps=ctx.state.context,
                max_tokens=1024
            )
        )
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                           ctx.state.agent_messages.get("formulation", []))
        
        # Run the bash agent to formulate the command
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await bash_agent.run(  # type: ignore
            context_pkg.user_prompt,
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the formulated command in state
        ctx.state.agent_messages["formulation"] = result.new_messages()
        ctx.state.command = result.output.command
        
        # Continue to command execution
        return CommandExecution()


@dataclass
class CommandExecution(BaseNode[OperationsState]):
    """Executes the formulated bash command."""
    
    async def run(self, ctx: GraphRunContext[OperationsState]) -> OutputAnalysis:
        if not ctx.state.command:
            # If no command was formulated, return a dummy result
            ctx.state.command_output = "No command to execute"
            return OutputAnalysis()
        
        # Execute the bash command using the bash tool
        command_output = await bash_executor(ctx.state.command)
        
        # Store the command output in state
        ctx.state.command_output = command_output
        
        # Continue to output analysis
        return OutputAnalysis()


@dataclass
class OutputAnalysis(BaseNode[OperationsState, None, BashResult]):
    """Analyzes the command output and produces a final result."""
    
    async def run(self, ctx: GraphRunContext[OperationsState]) -> End[BashResult]:
        # Get bash agent from registry
        bash_agent = agent_registry.get_agent("bash_executor")
        
        # Build optimized context for output analysis
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="bash_executor",
                task="Analyze command output",
                deps={
                    "command": ctx.state.command,
                    "output": ctx.state.command_output,
                    **ctx.state.context
                },
                max_tokens=1024
            )
        )
        
        # Format the command and output as XML
        command_data = {
            "command": ctx.state.command,
            "output": ctx.state.command_output
        }
        formatted_data = format_as_xml(command_data)
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                           ctx.state.agent_messages.get("analysis", []))
        
        # Run the bash agent to analyze the output
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await bash_agent.run(  # type: ignore
            f"Analyze this command and output:\n{formatted_data}",
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the analysis in state
        ctx.state.agent_messages["analysis"] = result.new_messages()
        ctx.state.analysis = {
            "explanation": result.output.explanation
        }
        
        # Add null checks before creating the final result
        command = ctx.state.command or ""
        output = ctx.state.command_output or ""
        explanation = ctx.state.analysis.get("explanation") if ctx.state.analysis else ""
        
        # Create the final bash result
        final_result = BashResult(
            command=command,
            output=output,
            exit_code=0 if "error" not in output.lower() else 1,
            success="error" not in output.lower(),
            explanation=explanation
        )
        
        ctx.state.final_result = final_result
        
        # End the graph with the final bash result
        return End(final_result)


# Create the operations graph
operations_graph = Graph(
    nodes=[BashCommandFormulation, CommandExecution, OutputAnalysis],
    state_type=OperationsState
)