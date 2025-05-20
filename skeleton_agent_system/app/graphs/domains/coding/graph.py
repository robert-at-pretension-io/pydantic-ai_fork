from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, cast, Optional, List, Sequence, TYPE_CHECKING

from pydantic_ai import format_as_xml, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.settings import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.core.base_agent import agent_registry
from app.core.context import context_builder, ContextBuildRequest
from app.models.outputs import CodeResult
from app.models.states import CodingState


@dataclass
class CodeTaskAnalysis(BaseNode[CodingState]):
    """Analyzes the coding task to understand requirements and approach."""
    
    async def run(self, ctx: GraphRunContext[CodingState]) -> CodeGeneration:
        # Get code completion agent from registry
        code_agent = agent_registry.get_agent("code_completion")
        
        # Build optimized context for task analysis
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="code_completion",
                task=f"Analyze coding task: {ctx.state.query}",
                deps=ctx.state.context,
                max_tokens=1024
            )
        )
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                           ctx.state.agent_messages.get("analysis", []))
        
        # Run the code agent to analyze the task
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await code_agent.run(  # type: ignore
            context_pkg.user_prompt,
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the analysis in state
        ctx.state.agent_messages["analysis"] = result.new_messages()
        ctx.state.task_analysis = {
            "description": ctx.state.query,
            "analysis": result.output.explanation if hasattr(result.output, "explanation") else "",
            "approach": "Identified from analysis"
        }
        
        # Continue to code generation
        return CodeGeneration()


@dataclass
class CodeGeneration(BaseNode[CodingState]):
    """Generates code based on the task analysis."""
    
    async def run(self, ctx: GraphRunContext[CodingState]) -> TestGeneration:
        # Get code completion agent from registry
        code_agent = agent_registry.get_agent("code_completion")
        
        # Get the task analysis safely
        task_analysis = ctx.state.task_analysis or {}
        
        # Build optimized context for code generation
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="code_completion",
                task="Generate code",
                deps={
                    "analysis": task_analysis,
                    **ctx.state.context
                },
                max_tokens=2048
            )
        )
        
        # Format the task analysis as XML
        formatted_analysis = format_as_xml(task_analysis)
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                           ctx.state.agent_messages.get("generation", []))
        
        # Run the code agent to generate code
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await code_agent.run(  # type: ignore
            f"Generate code for this task:\n{formatted_analysis}",
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the generated code in state
        ctx.state.agent_messages["generation"] = result.new_messages()
        ctx.state.generated_code = result.output.code if hasattr(result.output, "code") else ""
        
        # Continue to test generation
        return TestGeneration()


@dataclass
class TestGeneration(BaseNode[CodingState, None, CodeResult]):
    """Generates tests for the code and finalizes the result."""
    
    async def run(self, ctx: GraphRunContext[CodingState]) -> End[CodeResult]:
        # Get code completion agent from registry
        code_agent = agent_registry.get_agent("code_completion")
        
        # Get the task analysis and generated code safely
        task_analysis = ctx.state.task_analysis or {}
        generated_code = ctx.state.generated_code or ""
        
        # Build optimized context for test generation
        # Convert GraphRunContext to RunContext for context_builder
        run_ctx = cast(RunContext[Any], ctx) if ctx else None
        
        context_pkg = await context_builder.build_context(
            run_ctx,
            ContextBuildRequest(
                agent_name="code_completion",
                task="Generate tests",
                deps={
                    "code": generated_code,
                    "analysis": task_analysis,
                    **ctx.state.context
                },
                max_tokens=1024
            )
        )
        
        # Type cast to avoid mypy errors with system_prompt
        system_prompt = cast(Optional[str], context_pkg.system_prompt)
        
        # Cast message history to the correct type
        message_history = cast(Optional[List[ModelRequest | ModelResponse]], 
                          ctx.state.agent_messages.get("testing", []))
        
        # Run the code agent to generate tests
        # Use type ignore to bypass complex type checks with pydantic-ai's Agent
        result = await code_agent.run(  # type: ignore
            f"Generate tests for this code:\n```\n{generated_code}\n```",
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Store the test results in state
        ctx.state.agent_messages["testing"] = result.new_messages()
        ctx.state.test_results = {
            "tests": result.output.tests if hasattr(result.output, "tests") else "",
            "status": "Generated"
        }
        
        # Get explanation safely
        explanation = task_analysis.get("analysis", "") if task_analysis else ""
        tests = ctx.state.test_results.get("tests", "") if ctx.state.test_results else ""
        
        # Create the final code result
        final_code = CodeResult(
            task=ctx.state.query,
            code=generated_code,
            explanation=explanation,
            tests=tests
        )
        
        ctx.state.final_code = final_code
        
        # End the graph with the final code
        return End(final_code)


# Create the coding graph
coding_graph = Graph(
    nodes=[CodeTaskAnalysis, CodeGeneration, TestGeneration],
    state_type=CodingState
)