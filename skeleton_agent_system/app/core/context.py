from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING, cast
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, format_as_xml
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits


class ContextBuildRequest(BaseModel):
    """Request to build context for a specialized agent."""
    agent_name: str
    task: str
    deps: Dict[str, Any]
    hint_files: Optional[List[str]] = None
    max_tokens: int = 2048


class ContextPackage(BaseModel):
    """Optimized context package for a specialized agent."""
    system_prompt: str
    user_prompt: str
    memory: List[str] = Field(default_factory=list, description="Relevant snippets/RAG chunks")
    token_estimate: int


class ContextBuilder:
    """
    Builds optimized context packages for specialized agents.
    Acts as an intermediary that ensures agents get only relevant context.
    """
    
    def __init__(self) -> None:
        """Initialize the context builder with a dedicated LLM."""
        # Use type casting to avoid mypy checking model name against literal list
        model_name = cast(str, "openai:gpt-4o")
        model_settings = cast(ModelSettings, {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "seed": 42
        })
        self.agent = Agent(
            model=model_name,
            output_type=ContextPackage,
            model_settings=model_settings,
            system_prompt="""You build *concise* prompt bundles for other agents.
1. Classify the task.
2. Select only the most relevant dependencies/snippets.
3. Output a JSON ContextPackage that keeps total tokens under {max_tokens}.
NEVER solve the task itself; your job is to prepare context."""
        )
        
    async def build_context(
        self, 
        ctx: Optional[RunContext[Any]], 
        req: ContextBuildRequest
    ) -> ContextPackage:
        """
        Build an optimized context package for a specialized agent.
        
        Args:
            ctx: The run context from the caller
            req: The context build request
            
        Returns:
            A context package with system prompt, user prompt, and memory
        """
        formatted_request = format_as_xml(req.model_dump())
        result = await self.agent.run(
            f"Build context for agent: {formatted_request}",
            usage=ctx.usage if ctx else None
        )
        return result.output


# Singleton instance
context_builder = ContextBuilder()