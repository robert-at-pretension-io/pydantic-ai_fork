from __future__ import annotations

from typing import List, cast

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.models.outputs import CalculationResult

# Create a deterministic calculator agent
# Use type casting to avoid mypy checking model name against literal list
model_name = cast(str, "anthropic:claude-3-sonnet")
model_settings = cast(ModelSettings, {
    "temperature": 0,
    "top_p": 1,
    "do_sample": False,
    "seed": 42
})

calculator_agent = Agent(
    model=model_name,
    output_type=CalculationResult,
    system_prompt="""You are a specialized calculation agent.
Your task is to solve mathematical problems accurately and show your work step by step.
Always verify your calculations and provide the final answer.
Be precise with units and mathematical notation.
""",
    model_settings=model_settings
)