from __future__ import annotations

import importlib
import yaml  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

AgentT = TypeVar("AgentT", bound=Agent)


class AgentRegistry:
    """
    A registry for all agents in the system.
    Loads agent configurations from YAML and provides
    methods to get agents by name.
    """

    def __init__(self, config_path: Path | str = "config/agents.yaml") -> None:
        self.config_path = Path(config_path)
        self.agents_config: Dict[str, Dict[str, Any]] = {}
        self.loaded_agents: Dict[str, Agent] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load the agent configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.agents_config = config.get("agents", {})

    def get_agent(self, name: str) -> Agent:
        """
        Get an agent by name. Loads the agent if it's not already loaded.
        
        Args:
            name: The name of the agent as defined in the config file
            
        Returns:
            The agent instance
        """
        if name not in self.agents_config:
            raise ValueError(f"Agent '{name}' not defined in config")
            
        if name not in self.loaded_agents:
            self._load_agent(name)
            
        return self.loaded_agents[name]

    def _load_agent(self, name: str) -> None:
        """
        Load an agent by name from its module.
        
        Args:
            name: The name of the agent as defined in the config file
        """
        config = self.agents_config[name]
        module_name = config["module"]
        class_name = config["class"]
        
        module = importlib.import_module(module_name)
        self.loaded_agents[name] = getattr(module, class_name)
        
    def create_deterministic_agent(
        self, 
        name: str, 
        model: str,
        output_type: Type[Any],
        system_prompt: str,
        tools: Optional[List[Any]] = None
    ) -> Agent:
        """
        Create a deterministic agent with fixed model settings.
        
        Args:
            name: A unique name for the agent
            model: The model to use (e.g., 'openai:gpt-4o')
            output_type: The pydantic model defining the output structure
            system_prompt: The system prompt for the agent
            tools: Optional list of tools for the agent
            
        Returns:
            A configured deterministic agent
        """
        # Create deterministic agent with settings
        # Use type casting to avoid mypy checking model name against literal list
        model_name = cast(str, model)
        model_settings = cast(ModelSettings, {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "seed": 42
        })
        agent = Agent(
            model=model_name,
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=model_settings
        )
        
        # Register tools if provided
        if tools:
            for tool in tools:
                # Using private API to register tools
                # This is a workaround for type checking issues
                if hasattr(agent, "_register_tool"):
                    getattr(agent, "_register_tool")(tool)
                
        # Register the agent in the registry
        self.loaded_agents[name] = agent
        
        return agent


# Singleton instance
agent_registry = AgentRegistry()