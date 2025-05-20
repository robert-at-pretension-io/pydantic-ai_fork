from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Generic, Optional, TypeVar, Union, cast, Dict
import importlib.util

from pydantic_graph import BaseNode, End, GraphRunContext

# Check if logfire is available
logfire_available = importlib.util.find_spec("logfire") is not None
if logfire_available:
    import logfire  # type: ignore

# Set up logger
logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
NextNodeT = TypeVar("NextNodeT")


@dataclass
class TracedNode(BaseNode[StateT, OutputT, NextNodeT], Generic[StateT, OutputT, NextNodeT]):
    """
    A wrapper node that adds tracing for state transitions in the graph.
    This allows better visibility into the graph execution flow.
    
    Usage:
        # Instead of directly using a node
        @dataclass
        class MyNode(BaseNode[MyState]):
            async def run(self, ctx: GraphRunContext[MyState]) -> NextNode:
                # logic...
                return NextNode()
        
        # Wrap it with tracing
        @dataclass
        class MyNode(TracedNode[MyState]):
            async def process(self, ctx: GraphRunContext[MyState]) -> NextNode:
                # Same logic but with automatic tracing
                return NextNode()
    """
    
    node_attributes: Optional[dict[str, Any]] = None
    
    async def run(self, ctx: GraphRunContext[StateT, OutputT]) -> Union[BaseNode[StateT, OutputT, Any], End[NextNodeT]]:
        """Run the node with tracing."""
        node_name = self.__class__.__name__
        attrs = self.node_attributes or {}
        
        # Log node execution
        node_log_attrs = {
            "node_type": node_name,
            "state_type": ctx.state.__class__.__name__,
            **attrs
        }
        
        if logfire_available:
            # Create a span for the node execution
            with logfire.span(f"graph_node:{node_name}", **node_log_attrs):
                # Log the state transition
                logfire.info(
                    f"Executing node: {node_name}",
                    node=node_name,
                    state=ctx.state.__class__.__name__
                )
                
                # Execute the actual node logic
                result = await self.process(ctx)
                
                # Log the next node
                next_node = result.__class__.__name__ if result else "None"
                logfire.info(
                    f"Node {node_name} -> {next_node}",
                    current_node=node_name,
                    next_node=next_node
                )
                
                return result
        else:
            # Log with standard logger when logfire not available
            logger.info(f"Executing node: {node_name}")
            
            # Execute the actual node logic
            result = await self.process(ctx)
            
            # Log the next node
            next_node = result.__class__.__name__ if result else "None"
            logger.info(f"Node {node_name} -> {next_node}")
            
            return result
    
    async def process(self, ctx: GraphRunContext[StateT, OutputT]) -> Union[BaseNode[StateT, OutputT, Any], End[NextNodeT]]:
        """
        Override this method to implement the node's logic.
        This is called by the run method with tracing already set up.
        """
        raise NotImplementedError("Subclasses must implement this method")