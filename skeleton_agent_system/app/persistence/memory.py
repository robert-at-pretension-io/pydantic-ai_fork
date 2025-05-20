from __future__ import annotations

from typing import Dict, List, Optional, Any, TypeVar, Generic, cast
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.in_mem import SimpleStatePersistence, FullStatePersistence

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


class MemoryPersistenceManager:
    """
    Manages in-memory persistence of graph states.
    
    This class provides a simple way to create and manage
    in-memory persistence for graph runs.
    """
    
    def __init__(self) -> None:
        """Initialize the persistence manager."""
        self.simple_persistence: Dict[str, SimpleStatePersistence] = {}
        self.full_persistence: Dict[str, FullStatePersistence] = {}
    
    def get_simple_persistence(self, run_id: str) -> SimpleStatePersistence:
        """
        Get a simple persistence instance for a run ID.
        Creates a new one if it doesn't exist.
        
        Args:
            run_id: The unique ID for the run
            
        Returns:
            A SimpleStatePersistence instance
        """
        if run_id not in self.simple_persistence:
            self.simple_persistence[run_id] = SimpleStatePersistence()
        
        return self.simple_persistence[run_id]
    
    def get_full_persistence(self, run_id: str) -> FullStatePersistence:
        """
        Get a full persistence instance for a run ID.
        Creates a new one if it doesn't exist.
        
        Args:
            run_id: The unique ID for the run
            
        Returns:
            A FullStatePersistence instance
        """
        if run_id not in self.full_persistence:
            self.full_persistence[run_id] = FullStatePersistence()
        
        return self.full_persistence[run_id]
    
    def cleanup_run(self, run_id: str) -> None:
        """
        Remove persistence instances for a run ID.
        
        Args:
            run_id: The unique ID for the run
        """
        if run_id in self.simple_persistence:
            del self.simple_persistence[run_id]
        
        if run_id in self.full_persistence:
            del self.full_persistence[run_id]


# Singleton instance
memory_persistence = MemoryPersistenceManager()