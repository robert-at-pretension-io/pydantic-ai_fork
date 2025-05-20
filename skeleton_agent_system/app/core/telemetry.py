from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, cast

from pydantic_ai.usage import Usage


@dataclass
class RunMetrics:
    """Metrics for a single run."""
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    usage: Optional[Usage] = None
    agent_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class TelemetryManager:
    """
    Manages telemetry for agent runs.
    
    This class provides methods to track metrics and logs
    for agent runs.
    """
    
    def __init__(self, logger_name: str = "agent_system"):
        """Initialize the telemetry manager."""
        self.logger = logging.getLogger(logger_name)
        self.runs: Dict[str, RunMetrics] = {}
    
    def start_run(self, run_id: str) -> None:
        """
        Start tracking a new run.
        
        Args:
            run_id: The unique ID for the run
        """
        self.runs[run_id] = RunMetrics(
            run_id=run_id,
            start_time=time.time()
        )
        self.logger.info(f"Started run: {run_id}")
    
    def end_run(self, run_id: str, usage: Optional[Usage] = None) -> None:
        """
        End tracking for a run.
        
        Args:
            run_id: The unique ID for the run
            usage: Optional usage metrics
        """
        if run_id not in self.runs:
            self.logger.warning(f"Cannot end run {run_id} - not found")
            return
        
        run = self.runs[run_id]
        run.end_time = time.time()
        run.duration_ms = (run.end_time - run.start_time) * 1000
        
        if usage:
            run.usage = usage
        
        self.logger.info(
            f"Ended run: {run_id} - Duration: {run.duration_ms:.2f}ms - "
            f"Tokens: {run.usage.total_tokens if run.usage else 'N/A'}"
        )
    
    def record_agent_metrics(self, run_id: str, agent_name: str, metrics: Dict[str, Any]) -> None:
        """
        Record metrics for an agent within a run.
        
        Args:
            run_id: The unique ID for the run
            agent_name: The name of the agent
            metrics: The metrics to record
        """
        if run_id not in self.runs:
            self.logger.warning(f"Cannot record agent metrics for {run_id} - not found")
            return
        
        self.runs[run_id].agent_metrics[agent_name] = metrics
        self.logger.debug(f"Recorded metrics for agent {agent_name} in run {run_id}")
    
    def record_error(self, run_id: str, error: str) -> None:
        """
        Record an error for a run.
        
        Args:
            run_id: The unique ID for the run
            error: The error message
        """
        if run_id not in self.runs:
            self.logger.warning(f"Cannot record error for {run_id} - not found")
            return
        
        self.runs[run_id].errors.append(error)
        self.logger.error(f"Error in run {run_id}: {error}")
    
    def get_run_metrics(self, run_id: str) -> Optional[RunMetrics]:
        """
        Get metrics for a run.
        
        Args:
            run_id: The unique ID for the run
            
        Returns:
            The run metrics or None if not found
        """
        return self.runs.get(run_id)
    
    def cleanup_run(self, run_id: str) -> None:
        """
        Remove metrics for a run.
        
        Args:
            run_id: The unique ID for the run
        """
        if run_id in self.runs:
            del self.runs[run_id]
            self.logger.debug(f"Cleaned up metrics for run {run_id}")


# Singleton instance
telemetry = TelemetryManager()