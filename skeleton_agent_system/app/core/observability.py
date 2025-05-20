from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
import importlib.util

# Check if logfire is available
logfire_available = importlib.util.find_spec("logfire") is not None
if logfire_available:
    import logfire  # type: ignore
    from pydantic_ai.models.instrumented import InstrumentationSettings

# Set up logger
logger = logging.getLogger(__name__)

from app.core.telemetry import telemetry


class Observability:
    """
    Comprehensive observability setup for the agent system
    using Logfire and OpenTelemetry.
    """
    
    initialized: bool = False
    
    @classmethod
    def initialize(
        cls,
        project_name: str = "agent-system",
        environment: str = "development",
        send_to_logfire: bool = True,
        capture_http: bool = True,
        include_binary_content: bool = False,
    ) -> None:
        """
        Initialize the observability system.
        
        Args:
            project_name: The name of the project
            environment: The environment (development, staging, production)
            send_to_logfire: Whether to send data to Logfire (if token is available)
            capture_http: Whether to capture HTTP requests to model providers
            include_binary_content: Whether to include binary content in traces
        """
        if cls.initialized:
            return
        
        if not logfire_available:
            logger.warning(
                "Logfire not available. Install with 'pip install logfire' for enhanced observability."
            )
            cls.initialized = True
            return
            
        try:
            # Configure Logfire
            send_option = "if-token-present" if send_to_logfire else False
            logfire.configure(
                send_to_logfire=send_option,
                service_name=project_name,
                environment=environment,
            )
            
            # Enable PydanticAI instrumentation
            # As of the current version, binary content flag may not be supported yet
            instrumentation_settings = InstrumentationSettings()
            logfire.instrument_pydantic_ai(settings=instrumentation_settings)
            
            # Enable HTTP instrumentation to see raw API calls to model providers
            if capture_http:
                logfire.instrument_httpx(capture_all=True)
            
            # Log initialization
            with logfire.span("observability_initialized"):
                logfire.info(
                    "Observability initialized",
                    project_name=project_name,
                    environment=environment,
                    logfire_enabled=send_to_logfire,
                    http_capture=capture_http,
                )
                
            cls.initialized = True
            
        except Exception as e:
            logger.warning(f"Failed to initialize Logfire: {str(e)}")
            # Still mark as initialized to avoid repeated attempts
            cls.initialized = True
    
    @classmethod
    def start_trace(cls, run_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a trace for a run.
        
        Args:
            run_id: The unique ID for the run
            attributes: Optional attributes to add to the trace
        """
        if not cls.initialized:
            cls.initialize()
        
        # Start the run in the telemetry system
        telemetry.start_run(run_id)
        
        if logfire_available:
            # Create a span for the run
            attrs = attributes or {}
            with logfire.span(f"agent_run:{run_id}", **attrs):
                logfire.info(f"Starting agent run: {run_id}")
        else:
            logger.info(f"Starting agent run: {run_id}")
    
    @classmethod
    def end_trace(cls, run_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        """
        End a trace for a run.
        
        Args:
            run_id: The unique ID for the run
            result: Optional result to record
        """
        if not cls.initialized:
            return
        
        # End the run in the telemetry system
        telemetry.end_run(run_id)
        
        # Log the result
        if logfire_available and result:
            with logfire.span(f"agent_result:{run_id}"):
                logfire.info(
                    f"Agent run completed: {run_id}",
                    result=result,
                )
        else:
            logger.info(f"Agent run completed: {run_id}")
    
    @classmethod
    def record_error(cls, run_id: str, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error for a run.
        
        Args:
            run_id: The unique ID for the run
            error: The error message
            details: Optional error details
        """
        if not cls.initialized:
            return
        
        # Record the error in the telemetry system
        telemetry.record_error(run_id, error)
        
        # Log the error
        if logfire_available:
            with logfire.span(f"agent_error:{run_id}"):
                logfire.error(
                    f"Error in agent run {run_id}: {error}",
                    error_details=details or {},
                )
        else:
            logger.error(f"Error in agent run {run_id}: {error}")


# Initialize singleton instance
observability = Observability()