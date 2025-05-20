from __future__ import annotations

import functools
import inspect
import time
import logging
from typing import Any, Callable, TypeVar, cast, Awaitable, Optional, Dict
import importlib.util

# Check if logfire is available
logfire_available = importlib.util.find_spec("logfire") is not None
if logfire_available:
    import logfire  # type: ignore

# Set up logger
logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


def trace_function(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace function calls with OpenTelemetry.
    
    Args:
        name: Optional custom name for the span
        
    Returns:
        Decorator function
    """
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function name
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            
            start_time = time.time()
            
            if logfire_available:
                # Start span
                with logfire.span(f"function:{span_name}"):
                    # Log function call
                    logfire.info(f"Calling {span_name}")
                    
                    # Execute function
                    try:
                        result = func(*args, **kwargs)
                        
                        # Calculate execution time
                        elapsed = time.time() - start_time
                        
                        # Log success
                        logfire.info(
                            f"Function {span_name} completed in {elapsed:.4f}s",
                            duration=elapsed
                        )
                        
                        return result
                    except Exception as e:
                        # Log error
                        elapsed = time.time() - start_time
                        logfire.error(
                            f"Function {span_name} failed after {elapsed:.4f}s: {str(e)}",
                            duration=elapsed,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise
            else:
                # Standard logging without logfire
                logger.info(f"Calling {span_name}")
                
                # Execute function
                try:
                    result = func(*args, **kwargs)
                    
                    # Calculate execution time
                    elapsed = time.time() - start_time
                    
                    # Log success
                    logger.info(f"Function {span_name} completed in {elapsed:.4f}s")
                    
                    return result
                except Exception as e:
                    # Log error
                    elapsed = time.time() - start_time
                    logger.error(
                        f"Function {span_name} failed after {elapsed:.4f}s: {str(e)}"
                    )
                    raise
                
        return cast(F, wrapper)
    
    return decorator


def trace_async_function(name: Optional[str] = None) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to trace async function calls with OpenTelemetry.
    
    Args:
        name: Optional custom name for the span
        
    Returns:
        Decorator function
    """
    
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function name
            span_name = name or f"{func.__module__}.{func.__qualname__}"
            
            start_time = time.time()
            
            if logfire_available:
                # Start span
                with logfire.span(f"async_function:{span_name}"):
                    # Log function call
                    logfire.info(f"Calling async {span_name}")
                    
                    # Execute function
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Calculate execution time
                        elapsed = time.time() - start_time
                        
                        # Log success
                        logfire.info(
                            f"Async function {span_name} completed in {elapsed:.4f}s",
                            duration=elapsed
                        )
                        
                        return result
                    except Exception as e:
                        # Log error
                        elapsed = time.time() - start_time
                        logfire.error(
                            f"Async function {span_name} failed after {elapsed:.4f}s: {str(e)}",
                            duration=elapsed,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise
            else:
                # Standard logging without logfire
                logger.info(f"Calling async {span_name}")
                
                # Execute function
                try:
                    result = await func(*args, **kwargs)
                    
                    # Calculate execution time
                    elapsed = time.time() - start_time
                    
                    # Log success
                    logger.info(f"Async function {span_name} completed in {elapsed:.4f}s")
                    
                    return result
                except Exception as e:
                    # Log error
                    elapsed = time.time() - start_time
                    logger.error(
                        f"Async function {span_name} failed after {elapsed:.4f}s: {str(e)}"
                    )
                    raise
                
        return cast(AsyncF, wrapper)
    
    return decorator