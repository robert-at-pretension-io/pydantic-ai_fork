# Nested Subgraph Routing Architecture

This document outlines strategies for implementing a nested routing agent architecture where subgraphs can process parts of a complex task and then hand control back to the parent graph for further processing.

## Core Concepts

The key innovation in this architecture is the ability for a subgraph to recognize its domain boundaries and signal to the parent router that additional processing by other specialized agents is needed. This creates a flexible, composable system where complex tasks can be broken down and handled by the most appropriate specialized agents.

## Design Principles

The architecture follows these key design principles to ensure robustness and maintainability:

1. **Single Source of Truth**: Use a unified state model rather than spreading workflow state across multiple objects.
2. **Bounded Context and Memory Hygiene**: Implement strict memory bounds to prevent token explosion.
3. **Structured Error Handling**: Clear failure modes and retry semantics that all components understand.
4. **Static Domain Contracts**: Use enums and registries rather than magic strings for domain identification.
5. **Explicit State Transitions**: Make all workflow state transitions visible and traceable.

## Single Source of Truth for Workflow State

The router uses a unified state model to avoid ambiguity and state inconsistencies:

```python
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

# Define domain types as enums rather than strings
class Domain(str, Enum):
    SEARCH = "search"
    CALCULATION = "calculation"
    TEXT_ANALYSIS = "text_analysis"
    CODE_GENERATION = "code_generation"
    PLANNING = "planning"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [d.value for d in cls]

# Define workflow states
class WorkflowState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"

# Unified RouterState that includes workflow tracking
@dataclass
class RouterState:
    """Single source of truth for the entire routing workflow."""
    query: str
    
    # Core workflow tracking
    domain_states: Dict[Domain, WorkflowState] = field(default_factory=dict)
    domain_results: Dict[Domain, Any] = field(default_factory=dict)
    domain_errors: Dict[Domain, str] = field(default_factory=dict)
    domain_attempts: Dict[Domain, int] = field(default_factory=lambda: defaultdict(int))
    
    # Execution plan
    execution_sequence: List[Domain] = field(default_factory=list)
    current_domain_index: int = 0
    
    # Results aggregation
    final_response: Optional[str] = None
    
    # Metadata
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Utility properties and methods
    @property
    def current_domain(self) -> Optional[Domain]:
        """Get the current domain being processed."""
        if 0 <= self.current_domain_index < len(self.execution_sequence):
            return self.execution_sequence[self.current_domain_index]
        return None
    
    @property
    def completed_domains(self) -> Set[Domain]:
        """Get all domains that have been completed."""
        return {d for d, s in self.domain_states.items() 
                if s == WorkflowState.COMPLETE}
    
    @property
    def workflow_complete(self) -> bool:
        """Check if the workflow is complete."""
        for domain in self.execution_sequence:
            if self.domain_states[domain] not in [
                WorkflowState.COMPLETE, 
                WorkflowState.FAILED,
                WorkflowState.SKIPPED
            ]:
                return False
        return True
```

## Bounded Context and Memory Hygiene

To prevent token explosion with large LLM contexts, the architecture implements strict memory bounds:

```python
from typing import Any, Dict, List
import json

# Define size constraints
MAX_RESULTS_SIZE_BYTES = 8 * 1024  # 8KB per domain result
MAX_METADATA_SIZE_BYTES = 2 * 1024  # 2KB per domain metadata
MAX_RESULTS_COUNT = 20  # Maximum number of items in a list result

class MemoryConstraints:
    """Utilities for enforcing memory constraints."""
    
    @staticmethod
    def truncate_text(text: str, max_bytes: int) -> str:
        """Truncate text to a maximum byte size."""
        encoded = text.encode('utf-8')
        if len(encoded) <= max_bytes:
            return text
        
        # Try to truncate at a sentence boundary
        truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.75:  # If we find a period in the last quarter
            return truncated[:last_period + 1] + "... [truncated]"
        
        return truncated + "... [truncated]"
    
    @staticmethod
    def truncate_list(items: List[Any], max_count: int) -> List[Any]:
        """Truncate a list to a maximum number of items."""
        if len(items) <= max_count:
            return items
        
        return items[:max_count - 1] + ["... and {} more items [truncated]".format(len(items) - max_count + 1)]
    
    @staticmethod
    def truncate_dict(data: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
        """Truncate a dictionary to fit within a maximum byte size."""
        serialized = json.dumps(data)
        if len(serialized.encode('utf-8')) <= max_bytes:
            return data
        
        # Create a summarized version
        result = {}
        used_bytes = 0
        
        # First pass: include smaller values
        for key, value in sorted(data.items(), key=lambda x: len(json.dumps(x[1]).encode('utf-8'))):
            value_json = json.dumps(value)
            value_bytes = len(value_json.encode('utf-8'))
            
            # Skip very large values in first pass
            if value_bytes > max_bytes / 3:
                continue
                
            # Check if we can add this value
            if used_bytes + value_bytes <= max_bytes * 0.8:  # Leave some room for truncation markers
                result[key] = value
                used_bytes += value_bytes
        
        # Second pass: include truncated versions of larger values
        for key, value in data.items():
            if key in result:
                continue
                
            value_type = type(value)
            if value_type == str:
                # Truncate text value
                max_item_bytes = (max_bytes - used_bytes) // (len(data) - len(result))
                truncated = MemoryConstraints.truncate_text(value, max_item_bytes)
                result[key] = truncated
            elif value_type == list:
                # Truncate list value
                max_items = MAX_RESULTS_COUNT // (len(data) - len(result))
                truncated = MemoryConstraints.truncate_list(value, max_items)
                result[key] = truncated
            elif value_type == dict:
                # Truncate dict value
                max_item_bytes = (max_bytes - used_bytes) // (len(data) - len(result))
                result[key] = MemoryConstraints.truncate_dict(value, max_item_bytes)
            else:
                # For other types, use string representation
                result[key] = str(value) + " [converted to string]"
            
            # Recalculate used bytes
            result_json = json.dumps(result)
            used_bytes = len(result_json.encode('utf-8'))
            
            # If we've exceeded the limit, break
            if used_bytes >= max_bytes:
                result["_truncated"] = True
                break
        
        return result
```

## Failure and Retry Semantics

The architecture implements comprehensive failure handling with clear retry policies:

```python
from enum import Enum
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
import time
import logging

# Define failure types for more granular handling
class FailureType(str, Enum):
    TIMEOUT = "timeout"  # Operation timed out
    API_ERROR = "api_error"  # API returned an error
    RATE_LIMIT = "rate_limit"  # Rate limited by provider
    INVALID_INPUT = "invalid_input"  # Input to the domain was invalid
    INTERNAL_ERROR = "internal_error"  # Internal processing error
    DEPENDENCY_FAILED = "dependency_failed"  # Required dependency failed
    UNKNOWN = "unknown"  # Unspecified error

# Define retry strategies
class RetryStrategy(str, Enum):
    NONE = "none"  # No retry
    IMMEDIATE = "immediate"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Linear backoff

@dataclass
class RetryPolicy:
    """Policy for retrying failed operations."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_ms: int = 500  # Base delay in milliseconds
    max_delay_ms: int = 10000  # Maximum delay in milliseconds
    jitter_ms: int = 100  # Random jitter to add/subtract from delay
    timeout_ms: int = 30000  # Timeout for the entire operation
    
    # Maps failure types to whether they should be retried
    retry_on: Dict[FailureType, bool] = field(default_factory=lambda: {
        FailureType.TIMEOUT: True,
        FailureType.API_ERROR: True,
        FailureType.RATE_LIMIT: True,
        FailureType.INVALID_INPUT: False,  # Don't retry invalid inputs
        FailureType.INTERNAL_ERROR: True,
        FailureType.DEPENDENCY_FAILED: False,  # Don't retry if dependency failed
        FailureType.UNKNOWN: True
    })
    
    def calculate_delay_ms(self, attempt: int) -> int:
        """Calculate delay for a retry attempt."""
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay_ms * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay_ms * (2 ** (attempt - 1))
        else:  # Unhandled
            delay = self.base_delay_ms
        
        # Add jitter
        import random
        jitter = random.randint(-self.jitter_ms, self.jitter_ms)
        delay += jitter
        
        # Ensure within bounds
        return min(max(1, delay), self.max_delay_ms)
    
    def should_retry(self, failure_type: FailureType, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        return self.retry_on.get(failure_type, False)

@dataclass
class RetryManager:
    """Manages retries for domain functions."""
    # Default policies for each domain
    domain_policies: Dict[Domain, RetryPolicy] = field(default_factory=dict)
    # Logger for retry events
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("retry_manager"))
    
    async def execute_with_retry(
        self,
        domain: Domain,
        func: Any,  # Domain function
        state: RouterState,
        policy: Optional[RetryPolicy] = None
    ) -> Optional[Any]:
        """Execute a domain function with retry logic."""
        # Get the appropriate policy
        retry_policy = policy or self.domain_policies.get(domain, RetryPolicy())
        
        # Start timing
        start_time = time.time()
        timeout_sec = retry_policy.timeout_ms / 1000
        
        # Track the attempt number
        attempt = state.record_attempt(domain)
        
        while attempt <= retry_policy.max_attempts:
            try:
                # Update state to in-progress
                state.update_domain_state(domain, WorkflowState.IN_PROGRESS)
                
                # Check overall timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_sec:
                    raise asyncio.TimeoutError(
                        f"Timeout after {elapsed:.2f}s (limit: {timeout_sec}s)")
                
                # Execute the function with remaining timeout
                remaining = max(0.1, timeout_sec - elapsed)
                result = await asyncio.wait_for(func(state), timeout=remaining)
                
                # Success - update state and return result
                state.update_domain_state(domain, WorkflowState.COMPLETE)
                return result
                
            except Exception as e:
                # Determine failure type and handle accordingly
                failure_type = self._classify_exception(e)
                
                # Log the failure
                self.logger.warning(
                    f"Domain {domain.value} failed (attempt {attempt}/{retry_policy.max_attempts}): "
                    f"{failure_type.value} - {str(e)}"
                )
                
                # Check if we should retry
                if not retry_policy.should_retry(failure_type, attempt):
                    state.update_domain_state(domain, WorkflowState.FAILED)
                    state.domain_errors[domain] = str(e)
                    return None
                
                # Increment attempt count for next iteration
                attempt = state.record_attempt(domain)
                
                # Calculate delay and wait
                delay_ms = retry_policy.calculate_delay_ms(attempt-1)
                await asyncio.sleep(delay_ms / 1000)
```

## Static Domain Contracts

Replace magic strings with compile-time checkable enums and contracts:

```python
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
from pydantic import BaseModel, Field
import functools

# Domain enum instead of magic strings
class Domain(str, Enum):
    SEARCH = "search"
    CALCULATION = "calculation"
    TEXT_ANALYSIS = "text_analysis"
    CODE_GENERATION = "code_generation"
    PLANNING = "planning"

class DomainContractRegistry:
    """Registry for domain contracts."""
    _input_contracts: Dict[Domain, Type[BaseModel]] = {}
    _output_contracts: Dict[Domain, Type[BaseModel]] = {}
    _description: Dict[Domain, str] = {}
    
    @classmethod
    def register_domain(
        cls, 
        domain: Domain, 
        input_model: Type[BaseModel],
        output_model: Type[BaseModel],
        description: str
    ) -> None:
        """Register a domain contract."""
        cls._input_contracts[domain] = input_model
        cls._output_contracts[domain] = output_model
        cls._description[domain] = description
    
    @classmethod
    def get_input_contract(cls, domain: Domain) -> Type[BaseModel]:
        """Get the input contract for a domain."""
        if domain not in cls._input_contracts:
            raise ValueError(f"No input contract registered for domain {domain}")
        return cls._input_contracts[domain]
    
    @classmethod
    def get_output_contract(cls, domain: Domain) -> Type[BaseModel]:
        """Get the output contract for a domain."""
        if domain not in cls._output_contracts:
            raise ValueError(f"No output contract registered for domain {domain}")
        return cls._output_contracts[domain]
    
    @classmethod
    def validate_input(cls, domain: Domain, data: Dict[str, Any]) -> BaseModel:
        """Validate input data against a domain's contract."""
        contract = cls.get_input_contract(domain)
        return contract.model_validate(data)
    
    @classmethod
    def validate_output(cls, domain: Domain, data: Dict[str, Any]) -> BaseModel:
        """Validate output data against a domain's contract."""
        contract = cls.get_output_contract(domain)
        return contract.model_validate(data)

# Decorator for enforcing domain contracts
def enforce_domain_contract(domain: Domain):
    """Decorator to enforce domain contracts on functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the contracts
            input_contract = DomainContractRegistry.get_input_contract(domain)
            output_contract = DomainContractRegistry.get_output_contract(domain)
            
            # Extract the RouterState argument
            state = None
            for arg in args:
                if isinstance(arg, RouterState):
                    state = arg
                    break
            
            if state is None:
                for _, value in kwargs.items():
                    if isinstance(value, RouterState):
                        state = value
                        break
            
            if state is None:
                raise ValueError(f"No RouterState argument found for {func.__name__}")
            
            # Execute the function with validated input
            result = await func(*args, **kwargs)
            
            # Validate the output
            validated_output = output_contract.model_validate(result)
            
            return validated_output
        return wrapper
    return decorator

# Example domain-specific contracts
class SearchInput(BaseModel):
    """Input contract for search domain."""
    query: str
    max_results: int = Field(default=10)
    
class SearchOutput(BaseModel):
    """Output contract for search domain."""
    results: List[str]
    source_urls: Optional[List[str]] = None
    confidence: float = 1.0

# Register the domain
DomainContractRegistry.register_domain(
    domain=Domain.SEARCH,
    input_model=SearchInput,
    output_model=SearchOutput,
    description="Search domain for retrieving information from external sources"
)
```

## Complete Implementation Example

Here's a complete implementation showcasing all the design principles working together:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar, Union
import asyncio
import logging
import time
import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent, format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


# ======= Domain Definition =======

class Domain(str, Enum):
    SEARCH = "search"
    CALCULATION = "calculation"
    TEXT_ANALYSIS = "text_analysis"
    CODE_GENERATION = "code_generation"
    PLANNING = "planning"


class ProcessingStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    ERROR = "error"


class WorkflowState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# ======= Failure Handling =======

class FailureType(str, Enum):
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    INTERNAL_ERROR = "internal_error"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


class RetryStrategy(str, Enum):
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryPolicy:
    """Policy for retrying failed operations."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay_ms: int = 500
    max_delay_ms: int = 10000
    jitter_ms: int = 100
    timeout_ms: int = 30000
    
    # Maps failure types to whether they should be retried
    retry_on: Dict[FailureType, bool] = field(default_factory=lambda: {
        FailureType.TIMEOUT: True,
        FailureType.API_ERROR: True,
        FailureType.RATE_LIMIT: True,
        FailureType.INVALID_INPUT: False,
        FailureType.INTERNAL_ERROR: True,
        FailureType.DEPENDENCY_FAILED: False,
        FailureType.UNKNOWN: True
    })
    
    def calculate_delay_ms(self, attempt: int) -> int:
        """Calculate delay for a retry attempt."""
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay_ms * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay_ms * (2 ** (attempt - 1))
        else:  # CUSTOM or unhandled
            delay = self.base_delay_ms
        
        # Add jitter
        import random
        jitter = random.randint(-self.jitter_ms, self.jitter_ms)
        delay += jitter
        
        # Ensure within bounds
        return min(max(1, delay), self.max_delay_ms)
    
    def should_retry(self, failure_type: FailureType, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        return self.retry_on.get(failure_type, False)


# ======= Memory Management =======

class MemoryConstraints:
    """Utilities for enforcing memory constraints."""
    
    @staticmethod
    def truncate_text(text: str, max_bytes: int) -> str:
        """Truncate text to a maximum byte size."""
        encoded = text.encode('utf-8')
        if len(encoded) <= max_bytes:
            return text
        
        # Truncate at a sentence boundary if possible
        truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.75:
            return truncated[:last_period + 1] + "... [truncated]"
        
        return truncated + "... [truncated]"
    
    @staticmethod
    def truncate_list(items: List[Any], max_count: int) -> List[Any]:
        """Truncate a list to a maximum number of items."""
        if len(items) <= max_count:
            return items
        
        return items[:max_count - 1] + ["... and {} more items [truncated]".format(len(items) - max_count + 1)]
    
    @staticmethod
    def truncate_dict(data: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
        """Truncate a dictionary to fit within a maximum byte size."""
        serialized = json.dumps(data)
        if len(serialized.encode('utf-8')) <= max_bytes:
            return data
        
        # Create a summarized version
        result = {}
        used_bytes = 0
        
        # First pass: include smaller values
        for key, value in sorted(data.items(), key=lambda x: len(json.dumps(x[1]).encode('utf-8'))):
            value_json = json.dumps(value)
            value_bytes = len(value_json.encode('utf-8'))
            
            if value_bytes > max_bytes / 3:
                continue
                
            if used_bytes + value_bytes <= max_bytes * 0.8:
                result[key] = value
                used_bytes += value_bytes
        
        # Second pass: include truncated versions of larger values
        for key, value in data.items():
            if key in result:
                continue
                
            value_type = type(value)
            if value_type == str:
                max_item_bytes = (max_bytes - used_bytes) // (len(data) - len(result))
                truncated = MemoryConstraints.truncate_text(value, max_item_bytes)
                result[key] = truncated
            elif value_type == list:
                max_items = 20 // (len(data) - len(result))
                truncated = MemoryConstraints.truncate_list(value, max_items)
                result[key] = truncated
            elif value_type == dict:
                max_item_bytes = (max_bytes - used_bytes) // (len(data) - len(result))
                result[key] = MemoryConstraints.truncate_dict(value, max_item_bytes)
            else:
                result[key] = str(value) + " [converted to string]"
            
            result_json = json.dumps(result)
            used_bytes = len(result_json.encode('utf-8'))
            
            if used_bytes >= max_bytes:
                result["_truncated"] = True
                break
        
        return result


# ======= Router State =======

@dataclass
class RouterState:
    """Single source of truth for the router workflow."""
    query: str
    
    # Core workflow tracking
    domain_states: Dict[Domain, WorkflowState] = field(default_factory=dict)
    domain_results: Dict[Domain, Any] = field(default_factory=dict)
    domain_errors: Dict[Domain, str] = field(default_factory=dict)
    domain_attempts: Dict[Domain, int] = field(default_factory=lambda: {})
    
    # Execution plan
    execution_sequence: List[Domain] = field(default_factory=list)
    current_domain_index: int = 0
    
    # Memory constraints
    max_result_size_bytes: int = 8 * 1024  # 8KB per domain result
    
    # Results aggregation
    final_response: Optional[str] = None
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default domain states if not provided."""
        for domain in Domain:
            if domain not in self.domain_states:
                self.domain_states[domain] = WorkflowState.PENDING
    
    @property
    def current_domain(self) -> Optional[Domain]:
        """Get the current domain being processed."""
        if 0 <= self.current_domain_index < len(self.execution_sequence):
            return self.execution_sequence[self.current_domain_index]
        return None
    
    @property
    def next_domain(self) -> Optional[Domain]:
        """Get the next domain to process."""
        next_idx = self.current_domain_index + 1
        if next_idx < len(self.execution_sequence):
            return self.execution_sequence[next_idx]
        return None
    
    @property
    def completed_domains(self) -> Set[Domain]:
        """Get all domains that have been completed."""
        return {d for d, s in self.domain_states.items() 
                if s == WorkflowState.COMPLETE}
    
    @property
    def failed_domains(self) -> Set[Domain]:
        """Get all domains that have failed."""
        return {d for d, s in self.domain_states.items() 
                if s == WorkflowState.FAILED}
    
    @property
    def pending_domains(self) -> List[Domain]:
        """Get all domains that are pending execution in the sequence."""
        pending = []
        for i, domain in enumerate(self.execution_sequence):
            if i > self.current_domain_index and self.domain_states[domain] == WorkflowState.PENDING:
                pending.append(domain)
        return pending
    
    @property
    def workflow_complete(self) -> bool:
        """Check if the workflow is complete."""
        for domain in self.execution_sequence:
            if self.domain_states[domain] not in [
                WorkflowState.COMPLETE, 
                WorkflowState.FAILED,
                WorkflowState.SKIPPED
            ]:
                return False
        return True
    
    def update_domain_state(self, domain: Domain, state: WorkflowState) -> None:
        """Update the state of a domain."""
        self.domain_states[domain] = state
        
    def record_attempt(self, domain: Domain) -> int:
        """Record an attempt for a domain and return the attempt count."""
        if domain not in self.domain_attempts:
            self.domain_attempts[domain] = 0
        self.domain_attempts[domain] += 1
        return self.domain_attempts[domain]
    
    def add_domain_to_execution(self, domain: Domain, index: Optional[int] = None) -> None:
        """Add a domain to the execution sequence."""
        if domain in self.execution_sequence:
            return
            
        if index is None:
            self.execution_sequence.append(domain)
        else:
            self.execution_sequence.insert(index, domain)
    
    def add_domain_result(self, domain: Domain, result: Any) -> None:
        """Add a domain result with memory constraints."""
        # Apply memory constraints to different result types
        if isinstance(result, dict):
            self.domain_results[domain] = MemoryConstraints.truncate_dict(
                result, self.max_result_size_bytes)
        elif isinstance(result, list):
            self.domain_results[domain] = MemoryConstraints.truncate_list(
                result, 20)  # Max 20 items
        elif isinstance(result, str):
            self.domain_results[domain] = MemoryConstraints.truncate_text(
                result, self.max_result_size_bytes)
        else:
            # For other types, store as is but convert if too large
            try:
                result_json = json.dumps(result)
                if len(result_json.encode('utf-8')) > self.max_result_size_bytes:
                    self.domain_results[domain] = str(result) + " [converted due to size]"
                else:
                    self.domain_results[domain] = result
            except (TypeError, OverflowError):
                # Handle non-serializable objects
                self.domain_results[domain] = str(result) + " [converted due to type]"


# ======= Domain Contracts =======

class SearchInput(BaseModel):
    """Input contract for search domain."""
    query: str
    max_results: int = Field(default=10)
    
class SearchOutput(BaseModel):
    """Output contract for search domain."""
    results: List[str]
    source_urls: Optional[List[str]] = None
    confidence: float = 1.0

class CalculationInput(BaseModel):
    """Input contract for calculation domain."""
    query: str
    precision: int = Field(default=2)
    
class CalculationOutput(BaseModel):
    """Output contract for calculation domain."""
    result: float
    steps: List[str]
    input_interpreted: str


# ======= Retry Management =======

@dataclass
class RetryManager:
    """Manages retries for domain functions."""
    # Default policies for each domain
    domain_policies: Dict[Domain, RetryPolicy] = field(default_factory=dict)
    # Logger for retry events
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("retry_manager"))
    
    def __post_init__(self):
        """Initialize default policies for all domains."""
        for domain in Domain:
            if domain not in self.domain_policies:
                self.domain_policies[domain] = RetryPolicy()
    
    async def execute_with_retry(
        self,
        domain: Domain,
        func: Any,  # Domain function
        state: RouterState,
        policy: Optional[RetryPolicy] = None
    ) -> Optional[Any]:
        """Execute a domain function with retry logic."""
        # Get the appropriate policy
        retry_policy = policy or self.domain_policies.get(domain, RetryPolicy())
        
        # Start timing
        start_time = time.time()
        timeout_sec = retry_policy.timeout_ms / 1000
        
        # Track the attempt number
        attempt = state.record_attempt(domain)
        
        while attempt <= retry_policy.max_attempts:
            try:
                # Update state to in-progress
                state.update_domain_state(domain, WorkflowState.IN_PROGRESS)
                
                # Check overall timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_sec:
                    raise asyncio.TimeoutError(
                        f"Timeout after {elapsed:.2f}s (limit: {timeout_sec}s)")
                
                # Execute the function with remaining timeout
                remaining = max(0.1, timeout_sec - elapsed)
                result = await asyncio.wait_for(func(state), timeout=remaining)
                
                # Success - update state and return result
                state.update_domain_state(domain, WorkflowState.COMPLETE)
                return result
                
            except Exception as e:
                # Determine failure type
                failure_type = self._classify_exception(e)
                
                # Log the failure
                self.logger.warning(
                    f"Domain {domain.value} failed (attempt {attempt}/{retry_policy.max_attempts}): "
                    f"{failure_type.value} - {str(e)}"
                )
                
                # Check if we should retry
                if not retry_policy.should_retry(failure_type, attempt):
                    state.update_domain_state(domain, WorkflowState.FAILED)
                    state.domain_errors[domain] = str(e)
                    self.logger.error(
                        f"Domain {domain.value} failed permanently after {attempt} attempts: {str(e)}"
                    )
                    return None
                
                # Increment attempt count for next iteration
                attempt = state.record_attempt(domain)
                
                # Calculate delay and wait
                delay_ms = retry_policy.calculate_delay_ms(attempt-1)
                self.logger.info(
                    f"Retrying domain {domain.value} in {delay_ms}ms (attempt {attempt})"
                )
                await asyncio.sleep(delay_ms / 1000)
        
        # If we get here, we've exceeded max attempts
        state.update_domain_state(domain, WorkflowState.FAILED)
        state.domain_errors[domain] = f"Exceeded maximum attempts ({retry_policy.max_attempts})"
        return None
    
    def _classify_exception(self, exc: Exception) -> FailureType:
        """Classify an exception into a failure type."""
        if isinstance(exc, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        
        error_msg = str(exc).lower()
        if "rate limit" in error_msg or "too many requests" in error_msg:
            return FailureType.RATE_LIMIT
        elif "invalid input" in error_msg or "validation error" in error_msg:
            return FailureType.INVALID_INPUT
        elif "api" in error_msg and ("error" in error_msg or "exception" in error_msg):
            return FailureType.API_ERROR
        elif "dependency" in error_msg or "required domain" in error_msg:
            return FailureType.DEPENDENCY_FAILED
        
        return FailureType.INTERNAL_ERROR


# ======= Domain Implementation =======

async def execute_search(state: RouterState) -> Dict[str, Any]:
    """Execute a search operation."""
    search_agent = Agent(
        'openai:gpt-4o',
        output_type=SearchOutput,
        system_prompt="You are a search specialist. Find relevant information for queries."
    )
    
    result = await search_agent.run(f"Find information about: {state.query}")
    
    return result.output.model_dump()

async def execute_calculation(state: RouterState) -> Dict[str, Any]:
    """Execute a calculation operation."""
    calc_agent = Agent(
        'openai:gpt-4o',
        output_type=CalculationOutput,
        system_prompt="You are a calculation specialist. Solve mathematical problems step by step."
    )
    
    result = await calc_agent.run(f"Calculate the answer to: {state.query}")
    
    return result.output.model_dump()


# ======= Graph Nodes =======

@dataclass
class AnalyzeQuery(BaseNode[RouterState]):
    """Initial node that analyzes the query to determine required domains."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "PlanExecution":
        # Create analysis agent
        analysis_agent = Agent(
            'openai:gpt-4o',
            output_type=Dict[str, Any],
            system_prompt="You analyze queries to determine which specialized domains to use."
        )
        
        result = await analysis_agent.run(
            f"""
            Analyze this query: {ctx.state.query}
            
            Determine which specialized domains are needed to answer this query.
            Choose from: {[d.value for d in Domain]}
            
            Respond with:
            - "domains_needed": array of domain names, in the order they should be executed
            - "reasoning": brief explanation of your selection
            """
        )
        
        # Extract domains from response
        domains_needed = result.output.get("domains_needed", ["search"])
        
        # Validate and convert domain strings to enum values
        execution_sequence = []
        for domain_str in domains_needed:
            try:
                domain = Domain(domain_str.lower())
                execution_sequence.append(domain)
            except ValueError:
                # Skip invalid domains
                continue
        
        # Ensure at least search domain is included if no valid domains
        if not execution_sequence:
            execution_sequence = [Domain.SEARCH]
        
        # Update state with execution plan
        ctx.state.execution_sequence = execution_sequence
        
        return PlanExecution()


@dataclass
class PlanExecution(BaseNode[RouterState]):
    """Plans the execution sequence for the domains."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "ExecuteNextDomain":
        # If no execution sequence is set, default to search
        if not ctx.state.execution_sequence:
            ctx.state.execution_sequence = [Domain.SEARCH]
        
        # Reset the current domain index
        ctx.state.current_domain_index = 0
        
        return ExecuteNextDomain()


@dataclass
class ExecuteNextDomain(BaseNode[RouterState]):
    """Executes the next domain in the sequence."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> Union["DomainNode", "FinalizeResponse"]:
        # Check if we've completed all domains
        if ctx.state.current_domain_index >= len(ctx.state.execution_sequence):
            return FinalizeResponse()
        
        # Get the current domain to execute
        current_domain = ctx.state.execution_sequence[ctx.state.current_domain_index]
        
        # Return the appropriate domain node
        return DomainNode(domain=current_domain)


@dataclass
class DomainNode(BaseNode[RouterState]):
    """Generic domain execution node."""
    domain: Domain
    retry_manager: RetryManager = field(default_factory=RetryManager)
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "ProcessDomainResult":
        # Execute the domain function with retry
        domain_func = self._get_domain_function()
        
        result = await self.retry_manager.execute_with_retry(
            domain=self.domain,
            func=domain_func,
            state=ctx.state
        )
        
        # Store result if successful
        if result is not None:
            ctx.state.add_domain_result(self.domain, result)
        
        return ProcessDomainResult()
    
    def _get_domain_function(self):
        """Get the appropriate domain function."""
        if self.domain == Domain.SEARCH:
            return execute_search
        elif self.domain == Domain.CALCULATION:
            return execute_calculation
        else:
            # Fallback for unsupported domains
            async def unsupported_domain(_):
                raise ValueError(f"Unsupported domain: {self.domain}")
            return unsupported_domain


@dataclass
class ProcessDomainResult(BaseNode[RouterState]):
    """Processes the result from a domain execution."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> Union["ModifyExecution", "ExecuteNextDomain"]:
        # Get the current domain that was just executed
        current_domain = ctx.state.current_domain
        
        if not current_domain:
            return ExecuteNextDomain()
        
        # Check if we need to modify the execution plan based on this domain's results
        if current_domain == Domain.SEARCH and Domain.CALCULATION not in ctx.state.execution_sequence:
            # Check if search results suggest we need calculation
            search_results = ctx.state.domain_results.get(Domain.SEARCH, {})
            if isinstance(search_results, dict) and search_results.get("results"):
                # Simple heuristic: if any search result has numbers, consider calculation
                has_numbers = any(
                    any(char.isdigit() for char in result) 
                    for result in search_results.get("results", [])
                )
                if has_numbers and "calculate" in ctx.state.query.lower():
                    return ModifyExecution(add_domains=[Domain.CALCULATION])
        
        # Move to the next domain in the sequence
        ctx.state.current_domain_index += 1
        
        return ExecuteNextDomain()


@dataclass
class ModifyExecution(BaseNode[RouterState]):
    """Modifies the execution plan based on domain results."""
    add_domains: List[Domain] = field(default_factory=list)
    remove_domains: List[Domain] = field(default_factory=list)
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> "ExecuteNextDomain":
        # Add new domains to the execution sequence
        for domain in self.add_domains:
            if domain not in ctx.state.execution_sequence:
                ctx.state.add_domain_to_execution(domain)
        
        # Remove domains from the execution sequence
        for domain in self.remove_domains:
            if domain in ctx.state.execution_sequence:
                idx = ctx.state.execution_sequence.index(domain)
                if idx > ctx.state.current_domain_index:
                    # Only remove if we haven't executed it yet
                    ctx.state.execution_sequence.pop(idx)
        
        # Move to the next domain
        ctx.state.current_domain_index += 1
        
        return ExecuteNextDomain()


@dataclass
class FinalizeResponse(BaseNode[RouterState, None, str]):
    """Finalizes the response by aggregating results from all domains."""
    
    async def run(self, ctx: GraphRunContext[RouterState]) -> End[str]:
        # Record end time
        ctx.state.end_time = time.time()
        
        # Create an agent to synthesize all results
        synthesis_agent = Agent(
            'openai:gpt-4o',
            output_type=str,
            system_prompt="You synthesize results from multiple specialized domains into a coherent response."
        )
        
        # Format the domain results
        domain_data = {}
        for domain, result in ctx.state.domain_results.items():
            if result:
                domain_data[domain.value] = result
        
        # Add metadata about failures
        metadata = {
            "query": ctx.state.query,
            "execution_time_ms": int((ctx.state.end_time - ctx.state.start_time) * 1000),
            "domains_executed": [d.value for d in ctx.state.completed_domains],
            "failures": {d.value: err for d, err in ctx.state.domain_errors.items()}
        }
        
        # Request synthesis
        result = await synthesis_agent.run(
            f"""
            Query: {ctx.state.query}
            
            Metadata:
            {format_as_xml(metadata)}
            
            Results from specialized domains:
            {format_as_xml(domain_data)}
            
            Synthesize these results into a comprehensive, coherent response that fully answers the query.
            If some domains failed, work with the successful results and acknowledge any limitations.
            """
        )
        
        # Store and return the final response
        ctx.state.final_response = result.output
        
        return End(result.output)


# ======= Graph Definition =======

router_graph = Graph(
    nodes=[
        AnalyzeQuery, 
        PlanExecution, 
        ExecuteNextDomain, 
        DomainNode,
        ProcessDomainResult,
        ModifyExecution,
        FinalizeResponse
    ],
    state_type=RouterState
)


# ======= Main Entry Point =======

async def process_query(query: str) -> str:
    """Process a query through the improved router architecture."""
    # Initialize router state
    state = RouterState(query=query)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the graph
    result = await router_graph.run(AnalyzeQuery(), state=state)
    
    # Log workflow summary
    domains_executed = [d.value for d in state.completed_domains]
    execution_time = (state.end_time or time.time()) - state.start_time
    logging.info(f"Query processed in {execution_time:.2f}s using domains: {domains_executed}")
    
    if state.domain_errors:
        logging.warning(f"Errors encountered: {state.domain_errors}")
    
    return result.output
```

## Key Insights for Robust Nested Agent Routing

From our extensive exploration, here are the key insights for implementing an effective nested routing architecture:

### 1. Single Source of Truth for Workflow State

Maintain a unified state model that represents all aspects of the workflow. This avoids ambiguity, prevents state drift, and makes debugging easier.

### 2. Bounded Context and Memory Management

- Implement strict memory bounds from the beginning
- Use explicit truncation strategies for different data types
- Always truncate at semantically meaningful boundaries
- Add indicators when content has been truncated

### 3. Failure & Retry Semantics

Create a comprehensive failure handling system with:
- Granular failure type classification
- Per-domain retry policies
- Exponential backoff with jitter
- Failure records in the workflow state
- Consistent logging

### 4. Static Domain Contracts

Replace magic strings with:
- Enum-based domain types
- Explicit input/output contracts using Pydantic models
- Contract validation decorators
- Central registry for domain metadata

### 5. Explicit State Transitions

Make all workflow state transitions clear and traceable:
- Define workflow states as enums
- Record state changes with timestamps
- Track domain-specific state separately from overall workflow state

## Implementation Recommendations

1. **Start with Enums and Contracts**: Define domain types, workflow states, and contracts before implementing any business logic.

2. **Build Router-First**: Implement the parent router with basic routing logic before adding complex subgraphs.

3. **Add One Domain at a Time**: Implement and test each specialized domain one at a time, ensuring it follows all the established patterns.

4. **Test Failure Cases Explicitly**: Create test cases for timeouts, API errors, and other failure types to ensure proper handling.

5. **Monitor Memory Usage**: Track token counts and memory usage in production to fine-tune truncation thresholds.

6. **Create Visualization Tools**: Build tools to visualize the execution path and state transitions for debugging complex workflows.

7. **Document Domain Boundaries**: Make domain responsibilities explicit, with clear handoff points.

This refined architecture creates a robust foundation for complex multi-domain problem solving while ensuring maintainability, predictability, and efficient resource usage.