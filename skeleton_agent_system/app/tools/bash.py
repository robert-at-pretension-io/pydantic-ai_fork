from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from typing import List, Optional

async def bash_executor(command: str, timeout: int = 30) -> str:
    """
    Executes a bash command and returns the output.
    
    Args:
        command: The bash command to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        The command output (stdout and stderr combined)
    """
    # SECURITY: In a production environment, you would implement proper sandboxing
    # and validation before executing shell commands.
    
    # Simple validation
    if any(unsafe_cmd in command.lower() for unsafe_cmd in ['rm -rf', 'sudo', 'chmod 777']):
        return "ERROR: Command contains potentially unsafe operations"
    
    try:
        # Use asyncio to run the command with a timeout
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for the command to complete with timeout
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return f"ERROR: Command timed out after {timeout} seconds"
        
        # Combine stdout and stderr
        output = stdout.decode().strip() + ("\n" + stderr.decode().strip() if stderr else "")
        return output
    
    except Exception as e:
        return f"ERROR: Failed to execute command: {str(e)}"