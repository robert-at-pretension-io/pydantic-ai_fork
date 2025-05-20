from __future__ import annotations

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any

async def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search and returns results.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        A list of search results with title, snippet, and URL
    """
    # MOCK: In a real implementation, this would call a search API
    # such as Google Search, Bing, or DuckDuckGo.
    
    # Simulate search results
    results = []
    for i in range(1, num_results + 1):
        results.append({
            "title": f"Result {i} for: {query}",
            "snippet": f"This is search result {i} for the query: {query}. It contains relevant information.",
            "url": f"https://example.com/result/{i}"
        })
    
    return results


async def fetch_url(url: str, timeout: int = 30) -> str:
    """
    Fetches content from a URL.
    
    Args:
        url: The URL to fetch
        timeout: Timeout in seconds
        
    Returns:
        The text content from the URL
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    return f"ERROR: HTTP status {response.status}"
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type or 'application/json' in content_type:
                    return await response.text()
                else:
                    return f"ERROR: Unsupported content type: {content_type}"
    
    except aiohttp.ClientError as e:
        return f"ERROR: Failed to fetch URL: {str(e)}"
    except asyncio.TimeoutError:
        return f"ERROR: Request timed out after {timeout} seconds"