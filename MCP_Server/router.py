from __future__ import annotations
from fastmcp import FastMCP
from datetime import datetime
import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
import requests
from ..System_Prompts import MCP_ROUTER_PROMPT




api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

base_url = os.getenv("URL")
if not base_url:
    raise RuntimeError("Missing API endpoint URL environment variable.")

# You may want to validate that OPENAI_MODEL is set as well
model = os.getenv("MODEL")
if not model:
    raise RuntimeError("Missing OPENAI_MODEL environment variable.")

print(f"[config] Using model: {model}")
print(f"[config] Using base URL: {base_url}")

client = OpenAI(api_key=api_key, base_url=base_url)


resp = client.chat.completions.create(
    model=model,
    temperature=0,
    messages=[
        {"role": "system", "content": MCP_ROUTER_PROMPT},
        {"role": "user", "content": "HI! How are you?"},
    ],
)

print(resp.choices[0].message.content)



import json
import logging
from typing import List, Dict, Any, Optional
from anthropic import Anthropic, AsyncAnthropic
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class FilteredToolset:
    """Result of tool filtering operation"""
    tools: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    filter_time_ms: Optional[float] = None


class MCPToolRouter:
    """
    Intelligent tool router that uses an LLM to filter available tools
    based on user query context, reducing context window pollution.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-20250514",
        max_tools: int = 5,
        cache_enabled: bool = True,
        fallback_toolset: Optional[List[str]] = None
    ):
        """
        Initialize the router.
        
        Args:
            api_key: Anthropic API key
            model: Model to use for filtering (should be fast/cheap)
            max_tools: Maximum number of tools to return
            cache_enabled: Whether to cache tool catalog in prompt
            fallback_toolset: Tool names to return if filtering fails
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tools = max_tools
        self.cache_enabled = cache_enabled
        self.fallback_toolset = fallback_toolset or []
        
        # Cache for tool catalog string
        self._tool_catalog_cache: Optional[str] = None
        self._tool_index_map: Dict[int, str] = {}
    
    def _build_tool_catalog(self, tools: List[Dict[str, Any]]) -> str:
        """
        Build a concise catalog of tools for the filtering prompt.
        
        Args:
            tools: List of MCP tool definitions
            
        Returns:
            Formatted string of tool descriptions
        """
        catalog_lines = []
        self._tool_index_map = {}
        
        for idx, tool in enumerate(tools):
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "No description")
            
            # Truncate long descriptions
            if len(tool_desc) > 150:
                tool_desc = tool_desc[:147] + "..."
            
            catalog_lines.append(f"{idx}. {tool_name}: {tool_desc}")
            self._tool_index_map[idx] = tool_name
        
        return "\n".join(catalog_lines)
    
    def _build_filter_prompt(self, query: str, tool_catalog: str) -> str:
        """
        Build the prompt for the filtering LLM.
        
        Args:
            query: User's query/request
            tool_catalog: Formatted catalog of available tools
            
        Returns:
            Complete prompt for filtering
        """
        return f"""You are a tool selection assistant. Your job is to analyze a user query and select the most relevant tools from an available catalog.

User Query: "{query}"

Available Tools:
{tool_catalog}

Task: Select the {self.max_tools} most relevant tools that would help accomplish this query. Consider:
- Direct relevance to the query's intent
- Necessary prerequisite tools (e.g., if querying data, you need retrieval before analysis)
- Complementary tools that work together

Return ONLY a JSON array of tool indices (0-based). No explanation, no markdown formatting.

Example response format: [0, 3, 7]

If fewer than {self.max_tools} tools are relevant, return only those that are truly needed.
"""
    
    async def filter_tools(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        include_reasoning: bool = False
    ) -> FilteredToolset:
        """
        Filter tools based on query relevance using LLM.
        
        Args:
            query: User's query/request
            available_tools: Complete list of available MCP tools
            include_reasoning: Whether to request reasoning from the model
            
        Returns:
            FilteredToolset containing selected tools and metadata
            
        Raises:
            ValueError: If filtering fails and no fallback is available
        """
        import time
        start_time = time.time()
        
        try:
            # Build tool catalog (cache if enabled)
            if self.cache_enabled and self._tool_catalog_cache:
                tool_catalog = self._tool_catalog_cache
            else:
                tool_catalog = self._build_tool_catalog(available_tools)
                if self.cache_enabled:
                    self._tool_catalog_cache = tool_catalog
            
            # Build filtering prompt
            prompt = self._build_filter_prompt(query, tool_catalog)
            
            # Call filtering model
            logger.debug(f"Filtering tools with {self.model}")
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=100,  # Just need array of indices
                temperature=0,  # Deterministic selection
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract response text
            response_text = response.content[0].text.strip()
            
            # Parse indices (handle markdown wrapping)
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            selected_indices = json.loads(response_text)
            
            # Validate indices
            if not isinstance(selected_indices, list):
                raise ValueError(f"Expected list, got {type(selected_indices)}")
            
            if not all(isinstance(i, int) for i in selected_indices):
                raise ValueError(f"All indices must be integers: {selected_indices}")
            
            # Build filtered toolset
            filtered_tools = []
            for idx in selected_indices:
                if 0 <= idx < len(available_tools):
                    filtered_tools.append(available_tools[idx])
                else:
                    logger.warning(f"Invalid tool index {idx}, skipping")
            
            filter_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Filtered {len(available_tools)} tools → {len(filtered_tools)} tools "
                f"in {filter_time:.1f}ms"
            )
            
            return FilteredToolset(
                tools=filtered_tools,
                filter_time_ms=filter_time
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Tool filtering failed: {e}", exc_info=True)
            return await self._apply_fallback(available_tools, str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error during filtering: {e}", exc_info=True)
            return await self._apply_fallback(available_tools, str(e))
    
    async def _apply_fallback(
        self,
        available_tools: List[Dict[str, Any]],
        error_msg: str
    ) -> FilteredToolset:
        """
        Apply fallback strategy when filtering fails.
        
        Args:
            available_tools: All available tools
            error_msg: Error message from failed filtering
            
        Returns:
            FilteredToolset with fallback tools
        """
        if not self.fallback_toolset:
            # No fallback defined - return all tools
            logger.warning(
                f"Filtering failed ({error_msg}), returning all {len(available_tools)} tools"
            )
            return FilteredToolset(
                tools=available_tools,
                reasoning=f"Fallback: filtering failed - {error_msg}"
            )
        
        # Return predefined fallback tools
        fallback_tools = [
            tool for tool in available_tools
            if tool.get("name") in self.fallback_toolset
        ]
        
        logger.warning(
            f"Filtering failed ({error_msg}), using {len(fallback_tools)} fallback tools"
        )
        
        return FilteredToolset(
            tools=fallback_tools,
            reasoning=f"Fallback: using predefined toolset - {error_msg}"
        )
    
    def clear_cache(self):
        """Clear cached tool catalog (call when tools change)"""
        self._tool_catalog_cache = None
        self._tool_index_map = {}


class MCPRouterSession:
    """
    Represents a filtered MCP session with context-specific tools.
    """
    
    def __init__(
        self,
        session_id: str,
        filtered_tools: List[Dict[str, Any]],
        original_query: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.tools = filtered_tools
        self.original_query = original_query
        self.metadata = metadata or {}
        self.call_count = 0
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the filtered tool list for this session"""
        return self.tools
    
    def increment_call_count(self):
        """Track how many times tools are called in this session"""
        self.call_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session info"""
        return {
            "session_id": self.session_id,
            "tool_count": len(self.tools),
            "tool_names": [t.get("name") for t in self.tools],
            "original_query": self.original_query,
            "call_count": self.call_count,
            "metadata": self.metadata
        }