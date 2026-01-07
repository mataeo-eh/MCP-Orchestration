import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
from .router import MCPToolRouter, MCPRouterSession

logger = logging.getLogger(__name__)


class RoutedMCPServer:
    """
    MCP Server wrapper that uses intelligent tool routing.
    """
    
    def __init__(
        self,
        mcp_server_instance,
        router: MCPToolRouter,
        enable_routing: bool = True
    ):
        """
        Initialize routed MCP server.
        
        Args:
            mcp_server_instance: Your actual MCP server implementation
            router: MCPToolRouter instance for filtering
            enable_routing: Whether to enable routing (useful for A/B testing)
        """
        self.server = mcp_server_instance
        self.router = router
        self.enable_routing = enable_routing
        
        # Session management
        self.sessions: Dict[str, MCPRouterSession] = {}
    
    async def create_session(
        self,
        user_query: str,
        session_id: Optional[str] = None
    ) -> MCPRouterSession:
        """
        Create a new filtered session based on user query.
        
        Args:
            user_query: The user's query/intent
            session_id: Optional session ID (generates one if not provided)
            
        Returns:
            MCPRouterSession with filtered tools
        """
        session_id = session_id or str(uuid4())
        
        # Get all available tools from the server
        all_tools = await self.server.list_tools()
        
        if self.enable_routing:
            # Filter tools based on query
            filtered_result = await self.router.filter_tools(
                query=user_query,
                available_tools=all_tools
            )
            
            tools = filtered_result.tools
            metadata = {
                "filtered": True,
                "original_count": len(all_tools),
                "filtered_count": len(tools),
                "filter_time_ms": filtered_result.filter_time_ms
            }
            
            logger.info(
                f"Session {session_id}: Filtered {len(all_tools)} → {len(tools)} tools"
            )
        else:
            # No filtering - use all tools
            tools = all_tools
            metadata = {"filtered": False, "tool_count": len(all_tools)}
        
        # Create session
        session = MCPRouterSession(
            session_id=session_id,
            filtered_tools=tools,
            original_query=user_query,
            metadata=metadata
        )
        
        self.sessions[session_id] = session
        return session
    
    async def call_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool within a session context.
        
        Args:
            session_id: Session identifier
            tool_name: Name of tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        session = self.sessions.get(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Verify tool is in session's filtered set
        available_tool_names = [t.get("name") for t in session.tools]
        
        if tool_name not in available_tool_names:
            logger.warning(
                f"Tool '{tool_name}' not in filtered set for session {session_id}. "
                f"Available: {available_tool_names}"
            )
            # You could either:
            # 1. Reject the call
            # 2. Allow it anyway (permissive)
            # 3. Suggest alternatives
            # For now, we'll allow it but log
        
        # Increment call counter
        session.increment_call_count()
        
        # Delegate to actual MCP server
        result = await self.server.call_tool(tool_name, arguments)
        
        return result
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about routing effectiveness"""
        total_sessions = len(self.sessions)
        filtered_sessions = sum(
            1 for s in self.sessions.values()
            if s.metadata.get("filtered", False)
        )
        
        if filtered_sessions > 0:
            avg_reduction = sum(
                s.metadata.get("original_count", 0) - s.metadata.get("filtered_count", 0)
                for s in self.sessions.values()
                if s.metadata.get("filtered", False)
            ) / filtered_sessions
        else:
            avg_reduction = 0
        
        return {
            "total_sessions": total_sessions,
            "filtered_sessions": filtered_sessions,
            "avg_tools_filtered_out": avg_reduction,
            "total_tool_calls": sum(s.call_count for s in self.sessions.values())
        }