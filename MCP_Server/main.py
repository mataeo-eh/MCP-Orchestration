import asyncio
import logging
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
load_dotenv()  # reads .env into environment
from router import MCPToolRouter
from MCP_Routed_Server import RoutedMCPServer
from openai import OpenAI, AsyncOpenAI
from server import YourMCPServer  # Your actual MCP server
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load API key from env
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
# Load abse URL to use from env
base_url = os.getenv("URL")
if not base_url:
    raise RuntimeError("Missing API endpoint URL environment variable.")
# Load the model selected to use from env
# You may want to validate that OPENAI_MODEL is set as well
model = os.getenv("MODEL")
if not model:
    raise RuntimeError("Missing OPENAI_MODEL environment variable.")


async def main():
    # Initialize your base MCP server (with all your tools)
    mcp_server = YourMCPServer()
    await mcp_server.initialize()
    
    # Initialize router with Haiku
    router = MCPToolRouter(
        api_key=api_key,
        model=model,
        max_tools=5,
        fallback_toolset=[]  # Some safe default tools
    )
    
    # Wrap server with router
    routed_server = RoutedMCPServer(
        mcp_server_instance=mcp_server,
        router=router,
        enable_routing=True
    )
    
    # Example usage
    user_query = "Find all courses related to machine learning and show me the prerequisites"
    
    # Create filtered session
    session = await routed_server.create_session(user_query)
    
    logger.info(f"Session created with {len(session.tools)} tools")
    logger.info(f"Available tools: {[t['name'] for t in session.tools]}")
    
    # Now call tools within this session
    result = await routed_server.call_tool(
        session_id=session.session_id,
        tool_name="search_courses",
        arguments={"query": "machine learning"}
    )
    
    logger.info(f"Tool result: {result}")
    
    # Get stats
    stats = routed_server.get_session_stats()
    logger.info(f"Routing stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())