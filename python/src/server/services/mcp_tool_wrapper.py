"""
MCP Tool Wrapper
Creates Archon MCP tools that wrap discovered MCP servers
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from .mcp_discovery_service import mcp_discovery_service

logger = logging.getLogger(__name__)

class MCPToolWrapper:
    """Wraps discovered MCP tools as Archon MCP tools"""
    
    def __init__(self):
        self.wrapped_tools = {}
        
    async def initialize(self):
        """Initialize the wrapper and start discovering tools"""
        # Ensure MCP discovery service is running
        if not mcp_discovery_service.running:
            await mcp_discovery_service.start()
            
        # Wait for initial discovery
        await asyncio.sleep(5)
        
        # Create wrapped tools
        await self.create_wrapped_tools()
        
    async def create_wrapped_tools(self):
        """Create wrapped tools for all discovered MCP tools"""
        discovered_tools = mcp_discovery_service.get_discovered_tools()
        
        for tool_name, tool in discovered_tools.items():
            wrapped_name = f"mcp_wrapped__{tool_name}"
            self.wrapped_tools[wrapped_name] = self._create_tool_wrapper(tool)
            
        logger.info(f"Created {len(self.wrapped_tools)} wrapped MCP tools")
        
    def _create_tool_wrapper(self, tool):
        """Create a wrapper function for a specific MCP tool"""
        
        async def wrapped_tool(**kwargs):
            """Wrapped MCP tool execution"""
            try:
                result = await mcp_discovery_service.call_tool(tool.name, kwargs)
                return {
                    "success": True,
                    "result": result,
                    "tool": tool.name,
                    "server": tool.server
                }
            except Exception as e:
                logger.error(f"Wrapped tool {tool.name} failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tool": tool.name,
                    "server": tool.server
                }
                
        # Add metadata to the function
        wrapped_tool.__name__ = f"mcp_wrapped__{tool.name}"
        wrapped_tool.__doc__ = f"Wrapped MCP tool: {tool.description} (from {tool.server})"
        wrapped_tool._mcp_tool_info = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "server": tool.server
        }
        
        return wrapped_tool
        
    def get_wrapped_tools(self) -> Dict[str, Any]:
        """Get all wrapped tools"""
        return self.wrapped_tools
        
    def get_tool_info(self, wrapped_name: str) -> Optional[Dict[str, Any]]:
        """Get info about a wrapped tool"""
        tool = self.wrapped_tools.get(wrapped_name)
        if tool and hasattr(tool, '_mcp_tool_info'):
            return tool._mcp_tool_info
        return None
        
    async def refresh_tools(self):
        """Refresh wrapped tools after discovery updates"""
        await self.create_wrapped_tools()

# Global wrapper instance
mcp_tool_wrapper = MCPToolWrapper()

# Tool wrapper functions that can be used as MCP tools
async def claude_flow_swarm_init(**kwargs):
    """Initialize Claude Flow swarm via wrapped MCP"""
    return await mcp_discovery_service.call_tool("claude-flow__swarm_init", kwargs)

async def claude_flow_agent_spawn(**kwargs):
    """Spawn Claude Flow agents via wrapped MCP"""
    return await mcp_discovery_service.call_tool("claude-flow__agent_spawn", kwargs)

async def claude_flow_task_orchestrate(**kwargs):
    """Orchestrate Claude Flow tasks via wrapped MCP"""
    return await mcp_discovery_service.call_tool("claude-flow__task_orchestrate", kwargs)

async def claude_flow_swarm_status(**kwargs):
    """Get Claude Flow swarm status via wrapped MCP"""
    return await mcp_discovery_service.call_tool("claude-flow__swarm_status", kwargs)

async def ruv_swarm_init(**kwargs):
    """Initialize Ruv Swarm via wrapped MCP"""
    return await mcp_discovery_service.call_tool("ruv-swarm__swarm_init", kwargs)

async def ruv_swarm_spawn(**kwargs):
    """Spawn Ruv Swarm agents via wrapped MCP"""
    return await mcp_discovery_service.call_tool("ruv-swarm__agent_spawn", kwargs)

async def flow_nexus_deploy(**kwargs):
    """Deploy via Flow Nexus wrapped MCP"""
    return await mcp_discovery_service.call_tool("flow-nexus__deploy", kwargs)

async def serena_analyze(**kwargs):
    """Analyze code via Serena wrapped MCP"""
    return await mcp_discovery_service.call_tool("serena__analyze_codebase", kwargs)

async def serena_semantic_search(**kwargs):
    """Semantic search via Serena wrapped MCP"""
    return await mcp_discovery_service.call_tool("serena__semantic_search", kwargs)

# Registry of common wrapped tools
COMMON_WRAPPED_TOOLS = {
    "claude_flow_swarm_init": claude_flow_swarm_init,
    "claude_flow_agent_spawn": claude_flow_agent_spawn,
    "claude_flow_task_orchestrate": claude_flow_task_orchestrate,
    "claude_flow_swarm_status": claude_flow_swarm_status,
    "ruv_swarm_init": ruv_swarm_init,
    "ruv_swarm_spawn": ruv_swarm_spawn,
    "flow_nexus_deploy": flow_nexus_deploy,
    "serena_analyze": serena_analyze,
    "serena_semantic_search": serena_semantic_search
}

async def get_all_wrapped_tools():
    """Get all available wrapped MCP tools"""
    # Ensure wrapper is initialized
    if not mcp_tool_wrapper.wrapped_tools:
        await mcp_tool_wrapper.initialize()
        
    # Combine common tools and dynamic tools
    all_tools = COMMON_WRAPPED_TOOLS.copy()
    all_tools.update(mcp_tool_wrapper.get_wrapped_tools())
    
    return all_tools

async def call_wrapped_tool(tool_name: str, **kwargs):
    """Call a wrapped MCP tool by name"""
    all_tools = await get_all_wrapped_tools()
    
    if tool_name in all_tools:
        return await all_tools[tool_name](**kwargs)
    else:
        raise ValueError(f"Wrapped tool {tool_name} not found")

def get_wrapped_tool_info() -> List[Dict[str, Any]]:
    """Get information about all wrapped tools"""
    info = []
    
    # Add common tools info
    for name, func in COMMON_WRAPPED_TOOLS.items():
        info.append({
            "name": name,
            "description": func.__doc__ or f"Wrapped MCP tool: {name}",
            "type": "common_wrapped",
            "parameters": {}  # Could be enhanced with parameter introspection
        })
        
    # Add dynamic tools info
    for name, func in mcp_tool_wrapper.get_wrapped_tools().items():
        tool_info = mcp_tool_wrapper.get_tool_info(name)
        if tool_info:
            info.append({
                "name": name,
                "description": tool_info["description"],
                "type": "dynamic_wrapped",
                "parameters": tool_info["parameters"],
                "server": tool_info["server"],
                "original_name": tool_info["name"]
            })
            
    return info