"""
Enhanced Archon MCP with Auto-Discovered Tool Integration
Extends the existing Archon MCP server with wrapped tools from discovered MCP servers
Integrated with Process Pool Manager for memory optimization
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from .mcp_discovery_service import mcp_discovery_service
from .mcp_tool_wrapper import mcp_tool_wrapper, get_all_wrapped_tools, call_wrapped_tool, get_wrapped_tool_info
from .process_pool_manager import process_pool_manager
from .cli_tool_discovery_service import cli_discovery_service
from .serena_wrapper_service import serena_wrapper_service

logger = logging.getLogger(__name__)

class EnhancedArchonMCPHandler:
    """Enhanced MCP handler that includes auto-discovered tools"""
    
    def __init__(self, base_handler):
        self.base_handler = base_handler  # Original Archon MCP handler
        self.enhanced_tools = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize enhanced MCP handler"""
        try:
            # Start MCP discovery service
            await mcp_discovery_service.start()
            
            # Initialize tool wrapper
            await mcp_tool_wrapper.initialize()
            
            # Load enhanced tools
            await self.load_enhanced_tools()
            
            self.initialized = True
            logger.info("Enhanced Archon MCP handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced MCP handler: {e}")
            
    async def load_enhanced_tools(self):
        """Load all enhanced tools"""
        try:
            # Get wrapped tools
            wrapped_tools = await get_all_wrapped_tools()
            
            # Create enhanced tool definitions
            for tool_name, tool_func in wrapped_tools.items():
                self.enhanced_tools[tool_name] = {
                    "function": tool_func,
                    "metadata": {
                        "name": tool_name,
                        "description": tool_func.__doc__ or f"Enhanced tool: {tool_name}",
                        "type": "wrapped_mcp"
                    }
                }
                
            logger.info(f"Loaded {len(self.enhanced_tools)} enhanced tools")
            
        except Exception as e:
            logger.error(f"Failed to load enhanced tools: {e}")
            
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request with enhanced tool support"""
        try:
            method = request.get("method")
            
            if method == "tools/list":
                return await self.handle_tools_list(request)
            elif method == "tools/call":
                return await self.handle_tool_call(request)
            else:
                # Delegate to base handler for other requests
                if hasattr(self.base_handler, 'handle_request'):
                    return await self.base_handler.handle_request(request)
                else:
                    return {"error": {"code": -32601, "message": "Method not found"}}
                    
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return {"error": {"code": -32603, "message": "Internal error"}}
            
    async def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request with enhanced tools"""
        try:
            # Get base tools from original handler
            base_response = {"result": {"tools": []}}
            if hasattr(self.base_handler, 'handle_tools_list'):
                base_response = await self.base_handler.handle_tools_list(request)
            elif hasattr(self.base_handler, 'handle_request'):
                base_response = await self.base_handler.handle_request(request)
                
            base_tools = base_response.get("result", {}).get("tools", [])
            
            # Add enhanced tools
            enhanced_tool_list = []
            
            # Add wrapped MCP tools
            wrapped_info = get_wrapped_tool_info()
            for tool_info in wrapped_info:
                enhanced_tool_list.append({
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": tool_info.get("parameters", {}),
                        "additionalProperties": True
                    }
                })
                
            # Add Claude Flow integration tools
            claude_flow_tools = [
                {
                    "name": "claude_flow_swarm_init_enhanced",
                    "description": "Initialize Claude Flow swarm with Archon integration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topology": {"type": "string", "description": "Swarm topology"},
                            "max_agents": {"type": "integer", "description": "Maximum agents"},
                            "archon_project_id": {"type": "string", "description": "Archon project ID"}
                        }
                    }
                },
                {
                    "name": "claude_flow_agent_spawn_enhanced", 
                    "description": "Spawn Claude Flow agents with Archon task integration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agents": {"type": "string", "description": "Comma-separated agent types"},
                            "objective": {"type": "string", "description": "Objective for agents"},
                            "archon_task_id": {"type": "string", "description": "Archon task ID"}
                        }
                    }
                },
                {
                    "name": "mcp_discovery_status",
                    "description": "Get status of auto-discovered MCP servers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "mcp_server_control",
                    "description": "Control MCP servers (start/stop/restart)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "server_name": {"type": "string", "description": "Name of MCP server"},
                            "action": {"type": "string", "enum": ["start", "stop", "restart"], "description": "Action to perform"}
                        },
                        "required": ["server_name", "action"]
                    }
                }
            ]
            
            # Combine all tools
            all_tools = base_tools + enhanced_tool_list + claude_flow_tools
            
            return {
                "result": {
                    "tools": all_tools
                }
            }
            
        except Exception as e:
            logger.error(f"Error in handle_tools_list: {e}")
            return {"error": {"code": -32603, "message": "Failed to list tools"}}
            
    async def handle_tool_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request with enhanced tools"""
        try:
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Check if it's an enhanced tool
            if tool_name in self.enhanced_tools:
                result = await self.enhanced_tools[tool_name]["function"](**arguments)
                return {"result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
                
            # Check for special enhanced tools
            elif tool_name == "claude_flow_swarm_init_enhanced":
                return await self.handle_claude_flow_swarm_init_enhanced(arguments)
            elif tool_name == "claude_flow_agent_spawn_enhanced":
                return await self.handle_claude_flow_agent_spawn_enhanced(arguments)
            elif tool_name == "mcp_discovery_status":
                return await self.handle_mcp_discovery_status(arguments)
            elif tool_name == "mcp_server_control":
                return await self.handle_mcp_server_control(arguments)
                
            # Check wrapped tools
            try:
                result = await call_wrapped_tool(tool_name, **arguments)
                return {"result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
            except ValueError:
                pass  # Tool not found, try base handler
                
            # Delegate to base handler
            if hasattr(self.base_handler, 'handle_tool_call'):
                return await self.base_handler.handle_tool_call(request)
            elif hasattr(self.base_handler, 'handle_request'):
                return await self.base_handler.handle_request(request)
            else:
                return {"error": {"code": -32601, "message": "Tool not found"}}
                
        except Exception as e:
            logger.error(f"Error in handle_tool_call: {e}")
            return {"error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"}}
            
    async def handle_claude_flow_swarm_init_enhanced(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle enhanced Claude Flow swarm initialization"""
        try:
            # Extract parameters
            topology = arguments.get("topology", "mesh")
            max_agents = arguments.get("max_agents", 5)
            archon_project_id = arguments.get("archon_project_id")
            
            # Initialize swarm via wrapped tool
            swarm_result = await call_wrapped_tool("claude_flow_swarm_init", 
                                                 topology=topology,
                                                 maxAgents=max_agents)
            
            # If Archon project ID provided, create integration task
            if archon_project_id and swarm_result.get("success"):
                try:
                    # Import here to avoid circular imports
                    from src.server.unified_archon_mcp import create_task
                    
                    await create_task(
                        project_id=archon_project_id,
                        title="Claude Flow Swarm Integration",
                        description=f"Integration with Claude Flow swarm (topology: {topology}, max_agents: {max_agents})",
                        assignee="Claude Flow Expert",
                        feature="swarm-integration"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Archon integration task: {e}")
                    
            return {"result": {"content": [{"type": "text", "text": json.dumps(swarm_result, indent=2)}]}}
            
        except Exception as e:
            logger.error(f"Enhanced swarm init failed: {e}")
            return {"error": {"code": -32603, "message": f"Enhanced swarm init failed: {str(e)}"}}
            
    async def handle_claude_flow_agent_spawn_enhanced(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle enhanced Claude Flow agent spawning"""
        try:
            # Extract parameters
            agents = arguments.get("agents", "")
            objective = arguments.get("objective", "")
            archon_task_id = arguments.get("archon_task_id")
            
            # Spawn agents via wrapped tool
            spawn_result = await call_wrapped_tool("claude_flow_agent_spawn",
                                                 agents=agents,
                                                 objective=objective)
            
            # If Archon task ID provided, update task status
            if archon_task_id and spawn_result.get("success"):
                try:
                    from src.server.unified_archon_mcp import update_task
                    
                    await update_task(
                        task_id=archon_task_id,
                        status="doing",
                        description=f"Updated: Claude Flow agents spawned - {agents}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Archon task: {e}")
                    
            return {"result": {"content": [{"type": "text", "text": json.dumps(spawn_result, indent=2)}]}}
            
        except Exception as e:
            logger.error(f"Enhanced agent spawn failed: {e}")
            return {"error": {"code": -32603, "message": f"Enhanced agent spawn failed: {str(e)}"}}
            
    async def handle_mcp_discovery_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP discovery status request"""
        try:
            status = {
                "discovery_service_running": mcp_discovery_service.running,
                "servers": mcp_discovery_service.get_server_status(),
                "discovered_tools_count": len(mcp_discovery_service.get_discovered_tools()),
                "enhanced_tools_count": len(self.enhanced_tools),
                "initialized": self.initialized
            }
            
            return {"result": {"content": [{"type": "text", "text": json.dumps(status, indent=2, default=str)}]}}
            
        except Exception as e:
            logger.error(f"Failed to get MCP discovery status: {e}")
            return {"error": {"code": -32603, "message": f"Failed to get status: {str(e)}"}}
            
    async def handle_mcp_server_control(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP server control request"""
        try:
            server_name = arguments.get("server_name")
            action = arguments.get("action")
            
            if not server_name or not action:
                return {"error": {"code": -32602, "message": "Missing required parameters"}}
                
            result = None
            if action == "start":
                result = await mcp_discovery_service.start_server(server_name)
            elif action == "stop":
                result = await mcp_discovery_service.stop_server(server_name)
            elif action == "restart":
                await mcp_discovery_service.stop_server(server_name)
                await asyncio.sleep(2)
                result = await mcp_discovery_service.start_server(server_name)
            else:
                return {"error": {"code": -32602, "message": "Invalid action"}}
                
            return {"result": {"content": [{"type": "text", "text": json.dumps({
                "success": result,
                "server": server_name,
                "action": action
            }, indent=2)}]}}
            
        except Exception as e:
            logger.error(f"MCP server control failed: {e}")
            return {"error": {"code": -32603, "message": f"Server control failed: {str(e)}"}}
            
    async def refresh_tools(self):
        """Refresh enhanced tools after discovery updates"""
        try:
            await mcp_tool_wrapper.refresh_tools()
            await self.load_enhanced_tools()
            logger.info("Enhanced tools refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")

# Global enhanced handler instance
enhanced_archon_mcp = None

async def get_enhanced_archon_mcp(base_handler=None):
    """Get the global enhanced Archon MCP handler"""
    global enhanced_archon_mcp
    
    if enhanced_archon_mcp is None:
        enhanced_archon_mcp = EnhancedArchonMCPHandler(base_handler)
        await enhanced_archon_mcp.initialize()
        
    return enhanced_archon_mcp

# Convenience functions for direct tool access
async def enhanced_claude_flow_swarm_init(**kwargs):
    """Direct access to enhanced Claude Flow swarm init"""
    handler = await get_enhanced_archon_mcp()
    return await handler.handle_claude_flow_swarm_init_enhanced(kwargs)

async def enhanced_claude_flow_agent_spawn(**kwargs):
    """Direct access to enhanced Claude Flow agent spawn"""
    handler = await get_enhanced_archon_mcp()
    return await handler.handle_claude_flow_agent_spawn_enhanced(kwargs)