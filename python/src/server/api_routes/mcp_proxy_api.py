"""
MCP Proxy API Routes
Exposes the auto-discovered MCP tools through Archon's API
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.mcp_discovery_service import mcp_discovery_service, MCPServerConfig, MCPTool, MCPServerStatus
from ..services.cli_tool_discovery_service import cli_discovery_service, CLIToolConfig, CLICommand, CLIToolStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["MCP Proxy"])

# Pydantic models for API
class MCPServerConfigRequest(BaseModel):
    name: str
    command: str
    args: List[str]
    transport: str = "stdio"
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    auto_start: bool = True
    health_check_interval: int = 30
    retry_attempts: int = 3
    timeout: int = 10

class MCPToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call (format: server__tool)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")

class MCPToolResponse(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    server: str
    tool_type: str = "mcp"  # "mcp" or "cli"

class CLICommandResponse(BaseModel):
    name: str
    description: str
    usage: str
    options: Dict[str, Any]
    tool: str
    tool_type: str = "cli"

class MCPServerStatusResponse(BaseModel):
    name: str
    status: str
    pid: Optional[int] = None
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None
    tools_count: int = 0

class MCPDiscoveryStatusResponse(BaseModel):
    total_servers: int
    running_servers: int
    total_tools: int
    total_cli_tools: int
    servers: List[MCPServerStatusResponse]
    cli_tools_status: Dict[str, str]

@router.on_event("startup")
async def startup_mcp_discovery():
    """Start MCP and CLI discovery services when API starts"""
    try:
        await mcp_discovery_service.start()
        logger.info("MCP Auto-Discovery Service started successfully")
        
        await cli_discovery_service.start()
        logger.info("CLI Tool Discovery Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start discovery services: {e}")

@router.on_event("shutdown") 
async def shutdown_mcp_discovery():
    """Stop MCP discovery service when API shuts down"""
    try:
        await mcp_discovery_service.stop()
        logger.info("MCP Auto-Discovery Service stopped")
    except Exception as e:
        logger.error(f"Error stopping MCP Auto-Discovery Service: {e}")

@router.get("/status", response_model=MCPDiscoveryStatusResponse)
async def get_mcp_status():
    """Get overall MCP and CLI discovery status"""
    try:
        # Get MCP server status
        server_statuses = mcp_discovery_service.get_server_status()
        discovered_tools = mcp_discovery_service.get_discovered_tools()
        
        # Get CLI tool status
        cli_tool_statuses = cli_discovery_service.get_tool_status()
        cli_commands = cli_discovery_service.get_discovered_commands()
        
        servers = []
        running_count = 0
        
        for name, status in server_statuses.items():
            server_tools_count = len([
                tool for tool in discovered_tools.values()
                if tool.server == name
            ])
            
            servers.append(MCPServerStatusResponse(
                name=status.name,
                status=status.status,
                pid=status.pid,
                last_health_check=datetime.fromtimestamp(status.last_health_check) if status.last_health_check else None,
                error_message=status.error_message,
                tools_count=server_tools_count
            ))
            
            if status.status == "running":
                running_count += 1
        
        # Build CLI tools status dict
        cli_status = {}
        for name, status in cli_tool_statuses.items():
            cli_status[name] = "available" if status.available else "unavailable"
        
        return MCPDiscoveryStatusResponse(
            total_servers=len(server_statuses),
            running_servers=running_count,
            total_tools=len(discovered_tools),
            total_cli_tools=len(cli_commands),
            servers=servers,
            cli_tools_status=cli_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get MCP status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get MCP status")

@router.get("/tools")
async def get_discovered_tools(server: Optional[str] = None, include_cli: bool = True):
    """Get all discovered MCP tools and CLI commands"""
    try:
        # Get MCP tools
        discovered_tools = mcp_discovery_service.get_discovered_tools(server)
        mcp_tools = [
            MCPToolResponse(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                server=tool.server,
                tool_type="mcp"
            )
            for tool in discovered_tools.values()
        ]
        
        # Get CLI commands if requested
        cli_tools = []
        if include_cli:
            cli_commands = cli_discovery_service.get_discovered_commands(server)
            cli_tools = [
                CLICommandResponse(
                    name=cmd.name,
                    description=cmd.description,
                    usage=cmd.usage,
                    options=cmd.options,
                    tool=cmd.tool,
                    tool_type="cli"
                )
                for cmd in cli_commands.values()
            ]
        
        # Combine both types
        return {
            "mcp_tools": mcp_tools,
            "cli_tools": cli_tools,
            "total_tools": len(mcp_tools) + len(cli_tools)
        }
        
    except Exception as e:
        logger.error(f"Failed to get discovered tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to get discovered tools")

@router.post("/tools/call")
async def call_tool(request: MCPToolCallRequest):
    """Call a discovered MCP tool or CLI command"""
    try:
        # First try MCP tools
        try:
            result = await mcp_discovery_service.call_tool(
                request.tool_name,
                request.parameters
            )
            
            return {
                "success": True,
                "result": result,
                "tool": request.tool_name,
                "tool_type": "mcp",
                "timestamp": datetime.now().isoformat()
            }
        except ValueError:
            # Tool not found in MCP, try CLI commands
            pass
        
        # Try CLI commands
        result = await cli_discovery_service.execute_command(
            request.tool_name,
            request.parameters
        )
        
        return {
            "success": result.get("success", False),
            "result": result,
            "tool": request.tool_name,
            "tool_type": "cli",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Tool not found in MCP or CLI: {e}")
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")
    except RuntimeError as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error calling tool: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/servers", response_model=List[MCPServerStatusResponse])
async def get_servers():
    """Get all configured MCP servers and their status"""
    try:
        server_statuses = mcp_discovery_service.get_server_status()
        discovered_tools = mcp_discovery_service.get_discovered_tools()
        
        servers = []
        for name, status in server_statuses.items():
            server_tools_count = len([
                tool for tool in discovered_tools.values()
                if tool.server == name
            ])
            
            servers.append(MCPServerStatusResponse(
                name=status.name,
                status=status.status,
                pid=status.pid,
                last_health_check=datetime.fromtimestamp(status.last_health_check) if status.last_health_check else None,
                error_message=status.error_message,
                tools_count=server_tools_count
            ))
            
        return servers
        
    except Exception as e:
        logger.error(f"Failed to get servers: {e}")
        raise HTTPException(status_code=500, detail="Failed to get servers")

@router.post("/servers/{server_name}/start")
async def start_server(server_name: str, background_tasks: BackgroundTasks):
    """Start a specific MCP server"""
    try:
        # Run in background to avoid blocking
        background_tasks.add_task(mcp_discovery_service.start_server, server_name)
        
        return {
            "success": True,
            "message": f"Starting server {server_name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start server {server_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {e}")

@router.post("/servers/{server_name}/stop")
async def stop_server(server_name: str, background_tasks: BackgroundTasks):
    """Stop a specific MCP server"""
    try:
        # Run in background to avoid blocking
        background_tasks.add_task(mcp_discovery_service.stop_server, server_name)
        
        return {
            "success": True,
            "message": f"Stopping server {server_name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop server {server_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {e}")

@router.post("/servers/{server_name}/restart")
async def restart_server(server_name: str, background_tasks: BackgroundTasks):
    """Restart a specific MCP server"""
    try:
        async def restart_task():
            await mcp_discovery_service.stop_server(server_name)
            await asyncio.sleep(2)  # Wait between stop and start
            await mcp_discovery_service.start_server(server_name)
            
        background_tasks.add_task(restart_task)
        
        return {
            "success": True,
            "message": f"Restarting server {server_name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restart server {server_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart server: {e}")

@router.post("/servers")
async def add_server(config: MCPServerConfigRequest):
    """Add a new MCP server configuration"""
    try:
        server_config = MCPServerConfig(
            name=config.name,
            command=config.command,
            args=config.args,
            transport=config.transport,
            url=config.url,
            env=config.env,
            auto_start=config.auto_start,
            health_check_interval=config.health_check_interval,
            retry_attempts=config.retry_attempts,
            timeout=config.timeout
        )
        
        success = await mcp_discovery_service.add_server(server_config)
        
        if success:
            return {
                "success": True,
                "message": f"Server {config.name} added successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add server")
            
    except Exception as e:
        logger.error(f"Failed to add server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/servers/{server_name}")
async def remove_server(server_name: str):
    """Remove an MCP server configuration"""
    try:
        success = await mcp_discovery_service.remove_server(server_name)
        
        if success:
            return {
                "success": True,
                "message": f"Server {server_name} removed successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Server not found")
            
    except Exception as e:
        logger.error(f"Failed to remove server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}/tools", response_model=List[MCPToolResponse])
async def get_server_tools(server_name: str):
    """Get tools for a specific server"""
    try:
        discovered_tools = mcp_discovery_service.get_discovered_tools(server_name)
        
        return [
            MCPToolResponse(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                server=tool.server
            )
            for tool in discovered_tools.values()
        ]
        
    except Exception as e:
        logger.error(f"Failed to get tools for server {server_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get server tools")

@router.post("/discovery/refresh")
async def refresh_discovery(background_tasks: BackgroundTasks):
    """Manually trigger tool discovery refresh"""
    try:
        async def refresh_task():
            server_statuses = mcp_discovery_service.get_server_status()
            for name in server_statuses:
                if server_statuses[name].status == "running":
                    await mcp_discovery_service.discover_server_tools(name)
                    
        background_tasks.add_task(refresh_task)
        
        return {
            "success": True,
            "message": "Discovery refresh initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh discovery: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh discovery")

@router.get("/cli/status")
async def get_cli_status():
    """Get CLI tool discovery status"""
    try:
        cli_statuses = cli_discovery_service.get_tool_status()
        cli_commands = cli_discovery_service.get_discovered_commands()
        
        status_info = []
        available_count = 0
        
        for name, status in cli_statuses.items():
            command_count = len([cmd for cmd in cli_commands.values() if cmd.tool == name])
            status_info.append({
                "name": status.name,
                "available": status.available,
                "version": status.version,
                "commands_count": command_count,
                "error_message": status.error_message,
                "last_check": datetime.fromtimestamp(status.last_check) if status.last_check else None
            })
            
            if status.available:
                available_count += 1
        
        return {
            "total_cli_tools": len(cli_statuses),
            "available_tools": available_count,
            "total_commands": len(cli_commands),
            "tools": status_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get CLI status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get CLI status")

@router.get("/cli/commands")
async def get_cli_commands(tool: Optional[str] = None):
    """Get all discovered CLI commands, optionally filtered by tool"""
    try:
        cli_commands = cli_discovery_service.get_discovered_commands(tool)
        
        return [
            {
                "name": cmd.name,
                "description": cmd.description,
                "usage": cmd.usage,
                "options": cmd.options,
                "tool": cmd.tool
            }
            for cmd in cli_commands.values()
        ]
        
    except Exception as e:
        logger.error(f"Failed to get CLI commands: {e}")
        raise HTTPException(status_code=500, detail="Failed to get CLI commands")

@router.post("/cli/refresh")
async def refresh_cli_discovery(background_tasks: BackgroundTasks):
    """Manually trigger CLI tool discovery refresh"""
    try:
        background_tasks.add_task(cli_discovery_service.refresh_discovery)
        
        return {
            "success": True,
            "message": "CLI discovery refresh initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh CLI discovery: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh CLI discovery")

@router.get("/health")
async def health_check():
    """Health check endpoint for MCP and CLI discovery"""
    try:
        mcp_service_status = "running" if mcp_discovery_service.running else "stopped"
        cli_service_status = "running" if cli_discovery_service.running else "stopped"
        
        server_count = len(mcp_discovery_service.get_server_status())
        mcp_tool_count = len(mcp_discovery_service.get_discovered_tools())
        
        cli_statuses = cli_discovery_service.get_tool_status()
        cli_command_count = len(cli_discovery_service.get_discovered_commands())
        cli_available = sum(1 for status in cli_statuses.values() if status.available)
        
        return {
            "status": "healthy",
            "mcp_service_status": mcp_service_status,
            "cli_service_status": cli_service_status,
            "mcp_servers_configured": server_count,
            "mcp_tools_discovered": mcp_tool_count,
            "cli_tools_total": len(cli_statuses),
            "cli_tools_available": cli_available,
            "cli_commands_discovered": cli_command_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }