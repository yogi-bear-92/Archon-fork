"""
MCP Auto Tool Discovery Service
Dynamic discovery and proxying of MCP servers through Archon
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
import websockets
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str]
    transport: str = "stdio"  # stdio, http, websocket
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    auto_start: bool = True
    health_check_interval: int = 30
    retry_attempts: int = 3
    timeout: int = 10

@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    server: str
    method: str = "tools/call"

@dataclass
class MCPServerStatus:
    """Status of an MCP server"""
    name: str
    status: str  # running, stopped, error, connecting
    pid: Optional[int] = None
    last_health_check: Optional[float] = None
    error_message: Optional[str] = None
    tools: List[MCPTool] = None

class MCPAutoDiscoveryService:
    """Auto-discovery and management service for MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.server_status: Dict[str, MCPServerStatus] = {}
        self.discovered_tools: Dict[str, MCPTool] = {}
        self.config_file = Path("config/mcp_servers.json")
        self.running = False
        
        # Default servers we want to wrap
        self.default_servers = {
            "claude-flow": MCPServerConfig(
                name="claude-flow",
                command="npx",
                args=["claude-flow@alpha", "mcp", "start"],
                transport="stdio",
                auto_start=True
            ),
            "ruv-swarm": MCPServerConfig(
                name="ruv-swarm", 
                command="npx",
                args=["ruv-swarm", "mcp", "start"],
                transport="stdio",
                auto_start=True
            ),
            "flow-nexus": MCPServerConfig(
                name="flow-nexus",
                command="npx", 
                args=["flow-nexus@latest", "mcp", "start"],
                transport="stdio",
                auto_start=True
            ),
            "serena": MCPServerConfig(
                name="serena",
                command="uvx",
                args=[
                    "--from", 
                    "git+https://github.com/oraios/serena", 
                    "serena", 
                    "start-mcp-server", 
                    "--context", 
                    "ide-assistant", 
                    "--project", 
                    "/Users/yogi/Projects/Archon-fork"
                ],
                transport="stdio",
                auto_start=True
            )
        }
        
    async def start(self):
        """Start the auto-discovery service"""
        logger.info("Starting MCP Auto-Discovery Service")
        self.running = True
        
        # Load configuration
        await self.load_config()
        
        # Start default servers
        await self.start_default_servers()
        
        # Begin discovery and health monitoring
        asyncio.create_task(self.discovery_loop())
        asyncio.create_task(self.health_monitor_loop())
        
    async def stop(self):
        """Stop the auto-discovery service"""
        logger.info("Stopping MCP Auto-Discovery Service")
        self.running = False
        
        # Stop all managed servers
        await self.stop_all_servers()
        
    async def load_config(self):
        """Load MCP server configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                for name, config in config_data.items():
                    self.servers[name] = MCPServerConfig(**config)
                    
                logger.info(f"Loaded {len(self.servers)} MCP server configurations")
            except Exception as e:
                logger.error(f"Failed to load MCP config: {e}")
        else:
            # Use defaults
            self.servers = self.default_servers.copy()
            await self.save_config()
            
    async def save_config(self):
        """Save MCP server configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = {
                name: asdict(config) 
                for name, config in self.servers.items()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save MCP config: {e}")
            
    async def start_default_servers(self):
        """Start the default MCP servers"""
        for name, config in self.servers.items():
            if config.auto_start:
                await self.start_server(name)
                
    async def start_server(self, name: str) -> bool:
        """Start a specific MCP server"""
        if name not in self.servers:
            logger.error(f"Server {name} not configured")
            return False
            
        config = self.servers[name]
        
        try:
            # Check if already running
            if name in self.server_processes:
                process = self.server_processes[name]
                if process.poll() is None:  # Still running
                    logger.info(f"Server {name} already running")
                    return True
                    
            logger.info(f"Starting MCP server: {name}")
            
            # Prepare environment
            env = dict(os.environ)
            if config.env:
                env.update(config.env)
                
            # Start process
            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            self.server_processes[name] = process
            self.server_status[name] = MCPServerStatus(
                name=name,
                status="connecting",
                pid=process.pid
            )
            
            # Wait for startup and discover tools
            await asyncio.sleep(2)  # Give server time to start
            await self.discover_server_tools(name)
            
            logger.info(f"Started MCP server {name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server {name}: {e}")
            self.server_status[name] = MCPServerStatus(
                name=name,
                status="error", 
                error_message=str(e)
            )
            return False
            
    async def stop_server(self, name: str) -> bool:
        """Stop a specific MCP server"""
        if name not in self.server_processes:
            return True
            
        try:
            process = self.server_processes[name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
            del self.server_processes[name]
            self.server_status[name] = MCPServerStatus(
                name=name,
                status="stopped"
            )
            
            logger.info(f"Stopped MCP server: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server {name}: {e}")
            return False
            
    async def stop_all_servers(self):
        """Stop all managed MCP servers"""
        tasks = [self.stop_server(name) for name in list(self.server_processes.keys())]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def discover_server_tools(self, name: str):
        """Discover tools available from a specific server"""
        if name not in self.server_processes:
            return
            
        config = self.servers[name] 
        process = self.server_processes[name]
        
        try:
            if config.transport == "stdio":
                await self._discover_stdio_tools(name, process)
            elif config.transport == "http":
                await self._discover_http_tools(name, config)
            elif config.transport == "websocket":
                await self._discover_websocket_tools(name, config)
                
        except Exception as e:
            logger.error(f"Failed to discover tools for {name}: {e}")
            
    async def _discover_stdio_tools(self, name: str, process: subprocess.Popen):
        """Discover tools from stdio MCP server"""
        try:
            # Give the server more time to start up
            await asyncio.sleep(3)
            
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Process {name} died during startup")
                return
                
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "archon-mcp-discovery", "version": "1.0.0"}
                }
            }
            
            init_json = json.dumps(init_request) + "\n"
            process.stdin.write(init_json)
            process.stdin.flush()
            
            # Read response with timeout
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline), 
                timeout=5.0
            )
            
            if response_line.strip():
                try:
                    response = json.loads(response_line)
                    logger.debug(f"Initialize response from {name}: {response}")
                    
                    # Send tools list request
                    tools_request = {
                        "jsonrpc": "2.0", 
                        "id": 2,
                        "method": "tools/list",
                        "params": {}
                    }
                    
                    tools_json = json.dumps(tools_request) + "\n"
                    process.stdin.write(tools_json)
                    process.stdin.flush()
                    
                    # Read tools response with timeout
                    tools_response_line = await asyncio.wait_for(
                        asyncio.to_thread(process.stdout.readline),
                        timeout=5.0
                    )
                    
                    if tools_response_line.strip():
                        try:
                            tools_response = json.loads(tools_response_line)
                            await self._register_discovered_tools(name, tools_response)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in tools response from {name}: {e}")
                            logger.debug(f"Raw tools response: {tools_response_line}")
                    else:
                        logger.warning(f"Empty tools response from {name}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in init response from {name}: {e}")
                    logger.debug(f"Raw init response: {response_line}")
            else:
                logger.warning(f"Empty init response from {name}")
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while discovering tools for {name}")
        except Exception as e:
            logger.error(f"Failed to discover stdio tools for {name}: {e}")
            
    async def _discover_http_tools(self, name: str, config: MCPServerConfig):
        """Discover tools from HTTP MCP server"""
        if not config.url:
            return
            
        try:
            async with httpx.AsyncClient() as client:
                # Request tools list
                response = await client.post(
                    f"{config.url}/tools/list",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    timeout=config.timeout
                )
                
                if response.status_code == 200:
                    tools_response = response.json()
                    await self._register_discovered_tools(name, tools_response)
                    
        except Exception as e:
            logger.error(f"Failed to discover HTTP tools for {name}: {e}")
            
    async def _discover_websocket_tools(self, name: str, config: MCPServerConfig):
        """Discover tools from WebSocket MCP server"""
        if not config.url:
            return
            
        try:
            async with websockets.connect(config.url) as websocket:
                # Request tools list
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 1, 
                    "method": "tools/list",
                    "params": {}
                }
                
                await websocket.send(json.dumps(tools_request))
                response = await websocket.recv()
                tools_response = json.loads(response)
                
                await self._register_discovered_tools(name, tools_response)
                
        except Exception as e:
            logger.error(f"Failed to discover WebSocket tools for {name}: {e}")
            
    async def _register_discovered_tools(self, server_name: str, tools_response: Dict):
        """Register discovered tools from a server"""
        try:
            if "result" in tools_response and "tools" in tools_response["result"]:
                tools = tools_response["result"]["tools"]
                
                registered_count = 0
                for tool_def in tools:
                    tool = MCPTool(
                        name=f"{server_name}__{tool_def['name']}",
                        description=tool_def.get("description", ""),
                        parameters=tool_def.get("inputSchema", {}),
                        server=server_name,
                        method="tools/call"
                    )
                    
                    self.discovered_tools[tool.name] = tool
                    registered_count += 1
                    
                logger.info(f"Registered {registered_count} tools from {server_name}")
                
                # Update server status
                self.server_status[server_name] = MCPServerStatus(
                    name=server_name,
                    status="running",
                    pid=self.server_processes.get(server_name, {}).pid if server_name in self.server_processes else None,
                    last_health_check=time.time(),
                    tools=[tool for tool in self.discovered_tools.values() if tool.server == server_name]
                )
                
        except Exception as e:
            logger.error(f"Failed to register tools from {server_name}: {e}")
            
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a discovered tool through its MCP server"""
        if tool_name not in self.discovered_tools:
            raise ValueError(f"Tool {tool_name} not found")
            
        tool = self.discovered_tools[tool_name]
        server_name = tool.server
        
        if server_name not in self.server_processes:
            raise RuntimeError(f"Server {server_name} not running")
            
        config = self.servers[server_name]
        
        try:
            if config.transport == "stdio":
                return await self._call_stdio_tool(server_name, tool, parameters)
            elif config.transport == "http":
                return await self._call_http_tool(server_name, tool, parameters)
            elif config.transport == "websocket":
                return await self._call_websocket_tool(server_name, tool, parameters)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise
            
    async def _call_stdio_tool(self, server_name: str, tool: MCPTool, parameters: Dict) -> Dict:
        """Call tool via stdio transport"""
        process = self.server_processes[server_name]
        
        # Extract original tool name (remove server prefix)
        original_tool_name = tool.name.split("__", 1)[1] if "__" in tool.name else tool.name
        
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": tool.method,
            "params": {
                "name": original_tool_name,
                "arguments": parameters
            }
        }
        
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            if "result" in response:
                return response["result"]
            elif "error" in response:
                raise RuntimeError(f"Tool error: {response['error']}")
                
        raise RuntimeError("No response from server")
        
    async def _call_http_tool(self, server_name: str, tool: MCPTool, parameters: Dict) -> Dict:
        """Call tool via HTTP transport"""
        config = self.servers[server_name]
        original_tool_name = tool.name.split("__", 1)[1] if "__" in tool.name else tool.name
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": tool.method,
            "params": {
                "name": original_tool_name,
                "arguments": parameters
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.url}/tools/call",
                json=request,
                timeout=config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    raise RuntimeError(f"Tool error: {result['error']}")
                    
        raise RuntimeError("Failed to call HTTP tool")
        
    async def _call_websocket_tool(self, server_name: str, tool: MCPTool, parameters: Dict) -> Dict:
        """Call tool via WebSocket transport"""
        config = self.servers[server_name]
        original_tool_name = tool.name.split("__", 1)[1] if "__" in tool.name else tool.name
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": tool.method,
            "params": {
                "name": original_tool_name,
                "arguments": parameters
            }
        }
        
        async with websockets.connect(config.url) as websocket:
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            result = json.loads(response)
            
            if "result" in result:
                return result["result"]
            elif "error" in result:
                raise RuntimeError(f"Tool error: {result['error']}")
                
        raise RuntimeError("Failed to call WebSocket tool")
        
    async def discovery_loop(self):
        """Continuous discovery loop"""
        while self.running:
            try:
                # Rediscover tools for running servers
                for name in self.server_processes:
                    if name in self.server_processes:
                        process = self.server_processes[name]
                        if process.poll() is None:  # Still running
                            await self.discover_server_tools(name)
                            
                await asyncio.sleep(60)  # Rediscover every minute
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)
                
    async def health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while self.running:
            try:
                for name, config in self.servers.items():
                    await self.check_server_health(name, config)
                    
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
                
    async def check_server_health(self, name: str, config: MCPServerConfig):
        """Check health of a specific server"""
        try:
            if name in self.server_processes:
                process = self.server_processes[name]
                
                if process.poll() is None:  # Still running
                    self.server_status[name].status = "running"
                    self.server_status[name].last_health_check = time.time()
                else:
                    # Process died, restart if auto_start enabled
                    logger.warning(f"Server {name} died, restarting...")
                    del self.server_processes[name]
                    
                    if config.auto_start:
                        await self.start_server(name)
                    else:
                        self.server_status[name].status = "stopped"
            else:
                # Server not running but should be
                if config.auto_start and name in self.servers:
                    logger.info(f"Starting auto-restart for {name}")
                    await self.start_server(name)
                    
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            
    def get_server_status(self, name: Optional[str] = None) -> Union[MCPServerStatus, Dict[str, MCPServerStatus]]:
        """Get server status"""
        if name:
            return self.server_status.get(name)
        return self.server_status
        
    def get_discovered_tools(self, server: Optional[str] = None) -> Dict[str, MCPTool]:
        """Get discovered tools"""
        if server:
            return {
                name: tool 
                for name, tool in self.discovered_tools.items() 
                if tool.server == server
            }
        return self.discovered_tools
        
    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add a new MCP server configuration"""
        self.servers[config.name] = config
        await self.save_config()
        
        if config.auto_start:
            return await self.start_server(config.name)
        return True
        
    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration"""
        if name in self.servers:
            # Stop server if running
            await self.stop_server(name)
            
            # Remove configuration
            del self.servers[name]
            
            # Remove discovered tools
            tools_to_remove = [
                tool_name for tool_name, tool in self.discovered_tools.items()
                if tool.server == name
            ]
            for tool_name in tools_to_remove:
                del self.discovered_tools[tool_name]
                
            # Remove status
            if name in self.server_status:
                del self.server_status[name]
                
            await self.save_config()
            return True
            
        return False

# Global service instance
mcp_discovery_service = MCPAutoDiscoveryService()