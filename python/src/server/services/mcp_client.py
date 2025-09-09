"""
MCP Client for communicating with the Archon MCP Server

This module provides a client for the main API server to communicate
with the MCP server using the proper MCP protocol over Server-Sent Events (SSE).
"""

import asyncio
import json
import uuid
import logging
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime

from src.server.config.logfire_config import api_logger

logger = logging.getLogger(__name__)


class ArchonMCPClient:
    """
    Client for communicating with the Archon MCP Server using MCP protocol.
    Uses Server-Sent Events (SSE) transport as configured.
    """

    def __init__(self, host: str = "localhost", port: int = 8051):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.mcp_url = f"{self.base_url}/mcp"
        self.timeout = httpx.Timeout(30.0)
        self._session_id = None
        self._initialized = False
        
    def _generate_request_id(self) -> str:
        """Generate a unique request ID"""
        return str(uuid.uuid4())
    
    def _create_mcp_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an MCP JSON-RPC 2.0 request"""
        request = {
            "jsonrpc": "2.0",
            "id": self._generate_request_id(),
            "method": method
        }
        if params:
            request["params"] = params
        return request
    
    async def _send_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send an MCP request via SSE and get response"""
        try:
            headers = {
                "Accept": "application/json, text/event-stream",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json"
            }
            
            # Add session ID if we have one
            if self._session_id:
                headers["mcp-session-id"] = self._session_id
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Send POST request with proper headers
                async with client.stream("POST", self.mcp_url, json=request, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Capture session ID from response if this is initialization
                    if request.get("method") == "initialize" and "mcp-session-id" in response.headers:
                        self._session_id = response.headers["mcp-session-id"]
                        api_logger.info(f"MCP session established: {self._session_id}")
                    
                    # Read SSE response
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line.startswith("data: "):
                            try:
                                data = line[6:]  # Remove "data: " prefix
                                if data == "[DONE]":
                                    break
                                    
                                response_data = json.loads(data)
                                
                                # Check if this is the response to our request
                                if response_data.get("id") == request["id"]:
                                    return response_data
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    # If we didn't get a matching response
                    return {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {
                            "code": -32603,
                            "message": "No matching response received"
                        }
                    }
                    
        except httpx.TimeoutException:
            api_logger.error(f"Timeout communicating with MCP server at {self.mcp_url}")
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {
                    "code": -32603,
                    "message": "Request timeout"
                }
            }
        except Exception as e:
            api_logger.error(f"Error communicating with MCP server: {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {
                    "code": -32603,
                    "message": f"Communication error: {str(e)}"
                }
            }
    
    async def initialize(self) -> bool:
        """Initialize connection with MCP server"""
        if self._initialized:
            return True
            
        try:
            # Send initialize request
            init_request = self._create_mcp_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "archon-api-server",
                    "version": "1.0.0"
                }
            })
            
            response = await self._send_mcp_request(init_request)
            
            if "error" in response:
                api_logger.error(f"MCP initialize failed: {response['error']}")
                return False
                
            # Send initialized notification with session ID
            initialized_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            headers = {"Content-Type": "application/json"}
            if self._session_id:
                headers["mcp-session-id"] = self._session_id
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    await client.post(self.mcp_url, json=initialized_request, headers=headers)
            except Exception as e:
                api_logger.warning(f"Failed to send initialized notification: {e}")
            
            self._initialized = True
            api_logger.info("MCP client initialized successfully")
            return True
            
        except Exception as e:
            api_logger.error(f"MCP client initialization failed: {str(e)}")
            return False
    
    async def list_tools(self) -> Dict[str, Any]:
        """Get list of available tools from MCP server"""
        try:
            # Use the get_available_tools tool instead of the built-in method
            result = await self.call_tool("get_available_tools", {})
            
            if not result["success"]:
                api_logger.error(f"MCP get_available_tools failed: {result['error']}")
                return {
                    "tools": [],
                    "count": 0,
                    "error": result["error"]
                }
            
            # Parse the JSON response from the tool
            tool_data = json.loads(result["result"]["content"][0]["text"])
            
            if not tool_data.get("success", False):
                return {
                    "tools": [],
                    "count": 0,
                    "error": tool_data.get("error", "Unknown error")
                }
            
            tools = tool_data.get("tools", [])
            
            # Transform tools to our expected format
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}).get("properties", {}),
                    "required": tool.get("inputSchema", {}).get("required", [])
                }
                formatted_tools.append(formatted_tool)
            
            api_logger.info(f"Retrieved {len(formatted_tools)} tools from MCP server")
            
            return {
                "tools": formatted_tools,
                "count": len(formatted_tools),
                "error": None
            }
            
        except Exception as e:
            api_logger.error(f"Error listing MCP tools: {str(e)}")
            return {
                "tools": [],
                "count": 0,
                "error": str(e)
            }
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific MCP tool"""
        try:
            request = self._create_mcp_request("tools/call", {
                "name": name,
                "arguments": arguments
            })
            
            response = await self._send_mcp_request(request)
            
            if "error" in response:
                return {
                    "success": False,
                    "error": response["error"]["message"],
                    "result": None
                }
            
            result = response.get("result", {})
            
            return {
                "success": True,
                "error": None,
                "result": result
            }
            
        except Exception as e:
            api_logger.error(f"Error calling MCP tool {name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of MCP server"""
        try:
            # Try to call the health_check tool
            result = await self.call_tool("health_check", {})
            
            if result["success"]:
                health_data = json.loads(result["result"]["content"][0]["text"])
                return {
                    "success": True,
                    "healthy": health_data.get("success", False),
                    "status": health_data.get("status", "unknown"),
                    "details": health_data
                }
            else:
                return {
                    "success": False,
                    "healthy": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "healthy": False,
                "error": str(e)
            }


# Global client instance
_mcp_client = None
_client_initialized = False


async def get_mcp_client() -> ArchonMCPClient:
    """Get or create the global MCP client and ensure it's initialized"""
    global _mcp_client, _client_initialized
    
    if _mcp_client is None:
        _mcp_client = ArchonMCPClient()
    
    # Initialize if not done yet
    if not _client_initialized:
        try:
            success = await _mcp_client.initialize()
            _client_initialized = success
            if not success:
                api_logger.warning("MCP client initialization failed")
        except Exception as e:
            api_logger.error(f"MCP client initialization error: {str(e)}")
            _client_initialized = False
    
    return _mcp_client