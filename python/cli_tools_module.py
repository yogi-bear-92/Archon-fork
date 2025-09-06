"""
CLI Tools Module for Archon MCP Server

Integrates CLI tools from removed servers (claude-flow, flow-nexus, etc.)
as native MCP tools through HTTP-based discovery and execution.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

try:
    from mcp.types import Tool
except ImportError:
    # Fallback for testing
    class Tool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

logger = logging.getLogger(__name__)

# CLI tool configurations
CLI_TOOLS_CONFIG = {
    "claude-flow": {
        "command": "npx",
        "args": ["claude-flow@alpha"],
        "available_commands": {
            "swarm": {"description": "Multi-agent swarm coordination", "usage": "swarm <objective>"},
            "agent": {"description": "Agent management", "usage": "agent <action>"},
            "sparc": {"description": "SPARC development modes", "usage": "sparc <mode>"},
            "memory": {"description": "Persistent memory operations", "usage": "memory <operation>"},
            "coordination": {"description": "Swarm & agent orchestration", "usage": "coordination <action>"},
            "analysis": {"description": "Performance & token usage analytics", "usage": "analysis <type>"},
            "monitoring": {"description": "Real-time system monitoring", "usage": "monitoring <action>"}
        }
    },
    "flow-nexus": {
        "command": "npx", 
        "args": ["flow-nexus@latest"],
        "available_commands": {
            "deploy": {"description": "Deploy AI agents and workflows", "usage": "deploy <workflow>"},
            "orchestrate": {"description": "Orchestrate multi-agent workflows", "usage": "orchestrate <agents>"}
        }
    }
}

async def check_cli_tool_availability(tool_name: str, config: Dict[str, Any]) -> bool:
    """Check if a CLI tool is available"""
    try:
        cmd = config["command"]
        args = config["args"] + ["--version"]
        
        process = await asyncio.create_subprocess_exec(
            cmd, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
            return process.returncode == 0 or "version" in stdout.decode().lower()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False
            
    except Exception as e:
        logger.warning(f"CLI tool {tool_name} availability check failed: {e}")
        return False

async def execute_cli_command(tool_name: str, command: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a CLI command and return results"""
    if tool_name not in CLI_TOOLS_CONFIG:
        return {"success": False, "error": f"Unknown CLI tool: {tool_name}"}
    
    config = CLI_TOOLS_CONFIG[tool_name]
    
    # Build command arguments
    cmd_args = config["args"] + [command]
    
    # Add arguments from input
    if args:
        for key, value in args.items():
            if isinstance(value, bool) and value:
                cmd_args.append(f"--{key}")
            elif value is not None:
                cmd_args.extend([f"--{key}", str(value)])
    
    try:
        process = await asyncio.create_subprocess_exec(
            config["command"], *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "return_code": process.returncode
            }
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"success": False, "error": "Command timeout"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def register_cli_tools(mcp):
    """Register CLI tools as MCP tools"""
    
    logger.info("🔧 Registering CLI tools...")
    
    # Generate tools for each CLI command
    for tool_name, config in CLI_TOOLS_CONFIG.items():
        for cmd_name, cmd_info in config["available_commands"].items():
            tool_id = f"cli_{tool_name.replace('-', '_')}_{cmd_name}"
            
            # Create MCP tool
            @mcp.tool(name=tool_id, description=f"CLI Tool: {cmd_info['description']} (from {tool_name})")
            async def cli_tool_handler(
                tool_name=tool_name, 
                command=cmd_name,
                help: bool = False,
                **kwargs
            ) -> str:
                """Execute CLI tool command"""
                
                # Add help flag if requested
                args = dict(kwargs)
                if help:
                    args["help"] = True
                
                result = await execute_cli_command(tool_name, command, args)
                return json.dumps(result, indent=2)
            
            logger.info(f"  ✓ Registered: {tool_id}")
    
    # Add CLI status tool
    @mcp.tool(name="cli_tools_status", description="Get status of all CLI tools")
    async def cli_status() -> str:
        """Get CLI tools availability status"""
        status = {}
        
        for tool_name, config in CLI_TOOLS_CONFIG.items():
            available = await check_cli_tool_availability(tool_name, config)
            status[tool_name] = {
                "available": available,
                "commands": list(config["available_commands"].keys())
            }
        
        return json.dumps({
            "cli_tools": status,
            "total_tools": len(CLI_TOOLS_CONFIG),
            "available_tools": sum(1 for s in status.values() if s["available"])
        }, indent=2)
    
    logger.info(f"✅ CLI tools module registered with {len(CLI_TOOLS_CONFIG)} tools")

if __name__ == "__main__":
    # Test CLI tools availability
    async def test():
        print("Testing CLI tools availability...")
        for tool_name, config in CLI_TOOLS_CONFIG.items():
            available = await check_cli_tool_availability(tool_name, config)
            print(f"{tool_name}: {'✅ Available' if available else '❌ Not available'}")
    
    asyncio.run(test())