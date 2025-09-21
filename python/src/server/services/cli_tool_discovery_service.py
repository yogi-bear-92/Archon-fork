"""
CLI Tool Discovery Service
Discovery and wrapping of CLI-based tools like Claude Flow, Ruv Swarm, etc.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CLIToolConfig:
    """Configuration for a CLI tool"""
    name: str
    command: str
    args: List[str]
    help_command: Optional[List[str]] = None
    available_commands: Optional[Dict[str, dict]] = None
    description: str = ""
    timeout: int = 30

@dataclass
class CLICommand:
    """Represents a CLI command"""
    name: str
    description: str
    usage: str
    options: Dict[str, Any]
    tool: str

@dataclass
class CLIToolStatus:
    """Status of a CLI tool"""
    name: str
    available: bool
    version: Optional[str] = None
    last_check: Optional[float] = None
    error_message: Optional[str] = None
    commands: List[CLICommand] = None

class CLIToolDiscoveryService:
    """Discovery and management service for CLI tools"""
    
    def __init__(self):
        self.tools: Dict[str, CLIToolConfig] = {}
        self.tool_status: Dict[str, CLIToolStatus] = {}
        self.discovered_commands: Dict[str, CLICommand] = {}
        self.running = False
        
        # Define CLI tools we want to wrap
        self.default_tools = {
            "claude-flow": CLIToolConfig(
                name="claude-flow",
                command="npx",
                args=["claude-flow@alpha"],
                help_command=["--help"],
                description="Enterprise AI agent orchestration platform",
                available_commands={
                    "swarm": {
                        "description": "Multi-agent swarm coordination",
                        "usage": "swarm <objective> [options]",
                        "options": {
                            "strategy": "Execution strategy (research, development, analysis)",
                            "mode": "Coordination mode (centralized, distributed, mesh)",
                            "max_agents": "Maximum number of agents",
                            "parallel": "Enable parallel execution"
                        }
                    },
                    "agent": {
                        "description": "Agent management",
                        "usage": "agent <action>",
                        "options": {
                            "action": "spawn, list, terminate"
                        }
                    },
                    "sparc": {
                        "description": "SPARC development modes",
                        "usage": "sparc <mode>",
                        "options": {
                            "mode": "Development mode (17 available)"
                        }
                    },
                    "memory": {
                        "description": "Persistent memory operations",
                        "usage": "memory <action>",
                        "options": {
                            "action": "store, retrieve, search"
                        }
                    },
                    "coordination": {
                        "description": "Swarm & agent orchestration",
                        "usage": "coordination <command>",
                        "options": {
                            "command": "swarm-init, agent-spawn, task-orchestrate"
                        }
                    },
                    "analysis": {
                        "description": "Performance & token usage analytics",
                        "usage": "analysis <command>",
                        "options": {
                            "command": "token-usage, performance-report, bottleneck-detect"
                        }
                    },
                    "monitoring": {
                        "description": "Real-time system monitoring",
                        "usage": "monitoring <command>",
                        "options": {
                            "command": "agent-metrics, real-time-view, swarm-monitor"
                        }
                    }
                }
            ),
            "ruv-swarm": CLIToolConfig(
                name="ruv-swarm",
                command="npx",
                args=["ruv-swarm"],
                help_command=["--help"],
                description="Advanced swarm intelligence platform",
                available_commands={
                    "init": {
                        "description": "Initialize swarm environment",
                        "usage": "init [options]",
                        "options": {}
                    },
                    "spawn": {
                        "description": "Spawn swarm agents", 
                        "usage": "spawn <agents> [options]",
                        "options": {
                            "agents": "Agent types to spawn"
                        }
                    },
                    "coordinate": {
                        "description": "Coordinate swarm activities",
                        "usage": "coordinate <objective>",
                        "options": {}
                    }
                }
            ),
        }
        
    async def start(self):
        """Start the CLI tool discovery service"""
        logger.info("Starting CLI Tool Discovery Service")
        self.running = True
        
        # Load tools
        self.tools = self.default_tools.copy()
        
        # Discover available commands
        await self.discover_all_tools()
        
        logger.info("CLI Tool Discovery Service started")
        
    async def stop(self):
        """Stop the CLI tool discovery service"""
        logger.info("Stopping CLI Tool Discovery Service")
        self.running = False
        
    async def discover_all_tools(self):
        """Discover all configured CLI tools"""
        for tool_name, tool_config in self.tools.items():
            await self.discover_tool(tool_name, tool_config)
            
    async def discover_tool(self, tool_name: str, tool_config: CLIToolConfig):
        """Discover commands for a specific CLI tool"""
        try:
            logger.info(f"Discovering CLI tool: {tool_name}")
            
            # Check if tool is available
            help_command = tool_config.args + (tool_config.help_command or ["--help"])
            
            process = await asyncio.create_subprocess_exec(
                tool_config.command,
                *help_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=tool_config.timeout
                )
                
                if process.returncode == 0:
                    # Tool is available
                    help_output = stdout.decode('utf-8')
                    version = self.extract_version(help_output)
                    
                    # Create commands from configuration
                    commands = []
                    if tool_config.available_commands:
                        for cmd_name, cmd_info in tool_config.available_commands.items():
                            command = CLICommand(
                                name=f"{tool_name}__{cmd_name}",
                                description=cmd_info["description"],
                                usage=cmd_info["usage"],
                                options=cmd_info["options"],
                                tool=tool_name
                            )
                            commands.append(command)
                            self.discovered_commands[command.name] = command
                    
                    self.tool_status[tool_name] = CLIToolStatus(
                        name=tool_name,
                        available=True,
                        version=version,
                        last_check=time.time(),
                        commands=commands
                    )
                    
                    logger.info(f"✅ {tool_name} available (v{version}) with {len(commands)} commands")
                    
                else:
                    # Tool not available
                    error_output = stderr.decode('utf-8') if stderr else "Unknown error"
                    self.tool_status[tool_name] = CLIToolStatus(
                        name=tool_name,
                        available=False,
                        last_check=time.time(),
                        error_message=error_output
                    )
                    logger.warning(f"❌ {tool_name} not available: {error_output}")
                    
            except asyncio.TimeoutError:
                self.tool_status[tool_name] = CLIToolStatus(
                    name=tool_name,
                    available=False,
                    last_check=time.time(),
                    error_message="Timeout waiting for help command"
                )
                logger.warning(f"⏱️ {tool_name} timed out")
                
        except Exception as e:
            self.tool_status[tool_name] = CLIToolStatus(
                name=tool_name,
                available=False,
                last_check=time.time(),
                error_message=str(e)
            )
            logger.error(f"❌ Failed to discover {tool_name}: {e}")
            
    def extract_version(self, help_output: str) -> Optional[str]:
        """Extract version from help output"""
        lines = help_output.split('\n')
        for line in lines:
            if 'version' in line.lower() or 'v' in line:
                # Look for version patterns
                import re
                version_match = re.search(r'v?(\d+\.\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)', line)
                if version_match:
                    return version_match.group(1)
        return "unknown"
        
    async def execute_command(self, command_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a discovered CLI command"""
        if command_name not in self.discovered_commands:
            raise ValueError(f"Command {command_name} not found")
            
        command = self.discovered_commands[command_name]
        tool_config = self.tools[command.tool]
        
        # Build command line
        cmd_parts = [tool_config.command] + tool_config.args
        
        # Extract the actual command name (remove tool prefix)
        actual_cmd_name = command_name.split("__", 1)[1] if "__" in command_name else command_name
        cmd_parts.append(actual_cmd_name)
        
        # Add arguments
        for key, value in arguments.items():
            if key.startswith("--"):
                cmd_parts.append(key)
                if value not in [True, "true", ""]:
                    cmd_parts.append(str(value))
            elif key.startswith("-"):
                cmd_parts.append(key)
                if value not in [True, "true", ""]:
                    cmd_parts.append(str(value))
            else:
                # Positional argument or option without dashes
                if key in command.options:
                    cmd_parts.extend([f"--{key}", str(value)])
                else:
                    cmd_parts.append(str(value))
        
        try:
            logger.info(f"Executing: {' '.join(cmd_parts)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60  # Longer timeout for actual execution
            )
            
            result = {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else "",
                "command": ' '.join(cmd_parts)
            }
            
            if process.returncode == 0:
                logger.info(f"✅ Command {command_name} executed successfully")
            else:
                logger.warning(f"❌ Command {command_name} failed with code {process.returncode}")
                
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Command {command_name} timed out")
            return {
                "success": False,
                "error": "Command execution timed out",
                "command": ' '.join(cmd_parts)
            }
        except Exception as e:
            logger.error(f"❌ Failed to execute {command_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": ' '.join(cmd_parts)
            }
            
    def get_tool_status(self, tool_name: Optional[str] = None) -> Union[CLIToolStatus, Dict[str, CLIToolStatus]]:
        """Get CLI tool status"""
        if tool_name:
            return self.tool_status.get(tool_name)
        return self.tool_status
        
    def get_discovered_commands(self, tool_name: Optional[str] = None) -> Dict[str, CLICommand]:
        """Get discovered commands"""
        if tool_name:
            return {
                name: cmd for name, cmd in self.discovered_commands.items()
                if cmd.tool == tool_name
            }
        return self.discovered_commands
        
    async def refresh_discovery(self):
        """Refresh command discovery for all tools"""
        await self.discover_all_tools()

# Global service instance
cli_discovery_service = CLIToolDiscoveryService()