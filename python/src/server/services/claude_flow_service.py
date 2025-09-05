"""
Claude Flow Integration Service for Archon

This service provides native Claude Flow orchestration capabilities within Archon's
backend architecture, enabling seamless SPARC methodology and swarm coordination
integrated with Archon's task management and knowledge systems.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.logfire_config import get_logger

logger = get_logger(__name__)


class ClaudeFlowService:
    """Native Claude Flow orchestration service for Archon."""
    
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent.parent.parent
        self.claude_flow_config = self.base_path / ".claude-flow"
        self.memory_db = self.base_path / ".swarm" / "memory.db"
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.claude_flow_config / "config", exist_ok=True)
        os.makedirs(self.base_path / ".swarm", exist_ok=True)
        os.makedirs(self.base_path / "memory" / "agents", exist_ok=True)
        
    async def initialize_swarm(
        self, 
        topology: str = "adaptive",
        max_agents: int = 10,
        archon_integration: bool = True
    ) -> Dict[str, Any]:
        """Initialize Claude Flow swarm with Archon integration."""
        try:
            logger.info(f"Initializing Claude Flow swarm with topology: {topology}")
            
            # Create swarm configuration
            swarm_config = {
                "topology": topology,
                "maxAgents": max_agents,
                "archonIntegration": archon_integration,
                "timestamp": datetime.now().isoformat(),
                "coordination": {
                    "type": "mesh" if topology == "adaptive" else topology,
                    "fallback": "hierarchical"
                },
                "critical_rules": {
                    "archon_first": {
                        "enabled": True,
                        "priority": "absolute",
                        "description": "ALWAYS check Archon MCP server for task management before TodoWrite"
                    }
                }
            }
            
            # Save configuration
            config_path = self.claude_flow_config / "config" / "swarm_session.json"
            with open(config_path, 'w') as f:
                json.dump(swarm_config, f, indent=2)
            
            # Initialize swarm via Claude Flow CLI
            result = await self._run_claude_flow_command([
                "swarm", "init", 
                "--topology", topology,
                "--max-agents", str(max_agents),
                "--archon-integration"
            ])
            
            return {
                "status": "initialized",
                "config": swarm_config,
                "result": result,
                "session_id": f"archon-swarm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude Flow swarm: {e}")
            return {"status": "error", "error": str(e)}
    
    async def spawn_agents(
        self,
        objective: str,
        agents: List[str],
        strategy: str = "development",
        archon_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Spawn agents for a specific objective with Archon task integration."""
        try:
            logger.info(f"Spawning agents for objective: {objective}")
            
            # Prepare agent spawning configuration
            spawn_config = {
                "objective": objective,
                "agents": agents,
                "strategy": strategy,
                "archon_task_id": archon_task_id,
                "timestamp": datetime.now().isoformat(),
                "archon_integration": {
                    "task_management": True,
                    "rag_queries": True,
                    "status_updates": True
                }
            }
            
            # Build Claude Flow command
            cmd = [
                "swarm", objective,
                "--strategy", strategy,
                "--max-agents", str(len(agents)),
                "--archon-integration"
            ]
            
            if archon_task_id:
                cmd.extend(["--task-id", archon_task_id])
                
            # Execute swarm
            result = await self._run_claude_flow_command(cmd)
            
            return {
                "status": "spawned",
                "config": spawn_config,
                "result": result,
                "agents": agents
            }
            
        except Exception as e:
            logger.error(f"Failed to spawn agents: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_sparc_workflow(
        self,
        task: str,
        mode: str = "tdd",
        archon_project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute SPARC methodology workflow integrated with Archon."""
        try:
            logger.info(f"Executing SPARC {mode} workflow for task: {task}")
            
            # SPARC configuration with Archon integration
            sparc_config = {
                "task": task,
                "mode": mode,
                "archon_project_id": archon_project_id,
                "timestamp": datetime.now().isoformat(),
                "phases": ["specification", "pseudocode", "architecture", "refinement", "completion"],
                "archon_integration": {
                    "task_updates": True,
                    "rag_research": True,
                    "code_examples": True
                }
            }
            
            # Build SPARC command
            cmd = ["sparc", mode, task]
            
            if archon_project_id:
                cmd.extend(["--project-id", archon_project_id])
                
            # Execute SPARC workflow
            result = await self._run_claude_flow_command(cmd)
            
            return {
                "status": "executed",
                "config": sparc_config,
                "result": result,
                "workflow": "sparc"
            }
            
        except Exception as e:
            logger.error(f"Failed to execute SPARC workflow: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status and metrics."""
        try:
            result = await self._run_claude_flow_command(["status"])
            
            # Parse swarm status
            status_info = {
                "timestamp": datetime.now().isoformat(),
                "raw_status": result,
                "memory_available": self.memory_db.exists(),
                "config_present": (self.claude_flow_config / "config").exists()
            }
            
            return {"status": "success", "info": status_info}
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        try:
            # Read metrics from Claude Flow
            metrics_path = self.claude_flow_config / "metrics"
            metrics = {}
            
            if metrics_path.exists():
                for metrics_file in metrics_path.glob("*.json"):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics[metrics_file.stem] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not read metrics file {metrics_file}: {e}")
            
            return {"status": "success", "metrics": metrics}
            
        except Exception as e:
            logger.error(f"Failed to get agent metrics: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_hooks(self, hook_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Claude Flow hooks with context."""
        try:
            hooks_script = self.claude_flow_config / "hooks.js"
            
            if not hooks_script.exists():
                return {"status": "error", "error": "Hooks script not found"}
            
            # Execute hook with context
            cmd = ["node", str(hooks_script), hook_name]
            
            # Add context as command line arguments
            for key, value in context.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            result = await self._run_command(cmd)
            
            return {
                "status": "executed",
                "hook": hook_name,
                "result": result,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Failed to execute hook {hook_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def memory_operations(
        self, 
        operation: str, 
        key: Optional[str] = None, 
        value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform memory operations in Claude Flow."""
        try:
            cmd = ["memory", operation]
            
            if key:
                cmd.extend(["--key", key])
            if value:
                cmd.extend(["--value", json.dumps(value)])
            
            result = await self._run_claude_flow_command(cmd)
            
            return {
                "status": "success",
                "operation": operation,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Memory operation {operation} failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def neural_training(
        self,
        patterns: List[Dict[str, Any]],
        model_type: str = "performance"
    ) -> Dict[str, Any]:
        """Execute neural pattern training."""
        try:
            # Create temporary file with patterns
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(patterns, f, indent=2)
                patterns_file = f.name
            
            try:
                cmd = ["training", "neural-train", "--patterns-file", patterns_file, "--model", model_type]
                result = await self._run_claude_flow_command(cmd)
                
                return {
                    "status": "success",
                    "model_type": model_type,
                    "patterns_count": len(patterns),
                    "result": result
                }
            finally:
                os.unlink(patterns_file)
                
        except Exception as e:
            logger.error(f"Neural training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _run_claude_flow_command(self, args: List[str]) -> str:
        """Run Claude Flow command via CLI."""
        cmd = ["npx", "claude-flow@alpha"] + args
        return await self._run_command(cmd)
    
    async def _run_command(self, cmd: List[str]) -> str:
        """Run a command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else f"Command failed with code {process.returncode}"
                raise Exception(f"Command failed: {error_msg}")
            
            return stdout.decode()
            
        except Exception as e:
            logger.error(f"Command execution failed: {cmd}, error: {e}")
            raise


# Global service instance
claude_flow_service = ClaudeFlowService()