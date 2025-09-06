"""
Claude Flow Coordination Hooks for seamless swarm orchestration.

This module provides integration hooks for Claude Flow's swarm coordination system,
enabling the claude flow expert agent to leverage Claude Flow's advanced orchestration capabilities.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SwarmTopology:
    """Enum-like class for swarm topology types."""
    
    MESH = "mesh"
    HIERARCHICAL = "hierarchical" 
    RING = "ring"
    STAR = "star"
    ADAPTIVE = "adaptive"


class CoordinationStrategy:
    """Enum-like class for coordination strategies."""
    
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    BALANCED = "balanced"


class ClaudeFlowCoordinator:
    """
    Claude Flow coordination hooks for swarm orchestration.
    
    Provides seamless integration between the Claude Flow Expert Agent and Claude Flow's
    advanced swarm coordination capabilities including topology management,
    agent spawning, and performance monitoring.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize Claude Flow coordinator.
        
        Args:
            base_path: Base path for Claude Flow operations
        """
        self.base_path = base_path or Path.cwd()
        self.claude_flow_config = self.base_path / ".claude-flow"
        self.swarm_sessions: Dict[str, Dict[str, Any]] = {}
        self.current_session_id: Optional[str] = None
        self.metrics_enabled = True
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info("Claude Flow coordinator initialized")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.claude_flow_config / "config", exist_ok=True)
        os.makedirs(self.claude_flow_config / "metrics", exist_ok=True)
        os.makedirs(self.base_path / "memory" / "swarm", exist_ok=True)
    
    async def initialize_swarm(
        self,
        topology: str = SwarmTopology.ADAPTIVE,
        max_agents: int = 8,
        strategy: str = "balanced",
        archon_integration: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize a new Claude Flow swarm.
        
        Args:
            topology: Swarm topology type
            max_agents: Maximum number of agents
            strategy: Distribution strategy
            archon_integration: Enable Archon integration
            
        Returns:
            Swarm initialization result
        """
        try:
            session_id = f"archon-swarm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            logger.info(f"Initializing Claude Flow swarm: {session_id}")
            
            # Build initialization command
            cmd = [
                "npx", "claude-flow@alpha",
                "swarm", "init",
                "--topology", topology,
                "--max-agents", str(max_agents),
                "--strategy", strategy,
                "--session-id", session_id
            ]
            
            if archon_integration:
                cmd.append("--archon-integration")
            
            # Execute swarm initialization
            result = await self._run_claude_flow_command(cmd)
            
            # Store session information
            session_info = {
                "session_id": session_id,
                "topology": topology,
                "max_agents": max_agents,
                "strategy": strategy,
                "archon_integration": archon_integration,
                "created_at": datetime.now().isoformat(),
                "status": "initialized",
                "agents": []
            }
            
            self.swarm_sessions[session_id] = session_info
            self.current_session_id = session_id
            
            # Save configuration
            await self._save_session_config(session_info)
            
            return {
                "status": "success",
                "session_id": session_id,
                "topology": topology,
                "max_agents": max_agents,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize swarm: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def spawn_agents(
        self,
        objective: str,
        agents: List[str],
        strategy: str = CoordinationStrategy.ADAPTIVE,
        archon_task_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Spawn agents in the Claude Flow swarm.
        
        Args:
            objective: The task objective
            agents: List of agent types to spawn
            strategy: Coordination strategy
            archon_task_id: Optional Archon task ID
            session_id: Optional session ID (uses current if not provided)
            
        Returns:
            Agent spawning result
        """
        try:
            target_session = session_id or self.current_session_id
            
            if not target_session:
                logger.error("No active swarm session")
                return {"status": "error", "error": "No active swarm session"}
            
            logger.info(f"Spawning {len(agents)} agents for objective: {objective}")
            
            # Build spawn command
            cmd = [
                "npx", "claude-flow@alpha",
                "swarm", objective,
                "--strategy", strategy,
                "--max-agents", str(len(agents)),
                "--session-id", target_session
            ]
            
            # Add agent types
            if agents:
                cmd.extend(["--agent-types", ",".join(agents)])
            
            # Add Archon integration
            if archon_task_id:
                cmd.extend(["--archon-task-id", archon_task_id])
            
            # Execute agent spawning
            result = await self._run_claude_flow_command(cmd)
            
            # Update session information
            if target_session in self.swarm_sessions:
                session_info = self.swarm_sessions[target_session]
                session_info["agents"].extend(agents)
                session_info["last_objective"] = objective
                session_info["status"] = "active"
                
                await self._save_session_config(session_info)
            
            return {
                "status": "success",
                "session_id": target_session,
                "agents_spawned": agents,
                "objective": objective,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to spawn agents: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def coordinate_multi_agent(
        self,
        objective: str,
        agent_types: List[str],
        max_agents: int = 5,
        strategy: str = CoordinationStrategy.ADAPTIVE,
        archon_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for a complex task.
        
        Args:
            objective: The task objective
            agent_types: List of agent types to coordinate
            max_agents: Maximum agents to use
            strategy: Coordination strategy
            archon_task_id: Optional Archon task ID
            
        Returns:
            Coordination result
        """
        try:
            coordination_start = datetime.now()
            
            # Initialize swarm if needed
            if not self.current_session_id:
                init_result = await self.initialize_swarm(
                    topology=SwarmTopology.ADAPTIVE,
                    max_agents=max_agents
                )
                if init_result.get("status") != "success":
                    return init_result
            
            # Spawn agents
            spawn_result = await self.spawn_agents(
                objective=objective,
                agents=agent_types[:max_agents],
                strategy=strategy,
                archon_task_id=archon_task_id
            )
            
            if spawn_result.get("status") != "success":
                return spawn_result
            
            # Monitor coordination progress
            monitoring_result = await self._monitor_coordination_progress(
                session_id=self.current_session_id,
                timeout=300  # 5 minute timeout
            )
            
            coordination_time = (datetime.now() - coordination_start).total_seconds()
            
            return {
                "status": "success",
                "coordination_time": coordination_time,
                "session_id": self.current_session_id,
                "agents_coordinated": agent_types[:max_agents],
                "objective": objective,
                "spawn_result": spawn_result,
                "monitoring_result": monitoring_result,
                "metrics": await self._get_coordination_metrics()
            }
            
        except Exception as e:
            logger.error(f"Multi-agent coordination failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def monitor_coordination(
        self,
        session_id: Optional[str] = None,
        duration: int = 30
    ) -> Dict[str, Any]:
        """
        Monitor swarm coordination status and metrics.
        
        Args:
            session_id: Session ID to monitor (uses current if not provided)
            duration: Monitoring duration in seconds
            
        Returns:
            Monitoring results
        """
        try:
            target_session = session_id or self.current_session_id
            
            if not target_session:
                return {"status": "error", "error": "No active session"}
            
            # Build monitoring command
            cmd = [
                "npx", "claude-flow@alpha",
                "monitor",
                "--session-id", target_session,
                "--duration", str(duration)
            ]
            
            # Execute monitoring
            result = await self._run_claude_flow_command(cmd)
            
            # Get additional metrics
            metrics = await self._get_coordination_metrics(target_session)
            
            return {
                "status": "success",
                "session_id": target_session,
                "monitoring_result": result,
                "metrics": metrics,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Coordination monitoring failed: {e}")
            return {
                "status": "error", 
                "error": str(e)
            }
    
    async def execute_hooks(
        self,
        hook_type: str,
        context: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Claude Flow hooks with context.
        
        Args:
            hook_type: Type of hook to execute (pre-task, post-task, etc.)
            context: Hook execution context
            session_id: Optional session ID
            
        Returns:
            Hook execution result
        """
        try:
            target_session = session_id or self.current_session_id
            
            # Build hooks command
            cmd = [
                "npx", "claude-flow@alpha",
                "hooks", hook_type
            ]
            
            # Add context parameters
            for key, value in context.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            if target_session:
                cmd.extend(["--session-id", target_session])
            
            # Execute hooks
            result = await self._run_claude_flow_command(cmd)
            
            return {
                "status": "success",
                "hook_type": hook_type,
                "context": context,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Hook execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def memory_operations(
        self,
        operation: str,
        key: Optional[str] = None,
        value: Optional[Any] = None,
        namespace: str = "swarm"
    ) -> Dict[str, Any]:
        """
        Perform memory operations in Claude Flow.
        
        Args:
            operation: Memory operation (store, retrieve, list, search)
            key: Memory key
            value: Value to store
            namespace: Memory namespace
            
        Returns:
            Memory operation result
        """
        try:
            # Build memory command
            cmd = [
                "npx", "claude-flow@alpha", 
                "memory", operation,
                "--namespace", namespace
            ]
            
            if key:
                cmd.extend(["--key", key])
            if value is not None:
                cmd.extend(["--value", json.dumps(value)])
            
            # Execute memory operation
            result = await self._run_claude_flow_command(cmd)
            
            return {
                "status": "success",
                "operation": operation,
                "key": key,
                "namespace": namespace,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Memory operation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_swarm_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current swarm status.
        
        Args:
            session_id: Session ID to check (uses current if not provided)
            
        Returns:
            Swarm status information
        """
        try:
            target_session = session_id or self.current_session_id
            
            # Build status command
            cmd = ["npx", "claude-flow@alpha", "status"]
            
            if target_session:
                cmd.extend(["--session-id", target_session])
            
            # Execute status check
            result = await self._run_claude_flow_command(cmd)
            
            # Handle NPX unavailable gracefully
            if isinstance(result, str) and "npx_unavailable" in result:
                logger.info("Claude Flow unavailable (NPX not found)")
                return {
                    "status": "npx_unavailable",
                    "session_id": target_session,
                    "session_info": self.swarm_sessions.get(target_session, {}),
                    "message": "Claude Flow commands unavailable - NPX not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get session info
            session_info = self.swarm_sessions.get(target_session, {})
            
            return {
                "status": "success",
                "session_id": target_session,
                "session_info": session_info,
                "claude_flow_status": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def terminate_swarm(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Terminate a swarm session.
        
        Args:
            session_id: Session ID to terminate (uses current if not provided)
            
        Returns:
            Termination result
        """
        try:
            target_session = session_id or self.current_session_id
            
            if not target_session:
                return {"status": "error", "error": "No session to terminate"}
            
            # Build termination command
            cmd = [
                "npx", "claude-flow@alpha",
                "swarm", "terminate",
                "--session-id", target_session
            ]
            
            # Execute termination
            result = await self._run_claude_flow_command(cmd)
            
            # Update session info
            if target_session in self.swarm_sessions:
                self.swarm_sessions[target_session]["status"] = "terminated"
                self.swarm_sessions[target_session]["terminated_at"] = datetime.now().isoformat()
            
            # Clear current session if it was terminated
            if target_session == self.current_session_id:
                self.current_session_id = None
            
            return {
                "status": "success",
                "session_id": target_session,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to terminate swarm: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Helper methods
    
    async def _run_claude_flow_command(self, cmd: List[str]) -> str:
        """Run a Claude Flow command asynchronously."""
        try:
            logger.debug(f"Executing Claude Flow command: {' '.join(cmd)}")
            
            # Check if npx command is available and fix path issues
            if cmd[0] == "npx":
                npx_path = None
                # Try common npx locations
                for path in ["/opt/homebrew/bin/npx", "/usr/local/bin/npx", "/usr/bin/npx"]:
                    if os.path.exists(path):
                        npx_path = path
                        break
                
                if npx_path:
                    cmd[0] = npx_path
                else:
                    # NPX not found, return graceful error
                    logger.warning("NPX not found, claude-flow commands unavailable")
                    return '{"status": "npx_unavailable", "message": "NPX not found on system"}'
            
            # Set environment with proper PATH
            env = os.environ.copy()
            env['PATH'] = f"/opt/homebrew/bin:/usr/local/bin:/usr/bin:{env.get('PATH', '')}"
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path),
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else f"Command failed with code {process.returncode}"
                # Don't raise exception for NPX not found, return graceful error
                if "No such file or directory" in error_msg and "npx" in error_msg:
                    logger.warning(f"NPX command not available: {error_msg}")
                    return '{"status": "npx_unavailable", "message": "NPX command not available"}'
                raise Exception(f"Claude Flow command failed: {error_msg}")
            
            return stdout.decode().strip()
            
        except FileNotFoundError as e:
            # Handle NPX not found gracefully
            if "npx" in str(e):
                logger.warning(f"NPX not found: {e}")
                return '{"status": "npx_unavailable", "message": "NPX executable not found"}'
            logger.error(f"Claude Flow command execution failed: {cmd}, error: {e}")
            raise
        except Exception as e:
            logger.error(f"Claude Flow command execution failed: {cmd}, error: {e}")
            raise
    
    async def _save_session_config(self, session_info: Dict[str, Any]):
        """Save session configuration to disk."""
        try:
            config_file = self.claude_flow_config / "config" / f"{session_info['session_id']}.json"
            
            with open(config_file, 'w') as f:
                json.dump(session_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session config: {e}")
    
    async def _monitor_coordination_progress(
        self,
        session_id: str,
        timeout: int = 300,
        check_interval: int = 10
    ) -> Dict[str, Any]:
        """Monitor coordination progress with timeout."""
        try:
            start_time = datetime.now()
            progress_updates = []
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                # Check swarm status
                status_result = await self.get_swarm_status(session_id)
                
                if status_result.get("status") == "success":
                    progress_updates.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": status_result
                    })
                
                # Wait before next check
                await asyncio.sleep(check_interval)
            
            return {
                "status": "completed",
                "monitoring_duration": (datetime.now() - start_time).total_seconds(),
                "progress_updates": progress_updates
            }
            
        except Exception as e:
            logger.error(f"Coordination monitoring failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_coordination_metrics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get coordination metrics for a session."""
        try:
            target_session = session_id or self.current_session_id
            metrics = {}
            
            if not target_session:
                return metrics
            
            # Read metrics from Claude Flow
            metrics_path = self.claude_flow_config / "metrics"
            
            if metrics_path.exists():
                # Read system metrics
                system_metrics_file = metrics_path / "system-metrics.json"
                if system_metrics_file.exists():
                    try:
                        with open(system_metrics_file, 'r') as f:
                            system_metrics = json.load(f)
                        
                        # Get latest metrics
                        if system_metrics and isinstance(system_metrics, list):
                            latest_metrics = system_metrics[-1] if system_metrics else {}
                            metrics["system"] = latest_metrics
                            
                    except Exception as e:
                        logger.warning(f"Could not read system metrics: {e}")
                
                # Read session-specific metrics
                session_metrics_file = metrics_path / f"{target_session}-metrics.json"
                if session_metrics_file.exists():
                    try:
                        with open(session_metrics_file, 'r') as f:
                            session_metrics = json.load(f)
                        metrics["session"] = session_metrics
                        
                    except Exception as e:
                        logger.warning(f"Could not read session metrics: {e}")
            
            # Add basic session info
            if target_session in self.swarm_sessions:
                session_info = self.swarm_sessions[target_session]
                metrics["session_info"] = {
                    "agents_count": len(session_info.get("agents", [])),
                    "topology": session_info.get("topology"),
                    "status": session_info.get("status"),
                    "created_at": session_info.get("created_at")
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get coordination metrics: {e}")
            return {}
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active swarm sessions."""
        active_sessions = []
        
        for session_id, session_info in self.swarm_sessions.items():
            if session_info.get("status") in ["initialized", "active"]:
                active_sessions.append(session_info)
        
        return active_sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.swarm_sessions.get(session_id)
    
    async def cleanup_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old session data."""
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            sessions_to_remove = []
            
            for session_id, session_info in self.swarm_sessions.items():
                created_at = datetime.fromisoformat(session_info["created_at"])
                age_hours = (current_time - created_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                del self.swarm_sessions[session_id]
                
                # Remove config file
                config_file = self.claude_flow_config / "config" / f"{session_id}.json"
                if config_file.exists():
                    config_file.unlink()
                
                cleaned_count += 1
            
            return {
                "status": "success",
                "cleaned_sessions": cleaned_count,
                "remaining_sessions": len(self.swarm_sessions)
            }
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }