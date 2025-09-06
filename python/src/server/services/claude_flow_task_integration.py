"""
Claude Flow Task Integration Service

Provides seamless integration between Claude Flow swarm coordination
and Archon's task management system.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .claude_flow_service import claude_flow_service
from .projects.task_service import TaskService
from .projects.project_service import ProjectService
from ..config.logfire_config import get_logger

logger = get_logger(__name__)


class ClaudeFlowTaskIntegration:
    """Integrates Claude Flow orchestration with Archon task management."""
    
    def __init__(self):
        self.task_service = TaskService()
        self.project_service = ProjectService()
        self.active_swarms: Dict[str, Dict[str, Any]] = {}
        
    async def create_swarm_for_task(
        self,
        task_id: str,
        swarm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Claude Flow swarm for a specific Archon task."""
        try:
            logger.info(f"Creating swarm for task: {task_id}")
            
            # Get task details
            task = await self.task_service.get_task_by_id(task_id)
            if not task:
                return {"status": "error", "error": f"Task {task_id} not found"}
            
            # Initialize swarm with task context
            swarm_result = await claude_flow_service.initialize_swarm(
                topology=swarm_config.get("topology", "adaptive"),
                max_agents=swarm_config.get("max_agents", 10),
                archon_integration=True
            )
            
            if swarm_result["status"] != "initialized":
                return swarm_result
            
            # Create agents for the task
            objective = f"Complete Archon task: {task.get('title', 'Unnamed task')}"
            if task.get("description"):
                objective += f". Description: {task['description']}"
                
            # Determine appropriate agents based on task
            agents = self._select_agents_for_task(task)
            
            spawn_result = await claude_flow_service.spawn_agents(
                objective=objective,
                agents=agents,
                strategy="development",
                archon_task_id=task_id
            )
            
            if spawn_result["status"] == "spawned":
                # Store swarm information
                swarm_info = {
                    "task_id": task_id,
                    "session_id": swarm_result["session_id"],
                    "agents": agents,
                    "objective": objective,
                    "created_at": datetime.now().isoformat(),
                    "status": "active"
                }
                self.active_swarms[task_id] = swarm_info
                
                # Update task status to indicate swarm is working
                await self.task_service.update_task_status(task_id, "doing")
                await self._add_task_note(task_id, f"Claude Flow swarm initialized with {len(agents)} agents")
                
                return {
                    "status": "success",
                    "swarm_info": swarm_info,
                    "task": task
                }
            else:
                return spawn_result
                
        except Exception as e:
            logger.error(f"Failed to create swarm for task {task_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_sparc_for_project(
        self,
        project_id: str,
        tasks: List[str],
        sparc_mode: str = "tdd"
    ) -> Dict[str, Any]:
        """Execute SPARC workflow for project tasks."""
        try:
            logger.info(f"Executing SPARC workflow for project: {project_id}")
            
            # Get project details
            project = await self.project_service.get_project_by_id(project_id)
            if not project:
                return {"status": "error", "error": f"Project {project_id} not found"}
            
            # Create comprehensive task description
            task_descriptions = []
            for task_id in tasks:
                task = await self.task_service.get_task_by_id(task_id)
                if task:
                    task_descriptions.append(f"- {task.get('title', 'Unnamed task')}: {task.get('description', '')}")
            
            sparc_task = f"Project: {project.get('title', 'Unnamed project')}\n\nTasks:\n" + "\n".join(task_descriptions)
            
            # Execute SPARC workflow
            sparc_result = await claude_flow_service.execute_sparc_workflow(
                task=sparc_task,
                mode=sparc_mode,
                archon_project_id=project_id
            )
            
            if sparc_result["status"] == "executed":
                # Update all tasks to indicate SPARC is processing
                for task_id in tasks:
                    await self.task_service.update_task_status(task_id, "doing")
                    await self._add_task_note(task_id, f"SPARC {sparc_mode} workflow initiated")
                
                # Store SPARC session info
                sparc_info = {
                    "project_id": project_id,
                    "task_ids": tasks,
                    "mode": sparc_mode,
                    "task_description": sparc_task,
                    "created_at": datetime.now().isoformat(),
                    "status": "executing"
                }
                
                return {
                    "status": "success",
                    "sparc_info": sparc_info,
                    "project": project
                }
            else:
                return sparc_result
                
        except Exception as e:
            logger.error(f"Failed to execute SPARC for project {project_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def sync_swarm_progress(self, task_id: str) -> Dict[str, Any]:
        """Sync swarm progress with Archon task status."""
        try:
            if task_id not in self.active_swarms:
                return {"status": "error", "error": "No active swarm for task"}
            
            swarm_info = self.active_swarms[task_id]
            
            # Get swarm status from Claude Flow
            status_result = await claude_flow_service.get_swarm_status()
            metrics_result = await claude_flow_service.get_agent_metrics()
            
            # Update swarm info with latest status
            if status_result["status"] == "success":
                swarm_info["last_status"] = status_result["info"]
            if metrics_result["status"] == "success":
                swarm_info["metrics"] = metrics_result["metrics"]
            
            # Add progress note to task
            progress_note = f"Swarm progress update - Agents: {len(swarm_info['agents'])}, Status: Active"
            await self._add_task_note(task_id, progress_note)
            
            return {
                "status": "success",
                "swarm_info": swarm_info
            }
            
        except Exception as e:
            logger.error(f"Failed to sync swarm progress for task {task_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def complete_swarm_task(self, task_id: str, completion_result: Dict[str, Any]) -> Dict[str, Any]:
        """Complete a task when swarm finishes."""
        try:
            if task_id not in self.active_swarms:
                return {"status": "error", "error": "No active swarm for task"}
            
            swarm_info = self.active_swarms[task_id]
            
            # Mark swarm as completed
            swarm_info["status"] = "completed"
            swarm_info["completed_at"] = datetime.now().isoformat()
            swarm_info["result"] = completion_result
            
            # Update task status
            await self.task_service.update_task_status(task_id, "review")
            
            # Add completion note
            completion_note = f"Claude Flow swarm completed. Result: {completion_result.get('summary', 'Task processed successfully')}"
            await self._add_task_note(task_id, completion_note)
            
            # Remove from active swarms
            del self.active_swarms[task_id]
            
            return {
                "status": "success",
                "task_id": task_id,
                "completion_result": completion_result
            }
            
        except Exception as e:
            logger.error(f"Failed to complete swarm task {task_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _select_agents_for_task(self, task: Dict[str, Any]) -> List[str]:
        """Select appropriate agents based on task characteristics."""
        agents = ["coder", "reviewer"]  # Base agents
        
        # Analyze task content for agent selection
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        
        if any(keyword in task_text for keyword in ["test", "testing", "quality", "validation"]):
            agents.append("tester")
            
        if any(keyword in task_text for keyword in ["research", "investigate", "analyze", "study"]):
            agents.append("researcher")
            
        if any(keyword in task_text for keyword in ["design", "architecture", "system", "planning"]):
            agents.append("system-architect")
            
        if any(keyword in task_text for keyword in ["api", "backend", "server", "database"]):
            agents.append("backend-dev")
            
        if any(keyword in task_text for keyword in ["ui", "frontend", "interface", "component"]):
            agents.append("coder")  # Already included, but emphasize
            
        if any(keyword in task_text for keyword in ["machine learning", "ml", "ai", "model"]):
            agents.append("ml-developer")
            
        # Limit to reasonable number of agents
        return agents[:5]
    
    async def _add_task_note(self, task_id: str, note: str) -> None:
        """Add a note to a task."""
        try:
            # This would integrate with Archon's task notes system
            # For now, we'll log the note
            logger.info(f"Task {task_id} note: {note}")
        except Exception as e:
            logger.warning(f"Could not add note to task {task_id}: {e}")
    
    async def get_active_swarms(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active swarms."""
        return self.active_swarms.copy()
    
    async def terminate_swarm(self, task_id: str) -> Dict[str, Any]:
        """Terminate a swarm for a task."""
        try:
            if task_id not in self.active_swarms:
                return {"status": "error", "error": "No active swarm for task"}
            
            swarm_info = self.active_swarms[task_id]
            
            # Execute termination hook
            await claude_flow_service.execute_hooks("post-task", {
                "taskId": task_id,
                "terminated": True,
                "reason": "manual_termination"
            })
            
            # Mark as terminated
            swarm_info["status"] = "terminated"
            swarm_info["terminated_at"] = datetime.now().isoformat()
            
            # Update task
            await self._add_task_note(task_id, "Claude Flow swarm terminated")
            
            # Remove from active swarms
            del self.active_swarms[task_id]
            
            return {"status": "success", "task_id": task_id}
            
        except Exception as e:
            logger.error(f"Failed to terminate swarm for task {task_id}: {e}")
            return {"status": "error", "error": str(e)}


# Global service instance
claude_flow_task_integration = ClaudeFlowTaskIntegration()