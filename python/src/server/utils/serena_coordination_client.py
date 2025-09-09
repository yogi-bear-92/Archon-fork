"""
Serena Coordination Client SDK

Provides a high-level client interface for interacting with the Serena Claude Flow Expert Agent
coordination hooks system. This SDK simplifies integration with other agents and
external systems, providing semantic intelligence capabilities.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import httpx
from src.server.config.logfire_config import get_logger

logger = get_logger(__name__)


class CoordinationMode(Enum):
    """Coordination modes for different interaction patterns."""
    DIRECT = "direct"           # Direct API calls
    ASYNC = "async"            # Asynchronous execution
    WEBHOOK = "webhook"        # Webhook-based coordination
    STREAMING = "streaming"    # Real-time streaming updates


class TaskPhase(Enum):
    """Task execution phases."""
    PREPARATION = "preparation"
    EXECUTION = "execution"
    COMPLETION = "completion"
    COORDINATION = "coordination"


@dataclass
class CoordinationRequest:
    """Request for coordination services."""
    task_id: Optional[str] = None
    phase: TaskPhase = TaskPhase.EXECUTION
    mode: CoordinationMode = CoordinationMode.DIRECT
    context: Dict[str, Any] = None
    timeout: int = 30
    retry_count: int = 0


@dataclass 
class CoordinationResponse:
    """Response from coordination services."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    next_actions: List[str] = None
    
    def __post_init__(self):
        if self.next_actions is None:
            self.next_actions = []


class SerenaCoordinationClient:
    """
    High-level client for Serena Claude Flow Expert Agent coordination services.
    
    This client provides a simple interface for:
    - Semantic analysis coordination
    - Multi-agent workflow management
    - Memory synchronization
    - Performance monitoring
    - Knowledge persistence
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # HTTP client configuration
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout
        )
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.coordination_endpoint = "/serena/coordination"
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def close(self):
        """Close the client and cleanup resources."""
        await self.client.aclose()
        
    # ========================================================================
    # TASK COORDINATION METHODS
    # ========================================================================
    
    async def prepare_task(
        self,
        task_id: str,
        project_path: str = ".",
        target_files: List[str] = None,
        coordination_level: str = "individual",
        task_type: str = "general",
        metadata: Dict[str, Any] = None
    ) -> CoordinationResponse:
        """
        Prepare semantic analysis context for task execution.
        
        Args:
            task_id: Unique task identifier
            project_path: Path to project root
            target_files: List of files to analyze (optional)
            coordination_level: Level of coordination (individual, group, swarm, etc.)
            task_type: Type of task being executed
            metadata: Additional task metadata
            
        Returns:
            CoordinationResponse with preparation results
        """
        try:
            logger.info(f"Preparing task: {task_id}")
            
            request_data = {
                "task_context": {
                    "task_id": task_id,
                    "project_path": project_path,
                    "target_files": target_files or [],
                    "task_type": task_type,
                    "metadata": metadata or {}
                },
                "coordination_level": {
                    "level": coordination_level
                }
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/pre-task/semantic-preparation",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("success", False),
                data=response.get("data"),
                error=response.get("error"),
                execution_time=response.get("execution_time", 0.0),
                next_actions=response.get("next_actions", [])
            )
            
        except Exception as e:
            logger.error(f"Task preparation failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def complete_task(
        self,
        task_result: Dict[str, Any],
        execution_metrics: Dict[str, Any] = None
    ) -> CoordinationResponse:
        """
        Complete task and persist knowledge.
        
        Args:
            task_result: Results from task execution
            execution_metrics: Performance and execution metrics
            
        Returns:
            CoordinationResponse with completion results
        """
        try:
            task_id = task_result.get("task_id", "unknown")
            logger.info(f"Completing task: {task_id}")
            
            request_data = {
                "task_result": task_result,
                "execution_metrics": execution_metrics or {}
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/post-task/knowledge-persistence",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("success", False),
                data=response.get("data"),
                error=response.get("error"),
                execution_time=response.get("execution_time", 0.0),
                next_actions=response.get("next_actions", [])
            )
            
        except Exception as e:
            logger.error(f"Task completion failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def get_semantic_context(self, task_id: str) -> CoordinationResponse:
        """
        Get semantic context for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            CoordinationResponse with semantic context data
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/pre-task/semantic-context/{task_id}"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get semantic context for {task_id}: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # MULTI-AGENT COORDINATION METHODS
    # ========================================================================
    
    async def coordinate_workflow(
        self,
        workflow_name: str,
        workflow_description: str,
        participating_agents: List[str],
        workflow_phases: List[Dict[str, Any]] = None,
        coordination_level: str = "group",
        estimated_duration: int = 3600
    ) -> CoordinationResponse:
        """
        Coordinate multi-agent workflow execution.
        
        Args:
            workflow_name: Name of the workflow
            workflow_description: Description of workflow objectives
            participating_agents: List of agents participating in workflow
            workflow_phases: Phases of the workflow (optional)
            coordination_level: Level of coordination required
            estimated_duration: Estimated duration in seconds
            
        Returns:
            CoordinationResponse with coordination setup results
        """
        try:
            logger.info(f"Coordinating workflow: {workflow_name}")
            
            request_data = {
                "workflow_definition": {
                    "name": workflow_name,
                    "description": workflow_description,
                    "phases": workflow_phases or [],
                    "coordination_level": coordination_level,
                    "estimated_duration": estimated_duration
                },
                "participating_agents": {
                    "agents": participating_agents,
                    "coordinator": "serena-master"
                }
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/multi-agent/workflow-coordination",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("success", False),
                data=response.get("data"),
                error=response.get("error"),
                execution_time=response.get("execution_time", 0.0),
                next_actions=response.get("next_actions", [])
            )
            
        except Exception as e:
            logger.error(f"Workflow coordination failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def get_coordination_state(self, task_id: str) -> CoordinationResponse:
        """
        Get current coordination state for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            CoordinationResponse with coordination state data
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/multi-agent/coordination-state/{task_id}"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get coordination state for {task_id}: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def register_coordination_pattern(
        self,
        pattern_name: str,
        pattern_definition: Dict[str, Any]
    ) -> CoordinationResponse:
        """
        Register a new coordination pattern for reuse.
        
        Args:
            pattern_name: Name of the coordination pattern
            pattern_definition: Pattern definition and configuration
            
        Returns:
            CoordinationResponse with registration results
        """
        try:
            params = {
                "pattern_name": pattern_name
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/multi-agent/register-pattern",
                json=pattern_definition,
                params=params
            )
            
            return CoordinationResponse(
                success=response.get("status") == "success",
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to register coordination pattern {pattern_name}: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # MEMORY SYNCHRONIZATION METHODS
    # ========================================================================
    
    async def synchronize_memory(
        self,
        scope: str = "local",
        target_types: List[str] = None,
        force_sync: bool = False
    ) -> CoordinationResponse:
        """
        Synchronize memory and context across agents.
        
        Args:
            scope: Synchronization scope (local, distributed, global)
            target_types: Types of data to synchronize
            force_sync: Force synchronization even if recently synced
            
        Returns:
            CoordinationResponse with synchronization results
        """
        try:
            logger.info(f"Synchronizing memory with scope: {scope}")
            
            request_data = {
                "sync_context": {
                    "scope": scope,
                    "target_types": target_types or ["coordination_state", "semantic_context"],
                    "force_sync": force_sync
                }
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/memory/synchronization",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("success", False),
                data=response.get("data"),
                error=response.get("error"),
                execution_time=response.get("execution_time", 0.0),
                next_actions=response.get("next_actions", [])
            )
            
        except Exception as e:
            logger.error(f"Memory synchronization failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def get_memory_status(self) -> CoordinationResponse:
        """
        Get current memory synchronization status.
        
        Returns:
            CoordinationResponse with memory status data
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/memory/sync-status"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # PERFORMANCE MONITORING METHODS
    # ========================================================================
    
    async def monitor_performance(
        self,
        scope: str = "comprehensive",
        components: List[str] = None,
        auto_optimize: bool = False
    ) -> CoordinationResponse:
        """
        Monitor system performance and get optimization recommendations.
        
        Args:
            scope: Monitoring scope (basic, comprehensive, detailed)
            components: Components to monitor
            auto_optimize: Apply automatic optimizations
            
        Returns:
            CoordinationResponse with monitoring results
        """
        try:
            logger.info(f"Monitoring performance with scope: {scope}")
            
            request_data = {
                "monitoring_context": {
                    "scope": scope,
                    "components": components or ["system", "coordination", "semantic_analysis"],
                    "auto_optimize": auto_optimize,
                    "include_predictions": True
                }
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/performance/monitoring",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("success", False),
                data=response.get("data"),
                error=response.get("error"),
                execution_time=response.get("execution_time", 0.0),
                next_actions=response.get("next_actions", [])
            )
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def get_system_metrics(self) -> CoordinationResponse:
        """
        Get comprehensive system metrics.
        
        Returns:
            CoordinationResponse with metrics data
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/performance/metrics"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def get_system_health(self) -> CoordinationResponse:
        """
        Get current system health status.
        
        Returns:
            CoordinationResponse with health status data
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/performance/health"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def optimize_system(self, scope: str = "comprehensive") -> CoordinationResponse:
        """
        Manually trigger system optimization.
        
        Args:
            scope: Optimization scope
            
        Returns:
            CoordinationResponse with optimization results
        """
        try:
            logger.info(f"Triggering system optimization with scope: {scope}")
            
            request_data = {
                "optimization_scope": scope
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/optimize",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("status") == "optimization_completed",
                data=response,
                execution_time=response.get("execution_time", 0.0)
            )
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # LEARNING AND KNOWLEDGE METHODS
    # ========================================================================
    
    async def get_learning_patterns(
        self,
        limit: int = 100,
        task_type: Optional[str] = None,
        success_only: bool = False
    ) -> CoordinationResponse:
        """
        Get learning patterns from past task executions.
        
        Args:
            limit: Maximum number of patterns to return
            task_type: Filter by task type
            success_only: Return only successful patterns
            
        Returns:
            CoordinationResponse with learning patterns
        """
        try:
            params = {
                "limit": limit,
                "success_only": success_only
            }
            if task_type:
                params["task_type"] = task_type
                
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/post-task/learning-patterns",
                params=params
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get learning patterns: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # UTILITY AND MANAGEMENT METHODS
    # ========================================================================
    
    async def get_system_status(self) -> CoordinationResponse:
        """
        Get current system status and configuration.
        
        Returns:
            CoordinationResponse with system status
        """
        try:
            response = await self._make_request(
                "GET",
                f"{self.coordination_endpoint}/hooks/status"
            )
            
            return CoordinationResponse(
                success=True,
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def cleanup_task_resources(self, task_id: str) -> CoordinationResponse:
        """
        Manually cleanup resources for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            CoordinationResponse with cleanup results
        """
        try:
            response = await self._make_request(
                "DELETE",
                f"{self.coordination_endpoint}/hooks/cleanup/{task_id}"
            )
            
            return CoordinationResponse(
                success=response.get("status") == "success",
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to cleanup task resources for {task_id}: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def export_coordination_data(
        self,
        export_path: str,
        include_sensitive: bool = False
    ) -> CoordinationResponse:
        """
        Export coordination data for analysis or backup.
        
        Args:
            export_path: Path for export file
            include_sensitive: Include sensitive data in export
            
        Returns:
            CoordinationResponse with export status
        """
        try:
            request_data = {
                "export_path": export_path,
                "include_sensitive": include_sensitive
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/export",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("status") == "export_started",
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to export coordination data: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # EVENT HANDLING METHODS
    # ========================================================================
    
    def on_event(self, event_type: str, handler: Callable):
        """
        Register an event handler for coordination events.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function for event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def send_task_event(
        self,
        event_type: str,
        task_id: str,
        event_data: Dict[str, Any] = None
    ) -> CoordinationResponse:
        """
        Send a task event to the coordination system.
        
        Args:
            event_type: Type of event (task_started, task_completed, etc.)
            task_id: Task identifier
            event_data: Additional event data
            
        Returns:
            CoordinationResponse with event handling results
        """
        try:
            request_data = {
                "event_type": event_type,
                "task_id": task_id,
                **(event_data or {})
            }
            
            response = await self._make_request(
                "POST",
                f"{self.coordination_endpoint}/hooks/webhook/task-event",
                json=request_data
            )
            
            return CoordinationResponse(
                success=response.get("status") == "event_received",
                data=response,
                execution_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to send task event {event_type}: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # HIGH-LEVEL CONVENIENCE METHODS
    # ========================================================================
    
    async def execute_coordinated_task(
        self,
        task_id: str,
        task_function: Callable,
        project_path: str = ".",
        target_files: List[str] = None,
        coordination_level: str = "individual",
        task_type: str = "general",
        metadata: Dict[str, Any] = None
    ) -> CoordinationResponse:
        """
        Execute a task with full coordination lifecycle.
        
        This method handles:
        1. Task preparation and semantic analysis
        2. Task execution with your provided function
        3. Task completion and knowledge persistence
        4. Resource cleanup
        
        Args:
            task_id: Unique task identifier
            task_function: Async function to execute for the task
            project_path: Path to project root
            target_files: Files to analyze
            coordination_level: Level of coordination
            task_type: Type of task
            metadata: Additional metadata
            
        Returns:
            CoordinationResponse with complete task results
        """
        try:
            logger.info(f"Executing coordinated task: {task_id}")
            
            # Phase 1: Prepare task
            prep_result = await self.prepare_task(
                task_id=task_id,
                project_path=project_path,
                target_files=target_files,
                coordination_level=coordination_level,
                task_type=task_type,
                metadata=metadata
            )
            
            if not prep_result.success:
                return prep_result
            
            # Phase 2: Execute task
            start_time = datetime.now()
            try:
                task_result = await task_function()
                success = True
                error = None
            except Exception as e:
                task_result = {"error": str(e)}
                success = False
                error = str(e)
                
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Phase 3: Complete task
            task_completion_data = {
                "task_id": task_id,
                "success": success,
                "result": task_result,
                "error": error,
                "task_type": task_type
            }
            
            execution_metrics = {
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "preparation_data": prep_result.data
            }
            
            completion_result = await self.complete_task(
                task_result=task_completion_data,
                execution_metrics=execution_metrics
            )
            
            # Phase 4: Cleanup (optional, run in background)
            asyncio.create_task(self.cleanup_task_resources(task_id))
            
            return CoordinationResponse(
                success=success and completion_result.success,
                data={
                    "task_result": task_result,
                    "preparation_result": prep_result.data,
                    "completion_result": completion_result.data,
                    "execution_metrics": execution_metrics
                },
                error=error or completion_result.error,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Coordinated task execution failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    async def execute_multi_agent_workflow(
        self,
        workflow_name: str,
        workflow_steps: List[Dict[str, Any]],
        participating_agents: List[str],
        coordination_level: str = "group"
    ) -> CoordinationResponse:
        """
        Execute a complete multi-agent workflow.
        
        Args:
            workflow_name: Name of the workflow
            workflow_steps: List of workflow steps
            participating_agents: Agents participating in the workflow
            coordination_level: Level of coordination
            
        Returns:
            CoordinationResponse with workflow results
        """
        try:
            logger.info(f"Executing multi-agent workflow: {workflow_name}")
            
            # Setup coordination
            coord_result = await self.coordinate_workflow(
                workflow_name=workflow_name,
                workflow_description=f"Multi-agent workflow with {len(workflow_steps)} steps",
                participating_agents=participating_agents,
                workflow_phases=workflow_steps,
                coordination_level=coordination_level
            )
            
            if not coord_result.success:
                return coord_result
            
            # Execute workflow steps
            step_results = []
            for i, step in enumerate(workflow_steps):
                step_id = f"{workflow_name}_step_{i+1}"
                logger.info(f"Executing workflow step {i+1}: {step.get('name', step_id)}")
                
                # Each step would be coordinated separately
                # This is a simplified version - in practice, you'd have more complex coordination
                step_results.append({
                    "step_id": step_id,
                    "step_name": step.get("name", f"Step {i+1}"),
                    "status": "completed",
                    "result": step
                })
                
            return CoordinationResponse(
                success=True,
                data={
                    "workflow_name": workflow_name,
                    "coordination_result": coord_result.data,
                    "step_results": step_results,
                    "total_steps": len(workflow_steps),
                    "completed_steps": len(step_results)
                },
                execution_time=sum(step.get("execution_time", 0) for step in step_results)
            )
            
        except Exception as e:
            logger.error(f"Multi-agent workflow execution failed: {e}")
            return CoordinationResponse(success=False, error=str(e))
    
    # ========================================================================
    # INTERNAL UTILITY METHODS
    # ========================================================================
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Dict[str, Any] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to coordination API."""
        for attempt in range(self.retry_attempts):
            try:
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=json,
                    params=params
                )
                
                response.raise_for_status()
                
                if response.headers.get("content-type", "").startswith("application/json"):
                    return response.json()
                else:
                    return {"message": response.text}
                    
            except httpx.HTTPError as e:
                if attempt == self.retry_attempts - 1:
                    raise
                    
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                raise
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event_data))
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")


# ========================================================================
# CONVENIENCE FUNCTIONS AND UTILITIES
# ========================================================================

async def create_coordination_client(
    base_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    **kwargs
) -> SerenaCoordinationClient:
    """
    Create and return a configured coordination client.
    
    Args:
        base_url: Base URL for the coordination API
        api_key: API key for authentication
        **kwargs: Additional client configuration
        
    Returns:
        Configured SerenaCoordinationClient
    """
    return SerenaCoordinationClient(
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


async def execute_coordinated_task(
    task_id: str,
    task_function: Callable,
    client: Optional[SerenaCoordinationClient] = None,
    **kwargs
) -> CoordinationResponse:
    """
    Convenience function to execute a task with coordination.
    
    Args:
        task_id: Task identifier
        task_function: Function to execute
        client: Coordination client (creates new one if None)
        **kwargs: Additional task parameters
        
    Returns:
        CoordinationResponse with task results
    """
    if client is None:
        async with create_coordination_client() as client:
            return await client.execute_coordinated_task(
                task_id=task_id,
                task_function=task_function,
                **kwargs
            )
    else:
        return await client.execute_coordinated_task(
            task_id=task_id,
            task_function=task_function,
            **kwargs
        )


def coordination_decorator(
    task_id: Optional[str] = None,
    project_path: str = ".",
    coordination_level: str = "individual",
    task_type: str = "general",
    client: Optional[SerenaCoordinationClient] = None
):
    """
    Decorator for automatic task coordination.
    
    Args:
        task_id: Task identifier (auto-generated if None)
        project_path: Project path for analysis
        coordination_level: Level of coordination
        task_type: Type of task
        client: Coordination client
        
    Returns:
        Decorated function with coordination capabilities
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            actual_task_id = task_id or f"{func.__name__}_{int(datetime.now().timestamp())}"
            
            async def task_execution():
                return await func(*args, **kwargs)
            
            return await execute_coordinated_task(
                task_id=actual_task_id,
                task_function=task_execution,
                client=client,
                project_path=project_path,
                coordination_level=coordination_level,
                task_type=task_type
            )
        
        return wrapper
    return decorator


# ========================================================================
# EXAMPLE USAGE PATTERNS
# ========================================================================

async def example_basic_task_coordination():
    """Example of basic task coordination."""
    async with create_coordination_client() as client:
        # Prepare task
        prep_result = await client.prepare_task(
            task_id="example_task_001",
            project_path="/path/to/project",
            task_type="code_analysis"
        )
        
        if prep_result.success:
            print(f"Task prepared successfully: {prep_result.data}")
            
            # Simulate task execution
            task_result = {
                "task_id": "example_task_001",
                "success": True,
                "files_analyzed": 15,
                "patterns_found": ["MVC", "Repository"],
                "recommendations": ["Improve error handling", "Add unit tests"]
            }
            
            # Complete task
            completion_result = await client.complete_task(
                task_result=task_result,
                execution_metrics={"execution_time": 2.5, "memory_used": 45.2}
            )
            
            print(f"Task completed: {completion_result.success}")


async def example_multi_agent_workflow():
    """Example of multi-agent workflow coordination."""
    async with create_coordination_client() as client:
        workflow_result = await client.execute_multi_agent_workflow(
            workflow_name="code_refactoring_workflow",
            workflow_steps=[
                {"name": "analyze_code", "agent": "serena-master", "duration": 60},
                {"name": "generate_improvements", "agent": "coder", "duration": 120},
                {"name": "review_changes", "agent": "reviewer", "duration": 90},
                {"name": "test_changes", "agent": "tester", "duration": 180}
            ],
            participating_agents=["serena-master", "coder", "reviewer", "tester"]
        )
        
        print(f"Workflow completed: {workflow_result.success}")
        if workflow_result.data:
            print(f"Steps completed: {workflow_result.data['completed_steps']}/{workflow_result.data['total_steps']}")


@coordination_decorator(task_type="semantic_analysis")
async def example_decorated_function(file_path: str):
    """Example of using the coordination decorator."""
    # This function will automatically be wrapped with coordination
    # Simulate some work
    await asyncio.sleep(1)
    return {"analyzed_file": file_path, "complexity_score": 3.5}


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_basic_task_coordination())
    asyncio.run(example_multi_agent_workflow())
    
    # Example with decorator
    result = asyncio.run(example_decorated_function("example.py"))
    print(f"Decorated function result: {result}")