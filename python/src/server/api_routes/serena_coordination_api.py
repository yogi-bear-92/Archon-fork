"""
Serena Coordination API Routes

Provides REST API endpoints for interacting with the Serena Claude Flow Expert Agent
coordination hooks system. These endpoints enable external systems and
other agents to coordinate with Serena's semantic intelligence capabilities.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from src.server.services.serena_coordination_hooks import serena_coordination_hooks, CoordinationLevel, HookPhase
from src.server.config.logfire_config import get_logger

logger = get_logger(__name__)

# Create router
serena_coordination_router = APIRouter(
    prefix="/serena/coordination",
    tags=["serena-coordination"],
    responses={404: {"description": "Not found"}},
)


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class TaskContextModel(BaseModel):
    """Model for task context in coordination requests."""
    task_id: Optional[str] = None
    project_path: str = "."
    target_files: List[str] = []
    task_type: str = "general"
    priority: str = "medium"
    metadata: Dict[str, Any] = {}


class CoordinationLevelModel(BaseModel):
    """Model for coordination level specification."""
    level: str = Field(..., pattern="^(individual|pairwise|group|swarm|ecosystem)$")


class WorkflowDefinitionModel(BaseModel):
    """Model for multi-agent workflow definition."""
    workflow_id: Optional[str] = None
    name: str
    description: str = ""
    phases: List[Dict[str, Any]] = []
    requirements: Dict[str, Any] = {}
    coordination_level: str = "group"
    estimated_duration: int = 3600  # seconds


class AgentListModel(BaseModel):
    """Model for list of participating agents."""
    agents: List[str] = []
    coordinator: str = "serena-master"


class SyncContextModel(BaseModel):
    """Model for memory synchronization context."""
    scope: str = "local"
    target_types: List[str] = ["coordination_state", "semantic_context"]
    force_sync: bool = False
    timeout: int = 30


class MonitoringContextModel(BaseModel):
    """Model for performance monitoring context."""
    scope: str = "comprehensive"
    components: List[str] = ["system", "coordination", "semantic_analysis", "memory"]
    include_predictions: bool = True
    auto_optimize: bool = True


class HookExecutionResponse(BaseModel):
    """Standard response model for hook execution results."""
    success: bool
    execution_time: float
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    next_actions: List[str] = []


# ========================================================================
# PRE-TASK COORDINATION ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/pre-task/semantic-preparation",
    response_model=HookExecutionResponse,
    summary="Execute pre-task semantic preparation",
    description="Prepares semantic analysis context before task execution"
)
async def execute_pre_task_semantic_preparation(
    task_context: TaskContextModel,
    coordination_level: CoordinationLevelModel,
    background_tasks: BackgroundTasks
) -> HookExecutionResponse:
    """Execute pre-task semantic preparation hook."""
    try:
        logger.info(f"Pre-task semantic preparation requested for task: {task_context.task_id}")
        
        # Convert coordination level
        coord_level = CoordinationLevel(coordination_level.level)
        
        # Execute hook
        result = await serena_coordination_hooks.pre_task_semantic_preparation(
            task_context=task_context.dict(),
            coordination_level=coord_level
        )
        
        # Convert result to response model
        return HookExecutionResponse(
            success=result.success,
            execution_time=result.execution_time,
            data=result.data,
            error=result.error,
            retry_count=result.retry_count,
            next_actions=result.next_actions
        )
        
    except Exception as e:
        logger.error(f"Pre-task semantic preparation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/pre-task/semantic-context/{task_id}",
    response_model=Dict[str, Any],
    summary="Get semantic context for task",
    description="Retrieve semantic context prepared for a specific task"
)
async def get_task_semantic_context(task_id: str) -> Dict[str, Any]:
    """Get semantic context for a specific task."""
    try:
        if task_id not in serena_coordination_hooks.semantic_contexts:
            raise HTTPException(status_code=404, detail=f"Semantic context not found for task: {task_id}")
            
        context = serena_coordination_hooks.semantic_contexts[task_id]
        
        return {
            "task_id": task_id,
            "context": {
                "project_path": context.project_path,
                "file_count": len(context.file_paths),
                "symbol_count": len(context.symbol_map),
                "pattern_count": len(context.architecture_patterns),
                "complexity_metrics": context.complexity_metrics,
                "last_updated": context.last_updated.isoformat(),
                "context_hash": context.context_hash
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get semantic context for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# POST-TASK COORDINATION ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/post-task/knowledge-persistence",
    response_model=HookExecutionResponse,
    summary="Execute post-task knowledge persistence",
    description="Persist knowledge and insights after task completion"
)
async def execute_post_task_knowledge_persistence(
    task_result: Dict[str, Any],
    execution_metrics: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> HookExecutionResponse:
    """Execute post-task knowledge persistence hook."""
    try:
        logger.info(f"Post-task knowledge persistence requested for task: {task_result.get('task_id')}")
        
        # Execute hook
        result = await serena_coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        # Convert result to response model
        return HookExecutionResponse(
            success=result.success,
            execution_time=result.execution_time,
            data=result.data,
            error=result.error,
            retry_count=result.retry_count,
            next_actions=result.next_actions
        )
        
    except Exception as e:
        logger.error(f"Post-task knowledge persistence failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/post-task/learning-patterns",
    response_model=Dict[str, Any],
    summary="Get learning patterns",
    description="Retrieve stored learning patterns from task executions"
)
async def get_learning_patterns(
    limit: int = 100,
    task_type: Optional[str] = None,
    success_only: bool = False
) -> Dict[str, Any]:
    """Get learning patterns from task executions."""
    try:
        patterns_file = serena_coordination_hooks.memory_path / "learning_patterns.json"
        
        if not patterns_file.exists():
            return {"patterns": [], "total": 0, "filtered": 0}
            
        with open(patterns_file, 'r') as f:
            patterns_data = json.load(f)
            
        patterns = patterns_data.get("patterns", [])
        
        # Apply filters
        if task_type:
            patterns = [p for p in patterns if p.get("task_type") == task_type]
            
        if success_only:
            patterns = [p for p in patterns if p.get("success", False)]
            
        # Limit results
        filtered_count = len(patterns)
        patterns = patterns[-limit:] if limit > 0 else patterns
        
        return {
            "patterns": patterns,
            "total": len(patterns_data.get("patterns", [])),
            "filtered": filtered_count,
            "returned": len(patterns)
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# MULTI-AGENT COORDINATION ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/multi-agent/workflow-coordination",
    response_model=HookExecutionResponse,
    summary="Coordinate multi-agent workflow",
    description="Set up and coordinate complex multi-agent workflows"
)
async def coordinate_multi_agent_workflow(
    workflow_definition: WorkflowDefinitionModel,
    participating_agents: AgentListModel,
    background_tasks: BackgroundTasks
) -> HookExecutionResponse:
    """Coordinate multi-agent workflow execution."""
    try:
        logger.info(f"Multi-agent workflow coordination requested: {workflow_definition.name}")
        
        # Execute coordination hook
        result = await serena_coordination_hooks.coordinate_multi_agent_workflow(
            workflow_definition=workflow_definition.dict(),
            participating_agents=participating_agents.agents
        )
        
        # Convert result to response model
        return HookExecutionResponse(
            success=result.success,
            execution_time=result.execution_time,
            data=result.data,
            error=result.error,
            retry_count=result.retry_count,
            next_actions=result.next_actions
        )
        
    except Exception as e:
        logger.error(f"Multi-agent workflow coordination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/multi-agent/coordination-state/{task_id}",
    response_model=Dict[str, Any],
    summary="Get coordination state",
    description="Retrieve current coordination state for a task"
)
async def get_coordination_state(task_id: str) -> Dict[str, Any]:
    """Get coordination state for a specific task."""
    try:
        if task_id not in serena_coordination_hooks.coordination_states:
            raise HTTPException(status_code=404, detail=f"Coordination state not found for task: {task_id}")
            
        state = serena_coordination_hooks.coordination_states[task_id]
        
        return {
            "task_id": task_id,
            "state": {
                "agent_id": state.agent_id,
                "coordination_level": state.coordination_level.value,
                "active_agents": list(state.active_agents),
                "performance_metrics": state.performance_metrics,
                "error_count": state.error_count,
                "last_sync": state.last_sync.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get coordination state for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.post(
    "/hooks/multi-agent/register-pattern",
    response_model=Dict[str, Any],
    summary="Register coordination pattern",
    description="Register a new agent coordination pattern for reuse"
)
async def register_coordination_pattern(
    pattern_name: str,
    pattern_definition: Dict[str, Any]
) -> Dict[str, Any]:
    """Register a new agent coordination pattern."""
    try:
        result = await serena_coordination_hooks.register_agent_coordination_pattern(
            pattern_name=pattern_name,
            pattern_definition=pattern_definition
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to register coordination pattern {pattern_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# MEMORY SYNCHRONIZATION ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/memory/synchronization",
    response_model=HookExecutionResponse,
    summary="Execute memory synchronization",
    description="Synchronize memory and context across agents"
)
async def execute_memory_synchronization(
    sync_context: SyncContextModel,
    background_tasks: BackgroundTasks
) -> HookExecutionResponse:
    """Execute memory synchronization hook."""
    try:
        logger.info(f"Memory synchronization requested with scope: {sync_context.scope}")
        
        # Execute synchronization hook
        result = await serena_coordination_hooks.memory_synchronization_hook(
            sync_context=sync_context.dict()
        )
        
        # Convert result to response model
        return HookExecutionResponse(
            success=result.success,
            execution_time=result.execution_time,
            data=result.data,
            error=result.error,
            retry_count=result.retry_count,
            next_actions=result.next_actions
        )
        
    except Exception as e:
        logger.error(f"Memory synchronization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/memory/sync-status",
    response_model=Dict[str, Any],
    summary="Get memory synchronization status",
    description="Get current status of memory synchronization across agents"
)
async def get_memory_sync_status() -> Dict[str, Any]:
    """Get memory synchronization status."""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "coordination_states": len(serena_coordination_hooks.coordination_states),
            "semantic_contexts": len(serena_coordination_hooks.semantic_contexts),
            "active_hooks": len(serena_coordination_hooks.active_hooks),
            "last_sync_times": {}
        }
        
        # Get last sync times for each coordination state
        for task_id, state in serena_coordination_hooks.coordination_states.items():
            status["last_sync_times"][task_id] = state.last_sync.isoformat()
            
        return status
        
    except Exception as e:
        logger.error(f"Failed to get memory sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# PERFORMANCE MONITORING ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/performance/monitoring",
    response_model=HookExecutionResponse,
    summary="Execute performance monitoring",
    description="Monitor and optimize performance across coordination network"
)
async def execute_performance_monitoring(
    monitoring_context: MonitoringContextModel,
    background_tasks: BackgroundTasks
) -> HookExecutionResponse:
    """Execute performance monitoring hook."""
    try:
        logger.info(f"Performance monitoring requested with scope: {monitoring_context.scope}")
        
        # Execute monitoring hook
        result = await serena_coordination_hooks.performance_monitoring_hook(
            monitoring_context=monitoring_context.dict()
        )
        
        # Convert result to response model
        return HookExecutionResponse(
            success=result.success,
            execution_time=result.execution_time,
            data=result.data,
            error=result.error,
            retry_count=result.retry_count,
            next_actions=result.next_actions
        )
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/performance/metrics",
    response_model=Dict[str, Any],
    summary="Get coordination metrics",
    description="Get comprehensive coordination system metrics"
)
async def get_coordination_metrics() -> Dict[str, Any]:
    """Get comprehensive coordination system metrics."""
    try:
        metrics = await serena_coordination_hooks.get_coordination_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get coordination metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.get(
    "/hooks/performance/health",
    response_model=Dict[str, Any],
    summary="Get system health status",
    description="Get current health status of the coordination system"
)
async def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    try:
        # Collect basic metrics for health assessment
        monitoring_result = await serena_coordination_hooks.performance_monitoring_hook({
            "scope": "health_check",
            "components": ["system", "coordination"],
            "auto_optimize": False
        })
        
        if monitoring_result.success and monitoring_result.data:
            health_data = monitoring_result.data.get("health_analysis", {})
            return {
                "status": "healthy" if health_data.get("overall_score", 0) > 80 else "warning",
                "overall_score": health_data.get("overall_score", 0),
                "component_scores": health_data.get("component_scores", {}),
                "concerns": health_data.get("concerns", []),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "error": monitoring_result.error,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# UTILITY AND MANAGEMENT ENDPOINTS
# ========================================================================

@serena_coordination_router.get(
    "/hooks/status",
    response_model=Dict[str, Any],
    summary="Get hooks system status",
    description="Get current status of the coordination hooks system"
)
async def get_hooks_system_status() -> Dict[str, Any]:
    """Get current status of the coordination hooks system."""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "active_coordination_states": len(serena_coordination_hooks.coordination_states),
                "active_semantic_contexts": len(serena_coordination_hooks.semantic_contexts),
                "running_hooks": len(serena_coordination_hooks.active_hooks),
                "registered_error_patterns": len(serena_coordination_hooks.error_patterns)
            },
            "configuration": {
                "max_retry_attempts": serena_coordination_hooks.max_retry_attempts,
                "hook_timeout": serena_coordination_hooks.hook_timeout,
                "memory_sync_interval": serena_coordination_hooks.memory_sync_interval,
                "performance_check_interval": serena_coordination_hooks.performance_check_interval
            },
            "paths": {
                "hooks_path": str(serena_coordination_hooks.hooks_path),
                "memory_path": str(serena_coordination_hooks.memory_path),
                "metrics_path": str(serena_coordination_hooks.metrics_path)
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get hooks system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.post(
    "/hooks/export",
    response_model=Dict[str, Any],
    summary="Export coordination data",
    description="Export coordination data for analysis or backup"
)
async def export_coordination_data(
    export_path: str,
    background_tasks: BackgroundTasks,
    include_sensitive: bool = False
) -> Dict[str, Any]:
    """Export coordination data for analysis or backup."""
    try:
        # Execute export in background
        def run_export():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    serena_coordination_hooks.export_coordination_data(
                        export_path=export_path,
                        include_sensitive=include_sensitive
                    )
                )
                return result
            finally:
                loop.close()
                
        background_tasks.add_task(run_export)
        
        return {
            "status": "export_started",
            "export_path": export_path,
            "include_sensitive": include_sensitive,
            "message": "Export started in background. Check the specified path for completion."
        }
        
    except Exception as e:
        logger.error(f"Failed to start coordination data export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.delete(
    "/hooks/cleanup/{task_id}",
    response_model=Dict[str, Any],
    summary="Cleanup task resources",
    description="Manually cleanup resources for a specific task"
)
async def cleanup_task_resources(task_id: str) -> Dict[str, Any]:
    """Manually cleanup resources for a specific task."""
    try:
        cleanup_results = await serena_coordination_hooks._cleanup_task_resources(task_id)
        
        return {
            "status": "success",
            "task_id": task_id,
            "cleanup_results": cleanup_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup task resources for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@serena_coordination_router.post(
    "/hooks/optimize",
    response_model=Dict[str, Any],
    summary="Manual optimization trigger",
    description="Manually trigger system optimization"
)
async def trigger_manual_optimization(
    background_tasks: BackgroundTasks,
    optimization_scope: str = "comprehensive"
) -> Dict[str, Any]:
    """Manually trigger system optimization."""
    try:
        # Execute performance monitoring with auto-optimization enabled
        monitoring_result = await serena_coordination_hooks.performance_monitoring_hook({
            "scope": optimization_scope,
            "auto_optimize": True,
            "trigger": "manual_optimization"
        })
        
        if monitoring_result.success and monitoring_result.data:
            optimization_data = monitoring_result.data.get("auto_optimization_results", {})
            return {
                "status": "optimization_completed",
                "optimizations_applied": len(optimization_data.get("applied", [])),
                "optimizations_failed": len(optimization_data.get("failed", [])),
                "execution_time": monitoring_result.execution_time,
                "results": optimization_data
            }
        else:
            return {
                "status": "optimization_failed",
                "error": monitoring_result.error,
                "execution_time": monitoring_result.execution_time
            }
            
    except Exception as e:
        logger.error(f"Manual optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# WEBHOOK AND EVENT ENDPOINTS
# ========================================================================

@serena_coordination_router.post(
    "/hooks/webhook/task-event",
    response_model=Dict[str, Any],
    summary="Handle task event webhook",
    description="Handle incoming task events from other systems"
)
async def handle_task_event_webhook(
    event_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Handle incoming task events from other systems."""
    try:
        event_type = event_data.get("event_type", "unknown")
        task_id = event_data.get("task_id")
        
        logger.info(f"Received task event: {event_type} for task {task_id}")
        
        if event_type == "task_started":
            # Trigger pre-task preparation
            task_context = event_data.get("task_context", {})
            if task_id:
                task_context["task_id"] = task_id
                
            background_tasks.add_task(
                _execute_pre_task_preparation,
                task_context
            )
            
        elif event_type == "task_completed":
            # Trigger post-task knowledge persistence
            task_result = event_data.get("task_result", {})
            execution_metrics = event_data.get("execution_metrics", {})
            
            background_tasks.add_task(
                _execute_post_task_persistence,
                task_result,
                execution_metrics
            )
            
        elif event_type == "coordination_request":
            # Handle coordination request
            workflow_definition = event_data.get("workflow_definition", {})
            participating_agents = event_data.get("participating_agents", [])
            
            background_tasks.add_task(
                _execute_workflow_coordination,
                workflow_definition,
                participating_agents
            )
            
        return {
            "status": "event_received",
            "event_type": event_type,
            "task_id": task_id,
            "processing": "background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task event webhook handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# BACKGROUND TASK HELPERS
# ========================================================================

async def _execute_pre_task_preparation(task_context: Dict[str, Any]):
    """Background task for pre-task preparation."""
    try:
        from src.server.services.serena_coordination_hooks import CoordinationLevel
        
        result = await serena_coordination_hooks.pre_task_semantic_preparation(
            task_context=task_context,
            coordination_level=CoordinationLevel.INDIVIDUAL
        )
        
        logger.info(f"Background pre-task preparation completed: {result.success}")
        
    except Exception as e:
        logger.error(f"Background pre-task preparation failed: {e}")


async def _execute_post_task_persistence(
    task_result: Dict[str, Any],
    execution_metrics: Dict[str, Any]
):
    """Background task for post-task knowledge persistence."""
    try:
        result = await serena_coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        logger.info(f"Background post-task persistence completed: {result.success}")
        
    except Exception as e:
        logger.error(f"Background post-task persistence failed: {e}")


async def _execute_workflow_coordination(
    workflow_definition: Dict[str, Any],
    participating_agents: List[str]
):
    """Background task for workflow coordination."""
    try:
        result = await serena_coordination_hooks.coordinate_multi_agent_workflow(
            workflow_definition=workflow_definition,
            participating_agents=participating_agents
        )
        
        logger.info(f"Background workflow coordination completed: {result.success}")
        
    except Exception as e:
        logger.error(f"Background workflow coordination failed: {e}")