"""
Claude Flow API Routes for Archon Integration

This module provides REST API endpoints for Claude Flow orchestration
integrated with Archon's task management and knowledge systems.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging

from ..config.logfire_config import get_logger
from ..services.claude_flow_service import claude_flow_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/claude-flow", tags=["Claude Flow"])


# Pydantic models
class SwarmInitRequest(BaseModel):
    topology: str = Field(default="adaptive", description="Swarm topology (adaptive, mesh, hierarchical)")
    max_agents: int = Field(default=10, description="Maximum number of agents")
    archon_integration: bool = Field(default=True, description="Enable Archon integration")


class AgentSpawnRequest(BaseModel):
    objective: str = Field(..., description="Objective for the agents")
    agents: List[str] = Field(..., description="List of agent types to spawn")
    strategy: str = Field(default="development", description="Execution strategy")
    archon_task_id: Optional[str] = Field(None, description="Associated Archon task ID")


class SparcWorkflowRequest(BaseModel):
    task: str = Field(..., description="Task description")
    mode: str = Field(default="tdd", description="SPARC mode (tdd, batch, pipeline)")
    archon_project_id: Optional[str] = Field(None, description="Associated Archon project ID")


class HookExecutionRequest(BaseModel):
    hook_name: str = Field(..., description="Hook name to execute")
    context: Dict[str, Any] = Field(..., description="Hook execution context")


class MemoryOperationRequest(BaseModel):
    operation: str = Field(..., description="Memory operation (store, retrieve, search)")
    key: Optional[str] = Field(None, description="Memory key")
    value: Optional[Any] = Field(None, description="Value to store")


class NeuralTrainingRequest(BaseModel):
    patterns: List[Dict[str, Any]] = Field(..., description="Training patterns")
    model_type: str = Field(default="performance", description="Model type to train")


# API Endpoints
@router.post("/swarm/init")
async def initialize_swarm(request: SwarmInitRequest) -> Dict[str, Any]:
    """Initialize Claude Flow swarm with Archon integration."""
    try:
        logger.info(f"Initializing swarm: topology={request.topology}, agents={request.max_agents}")
        
        result = await claude_flow_service.initialize_swarm(
            topology=request.topology,
            max_agents=request.max_agents,
            archon_integration=request.archon_integration
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Swarm initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/spawn")
async def spawn_agents(request: AgentSpawnRequest) -> Dict[str, Any]:
    """Spawn agents for a specific objective."""
    try:
        logger.info(f"Spawning agents: {request.agents} for objective: {request.objective}")
        
        result = await claude_flow_service.spawn_agents(
            objective=request.objective,
            agents=request.agents,
            strategy=request.strategy,
            archon_task_id=request.archon_task_id
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Agent spawning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sparc/execute")
async def execute_sparc_workflow(request: SparcWorkflowRequest) -> Dict[str, Any]:
    """Execute SPARC methodology workflow."""
    try:
        logger.info(f"Executing SPARC workflow: mode={request.mode}, task={request.task}")
        
        result = await claude_flow_service.execute_sparc_workflow(
            task=request.task,
            mode=request.mode,
            archon_project_id=request.archon_project_id
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"SPARC workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_swarm_status() -> Dict[str, Any]:
    """Get current swarm status and metrics."""
    try:
        result = await claude_flow_service.get_swarm_status()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to get swarm status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_agent_metrics() -> Dict[str, Any]:
    """Get agent performance metrics."""
    try:
        result = await claude_flow_service.get_agent_metrics()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hooks/execute")
async def execute_hooks(request: HookExecutionRequest) -> Dict[str, Any]:
    """Execute Claude Flow hooks."""
    try:
        logger.info(f"Executing hook: {request.hook_name}")
        
        result = await claude_flow_service.execute_hooks(
            hook_name=request.hook_name,
            context=request.context
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Hook execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory")
async def memory_operations(request: MemoryOperationRequest) -> Dict[str, Any]:
    """Perform memory operations."""
    try:
        logger.info(f"Memory operation: {request.operation}")
        
        result = await claude_flow_service.memory_operations(
            operation=request.operation,
            key=request.key,
            value=request.value
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Memory operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neural/train")
async def neural_training(request: NeuralTrainingRequest) -> Dict[str, Any]:
    """Execute neural pattern training."""
    try:
        logger.info(f"Neural training: model_type={request.model_type}, patterns={len(request.patterns)}")
        
        result = await claude_flow_service.neural_training(
            patterns=request.patterns,
            model_type=request.model_type
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Neural training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/types")
async def get_available_agent_types() -> Dict[str, List[str]]:
    """Get available agent types organized by category."""
    return {
        "core": ["coder", "reviewer", "tester", "planner", "researcher"],
        "sparc": ["sparc-coord", "specification", "pseudocode", "architecture", "refinement", "sparc-coder"],
        "archon": ["archon_prp", "backend-dev", "ml-developer", "system-architect"],
        "swarm": ["hierarchical-coordinator", "mesh-coordinator", "adaptive-coordinator"],
        "github": ["code-review-swarm", "pr-manager", "issue-tracker", "release-manager"],
        "testing": ["tdd-london-swarm", "production-validator"],
        "specialized": ["mobile-dev", "cicd-engineer", "api-docs", "devops-architect"]
    }


@router.get("/sparc/modes")
async def get_sparc_modes() -> Dict[str, str]:
    """Get available SPARC workflow modes."""
    return {
        "tdd": "Test-driven development workflow",
        "batch": "Batch processing multiple tasks",
        "pipeline": "Full pipeline processing",
        "concurrent": "Concurrent multi-task processing",
        "spec-pseudocode": "Specification and pseudocode phases",
        "architect": "Architecture design phase",
        "integration": "Integration and completion phase"
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for Claude Flow service."""
    try:
        # Basic health check
        status = await claude_flow_service.get_swarm_status()
        return {
            "status": "healthy",
            "service": "claude-flow",
            "timestamp": status.get("info", {}).get("timestamp", "unknown")
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")