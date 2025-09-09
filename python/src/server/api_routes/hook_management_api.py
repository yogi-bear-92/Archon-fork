"""
Hook Management API Routes

API endpoints for managing Claude Code hooks and testing the integration system.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.server.config.logfire_config import get_logger, safe_span, safe_set_attribute
from src.server.services.hook_integration_service import get_hook_integration_service, HookExecutionResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api/hooks", tags=["Hook Management"])


class HookTestRequest(BaseModel):
    """Request model for testing hooks."""
    user_message: str = Field(..., description="User message to test with")
    conversation_context: Optional[str] = Field(None, description="Optional conversation context")
    project_context: Optional[str] = Field(None, description="Optional project context")
    async_execution: bool = Field(True, description="Whether to execute asynchronously")


class HookConfigurationUpdate(BaseModel):
    """Request model for updating hook configuration."""
    hook_name: str = Field(..., description="Name of the hook to update")
    enabled: bool = Field(..., description="Whether the hook should be enabled")
    timeout: Optional[int] = Field(None, description="Timeout in milliseconds")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Hook execution conditions")


class HookExecutionResponse(BaseModel):
    """Response model for hook execution results."""
    hook_name: str
    success: bool
    execution_time_ms: float
    output: Optional[Any] = None
    error: Optional[str] = None
    tasks_created: int = 0
    tasks_suggested: int = 0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime


@router.get("/", summary="Get hooks configuration")
async def get_hooks_configuration() -> Dict[str, Any]:
    """Get the current Claude Code hooks configuration."""
    with safe_span("get_hooks_configuration") as span:
        try:
            hook_service = get_hook_integration_service()
            config = await hook_service.get_hooks_configuration()
            
            safe_set_attribute(span, "hooks_count", len(config.get("hooks", {})))
            
            return {
                "success": True,
                "configuration": config,
                "hooks_count": len(config.get("hooks", {}))
            }
        except Exception as e:
            logger.error(f"Error getting hooks configuration: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", summary="Test hook execution")
async def test_hook_execution(request: HookTestRequest) -> HookExecutionResponse:
    """Test the post-user-prompt hook with a sample message."""
    with safe_span("test_hook_execution") as span:
        safe_set_attribute(span, "message_length", len(request.user_message))
        safe_set_attribute(span, "async_execution", request.async_execution)
        
        try:
            hook_service = get_hook_integration_service()
            
            result = await hook_service.execute_post_user_prompt_hook(
                user_message=request.user_message,
                conversation_context=request.conversation_context,
                project_context=request.project_context,
                async_execution=request.async_execution
            )
            
            safe_set_attribute(span, "execution_success", result.success)
            safe_set_attribute(span, "tasks_created", result.tasks_created)
            safe_set_attribute(span, "tasks_suggested", result.tasks_suggested)
            
            return HookExecutionResponse(
                hook_name=result.hook_name,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
                output=result.output,
                error=result.error,
                tasks_created=result.tasks_created,
                tasks_suggested=result.tasks_suggested,
                metadata=result.metadata,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error testing hook execution: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable/{hook_name}", summary="Enable a hook")
async def enable_hook(hook_name: str) -> Dict[str, Any]:
    """Enable a specific hook."""
    with safe_span("enable_hook") as span:
        safe_set_attribute(span, "hook_name", hook_name)
        
        try:
            hook_service = get_hook_integration_service()
            success = await hook_service.enable_hook(hook_name)
            
            safe_set_attribute(span, "success", success)
            
            if success:
                return {
                    "success": True,
                    "message": f"Hook '{hook_name}' enabled successfully",
                    "hook_name": hook_name,
                    "enabled": True
                }
            else:
                raise HTTPException(status_code=404, detail=f"Hook '{hook_name}' not found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error enabling hook: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable/{hook_name}", summary="Disable a hook")
async def disable_hook(hook_name: str) -> Dict[str, Any]:
    """Disable a specific hook."""
    with safe_span("disable_hook") as span:
        safe_set_attribute(span, "hook_name", hook_name)
        
        try:
            hook_service = get_hook_integration_service()
            success = await hook_service.disable_hook(hook_name)
            
            safe_set_attribute(span, "success", success)
            
            if success:
                return {
                    "success": True,
                    "message": f"Hook '{hook_name}' disabled successfully",
                    "hook_name": hook_name,
                    "enabled": False
                }
            else:
                raise HTTPException(status_code=404, detail=f"Hook '{hook_name}' not found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error disabling hook: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Get hook execution history")
async def get_hook_execution_history(hook_name: Optional[str] = None) -> Dict[str, Any]:
    """Get execution history for hooks."""
    with safe_span("get_hook_execution_history") as span:
        try:
            hook_service = get_hook_integration_service()
            history = await hook_service.get_hook_execution_history(hook_name)
            
            safe_set_attribute(span, "history_count", len(history))
            if hook_name:
                safe_set_attribute(span, "hook_name", hook_name)
            
            return {
                "success": True,
                "history": [
                    {
                        "hook_name": result.hook_name,
                        "success": result.success,
                        "execution_time_ms": result.execution_time_ms,
                        "tasks_created": result.tasks_created,
                        "tasks_suggested": result.tasks_suggested,
                        "error": result.error,
                        "metadata": result.metadata
                    }
                    for result in history
                ],
                "count": len(history),
                "filter": hook_name
            }
            
        except Exception as e:
            logger.error(f"Error getting hook execution history: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history", summary="Clear hook execution history")
async def clear_hook_execution_history() -> Dict[str, Any]:
    """Clear the hook execution history cache."""
    with safe_span("clear_hook_execution_history"):
        try:
            hook_service = get_hook_integration_service()
            await hook_service.clear_execution_cache()
            
            return {
                "success": True,
                "message": "Hook execution history cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Error clearing hook execution history: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-integration", summary="Test hook system integration")
async def test_hook_system_integration() -> Dict[str, Any]:
    """Test the entire hook system integration."""
    with safe_span("test_hook_system_integration"):
        try:
            hook_service = get_hook_integration_service()
            test_results = await hook_service.test_hook_integration()
            
            safe_set_attribute(span, "overall_status", test_results["overall_status"])
            
            return {
                "success": True,
                "test_results": test_results,
                "message": "Hook system integration test completed",
                "recommendations": _get_integration_recommendations(test_results)
            }
            
        except Exception as e:
            logger.error(f"Error testing hook system integration: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-background", summary="Execute hook in background")
async def execute_hook_background(
    request: HookTestRequest, 
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute the post-user-prompt hook in the background."""
    with safe_span("execute_hook_background"):
        try:
            hook_service = get_hook_integration_service()
            
            # Add background task
            background_tasks.add_task(
                _execute_hook_background_task,
                hook_service,
                request.user_message,
                request.conversation_context,
                request.project_context
            )
            
            return {
                "success": True,
                "message": "Hook execution started in background",
                "user_message": request.user_message[:100] + "..." if len(request.user_message) > 100 else request.user_message
            }
            
        except Exception as e:
            logger.error(f"Error executing hook in background: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _execute_hook_background_task(
    hook_service,
    user_message: str,
    conversation_context: Optional[str],
    project_context: Optional[str]
):
    """Background task for hook execution."""
    try:
        await hook_service.execute_post_user_prompt_hook(
            user_message=user_message,
            conversation_context=conversation_context,
            project_context=project_context,
            async_execution=True
        )
    except Exception as e:
        logger.error(f"Error in background hook execution: {e}")


def _get_integration_recommendations(test_results: Dict[str, Any]) -> List[str]:
    """Get recommendations based on integration test results."""
    recommendations = []
    
    if not test_results.get("configuration_valid"):
        recommendations.append("Update or create the .claude-hooks.json configuration file")
    
    if not test_results.get("hook_script_exists"):
        recommendations.append("Ensure the post_user_prompt_hook.py script exists in src/hooks/")
    
    if not test_results.get("task_service_available"):
        recommendations.append("Check task detection service dependencies and configuration")
    
    if not test_results.get("archon_available"):
        recommendations.append("Verify Archon MCP coordinator is properly configured")
    
    if not test_results.get("test_execution_success"):
        recommendations.append("Review hook execution logs for specific error details")
    
    if not recommendations:
        recommendations.append("Hook system integration is working correctly!")
    
    return recommendations