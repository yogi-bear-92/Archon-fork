"""
Interactive Task Creation API Routes

API endpoints for managing interactive task creation sessions where users
are prompted to confirm and provide details before tasks are created.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..config.logfire_config import get_logger, safe_span, safe_set_attribute
from ..services.interactive_task_service import get_interactive_task_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/interactive-tasks", tags=["Interactive Task Creation"])


class TaskDetectionRequest(BaseModel):
    """Request model for detecting potential tasks in user messages."""
    user_message: str = Field(..., description="User message to analyze for tasks")
    conversation_context: Optional[str] = Field(None, description="Optional conversation context")
    project_context: Optional[str] = Field(None, description="Optional project context")


class UserResponseRequest(BaseModel):
    """Request model for user responses in interactive task creation."""
    session_id: str = Field(..., description="Active task creation session ID")
    user_response: str = Field(..., description="User's response to questions/confirmations")


class TaskCreationResponse(BaseModel):
    """Response model for interactive task creation operations."""
    success: bool
    session_active: bool
    session_id: Optional[str] = None
    state: Optional[str] = None
    message: str
    action_required: bool = False
    questions: Optional[List[Dict[str, Any]]] = None
    task_preview: Optional[Dict[str, Any]] = None
    task_created: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/detect", summary="Detect potential tasks in user message")
async def detect_potential_tasks(request: TaskDetectionRequest) -> TaskCreationResponse:
    """
    Analyze a user message for potential tasks and initiate interactive creation process.
    
    This endpoint:
    1. Analyzes the user message for actionable tasks
    2. If tasks are found, creates an interactive session
    3. Returns a confirmation question for the user
    """
    with safe_span("detect_potential_tasks") as span:
        safe_set_attribute(span, "message_length", len(request.user_message))
        
        try:
            interactive_service = get_interactive_task_service()
            
            result = await interactive_service.detect_and_initiate_task_creation(
                user_message=request.user_message,
                conversation_context=request.conversation_context,
                project_context=request.project_context
            )
            
            safe_set_attribute(span, "has_potential_tasks", result.get("has_potential_tasks", False))
            
            if not result.get("has_potential_tasks", False):
                return TaskCreationResponse(
                    success=True,
                    session_active=False,
                    message=result.get("message", "No actionable tasks detected in your message."),
                    action_required=False
                )
            
            # Task detected, return confirmation request
            safe_set_attribute(span, "session_id", result.get("session_id"))
            safe_set_attribute(span, "task_confidence", result.get("primary_task", {}).get("confidence", 0))
            
            return TaskCreationResponse(
                success=True,
                session_active=True,
                session_id=result.get("session_id"),
                state="user_confirmation",
                message=result.get("confirmation_question", "Would you like to create this task?"),
                action_required=True,
                task_preview=result.get("primary_task")
            )
            
        except Exception as e:
            logger.error(f"Error detecting potential tasks: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/respond", summary="Process user response in interactive session")
async def process_user_response(request: UserResponseRequest) -> TaskCreationResponse:
    """
    Process a user's response in an active interactive task creation session.
    
    This endpoint handles:
    1. User confirmations (yes/no)
    2. Answers to clarifying questions
    3. Final task creation confirmation
    """
    with safe_span("process_user_response") as span:
        safe_set_attribute(span, "session_id", request.session_id)
        safe_set_attribute(span, "response_length", len(request.user_response))
        
        try:
            interactive_service = get_interactive_task_service()
            
            result = await interactive_service.process_user_response(
                session_id=request.session_id,
                user_response=request.user_response
            )
            
            safe_set_attribute(span, "session_active", result.get("session_active", False))
            safe_set_attribute(span, "state", result.get("state", "unknown"))
            
            if result.get("error"):
                if "not found" in result["error"].lower():
                    raise HTTPException(status_code=404, detail=result["error"])
                else:
                    raise HTTPException(status_code=400, detail=result["error"])
            
            return TaskCreationResponse(
                success=True,
                session_active=result.get("session_active", False),
                session_id=result.get("session_id"),
                state=result.get("state"),
                message=result.get("message", ""),
                action_required=result.get("session_active", False),
                questions=result.get("questions"),
                task_preview=result.get("task_preview"),
                task_created=result.get("task_created")
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", summary="Get active task creation sessions")
async def get_active_sessions() -> Dict[str, Any]:
    """Get all active interactive task creation sessions."""
    with safe_span("get_active_sessions"):
        try:
            interactive_service = get_interactive_task_service()
            sessions = await interactive_service.get_active_sessions()
            
            return {
                "success": True,
                "active_sessions": sessions,
                "count": len(sessions)
            }
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", summary="Get specific session details")
async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Get details about a specific interactive task creation session."""
    with safe_span("get_session_details") as span:
        safe_set_attribute(span, "session_id", session_id)
        
        try:
            interactive_service = get_interactive_task_service()
            
            if session_id not in interactive_service.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            
            session = interactive_service.active_sessions[session_id]
            
            return {
                "success": True,
                "session": {
                    "session_id": session_id,
                    "task_title": session.detected_task.title,
                    "task_description": session.detected_task.description,
                    "current_state": session.current_state.value,
                    "confidence_score": session.detected_task.confidence_score,
                    "category": session.detected_task.category,
                    "urgency": session.detected_task.urgency,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "questions_asked": session.questions_asked,
                    "user_responses": session.user_responses,
                    "gathered_details": session.gathered_details
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting session details: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", summary="Cancel task creation session")
async def cancel_session(session_id: str) -> Dict[str, Any]:
    """Cancel an active interactive task creation session."""
    with safe_span("cancel_session") as span:
        safe_set_attribute(span, "session_id", session_id)
        
        try:
            interactive_service = get_interactive_task_service()
            success = await interactive_service.cancel_session(session_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            
            safe_set_attribute(span, "cancelled", True)
            
            return {
                "success": True,
                "message": f"Task creation session {session_id} has been cancelled",
                "session_id": session_id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error cancelling session: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-workflow", summary="Test the complete interactive workflow")
async def test_interactive_workflow() -> Dict[str, Any]:
    """Test the complete interactive task creation workflow with sample data."""
    with safe_span("test_interactive_workflow"):
        try:
            interactive_service = get_interactive_task_service()
            
            # Step 1: Detect task
            test_message = "I need to implement a user authentication system with JWT tokens, password hashing, and email verification"
            
            detection_result = await interactive_service.detect_and_initiate_task_creation(
                user_message=test_message,
                conversation_context="Testing interactive workflow",
                project_context="Web application development"
            )
            
            if not detection_result.get("has_potential_tasks"):
                return {
                    "success": True,
                    "test_result": "no_tasks_detected",
                    "message": "Test message did not trigger task detection"
                }
            
            session_id = detection_result.get("session_id")
            
            # Step 2: Simulate user confirmation
            confirmation_result = await interactive_service.process_user_response(
                session_id=session_id,
                user_response="yes, please create this task"
            )
            
            # Step 3: Simulate answering questions (if any)
            if confirmation_result.get("questions"):
                question_response = await interactive_service.process_user_response(
                    session_id=session_id,
                    user_response="The system should support email/password authentication with bcrypt hashing, JWT tokens for session management, and email verification for new registrations"
                )
            else:
                question_response = confirmation_result
            
            # Step 4: Final confirmation (if needed)
            final_result = question_response
            if question_response.get("state") == "final_confirmation":
                final_result = await interactive_service.process_user_response(
                    session_id=session_id,
                    user_response="yes, create the task"
                )
            
            return {
                "success": True,
                "test_result": "workflow_completed",
                "session_id": session_id,
                "detection_result": detection_result,
                "final_state": final_result.get("state"),
                "task_created": final_result.get("task_created") is not None,
                "message": "Interactive workflow test completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error testing interactive workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="Check interactive task service health")
async def health_check() -> Dict[str, Any]:
    """Check the health of the interactive task creation service."""
    with safe_span("interactive_task_health_check"):
        try:
            interactive_service = get_interactive_task_service()
            active_sessions = await interactive_service.get_active_sessions()
            
            return {
                "success": True,
                "service_status": "healthy",
                "active_sessions_count": len(active_sessions),
                "timestamp": datetime.now().isoformat(),
                "features": {
                    "task_detection": "available",
                    "interactive_creation": "available",
                    "session_management": "available",
                    "archon_integration": "available"
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")