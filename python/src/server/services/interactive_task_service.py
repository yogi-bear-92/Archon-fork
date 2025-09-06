"""
Interactive Task Creation Service

This service provides a conversational approach to task creation where the system:
1. Detects potential tasks in user messages
2. Asks the user for confirmation 
3. Gathers additional details through questions
4. Creates tasks only when user explicitly confirms
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..config.logfire_config import get_logger, safe_span, safe_set_attribute
from ..services.llm_provider_service import get_llm_client
from ..unified_archon_mcp import ArchonMCPCoordinator
from .task_detection_service import TaskCandidate, TaskDetectionResult, get_task_detection_service

logger = get_logger(__name__)


class TaskCreationState(str, Enum):
    """States in the interactive task creation workflow."""
    INITIAL_DETECTION = "initial_detection"
    USER_CONFIRMATION = "user_confirmation"
    GATHERING_DETAILS = "gathering_details"
    FINAL_CONFIRMATION = "final_confirmation"
    TASK_CREATED = "task_created"
    USER_DECLINED = "user_declined"


@dataclass
class TaskGatheringSession:
    """Session for interactive task gathering."""
    session_id: str
    detected_task: TaskCandidate
    current_state: TaskCreationState
    gathered_details: Dict[str, Any]
    questions_asked: List[str]
    user_responses: List[str]
    created_at: datetime
    updated_at: datetime
    project_context: Optional[str] = None
    conversation_context: Optional[str] = None


class TaskQuestion(BaseModel):
    """A clarifying question about a task."""
    question: str = Field(..., description="The clarifying question to ask")
    category: str = Field(..., description="Category of question (scope, requirements, priority, etc.)")
    required: bool = Field(default=True, description="Whether this question must be answered")
    follow_up_questions: List[str] = Field(default_factory=list, description="Potential follow-up questions")


class TaskAnalysisRequest(BaseModel):
    """Request for analyzing and asking questions about a task."""
    task_title: str
    task_description: str
    user_message: str
    current_details: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: Optional[str] = None
    project_context: Optional[str] = None


class InteractiveTaskService:
    """
    Service for interactive task creation with user confirmation and detail gathering.
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, TaskGatheringSession] = {}
        self.task_detection_service = get_task_detection_service()
        
        # Question generation prompt template
        self.question_prompt = """
You are an intelligent project manager helping to gather complete requirements for a task. 
Your job is to ask the RIGHT questions to ensure the task is well-defined and actionable.

TASK INFORMATION:
Title: {task_title}
Description: {task_description}
Original User Message: {user_message}

CURRENT DETAILS GATHERED:
{current_details}

CONVERSATION CONTEXT:
{conversation_context}

PROJECT CONTEXT:
{project_context}

INSTRUCTIONS:
1. Analyze what information is still missing or unclear about this task
2. Ask 1-3 specific, actionable questions to clarify requirements
3. Focus on the most important missing details first
4. Make questions conversational but specific
5. Avoid asking questions if the information is already clear

AREAS TO CONSIDER (ask only if unclear):
- **Scope & Requirements**: What exactly needs to be built/fixed/implemented?
- **Acceptance Criteria**: How will we know when it's done?
- **Technical Approach**: Any specific technologies, frameworks, or methods?
- **Priority & Timeline**: How urgent is this? Any deadlines?
- **Dependencies**: What other tasks/systems does this depend on?
- **User Experience**: Who will use this and how?
- **Testing**: What kind of testing is needed?

IMPORTANT: Do NOT ask about task assignment - the coordinator agent will handle assignee selection automatically based on the task requirements.

Response format (JSON):
{{
    "needs_more_info": true/false,
    "questions": [
        {{
            "question": "What specific authentication methods should be supported (email/password, OAuth, etc.)?",
            "category": "requirements",
            "required": true,
            "follow_up_questions": ["Should we include social login options?", "What about two-factor authentication?"]
        }}
    ],
    "missing_areas": ["scope", "technical_approach", "timeline"],
    "confidence_score": 0.7,
    "ready_to_create": false,
    "reasoning": "Need to clarify authentication methods and technical requirements"
}}

IMPORTANT: If the task is already well-defined with clear requirements, set "ready_to_create": true and "needs_more_info": false.
"""


        # Final task creation prompt
        self.creation_prompt = """
Based on the conversation and gathered details, create a comprehensive task specification.

ORIGINAL TASK:
Title: {task_title}
Description: {task_description}

USER RESPONSES:
{user_responses}

GATHERED DETAILS:
{gathered_details}

Create a final task with:
1. Clear, actionable title (5-10 words)
2. Comprehensive description with acceptance criteria
3. Priority level and effort estimate
4. Feature category
5. Source references when relevant
6. Code examples when applicable

NOTE: Do NOT specify an assignee - the coordinator agent will determine the appropriate assignee based on the task requirements and available agents.

Response format (JSON):
{{
    "title": "Implement JWT authentication system",
    "description": "Create a complete authentication system with JWT tokens, including login, logout, registration, and password reset functionality. Must support email/password authentication with bcrypt hashing. Include middleware for protected routes and refresh token mechanism.\n\nAcceptance Criteria:\n- Users can register with email and password\n- Users can login and receive JWT tokens\n- Password reset functionality via email\n- Protected routes require valid JWT\n- Refresh token mechanism for security\n- Password hashing with bcrypt\n- Input validation and error handling",
    "urgency": "high",
    "estimated_effort": "large",
    "category": "feature",
    "sources": [
        {{"url": "https://jwt.io/introduction", "type": "documentation", "relevance": "JWT implementation guide"}},
        {{"url": "internal://auth-requirements.md", "type": "requirements", "relevance": "Authentication requirements document"}}
    ],
    "code_examples": [
        {{"file": "src/auth/base.py", "function": "BaseAuthProvider", "purpose": "Base authentication class to extend"}}
    ]
}}
"""

    async def detect_and_initiate_task_creation(
        self,
        user_message: str,
        conversation_context: Optional[str] = None,
        project_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect potential tasks and initiate interactive creation process.
        
        Returns a response that Claude Code can use to ask the user for confirmation.
        """
        with safe_span("interactive_task_detection") as span:
            safe_set_attribute(span, "message_length", len(user_message))
            
            try:
                # Use existing task detection service
                detection_result = await self.task_detection_service.detect_tasks_in_message(
                    user_message=user_message,
                    conversation_context=conversation_context,
                    project_context=project_context
                )
                
                safe_set_attribute(span, "tasks_detected", len(detection_result.tasks))
                
                if not detection_result.has_tasks:
                    return {
                        "has_potential_tasks": False,
                        "message": "No actionable tasks detected in your message.",
                        "detection_result": detection_result.dict() if hasattr(detection_result, 'dict') else None
                    }
                
                # Filter to high-confidence tasks for interactive creation
                high_confidence_tasks = [
                    task for task in detection_result.tasks 
                    if task.confidence_score >= 0.6
                ]
                
                if not high_confidence_tasks:
                    return {
                        "has_potential_tasks": False,
                        "message": "I detected some potential tasks but they seem unclear. Could you be more specific about what you'd like me to help you with?",
                        "low_confidence_tasks": len(detection_result.tasks)
                    }
                
                # Create response for user confirmation
                task_summaries = []
                for task in high_confidence_tasks:
                    task_summaries.append({
                        "title": task.title,
                        "description": task.description,
                        "confidence": task.confidence_score,
                        "category": task.category,
                        "urgency": task.urgency
                    })
                
                # Create session for the most confident task
                primary_task = max(high_confidence_tasks, key=lambda t: t.confidence_score)
                session_id = f"task_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                session = TaskGatheringSession(
                    session_id=session_id,
                    detected_task=primary_task,
                    current_state=TaskCreationState.USER_CONFIRMATION,
                    gathered_details={},
                    questions_asked=[],
                    user_responses=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    project_context=project_context,
                    conversation_context=conversation_context
                )
                
                self.active_sessions[session_id] = session
                
                safe_set_attribute(span, "session_created", session_id)
                safe_set_attribute(span, "primary_task_confidence", primary_task.confidence_score)
                
                logger.info(f"🎯 Task creation session initiated: {session_id} for task: {primary_task.title}")
                
                return {
                    "has_potential_tasks": True,
                    "session_id": session_id,
                    "primary_task": {
                        "title": primary_task.title,
                        "description": primary_task.description,
                        "confidence": primary_task.confidence_score,
                        "category": primary_task.category,
                        "urgency": primary_task.urgency
                    },
                    "additional_tasks": task_summaries[1:] if len(task_summaries) > 1 else [],
                    "user_confirmation_needed": True,
                    "confirmation_question": f"I detected a potential task: **{primary_task.title}**\n\n{primary_task.description}\n\nWould you like me to help you create this as a task in your project? (yes/no)"
                }
                
            except Exception as e:
                logger.error(f"❌ Error in interactive task detection: {e}")
                return {
                    "has_potential_tasks": False,
                    "error": str(e),
                    "message": "I encountered an error while analyzing your message for tasks."
                }

    async def process_user_response(
        self,
        session_id: str,
        user_response: str
    ) -> Dict[str, Any]:
        """
        Process user response in an active task creation session.
        """
        with safe_span("process_user_response") as span:
            safe_set_attribute(span, "session_id", session_id)
            safe_set_attribute(span, "response_length", len(user_response))
            
            if session_id not in self.active_sessions:
                return {
                    "error": "Task creation session not found or expired",
                    "session_active": False
                }
            
            session = self.active_sessions[session_id]
            session.updated_at = datetime.now()
            session.user_responses.append(user_response)
            
            try:
                if session.current_state == TaskCreationState.USER_CONFIRMATION:
                    return await self._handle_user_confirmation(session, user_response)
                
                elif session.current_state == TaskCreationState.GATHERING_DETAILS:
                    return await self._handle_detail_gathering(session, user_response)
                
                elif session.current_state == TaskCreationState.FINAL_CONFIRMATION:
                    return await self._handle_final_confirmation(session, user_response)
                
                else:
                    return {
                        "error": f"Invalid session state: {session.current_state}",
                        "session_active": False
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error processing user response: {e}")
                return {
                    "error": str(e),
                    "session_active": True,
                    "session_id": session_id
                }

    async def _handle_user_confirmation(
        self,
        session: TaskGatheringSession,
        user_response: str
    ) -> Dict[str, Any]:
        """Handle initial user confirmation."""
        response_lower = user_response.lower().strip()
        
        # Check for positive responses
        if any(word in response_lower for word in ['yes', 'y', 'sure', 'ok', 'okay', 'please', 'create', 'go ahead']):
            # User confirmed, start gathering details
            session.current_state = TaskCreationState.GATHERING_DETAILS
            
            # Generate initial clarifying questions
            questions_result = await self._generate_clarifying_questions(session)
            
            if questions_result.get("ready_to_create", False):
                # Task is already well-defined, go to final confirmation
                session.current_state = TaskCreationState.FINAL_CONFIRMATION
                return {
                    "session_active": True,
                    "session_id": session.session_id,
                    "state": "final_confirmation",
                    "message": f"Great! The task '{session.detected_task.title}' seems well-defined. Should I create it now with the following details?\n\n**Title**: {session.detected_task.title}\n**Description**: {session.detected_task.description}\n**Category**: {session.detected_task.category}\n**Urgency**: {session.detected_task.urgency}\n\nConfirm to create this task? (yes/no)",
                    "task_preview": {
                        "title": session.detected_task.title,
                        "description": session.detected_task.description,
                        "category": session.detected_task.category,
                        "urgency": session.detected_task.urgency
                    }
                }
            else:
                # Need to gather more details
                questions = questions_result.get("questions", [])
                if questions:
                    session.questions_asked.extend([q.get("question", "") for q in questions])
                    
                    question_text = "Great! I'd like to gather some more details to make this task as clear as possible:\n\n"
                    for i, q in enumerate(questions, 1):
                        question_text += f"**{i}.** {q.get('question', '')}\n"
                    
                    return {
                        "session_active": True,
                        "session_id": session.session_id,
                        "state": "gathering_details",
                        "message": question_text,
                        "questions": questions
                    }
                else:
                    # No questions generated, proceed to creation
                    return await self._create_task_from_session(session)
        
        # Check for negative responses
        elif any(word in response_lower for word in ['no', 'n', 'cancel', 'skip', 'not now', 'later']):
            session.current_state = TaskCreationState.USER_DECLINED
            self._cleanup_session(session.session_id)
            
            return {
                "session_active": False,
                "state": "declined",
                "message": "No problem! I won't create a task. Let me know if you change your mind or have other tasks you'd like to create."
            }
        
        else:
            # Unclear response, ask for clarification
            return {
                "session_active": True,
                "session_id": session.session_id,
                "state": "user_confirmation",
                "message": f"I'm not sure if you want me to create the task '{session.detected_task.title}'. Could you please respond with 'yes' to create it or 'no' to skip it?"
            }

    async def _handle_detail_gathering(
        self,
        session: TaskGatheringSession,
        user_response: str
    ) -> Dict[str, Any]:
        """Handle detail gathering phase."""
        # Store the user's response in gathered details
        question_count = len(session.questions_asked)
        response_count = len(session.user_responses)
        
        if question_count > 0:
            latest_question = session.questions_asked[-1]
            session.gathered_details[f"question_{question_count}"] = {
                "question": latest_question,
                "answer": user_response
            }
        
        
        # Generate next questions or move to final confirmation
        questions_result = await self._generate_clarifying_questions(session)
        
        if questions_result.get("ready_to_create", False):
            # We have enough information, move to final confirmation
            session.current_state = TaskCreationState.FINAL_CONFIRMATION
            
            # Generate final task preview
            task_preview = await self._generate_final_task(session)
            
            return {
                "session_active": True,
                "session_id": session.session_id,
                "state": "final_confirmation",
                "message": f"Perfect! Based on our conversation, here's the task I'll create:\n\n**Title**: {task_preview.get('title', session.detected_task.title)}\n\n**Description**: {task_preview.get('description', session.detected_task.description)}\n\n**Category**: {task_preview.get('category', session.detected_task.category)}\n**Priority**: {task_preview.get('urgency', session.detected_task.urgency)}\n\n*Note: The coordinator agent will automatically assign this task to the most appropriate team member based on the requirements.*\n\nShould I create this task? (yes/no)",
                "task_preview": task_preview
            }
        else:
            # Ask more questions
            questions = questions_result.get("questions", [])
            if questions:
                session.questions_asked.extend([q.get("question", "") for q in questions])
                
                question_text = "Thanks for that information! I have a few more questions:\n\n"
                for i, q in enumerate(questions, 1):
                    question_text += f"**{i}.** {q.get('question', '')}\n"
                
                return {
                    "session_active": True,
                    "session_id": session.session_id,
                    "state": "gathering_details",
                    "message": question_text,
                    "questions": questions
                }
            else:
                # No more questions, move to final confirmation
                session.current_state = TaskCreationState.FINAL_CONFIRMATION
                return await self._handle_final_confirmation(session, "ready")

    async def _handle_final_confirmation(
        self,
        session: TaskGatheringSession,
        user_response: str
    ) -> Dict[str, Any]:
        """Handle final confirmation before task creation."""
        if user_response.lower() == "ready":
            # Internal signal to proceed with confirmation
            task_preview = await self._generate_final_task(session)
            return {
                "session_active": True,
                "session_id": session.session_id,
                "state": "final_confirmation",
                "message": f"Here's the complete task I'll create:\n\n**Title**: {task_preview.get('title')}\n\n**Description**: {task_preview.get('description')}\n\nShould I create this task? (yes/no)",
                "task_preview": task_preview
            }
        
        response_lower = user_response.lower().strip()
        
        if any(word in response_lower for word in ['yes', 'y', 'create', 'go ahead', 'sure', 'ok']):
            # Create the task
            return await self._create_task_from_session(session)
        
        elif any(word in response_lower for word in ['no', 'n', 'cancel', 'skip']):
            session.current_state = TaskCreationState.USER_DECLINED
            self._cleanup_session(session.session_id)
            
            return {
                "session_active": False,
                "state": "declined",
                "message": "Understood! I won't create the task. Feel free to let me know if you'd like to create it later or have other tasks to work on."
            }
        
        else:
            return {
                "session_active": True,
                "session_id": session.session_id,
                "state": "final_confirmation",
                "message": "Please respond with 'yes' to create the task or 'no' to cancel."
            }

    async def _generate_clarifying_questions(
        self,
        session: TaskGatheringSession
    ) -> Dict[str, Any]:
        """Generate clarifying questions for the task."""
        try:
            # Prepare current details for the prompt
            current_details = {}
            for key, value in session.gathered_details.items():
                if isinstance(value, dict) and "answer" in value:
                    current_details[value.get("question", key)] = value["answer"]
                else:
                    current_details[key] = value
            
            prompt = self.question_prompt.format(
                task_title=session.detected_task.title,
                task_description=session.detected_task.description,
                user_message=" ".join(session.user_responses[:1]),  # Original user message
                current_details=json.dumps(current_details, indent=2),
                conversation_context=session.conversation_context or "No conversation context",
                project_context=session.project_context or "General development project"
            )
            
            async with get_llm_client() as client:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert project manager. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
        except Exception as e:
            logger.warning(f"⚠️ Error generating questions: {e}")
            # Fallback to ready for creation
            return {
                "needs_more_info": False,
                "questions": [],
                "ready_to_create": True,
                "reasoning": "Unable to generate questions, proceeding with available information"
            }

    async def _generate_final_task(
        self,
        session: TaskGatheringSession
    ) -> Dict[str, Any]:
        """Generate the final task specification."""
        try:
            # Prepare user responses and gathered details
            user_responses_text = "\n".join([
                f"Response {i+1}: {response}" 
                for i, response in enumerate(session.user_responses)
            ])
            
            gathered_details_text = json.dumps(session.gathered_details, indent=2)
            
            prompt = self.creation_prompt.format(
                task_title=session.detected_task.title,
                task_description=session.detected_task.description,
                user_responses=user_responses_text,
                gathered_details=gathered_details_text
            )
            
            async with get_llm_client() as client:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert project manager creating comprehensive task specifications. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
        except Exception as e:
            logger.warning(f"⚠️ Error generating final task: {e}")
            # Fallback to original task data
            return {
                "title": session.detected_task.title,
                "description": session.detected_task.description,
                "urgency": session.detected_task.urgency,
                "category": session.detected_task.category,
                "estimated_effort": session.detected_task.estimated_effort
            }

    async def _create_task_from_session(
        self,
        session: TaskGatheringSession
    ) -> Dict[str, Any]:
        """Create the actual task in Archon."""
        try:
            # Generate final task specification
            task_spec = await self._generate_final_task(session)
            
            # Create task using Archon MCP Coordinator
            coordinator = ArchonMCPCoordinator()
            
            # Get or create a project
            projects_result = await coordinator.list_projects()
            project_id = None
            
            if projects_result.get("success") and projects_result.get("projects"):
                # Use the first available project
                project_id = projects_result["projects"][0]["id"]
            else:
                # Create a default project
                create_result = await coordinator.create_project(
                    title="Interactive Tasks",
                    description="Tasks created through interactive Claude conversations"
                )
                if create_result.get("success"):
                    project_id = create_result.get("project_id")
            
            if not project_id:
                raise Exception("Could not find or create a project for the task")
            
            # Create the task - let coordinator agent decide assignee
            task_result = await coordinator.create_task(
                project_id=project_id,
                title=task_spec.get("title", session.detected_task.title),
                description=task_spec.get("description", session.detected_task.description),
                # No assignee specified - coordinator agent will decide
                feature=task_spec.get("category", session.detected_task.category),
                sources=task_spec.get("sources"),
                code_examples=task_spec.get("code_examples")
            )
            
            if task_result.get("success"):
                session.current_state = TaskCreationState.TASK_CREATED
                self._cleanup_session(session.session_id)
                
                logger.info(f"✅ Task created successfully: {task_result.get('task_id')}")
                
                return {
                    "session_active": False,
                    "state": "task_created",
                    "message": f"✅ Great! I've successfully created the task: **{task_spec.get('title')}**\n\nTask ID: {task_result.get('task_id')}\nProject: {project_id}\n\nThe task is now in your project and the coordinator agent will assign it to the most appropriate team member based on the requirements.",
                    "task_created": {
                        "task_id": task_result.get("task_id"),
                        "project_id": project_id,
                        "title": task_spec.get("title"),
                        "category": task_spec.get("category")
                    }
                }
            else:
                raise Exception(f"Task creation failed: {task_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Error creating task: {e}")
            return {
                "session_active": False,
                "state": "error",
                "error": str(e),
                "message": f"I encountered an error while creating the task: {str(e)}. Please try again or create the task manually."
            }

    def _cleanup_session(self, session_id: str):
        """Clean up an expired or completed session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.debug(f"🧹 Cleaned up task session: {session_id}")

    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active task creation sessions."""
        sessions = []
        for session_id, session in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "task_title": session.detected_task.title,
                "current_state": session.current_state.value,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "questions_asked": len(session.questions_asked),
                "responses_received": len(session.user_responses)
            })
        return sessions

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active task creation session."""
        if session_id in self.active_sessions:
            self._cleanup_session(session_id)
            logger.info(f"🚫 Task creation session cancelled: {session_id}")
            return True
        return False


# Global instance
_interactive_task_service: Optional[InteractiveTaskService] = None


def get_interactive_task_service() -> InteractiveTaskService:
    """Get global interactive task service instance."""
    global _interactive_task_service
    
    if _interactive_task_service is None:
        _interactive_task_service = InteractiveTaskService()
    
    return _interactive_task_service