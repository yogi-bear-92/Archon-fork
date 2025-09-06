"""
AI-Powered Task Detection Service for Claude Hook Integration

This service analyzes user messages to automatically detect when new tasks 
should be created in Archon projects, enabling seamless task management.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..config.logfire_config import get_logger, safe_span, safe_set_attribute
from ..services.llm_provider_service import get_llm_client
from .client_manager import get_supabase_client

logger = get_logger(__name__)


class TaskCandidate(BaseModel):
    """A potential task detected from user input."""
    title: str = Field(..., description="Brief, actionable task title")
    description: str = Field(..., description="Detailed task description")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence this is a real task")
    urgency: str = Field(..., description="Priority level: low, medium, high, critical")
    estimated_effort: str = Field(..., description="Effort estimate: small, medium, large")
    category: str = Field(..., description="Task category: feature, bug, refactor, research, etc.")
    context: str = Field(default="", description="Context where task was detected")
    suggested_assignee: str = Field(default="User", description="Suggested assignee")
    reasoning: str = Field(..., description="Why this was identified as a task")


class TaskDetectionResult(BaseModel):
    """Result of task detection analysis."""
    has_tasks: bool = Field(..., description="Whether any tasks were detected")
    tasks: List[TaskCandidate] = Field(default_factory=list)
    conversation_summary: str = Field(default="", description="Brief summary of the conversation")
    project_context: str = Field(default="", description="Detected project context")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in analysis")


class TaskDetectionService:
    """
    AI service that analyzes user messages and conversation context to detect 
    when new tasks should be created automatically.
    """
    
    def __init__(self):
        self._detection_cache: Dict[str, TaskDetectionResult] = {}
        
        # Task detection prompt template
        self.detection_prompt = """
You are an intelligent task detection agent for a project management system. Your job is to analyze user messages and determine if they contain requests for new tasks that should be tracked.

Analyze this user message and determine if it contains actionable tasks that should be created:

USER MESSAGE:
{user_message}

CONVERSATION CONTEXT (if available):
{conversation_context}

PROJECT CONTEXT (if available):
{project_context}

INSTRUCTIONS:
1. Look for explicit task requests ("can you", "please implement", "I need", "create", "build", "fix", etc.)
2. Look for implied tasks (problems mentioned, features discussed, bugs reported)
3. Distinguish between questions/discussions vs actionable work items
4. Consider urgency indicators (ASAP, urgent, by Friday, etc.)
5. Identify technical vs non-technical tasks

CRITERIA FOR TASK DETECTION:
- HIGH CONFIDENCE (0.8-1.0): Explicit requests with clear deliverables
- MEDIUM CONFIDENCE (0.5-0.7): Implied tasks or problems that need solving
- LOW CONFIDENCE (0.2-0.4): Discussions that might lead to tasks later
- NO TASK (0.0-0.1): Pure questions, explanations, or casual conversation

For each detected task, provide:
- Brief, actionable title (3-8 words)
- Clear description with acceptance criteria
- Confidence score (0.0-1.0)
- Urgency level (low/medium/high/critical)
- Effort estimate (small/medium/large)
- Category (feature/bug/refactor/research/docs/test/deployment)
- Suggested assignee if mentioned

Response format (JSON):
{{
    "has_tasks": true/false,
    "tasks": [
        {{
            "title": "Implement user authentication",
            "description": "Create JWT-based auth system with login/logout/register endpoints",
            "confidence_score": 0.9,
            "urgency": "high",
            "estimated_effort": "large",
            "category": "feature",
            "context": "user_request",
            "suggested_assignee": "Backend Developer",
            "reasoning": "Explicit request for authentication implementation with clear requirements"
        }}
    ],
    "conversation_summary": "User requested implementation of authentication system",
    "project_context": "web_application_development",
    "confidence_score": 0.85
}}

IMPORTANT: Only detect genuine actionable work items. Avoid creating tasks for:
- Simple questions that just need answers
- Casual conversations or discussions
- Already completed work
- Vague ideas without clear requirements
"""

    async def detect_tasks_in_message(
        self, 
        user_message: str,
        conversation_context: Optional[str] = None,
        project_context: Optional[str] = None
    ) -> TaskDetectionResult:
        """
        Analyze a user message to detect potential tasks.
        
        Args:
            user_message: The user's message to analyze
            conversation_context: Previous conversation context
            project_context: Current project context
            
        Returns:
            TaskDetectionResult with detected tasks and analysis
        """
        with safe_span("task_detection_analysis") as span:
            safe_set_attribute(span, "message_length", len(user_message))
            safe_set_attribute(span, "has_context", bool(conversation_context))
            
            try:
                # Check cache first
                cache_key = self._get_cache_key(user_message, conversation_context)
                if cache_key in self._detection_cache:
                    logger.debug("📋 Using cached task detection result")
                    return self._detection_cache[cache_key]
                
                # Prepare the prompt
                prompt = self.detection_prompt.format(
                    user_message=user_message,
                    conversation_context=conversation_context or "No previous context available",
                    project_context=project_context or "General development project"
                )
                
                # Get AI analysis
                async with get_llm_client() as client:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",  # Fast model for task detection
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert project manager and task analyst. Respond only with valid JSON."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.1,  # Low temperature for consistent analysis
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content
                
                # Parse the AI response
                try:
                    analysis_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Failed to parse AI response as JSON: {e}")
                    # Fallback to heuristic analysis
                    return await self._fallback_heuristic_analysis(user_message)
                
                # Create result object
                tasks = []
                for task_data in analysis_data.get("tasks", []):
                    try:
                        task = TaskCandidate(
                            title=task_data.get("title", "Untitled Task"),
                            description=task_data.get("description", ""),
                            confidence_score=float(task_data.get("confidence_score", 0.5)),
                            urgency=task_data.get("urgency", "medium"),
                            estimated_effort=task_data.get("estimated_effort", "medium"),
                            category=task_data.get("category", "feature"),
                            context=task_data.get("context", "user_message"),
                            suggested_assignee=task_data.get("suggested_assignee", "User"),
                            reasoning=task_data.get("reasoning", "AI detected actionable task")
                        )
                        tasks.append(task)
                    except Exception as e:
                        logger.warning(f"⚠️ Error parsing task data: {e}")
                        continue
                
                result = TaskDetectionResult(
                    has_tasks=analysis_data.get("has_tasks", False),
                    tasks=tasks,
                    conversation_summary=analysis_data.get("conversation_summary", ""),
                    project_context=analysis_data.get("project_context", ""),
                    confidence_score=float(analysis_data.get("confidence_score", 0.5))
                )
                
                # Cache the result
                self._detection_cache[cache_key] = result
                
                safe_set_attribute(span, "tasks_detected", len(tasks))
                safe_set_attribute(span, "overall_confidence", result.confidence_score)
                safe_set_attribute(span, "has_tasks", result.has_tasks)
                
                logger.info(f"🔍 Task detection complete: {len(tasks)} tasks detected "
                           f"(confidence: {result.confidence_score:.2f})")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Error in task detection analysis: {e}")
                # Fallback to heuristic analysis
                return await self._fallback_heuristic_analysis(user_message)
    
    async def create_detected_tasks(
        self,
        detection_result: TaskDetectionResult,
        project_id: Optional[str] = None,
        auto_threshold: float = 0.8
    ) -> List[Dict[str, str]]:
        """
        Create tasks in Archon based on detection results.
        
        Args:
            detection_result: Result from task detection
            project_id: Target project ID (will detect if not provided)
            auto_threshold: Confidence threshold for auto-creation
            
        Returns:
            List of created task IDs and statuses
        """
        with safe_span("create_detected_tasks") as span:
            created_tasks = []
            
            if not detection_result.has_tasks:
                logger.debug("📝 No tasks detected, skipping creation")
                return created_tasks
            
            # Get or detect project context
            if not project_id:
                project_id = await self._detect_current_project(detection_result.project_context)
            
            if not project_id:
                logger.warning("⚠️ No project context found, cannot create tasks")
                return created_tasks
            
            try:
                # Import here to avoid circular imports
                from ..unified_archon_mcp import ArchonMCPCoordinator
                coordinator = ArchonMCPCoordinator()
                
                for task in detection_result.tasks:
                    try:
                        # Determine if task should be auto-created or suggested
                        if task.confidence_score >= auto_threshold:
                            # Auto-create high-confidence tasks
                            result = await coordinator.create_task(
                                project_id=project_id,
                                title=task.title,
                                description=f"{task.description}\n\nAuto-detected from user message.\nReasoning: {task.reasoning}",
                                assignee=task.suggested_assignee,
                                task_order=self._get_priority_order(task.urgency),
                                feature=task.category
                            )
                            
                            if result.get("success"):
                                created_tasks.append({
                                    "task_id": result.get("task_id"),
                                    "title": task.title,
                                    "status": "auto_created",
                                    "confidence": task.confidence_score
                                })
                                logger.info(f"✅ Auto-created task: {task.title}")
                            else:
                                logger.error(f"❌ Failed to create task: {result.get('error')}")
                                
                        else:
                            # Store lower-confidence tasks as suggestions
                            await self._store_task_suggestion(task, project_id)
                            created_tasks.append({
                                "task_id": f"suggestion_{task.title.replace(' ', '_').lower()}",
                                "title": task.title,
                                "status": "suggested",
                                "confidence": task.confidence_score
                            })
                            logger.info(f"💡 Stored task suggestion: {task.title}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error creating task '{task.title}': {e}")
                        continue
                
                safe_set_attribute(span, "tasks_created", len(created_tasks))
                
            except Exception as e:
                logger.error(f"❌ Error in task creation process: {e}")
            
            return created_tasks
    
    # Private helper methods
    
    def _get_cache_key(self, message: str, context: Optional[str]) -> str:
        """Generate cache key for task detection."""
        import hashlib
        cache_data = f"{message}|{context or ''}"
        return hashlib.md5(cache_data.encode()).hexdigest()[:16]
    
    async def _fallback_heuristic_analysis(self, user_message: str) -> TaskDetectionResult:
        """Fallback heuristic analysis when AI is unavailable."""
        # Simple heuristic patterns for task detection
        task_indicators = [
            r'\b(?:can you|could you|please|implement|create|build|add|fix|update)\b',
            r'\b(?:need to|should|must|have to|want to)\b',
            r'\b(?:task|todo|feature|bug|issue|problem)\b',
            r'\b(?:develop|code|program|script|function)\b'
        ]
        
        has_task_indicators = any(
            re.search(pattern, user_message, re.IGNORECASE) 
            for pattern in task_indicators
        )
        
        if has_task_indicators:
            # Extract potential task title from message
            sentences = user_message.split('.')
            first_sentence = sentences[0].strip() if sentences else user_message[:100]
            
            task = TaskCandidate(
                title=f"Task from user message: {first_sentence[:50]}...",
                description=user_message,
                confidence_score=0.6,  # Medium confidence for heuristic
                urgency="medium",
                estimated_effort="medium",
                category="feature",
                suggested_assignee="User",
                reasoning="Heuristic pattern matching detected task indicators"
            )
            
            return TaskDetectionResult(
                has_tasks=True,
                tasks=[task],
                conversation_summary="User message contains potential task",
                project_context="general",
                confidence_score=0.6
            )
        else:
            return TaskDetectionResult(
                has_tasks=False,
                tasks=[],
                conversation_summary="No actionable tasks detected",
                project_context="general",
                confidence_score=0.9  # High confidence in no-task detection
            )
    
    async def _detect_current_project(self, project_context: str) -> Optional[str]:
        """Detect the current project context."""
        try:
            # Import here to avoid circular imports
            from ..unified_archon_mcp import ArchonMCPCoordinator
            coordinator = ArchonMCPCoordinator()
            
            # Get recent projects
            projects_result = await coordinator.list_projects()
            if not projects_result.get("success"):
                return None
                
            projects = projects_result.get("projects", [])
            if not projects:
                # Create a default project if none exists
                create_result = await coordinator.create_project(
                    title="General Development Tasks",
                    description="Auto-created project for task management from Claude conversations"
                )
                if create_result.get("success"):
                    return create_result.get("project_id")
                return None
            
            # Return the most recently updated project
            return projects[0].get("id") if projects else None
            
        except Exception as e:
            logger.error(f"❌ Error detecting current project: {e}")
            return None
    
    def _get_priority_order(self, urgency: str) -> int:
        """Convert urgency to task order priority."""
        urgency_map = {
            "critical": 100,
            "high": 80,
            "medium": 50,
            "low": 20
        }
        return urgency_map.get(urgency.lower(), 50)
    
    async def _store_task_suggestion(self, task: TaskCandidate, project_id: str):
        """Store a task suggestion for user review."""
        try:
            client = get_supabase_client()
            
            # Store in a task suggestions table (if it exists)
            # For now, log the suggestion
            logger.info(f"💡 Task suggestion stored: {task.title} "
                       f"(confidence: {task.confidence_score:.2f})")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store task suggestion: {e}")


# Global instance
_task_detection_service: Optional[TaskDetectionService] = None


def get_task_detection_service() -> TaskDetectionService:
    """Get global task detection service instance."""
    global _task_detection_service
    
    if _task_detection_service is None:
        _task_detection_service = TaskDetectionService()
    
    return _task_detection_service