#!/usr/bin/env python3
"""
Post-User-Prompt Hook for Claude Code Integration

This hook runs after each user prompt to evaluate if the message contains
new tasks that should be automatically created in Archon projects.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.services.interactive_task_service import get_interactive_task_service
from config.logfire_config import get_logger

logger = get_logger(__name__)


async def main():
    """
    Main hook execution function.
    
    Claude Code will call this script with the user's message as an argument.
    We analyze the message for potential tasks and create them in Archon if detected.
    """
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            logger.warning("No user message provided to post-user-prompt hook")
            return
        
        user_message = sys.argv[1]
        
        # Optional: Get conversation context if available
        conversation_context = None
        if len(sys.argv) > 2:
            try:
                conversation_context = json.loads(sys.argv[2])
            except json.JSONDecodeError:
                conversation_context = sys.argv[2]  # Treat as plain text
        
        # Optional: Get project context if available
        project_context = None
        if len(sys.argv) > 3:
            project_context = sys.argv[3]
        
        logger.info(f"🎣 Post-user-prompt hook triggered for message: {user_message[:100]}...")
        
        # Initialize interactive task service
        interactive_service = get_interactive_task_service()
        
        # Detect potential tasks and initiate interactive creation
        result = await interactive_service.detect_and_initiate_task_creation(
            user_message=user_message,
            conversation_context=conversation_context,
            project_context=project_context
        )
        
        if not result.get("has_potential_tasks", False):
            logger.info("📝 No actionable tasks detected in user message")
            
            # Output minimal result for Claude Code
            output = {
                "hook": "post-user-prompt",
                "success": True,
                "has_potential_tasks": False,
                "message": result.get("message", "No actionable tasks detected"),
                "action_required": False
            }
            print(json.dumps(output, indent=2))
            return
        
        # Task detected - output result for Claude Code to ask user
        primary_task = result.get("primary_task", {})
        logger.info(f"🎯 Detected potential task: {primary_task.get('title', 'Unknown')} (confidence: {primary_task.get('confidence', 0):.2f})")
        
        # Output structured result for Claude Code
        output = {
            "hook": "post-user-prompt",
            "success": True,
            "has_potential_tasks": True,
            "session_id": result.get("session_id"),
            "action_required": True,
            "user_confirmation_needed": True,
            "confirmation_question": result.get("confirmation_question"),
            "primary_task": primary_task,
            "additional_tasks_count": len(result.get("additional_tasks", [])),
            "interactive_session_active": True
        }
        print(json.dumps(output, indent=2))
    
    except Exception as e:
        logger.error(f"❌ Error in post-user-prompt hook: {e}", exc_info=True)
        # Output error for Claude Code
        error_result = {
            "hook": "post-user-prompt",
            "success": False,
            "error": str(e),
            "tasks_created": 0,
            "tasks_suggested": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())