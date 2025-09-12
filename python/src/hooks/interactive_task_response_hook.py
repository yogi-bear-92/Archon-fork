#!/usr/bin/env python3
"""
Interactive Task Response Hook for Claude Code

This hook processes user responses in interactive task creation sessions.
It's called when a user responds to task creation confirmations or questions.
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
    Main hook execution function for processing user responses.
    
    Expected arguments:
    1. session_id - Active task creation session ID
    2. user_response - User's response to confirmation or questions
    3. (optional) context - Additional context information
    """
    try:
        # Parse command line arguments
        if len(sys.argv) < 3:
            logger.warning("Insufficient arguments for interactive task response hook")
            print(json.dumps({
                "hook": "interactive-task-response",
                "success": False,
                "error": "Missing required arguments: session_id and user_response",
                "usage": "python3 interactive_task_response_hook.py <session_id> <user_response> [context]"
            }, indent=2))
            sys.exit(1)
        
        session_id = sys.argv[1]
        user_response = sys.argv[2]
        
        # Optional context
        context = None
        if len(sys.argv) > 3:
            context = sys.argv[3]
        
        logger.info(f"🎯 Interactive task response hook triggered for session: {session_id}")
        logger.info(f"📝 User response: {user_response[:100]}...")
        
        # Initialize interactive task service
        interactive_service = get_interactive_task_service()
        
        # Process the user's response
        result = await interactive_service.process_user_response(
            session_id=session_id,
            user_response=user_response
        )
        
        # Determine the type of response based on the result
        if result.get("error"):
            logger.error(f"❌ Error processing user response: {result['error']}")
            output = {
                "hook": "interactive-task-response",
                "success": False,
                "error": result["error"],
                "session_id": session_id,
                "session_active": result.get("session_active", False)
            }
        
        elif result.get("state") == "declined":
            logger.info("🚫 User declined task creation")
            output = {
                "hook": "interactive-task-response",
                "success": True,
                "session_active": False,
                "state": "declined",
                "message": result.get("message", "Task creation cancelled by user"),
                "action_required": False
            }
        
        elif result.get("state") == "task_created":
            logger.info(f"✅ Task successfully created: {result.get('task_created', {}).get('task_id')}")
            output = {
                "hook": "interactive-task-response",
                "success": True,
                "session_active": False,
                "state": "task_created",
                "message": result.get("message", "Task created successfully!"),
                "task_created": result.get("task_created"),
                "action_required": False
            }
        
        elif result.get("state") in ["gathering_details", "final_confirmation"]:
            logger.info(f"🔄 Session continuing in state: {result.get('state')}")
            output = {
                "hook": "interactive-task-response",
                "success": True,
                "session_active": True,
                "session_id": session_id,
                "state": result.get("state"),
                "message": result.get("message", ""),
                "questions": result.get("questions"),
                "task_preview": result.get("task_preview"),
                "action_required": True,
                "next_step": _get_next_step_instructions(result.get("state"))
            }
        
        else:
            # Unknown state
            logger.warning(f"⚠️ Unknown session state: {result.get('state')}")
            output = {
                "hook": "interactive-task-response",
                "success": True,
                "session_active": result.get("session_active", False),
                "session_id": session_id,
                "state": result.get("state", "unknown"),
                "message": result.get("message", "Session state unclear"),
                "action_required": result.get("session_active", False)
            }
        
        # Output result for Claude Code to parse
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        logger.error(f"❌ Error in interactive task response hook: {e}", exc_info=True)
        # Output error for Claude Code
        error_result = {
            "hook": "interactive-task-response",
            "success": False,
            "error": str(e),
            "session_id": sys.argv[1] if len(sys.argv) > 1 else None,
            "session_active": False
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def _get_next_step_instructions(state: str) -> str:
    """Get instructions for what the user should do next."""
    if state == "gathering_details":
        return "Please answer the questions above to help define the task requirements."
    elif state == "final_confirmation":
        return "Review the task details and respond with 'yes' to create or 'no' to cancel."
    else:
        return "Please respond to continue the task creation process."


if __name__ == "__main__":
    asyncio.run(main())