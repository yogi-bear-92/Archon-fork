"""
Hook Integration Service for Claude Code Hooks

This service manages the integration between Claude Code hooks and
the Archon system, providing utilities for hook coordination and management.
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.server.config.logfire_config import get_logger, safe_span, safe_set_attribute
from .task_detection_service import get_task_detection_service
from src.server.unified_archon_mcp import ArchonMCPCoordinator
from .client_manager import get_supabase_client

logger = get_logger(__name__)


@dataclass
class HookExecutionResult:
    """Result of a hook execution."""
    hook_name: str
    success: bool
    execution_time_ms: float
    output: Optional[str] = None
    error: Optional[str] = None
    tasks_created: int = 0
    tasks_suggested: int = 0
    metadata: Optional[Dict[str, Any]] = None


class HookIntegrationService:
    """
    Service for managing Claude Code hook integrations with Archon.
    
    This service provides utilities for:
    - Executing hooks programmatically
    - Managing hook configurations
    - Tracking hook performance and results
    - Integrating hook results with Archon project management
    """
    
    def __init__(self):
        self.hooks_config_path = Path(__file__).parent.parent.parent / ".claude-hooks.json"
        self._execution_cache: Dict[str, HookExecutionResult] = {}
        
    async def execute_post_user_prompt_hook(
        self,
        user_message: str,
        conversation_context: Optional[str] = None,
        project_context: Optional[str] = None,
        async_execution: bool = True
    ) -> HookExecutionResult:
        """
        Execute the post-user-prompt hook with the given message.
        
        Args:
            user_message: The user's message to analyze
            conversation_context: Optional conversation context
            project_context: Optional project context
            async_execution: Whether to execute asynchronously
            
        Returns:
            HookExecutionResult with execution details
        """
        with safe_span("hook_execution_post_user_prompt") as span:
            safe_set_attribute(span, "message_length", len(user_message))
            safe_set_attribute(span, "has_context", bool(conversation_context))
            safe_set_attribute(span, "async_execution", async_execution)
            
            start_time = datetime.now()
            
            try:
                # Prepare hook arguments
                hook_script = Path(__file__).parent.parent / "hooks" / "post_user_prompt_hook.py"
                args = ["python3", str(hook_script), user_message]
                
                if conversation_context:
                    args.append(json.dumps(conversation_context) if isinstance(conversation_context, dict) else conversation_context)
                
                if project_context:
                    args.append(project_context)
                
                # Execute the hook
                if async_execution:
                    result = await self._execute_hook_async(args)
                else:
                    result = await self._execute_hook_sync(args)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Parse hook output
                hook_output = None
                tasks_created = 0
                tasks_suggested = 0
                metadata = None
                
                if result.stdout:
                    try:
                        hook_output = json.loads(result.stdout)
                        tasks_created = hook_output.get("tasks_created", 0)
                        tasks_suggested = hook_output.get("tasks_suggested", 0)
                        metadata = {
                            "detection_summary": hook_output.get("detection_summary"),
                            "total_detected": hook_output.get("total_detected", 0),
                            "overall_confidence": hook_output.get("overall_confidence", 0.0),
                            "created_tasks": hook_output.get("created_tasks", [])
                        }
                    except json.JSONDecodeError:
                        hook_output = result.stdout
                
                execution_result = HookExecutionResult(
                    hook_name="post-user-prompt",
                    success=result.returncode == 0,
                    execution_time_ms=execution_time,
                    output=hook_output,
                    error=result.stderr if result.stderr else None,
                    tasks_created=tasks_created,
                    tasks_suggested=tasks_suggested,
                    metadata=metadata
                )
                
                # Cache the result
                cache_key = f"post-user-prompt-{hash(user_message)}"
                self._execution_cache[cache_key] = execution_result
                
                # Log execution details
                safe_set_attribute(span, "execution_time_ms", execution_time)
                safe_set_attribute(span, "success", execution_result.success)
                safe_set_attribute(span, "tasks_created", tasks_created)
                safe_set_attribute(span, "tasks_suggested", tasks_suggested)
                
                if execution_result.success:
                    logger.info(f"🎯 Post-user-prompt hook completed successfully: "
                               f"{tasks_created} created, {tasks_suggested} suggested "
                               f"({execution_time:.1f}ms)")
                else:
                    logger.warning(f"⚠️ Post-user-prompt hook failed: {execution_result.error}")
                
                return execution_result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"❌ Error executing post-user-prompt hook: {e}")
                
                return HookExecutionResult(
                    hook_name="post-user-prompt",
                    success=False,
                    execution_time_ms=execution_time,
                    error=str(e)
                )
    
    async def _execute_hook_async(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute hook asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=args,
            returncode=process.returncode,
            stdout=stdout.decode() if stdout else None,
            stderr=stderr.decode() if stderr else None
        )
    
    async def _execute_hook_sync(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute hook synchronously."""
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=30
        )
    
    async def get_hooks_configuration(self) -> Dict[str, Any]:
        """Get the current hooks configuration."""
        try:
            if self.hooks_config_path.exists():
                with open(self.hooks_config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Hooks configuration file not found: {self.hooks_config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error reading hooks configuration: {e}")
            return {}
    
    async def update_hooks_configuration(self, config: Dict[str, Any]) -> bool:
        """Update the hooks configuration file."""
        try:
            with open(self.hooks_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("✅ Hooks configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating hooks configuration: {e}")
            return False
    
    async def enable_hook(self, hook_name: str) -> bool:
        """Enable a specific hook."""
        config = await self.get_hooks_configuration()
        if hook_name in config.get("hooks", {}):
            config["hooks"][hook_name]["enabled"] = True
            return await self.update_hooks_configuration(config)
        return False
    
    async def disable_hook(self, hook_name: str) -> bool:
        """Disable a specific hook."""
        config = await self.get_hooks_configuration()
        if hook_name in config.get("hooks", {}):
            config["hooks"][hook_name]["enabled"] = False
            return await self.update_hooks_configuration(config)
        return False
    
    async def get_hook_execution_history(self, hook_name: Optional[str] = None) -> List[HookExecutionResult]:
        """Get execution history for hooks."""
        if hook_name:
            return [result for result in self._execution_cache.values() if result.hook_name == hook_name]
        return list(self._execution_cache.values())
    
    async def clear_execution_cache(self) -> None:
        """Clear the execution result cache."""
        self._execution_cache.clear()
        logger.info("🧹 Hook execution cache cleared")
    
    async def test_hook_integration(self) -> Dict[str, Any]:
        """Test the hook integration system."""
        logger.info("🧪 Testing hook integration system...")
        
        # Test configuration loading
        config = await self.get_hooks_configuration()
        config_valid = bool(config and "hooks" in config)
        
        # Test hook script existence
        hook_script = Path(__file__).parent.parent / "hooks" / "post_user_prompt_hook.py"
        script_exists = hook_script.exists()
        
        # Test task detection service
        task_service_available = True
        try:
            get_task_detection_service()
        except Exception:
            task_service_available = False
        
        # Test Archon MCP coordinator
        archon_available = True
        try:
            ArchonMCPCoordinator()
        except Exception:
            archon_available = False
        
        # Run a simple test execution
        test_execution_success = False
        if script_exists and task_service_available:
            try:
                result = await self.execute_post_user_prompt_hook(
                    user_message="This is a test message for hook integration",
                    async_execution=False
                )
                test_execution_success = result.success
            except Exception as e:
                logger.warning(f"Test execution failed: {e}")
        
        test_results = {
            "configuration_valid": config_valid,
            "hook_script_exists": script_exists,
            "task_service_available": task_service_available,
            "archon_available": archon_available,
            "test_execution_success": test_execution_success,
            "overall_status": all([
                config_valid,
                script_exists,
                task_service_available,
                archon_available,
                test_execution_success
            ])
        }
        
        logger.info(f"🧪 Hook integration test results: {test_results}")
        return test_results


# Global instance
_hook_integration_service: Optional[HookIntegrationService] = None


def get_hook_integration_service() -> HookIntegrationService:
    """Get global hook integration service instance."""
    global _hook_integration_service
    
    if _hook_integration_service is None:
        _hook_integration_service = HookIntegrationService()
    
    return _hook_integration_service