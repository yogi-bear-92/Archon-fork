"""
MCP Server for Archon (Microservices Version)

This is the MCP server that uses HTTP calls to other services
instead of importing heavy dependencies directly. This significantly reduces
the container size from 1.66GB to ~150MB.

Modules:
- RAG Module: RAG queries, search, and source management via HTTP
- Project Module: Task and project management via HTTP
- Health & Session: Local operations

Note: Crawling and document upload operations are handled directly by the
API service and frontend, not through MCP tools.
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mcp.server.fastmcp import Context, FastMCP

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Configure logging FIRST before any imports that might use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/mcp_server.log", mode="a")
        if os.path.exists("/tmp")
        else logging.NullHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Import Logfire configuration
from src.server.config.logfire_config import mcp_logger, setup_logfire

# Import service client for HTTP calls
from src.server.services.mcp_service_client import get_mcp_service_client

# Import session management
from src.server.services.mcp_session_manager import get_session_manager

# Global initialization lock and flag
_initialization_lock = threading.Lock()
_initialization_complete = False
_shared_context = None

server_host = "0.0.0.0"  # Listen on all interfaces

# Require ARCHON_MCP_PORT to be set
mcp_port = os.getenv("ARCHON_MCP_PORT")
if not mcp_port:
    raise ValueError(
        "ARCHON_MCP_PORT environment variable is required. "
        "Please set it in your .env file or environment. "
        "Default value: 8051"
    )
server_port = int(mcp_port)


@dataclass
class ArchonContext:
    """
    Context for MCP server.
    No heavy dependencies - just service client for HTTP calls.
    """

    service_client: Any
    health_status: dict = None
    startup_time: float = None

    def __post_init__(self):
        if self.health_status is None:
            self.health_status = {
                "status": "healthy",
                "api_service": False,
                "agents_service": False,
                "last_health_check": None,
            }
        if self.startup_time is None:
            self.startup_time = time.time()


async def perform_health_checks(context: ArchonContext):
    """Perform health checks on dependent services via HTTP."""
    try:
        # Check dependent services
        service_health = await context.service_client.health_check()

        context.health_status["api_service"] = service_health.get("api_service", False)
        context.health_status["agents_service"] = service_health.get("agents_service", False)

        # Overall status
        all_critical_ready = context.health_status["api_service"]

        context.health_status["status"] = "healthy" if all_critical_ready else "degraded"
        context.health_status["last_health_check"] = datetime.now().isoformat()

        if not all_critical_ready:
            logger.warning(f"Health check failed: {context.health_status}")
        else:
            logger.info("Health check passed - dependent services healthy")

    except Exception as e:
        logger.error(f"Health check error: {e}")
        context.health_status["status"] = "unhealthy"
        context.health_status["last_health_check"] = datetime.now().isoformat()


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ArchonContext]:
    """
    Lifecycle manager - no heavy dependencies.
    """
    global _initialization_complete, _shared_context

    # Quick check without lock
    if _initialization_complete and _shared_context:
        logger.info("♻️ Reusing existing context for new SSE connection")
        yield _shared_context
        return

    # Acquire lock for initialization
    with _initialization_lock:
        # Double-check pattern
        if _initialization_complete and _shared_context:
            logger.info("♻️ Reusing existing context for new SSE connection")
            yield _shared_context
            return

        logger.info("🚀 Starting MCP server...")

        try:
            # Initialize session manager
            logger.info("🔐 Initializing session manager...")
            session_manager = get_session_manager()
            logger.info("✓ Session manager initialized")

            # Initialize service client for HTTP calls
            logger.info("🌐 Initializing service client...")
            service_client = get_mcp_service_client()
            logger.info("✓ Service client initialized")

            # Create context
            context = ArchonContext(service_client=service_client)

            # Perform initial health check
            await perform_health_checks(context)

            logger.info("✓ MCP server ready")

            # Store context globally
            _shared_context = context
            _initialization_complete = True

            yield context

        except Exception as e:
            logger.error(f"💥 Critical error in lifespan setup: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up resources
            logger.info("🧹 Cleaning up MCP server...")
            logger.info("✅ MCP server shutdown complete")


# Define MCP instructions for Claude Code and other clients
MCP_INSTRUCTIONS = """
# Archon MCP Server Instructions

## 🚨 CRITICAL RULES (ALWAYS FOLLOW)
1. **Task Management**: ALWAYS use Archon MCP tools for task management.
   - Combine with your local TODO tools for granular tracking
   - First TODO: Update Archon task status
   - Last TODO: Update Archon with findings/completion

2. **Research First**: Before implementing, use perform_rag_query and search_code_examples
3. **Task-Driven Development**: Never code without checking current tasks first

## 📋 Core Workflow

### Task Management Cycle
1. **Get current task**: `get_task(task_id="...")`
2. **Mark as doing**: `update_task(task_id="...", status="doing")`
3. **Research phase**:
   - `perform_rag_query(query="...", match_count=5)`
   - `search_code_examples(query="...", match_count=3)`
4. **Implementation**: Code based on research findings
5. **Mark for review**: `update_task(task_id="...", status="review")`
6. **Get next task**: `list_tasks(filter_by="status", filter_value="todo")`

### Available Task Functions
- `create_task(project_id, title, description, assignee="User", ...)`
- `list_tasks(filter_by="status", filter_value="todo", project_id=None)`
- `get_task(task_id)`
- `update_task(task_id, title=None, status=None, assignee=None, ...)`
- `delete_task(task_id)`

## 🏗️ Project Management

### Project Functions
- `create_project(title, description, github_repo=None)` - Auto-detects GitHub repo if not provided
- `check_duplicate_projects(title, github_repo=None)` - Check for existing similar projects
- `merge_duplicate_projects(primary_project_id, duplicate_project_ids, merge_strategy="consolidate")` - Merge duplicate projects
- `cleanup_empty_projects(dry_run=False)` - Delete projects with no tasks
- `auto_detect_github_path(project_title, base_path=None)` - Auto-detect GitHub repository path
- `list_projects()`
- `get_project(project_id)`
- `update_project(project_id, title=None, description=None, ...)`
- `delete_project(project_id)`

### Document Functions
- `create_document(project_id, title, document_type, content=None, ...)`
- `list_documents(project_id)`
- `get_document(project_id, doc_id)`
- `update_document(project_id, doc_id, title=None, content=None, ...)`
- `delete_document(project_id, doc_id)`

## 🔍 Research Patterns
- **Architecture patterns**: `perform_rag_query(query="[tech] architecture patterns", match_count=5)`
- **Code examples**: `search_code_examples(query="[feature] implementation", match_count=3)`
- **Source discovery**: `get_available_sources()`
- Keep match_count around 3-5 for focused results

## 📊 Task Status Flow
`todo` → `doing` → `review` → `done`
- Only ONE task in 'doing' status at a time
- Use 'review' for completed work awaiting validation
- Mark tasks 'done' only after verification

## 💾 Version Management
- `create_version(project_id, field_name, content, change_summary)`
- `list_versions(project_id, field_name=None)`
- `get_version(project_id, field_name, version_number)`
- `restore_version(project_id, field_name, version_number)`
- Field names: "docs", "features", "data", "prd"

## 🎯 Best Practices
1. **Atomic Tasks**: Create tasks that take 1-4 hours
2. **Clear Descriptions**: Include acceptance criteria in task descriptions
3. **Use Features**: Group related tasks with feature labels
4. **Add Sources**: Link relevant documentation to tasks
5. **Track Progress**: Update task status as you work
"""

# Initialize the main FastMCP server with fixed configuration
try:
    logger.info("🏗️ MCP SERVER INITIALIZATION:")
    logger.info("   Server Name: archon-mcp-server")
    logger.info("   Description: MCP server using HTTP calls")

    mcp = FastMCP(
        "archon-mcp-server",
        description="MCP server for Archon - uses HTTP calls to other services",
        instructions=MCP_INSTRUCTIONS,
        lifespan=lifespan,
        host=server_host,
        port=server_port,
    )
    logger.info("✓ FastMCP server instance created successfully")

except Exception as e:
    logger.error(f"✗ Failed to create FastMCP server: {e}")
    logger.error(traceback.format_exc())
    raise


# Health check endpoint
@mcp.tool()
async def health_check(ctx: Context) -> str:
    """
    Check health status of MCP server and dependencies.

    Returns:
        JSON with health status, uptime, and service availability
    """
    try:
        # Try to get the lifespan context
        context = getattr(ctx.request_context, "lifespan_context", None)

        if context is None:
            # Server starting up
            return json.dumps({
                "success": True,
                "status": "starting",
                "message": "MCP server is initializing...",
                "timestamp": datetime.now().isoformat(),
            })

        # Server is ready - perform health checks
        if hasattr(context, "health_status") and context.health_status:
            await perform_health_checks(context)

            return json.dumps({
                "success": True,
                "health": context.health_status,
                "uptime_seconds": time.time() - context.startup_time,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            return json.dumps({
                "success": True,
                "status": "ready",
                "message": "MCP server is running",
                "timestamp": datetime.now().isoformat(),
            })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return json.dumps({
            "success": False,
            "error": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        })


# Session management endpoint
@mcp.tool()
async def session_info(ctx: Context) -> str:
    """
    Get current and active session information.

    Returns:
        JSON with active sessions count and server uptime
    """
    try:
        session_manager = get_session_manager()

        # Build session info
        session_info_data = {
            "active_sessions": session_manager.get_active_session_count(),
            "session_timeout": session_manager.timeout,
        }

        # Add server uptime
        context = getattr(ctx.request_context, "lifespan_context", None)
        if context and hasattr(context, "startup_time"):
            session_info_data["server_uptime_seconds"] = time.time() - context.startup_time

        return json.dumps({
            "success": True,
            "session_management": session_info_data,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Session info failed: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to get session info: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        })


# Create a special tool that lists all other tools
@mcp.tool()
async def get_available_tools(ctx: Context) -> str:
    """
    Get list of all available tools in this MCP server.
    This provides the functionality of tools/list as a regular tool.
    
    Returns:
        JSON list of tools with their names, descriptions, and parameters
    """
    try:
        tools_info = []
        
        # Get all registered tools from FastMCP
        if hasattr(mcp, '_tools') and mcp._tools:
            for tool_name, tool_func in mcp._tools.items():
                # Skip this tool to avoid recursion
                if tool_name == 'get_available_tools':
                    continue
                    
                # Extract tool info
                tool_info = {
                    "name": tool_name,
                    "description": getattr(tool_func, '__doc__', 'No description available').strip().split('\n')[0],
                    "inputSchema": {
                        "type": "object", 
                        "properties": {},
                        "required": []
                    }
                }
                
                # Try to get parameter info from function signature
                import inspect
                try:
                    sig = inspect.signature(tool_func)
                    properties = {}
                    required = []
                    
                    for param_name, param in sig.parameters.items():
                        if param_name in ['ctx', 'context']:  # Skip context parameters
                            continue
                            
                        param_info = {"type": "string"}  # Default type
                        
                        # Check if parameter is required
                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)
                        
                        # Try to infer type from annotation
                        if param.annotation != inspect.Parameter.empty:
                            if param.annotation == int:
                                param_info["type"] = "integer"
                            elif param.annotation == float:
                                param_info["type"] = "number"
                            elif param.annotation == bool:
                                param_info["type"] = "boolean"
                                
                        properties[param_name] = param_info
                    
                    tool_info["inputSchema"]["properties"] = properties
                    tool_info["inputSchema"]["required"] = required
                    
                except Exception as e:
                    logger.debug(f"Could not inspect tool {tool_name}: {e}")
                
                tools_info.append(tool_info)
        
        return json.dumps({
            "success": True,
            "tools": tools_info,
            "count": len(tools_info),
            "source": "mcp_server"
        })
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return json.dumps({
            "success": False,
            "error": f"Failed to list tools: {str(e)}",
            "tools": [],
            "count": 0
        })


# Import and register modules
def register_modules():
    """Register all MCP tool modules."""
    logger.info("🔧 Registering MCP tool modules...")

    modules_registered = 0

    # Import and register RAG module (HTTP-based version)
    try:
        from src.mcp_server.modules.rag_module import register_rag_tools

        register_rag_tools(mcp)
        modules_registered += 1
        logger.info("✓ RAG module registered (HTTP-based)")
    except ImportError as e:
        logger.warning(f"⚠ RAG module not available: {e}")
    except Exception as e:
        logger.error(f"✗ Error registering RAG module: {e}")
        logger.error(traceback.format_exc())

    # Import and register all feature tools - separated and focused

    # Project Management Tools
    try:
        from src.mcp_server.features.projects import register_project_tools

        register_project_tools(mcp)
        modules_registered += 1
        logger.info("✓ Project tools registered")
    except ImportError as e:
        # Module not found - this is acceptable in modular architecture
        logger.warning(f"⚠ Project tools module not available (optional): {e}")
    except (SyntaxError, NameError, AttributeError) as e:
        # Code errors that should not be ignored
        logger.error(f"✗ Code error in project tools - MUST FIX: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to prevent running with broken code
    except Exception as e:
        # Unexpected errors during registration
        logger.error(f"✗ Failed to register project tools: {e}")
        logger.error(traceback.format_exc())
        # Don't raise - allow other modules to register

    # Task Management Tools
    try:
        from src.mcp_server.features.tasks import register_task_tools

        register_task_tools(mcp)
        modules_registered += 1
        logger.info("✓ Task tools registered")
    except ImportError as e:
        logger.warning(f"⚠ Task tools module not available (optional): {e}")
    except (SyntaxError, NameError, AttributeError) as e:
        logger.error(f"✗ Code error in task tools - MUST FIX: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"✗ Failed to register task tools: {e}")
        logger.error(traceback.format_exc())

    # Document Management Tools
    try:
        from src.mcp_server.features.documents import register_document_tools

        register_document_tools(mcp)
        modules_registered += 1
        logger.info("✓ Document tools registered")
    except ImportError as e:
        logger.warning(f"⚠ Document tools module not available (optional): {e}")
    except (SyntaxError, NameError, AttributeError) as e:
        logger.error(f"✗ Code error in document tools - MUST FIX: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"✗ Failed to register document tools: {e}")
        logger.error(traceback.format_exc())

    # Version Management Tools
    try:
        from src.mcp_server.features.documents import register_version_tools

        register_version_tools(mcp)
        modules_registered += 1
        logger.info("✓ Version tools registered")
    except ImportError as e:
        logger.warning(f"⚠ Version tools module not available (optional): {e}")
    except (SyntaxError, NameError, AttributeError) as e:
        logger.error(f"✗ Code error in version tools - MUST FIX: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"✗ Failed to register version tools: {e}")
        logger.error(traceback.format_exc())

    # Feature Management Tools
    try:
        from src.mcp_server.features.feature_tools import register_feature_tools

        register_feature_tools(mcp)
        modules_registered += 1
        logger.info("✓ Feature tools registered")
    except ImportError as e:
        logger.warning(f"⚠ Feature tools module not available (optional): {e}")
    except (SyntaxError, NameError, AttributeError) as e:
        logger.error(f"✗ Code error in feature tools - MUST FIX: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"✗ Failed to register feature tools: {e}")
        logger.error(traceback.format_exc())

    # Claude Flow Integration Tools
    try:
        from src.mcp_server.features.claude_flow import register_claude_flow_tools
        from src.server.services.mcp_service_client import get_mcp_service_client
        
        # Get HTTP client for Claude Flow service calls
        service_client = get_mcp_service_client()
        register_claude_flow_tools(mcp)
        modules_registered += 1
        logger.info("✓ Claude Flow integration tools registered")
    except ImportError as e:
        logger.warning(f"⚠ Claude Flow tools not available: {e}")
    except Exception as e:
        logger.error(f"✗ Failed to register Claude Flow tools: {e}")
        logger.error(traceback.format_exc())

    # AI Tagging Tools
    try:
        from src.mcp_server.features.ai_tagging.ai_tagging_tools import register_ai_tagging_tools

        register_ai_tagging_tools(mcp)
        modules_registered += 1
        logger.info("✓ AI Tagging tools registered")

    except Exception as e:
        logger.error(f"✗ Failed to register AI Tagging tools: {e}")
        logger.error(traceback.format_exc())

    logger.info(f"📦 Total modules registered: {modules_registered}")

    if modules_registered == 0:
        logger.error("💥 No modules were successfully registered!")
        raise RuntimeError("No MCP modules available")


# Register all modules when this file is imported
try:
    register_modules()
except Exception as e:
    logger.error(f"💥 Critical error during module registration: {e}")
    logger.error(traceback.format_exc())
    raise


def main():
    """Main entry point for the MCP server."""
    try:
        # Initialize Logfire first
        setup_logfire(service_name="archon-mcp-server")

        logger.info("🚀 Starting Archon MCP Server")
        logger.info("   Mode: Streamable HTTP")
        logger.info(f"   URL: http://{server_host}:{server_port}/mcp")

        mcp_logger.info("🔥 Logfire initialized for MCP server")
        mcp_logger.info(f"🌟 Starting MCP server - host={server_host}, port={server_port}")

        # Use streamable-http transport for persistent HTTP server operation
        transport_mode = os.getenv("TRANSPORT", "streamable-http").lower()
        if transport_mode == "streamable-http":
            mcp.run(transport="streamable-http")
        else:
            mcp.run(transport="stdio")

    except Exception as e:
        mcp_logger.error(f"💥 Fatal error in main - error={str(e)}, error_type={type(e).__name__}")
        logger.error(f"💥 Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 MCP server stopped by user")
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
