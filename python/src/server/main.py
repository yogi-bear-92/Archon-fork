"""
FastAPI Backend for Archon Knowledge Engine

This is the main entry point for the Archon backend API.
It uses a modular approach with separate API modules for different functionality.

Modules:
- settings_api: Settings and credentials management
- mcp_api: MCP server management and tool execution
- knowledge_api: Knowledge base, crawling, and RAG operations
- projects_api: Project and task management with streaming
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_routes.agent_chat_api import router as agent_chat_router
from .api_routes.ai_tagging_api import ai_tagging_router
from .api_routes.bug_report_api import router as bug_report_router
from .api_routes.claude_flow_api import router as claude_flow_router
from .api_routes.hook_management_api import router as hook_management_router
from .api_routes.interactive_task_api import router as interactive_task_router
from .api_routes.internal_api import router as internal_router
from .api_routes.knowledge_api import router as knowledge_router
from .api_routes.mcp_api import router as mcp_router
from .api_routes.progress_api import router as progress_router
from .api_routes.projects_api import router as projects_router
from .api_routes.serena_api import router as serena_router
from .api_routes.serena_coordination_api import serena_coordination_router
from .api_routes.url_suggestion_api import router as url_suggestion_router
from .api_routes.github_integration import router as github_integration_router
from .api_routes.mcp_proxy_api import router as mcp_proxy_router

# Import modular API routers
from .api_routes.settings_api import router as settings_router

# Import Logfire configuration
from .config.logfire_config import api_logger, setup_logfire
from .services.background_task_manager import cleanup_task_manager
from .services.crawler_manager import cleanup_crawler, initialize_crawler

# Import utilities and core classes
from .services.credential_service import initialize_credentials

# Import URL detection middleware
from .middleware.url_detection_middleware import URLDetectionMiddleware

# Import missing dependencies that the modular APIs need
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError:
    # These are optional dependencies for full functionality
    AsyncWebCrawler = None
    BrowserConfig = None

# Logger will be initialized after credentials are loaded
logger = logging.getLogger(__name__)

# Set up logging configuration to reduce noise

# Override uvicorn's access log format to be less verbose
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.WARNING)  # Only log warnings and errors, not every request

# CrawlingContext has been replaced by CrawlerManager in services/crawler_manager.py

# Global flag to track if initialization is complete
_initialization_complete = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    global _initialization_complete
    _initialization_complete = False

    # Startup
    logger.info("🚀 Starting Archon backend...")

    try:
        # Validate configuration FIRST - check for anon vs service key
        from .config.config import get_config

        get_config()  # This will raise ConfigurationError if anon key detected

        # Initialize credentials from database FIRST - this is the foundation for everything else
        await initialize_credentials()

        # Now that credentials are loaded, we can properly initialize logging
        # This must happen AFTER credentials so LOGFIRE_ENABLED is set from database
        setup_logfire(service_name="archon-backend")

        # Now we can safely use the logger
        logger.info("✅ Credentials initialized")
        api_logger.info("🔥 Logfire initialized for backend")

        # Initialize crawling context
        try:
            await initialize_crawler()
        except Exception as e:
            api_logger.warning(f"Could not fully initialize crawling context: {str(e)}")

        # Make crawling context available to modules
        # Crawler is now managed by CrawlerManager

        api_logger.info("✅ Using polling for real-time updates")

        # Initialize prompt service
        try:
            from .services.prompt_service import prompt_service

            await prompt_service.load_prompts()
            api_logger.info("✅ Prompt service initialized")
        except Exception as e:
            api_logger.warning(f"Could not initialize prompt service: {e}")

        # Set the main event loop for background tasks
        try:
            from .services.background_task_manager import get_task_manager

            task_manager = get_task_manager()
            current_loop = asyncio.get_running_loop()
            task_manager.set_main_loop(current_loop)
            api_logger.info("✅ Main event loop set for background tasks")
        except Exception as e:
            api_logger.warning(f"Could not set main event loop: {e}")

        # Initialize URL detection service
        try:
            from .services.url_detection_service import get_url_detection_service
            
            url_detection_service = get_url_detection_service()
            api_logger.info("✅ URL detection service initialized")
        except Exception as e:
            api_logger.warning(f"Could not initialize URL detection service: {e}")

        # Initialize hook integration service
        try:
            from .services.hook_integration_service import get_hook_integration_service
            
            hook_service = get_hook_integration_service()
            api_logger.info("✅ Hook integration service initialized")
        except Exception as e:
            api_logger.warning(f"Could not initialize hook integration service: {e}")

        # Initialize interactive task service
        try:
            from .services.interactive_task_service import get_interactive_task_service
            
            interactive_task_service = get_interactive_task_service()
            api_logger.info("✅ Interactive task service initialized")
        except Exception as e:
            api_logger.warning(f"Could not initialize interactive task service: {e}")

        # Initialize GitHub monitoring service
        try:
            from .services.github_monitoring_service import get_github_monitoring_service
            
            github_monitoring_service = get_github_monitoring_service()
            api_logger.info("✅ GitHub monitoring service initialized")
        except Exception as e:
            api_logger.warning(f"Could not initialize GitHub monitoring service: {e}")

        # MCP Client functionality removed from architecture
        # Agents now use MCP tools directly

        # Mark initialization as complete
        _initialization_complete = True
        api_logger.info("🎉 Archon backend started successfully!")

    except Exception as e:
        api_logger.error(f"❌ Failed to start backend: {str(e)}")
        raise

    yield

    # Shutdown
    _initialization_complete = False
    api_logger.info("🛑 Shutting down Archon backend...")

    try:
        # MCP Client cleanup not needed

        # Cleanup crawling context
        try:
            await cleanup_crawler()
        except Exception as e:
            api_logger.warning("Could not cleanup crawling context", error=str(e))

        # Cleanup background task manager
        try:
            await cleanup_task_manager()
            api_logger.info("Background task manager cleaned up")
        except Exception as e:
            api_logger.warning("Could not cleanup background task manager", error=str(e))

        api_logger.info("✅ Cleanup completed")

    except Exception as e:
        api_logger.error(f"❌ Error during shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Archon Knowledge Engine API",
    description="Backend API for the Archon knowledge management and project automation platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add URL detection middleware
try:
    # URL detection is enabled by default but can be disabled via settings
    url_detection_enabled = os.getenv("URL_DETECTION_ENABLED", "true").lower() == "true"
    app.add_middleware(URLDetectionMiddleware, enabled=url_detection_enabled)
    
    if url_detection_enabled:
        api_logger.info("✅ URL detection middleware enabled")
    else:
        api_logger.info("ℹ️ URL detection middleware disabled")
except Exception as e:
    api_logger.warning(f"⚠️ Could not initialize URL detection middleware: {e}")


# Add middleware to skip logging for health checks
@app.middleware("http")
async def skip_health_check_logs(request, call_next):
    # Skip logging for health check endpoints
    if request.url.path in ["/health", "/api/health"]:
        # Temporarily suppress the log
        import logging

        logger = logging.getLogger("uvicorn.access")
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        response = await call_next(request)
        logger.setLevel(old_level)
        return response
    return await call_next(request)


# Include API routers
app.include_router(settings_router)
app.include_router(mcp_router)
# app.include_router(mcp_client_router)  # Removed - not part of new architecture
app.include_router(knowledge_router)
app.include_router(projects_router)
app.include_router(progress_router)
app.include_router(agent_chat_router)
app.include_router(claude_flow_router)
app.include_router(serena_router)
app.include_router(serena_coordination_router)
app.include_router(ai_tagging_router)
app.include_router(internal_router)
app.include_router(bug_report_router)
app.include_router(url_suggestion_router)
app.include_router(hook_management_router)
app.include_router(interactive_task_router)
app.include_router(github_integration_router)
app.include_router(mcp_proxy_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Archon Knowledge Engine API",
        "version": "1.0.0",
        "description": "Backend API for knowledge management and project automation",
        "status": "healthy",
        "modules": ["settings", "mcp", "knowledge", "projects", "url-detection"],
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint that indicates true readiness including credential loading."""
    from datetime import datetime

    # Check if initialization is complete
    if not _initialization_complete:
        return {
            "status": "initializing",
            "service": "archon-backend",
            "timestamp": datetime.now().isoformat(),
            "message": "Backend is starting up, credentials loading...",
            "ready": False,
        }

    # Check for required database schema
    schema_status = await _check_database_schema()
    if not schema_status["valid"]:
        return {
            "status": "migration_required",
            "service": "archon-backend",
            "timestamp": datetime.now().isoformat(),
            "ready": False,
            "migration_required": True,
            "message": schema_status["message"],
            "migration_instructions": "Open Supabase Dashboard → SQL Editor → Run: migration/add_source_url_display_name.sql",
            "schema_valid": False
        }

    return {
        "status": "healthy",
        "service": "archon-backend",
        "timestamp": datetime.now().isoformat(),
        "ready": True,
        "credentials_loaded": True,
        "schema_valid": True,
    }


# API health check endpoint (alias for /health at /api/health)
@app.get("/api/health")
async def api_health_check():
    """API health check endpoint - alias for /health."""
    return await health_check()


# Cache schema check result to avoid repeated database queries
_schema_check_cache = {"valid": None, "checked_at": 0}

async def _check_database_schema():
    """Check if required database schema exists - only for existing users who need migration."""
    import time

    # If we've already confirmed schema is valid, don't check again
    if _schema_check_cache["valid"] is True:
        return {"valid": True, "message": "Schema is up to date (cached)"}

    # If we recently failed, don't spam the database (wait at least 30 seconds)
    current_time = time.time()
    if (_schema_check_cache["valid"] is False and
        current_time - _schema_check_cache["checked_at"] < 30):
        return _schema_check_cache["result"]

    try:
        from .services.client_manager import get_supabase_client

        client = get_supabase_client()

        # Try to query the new columns directly - if they exist, schema is up to date
        test_query = client.table('archon_sources').select('source_url, source_display_name').limit(1).execute()

        # Cache successful result permanently
        _schema_check_cache["valid"] = True
        _schema_check_cache["checked_at"] = current_time

        return {"valid": True, "message": "Schema is up to date"}

    except Exception as e:
        error_msg = str(e).lower()

        # Log schema check error for debugging
        api_logger.debug(f"Schema check error: {type(e).__name__}: {str(e)}")

        # Check for specific error types based on PostgreSQL error codes and messages

        # Check for missing columns first (more specific than table check)
        missing_source_url = 'source_url' in error_msg and ('column' in error_msg or 'does not exist' in error_msg)
        missing_source_display = 'source_display_name' in error_msg and ('column' in error_msg or 'does not exist' in error_msg)

        # Also check for PostgreSQL error code 42703 (undefined column)
        is_column_error = '42703' in error_msg or 'column' in error_msg

        if (missing_source_url or missing_source_display) and is_column_error:
            result = {
                "valid": False,
                "message": "Database schema outdated - missing required columns from recent updates"
            }
            # Cache failed result with timestamp
            _schema_check_cache["valid"] = False
            _schema_check_cache["checked_at"] = current_time
            _schema_check_cache["result"] = result
            return result

        # Check for table doesn't exist (less specific, only if column check didn't match)
        # Look for relation/table errors specifically
        if ('relation' in error_msg and 'does not exist' in error_msg) or ('table' in error_msg and 'does not exist' in error_msg):
            # Table doesn't exist - not a migration issue, it's a setup issue
            return {"valid": True, "message": "Table doesn't exist - handled by startup error"}

        # Other errors don't necessarily mean migration needed
        result = {"valid": True, "message": f"Schema check inconclusive: {str(e)}"}
        # Don't cache inconclusive results - allow retry
        return result


# Export the app directly for uvicorn to use


def main():
    """Main entry point for running the server."""
    import uvicorn

    # Require ARCHON_SERVER_PORT to be set
    server_port = os.getenv("ARCHON_SERVER_PORT")
    if not server_port:
        raise ValueError(
            "ARCHON_SERVER_PORT environment variable is required. "
            "Please set it in your .env file or environment. "
            "Default value: 8181"
        )

    uvicorn.run(
        "src.server.main:app",
        host="0.0.0.0",
        port=int(server_port),
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
