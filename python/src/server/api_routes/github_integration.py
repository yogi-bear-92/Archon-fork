"""
GitHub Integration API Routes

Provides REST API endpoints for managing GitHub repository monitoring,
webhook handling, and automated maintenance tasks.
"""

import asyncio
import hmac
import hashlib
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.github_monitoring_service import GitHubMonitoringService, get_github_monitoring_service
from ..unified_archon_mcp import ArchonMCPCoordinator


class GitHubRepoConfig(BaseModel):
    """Configuration for GitHub repository monitoring."""
    repository_url: str = Field(..., description="Full GitHub repository URL")
    auto_docs: bool = Field(default=True, description="Enable automatic documentation generation")
    auto_changelog: bool = Field(default=True, description="Enable automatic changelog generation")
    auto_linting: bool = Field(default=False, description="Enable automatic linting on changes")
    webhook_secret: Optional[str] = Field(None, description="GitHub webhook secret for verification")
    monitored_paths: List[str] = Field(default=["src/", "docs/", "README.md"], description="Paths to monitor for changes")
    documentation_style: str = Field(default="comprehensive", description="Documentation generation style")
    linting_config: Dict[str, Any] = Field(default_factory=dict, description="Language-specific linting configuration")


class GitHubWebhookEvent(BaseModel):
    """GitHub webhook event payload."""
    action: str
    repository: Dict[str, Any]
    sender: Dict[str, Any]
    ref: Optional[str] = None
    commits: Optional[List[Dict[str, Any]]] = None
    pull_request: Optional[Dict[str, Any]] = None
    release: Optional[Dict[str, Any]] = None


class MonitoringStatus(BaseModel):
    """GitHub repository monitoring status."""
    repository: str
    is_active: bool
    last_processed: Optional[datetime]
    events_processed: int
    auto_docs_enabled: bool
    auto_changelog_enabled: bool
    auto_linting_enabled: bool
    webhook_configured: bool


router = APIRouter(prefix="/api/github", tags=["GitHub Integration"])


@router.post("/repositories", response_model=Dict[str, Any])
async def add_repository(
    config: GitHubRepoConfig,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Add a GitHub repository for monitoring."""
    try:
        result = await github_service.add_repository(
            repository_url=config.repository_url,
            auto_docs=config.auto_docs,
            auto_changelog=config.auto_changelog,
            auto_linting=config.auto_linting,
            webhook_secret=config.webhook_secret,
            monitored_paths=config.monitored_paths,
            documentation_style=config.documentation_style,
            linting_config=config.linting_config
        )
        
        return {
            "success": True,
            "message": "Repository added successfully",
            "repository": result["repository"],
            "monitoring_config": result["config"],
            "webhook_url": f"{request.base_url}api/github/webhook/{result['repository']}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add repository: {str(e)}")


@router.get("/repositories", response_model=List[MonitoringStatus])
async def list_repositories(
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """List all monitored GitHub repositories."""
    try:
        repositories = await github_service.get_monitored_repositories()
        
        return [
            MonitoringStatus(
                repository=repo["repository"],
                is_active=repo["is_active"],
                last_processed=repo.get("last_processed"),
                events_processed=repo.get("events_processed", 0),
                auto_docs_enabled=repo["config"].get("auto_docs", False),
                auto_changelog_enabled=repo["config"].get("auto_changelog", False),
                auto_linting_enabled=repo["config"].get("auto_linting", False),
                webhook_configured=bool(repo["config"].get("webhook_secret"))
            )
            for repo in repositories
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")


@router.get("/repositories/{repository_name}", response_model=MonitoringStatus)
async def get_repository_status(
    repository_name: str,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Get detailed status for a specific repository."""
    try:
        status = await github_service.get_repository_status(repository_name)
        if not status:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        return MonitoringStatus(
            repository=status["repository"],
            is_active=status["is_active"],
            last_processed=status.get("last_processed"),
            events_processed=status.get("events_processed", 0),
            auto_docs_enabled=status["config"].get("auto_docs", False),
            auto_changelog_enabled=status["config"].get("auto_changelog", False),
            auto_linting_enabled=status["config"].get("auto_linting", False),
            webhook_configured=bool(status["config"].get("webhook_secret"))
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repository status: {str(e)}")


@router.put("/repositories/{repository_name}", response_model=Dict[str, Any])
async def update_repository_config(
    repository_name: str,
    config: GitHubRepoConfig,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Update configuration for a monitored repository."""
    try:
        result = await github_service.update_repository_config(
            repository=repository_name,
            config_updates={
                "auto_docs": config.auto_docs,
                "auto_changelog": config.auto_changelog,
                "auto_linting": config.auto_linting,
                "webhook_secret": config.webhook_secret,
                "monitored_paths": config.monitored_paths,
                "documentation_style": config.documentation_style,
                "linting_config": config.linting_config
            }
        )
        
        return {
            "success": True,
            "message": "Repository configuration updated",
            "repository": repository_name,
            "updated_config": result["config"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update repository: {str(e)}")


@router.delete("/repositories/{repository_name}", response_model=Dict[str, Any])
async def remove_repository(
    repository_name: str,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Remove a repository from monitoring."""
    try:
        success = await github_service.remove_repository(repository_name)
        if not success:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        return {
            "success": True,
            "message": f"Repository {repository_name} removed from monitoring"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove repository: {str(e)}")


@router.post("/webhook/{repository_name}", response_model=Dict[str, Any])
async def handle_webhook(
    repository_name: str,
    request: Request,
    x_github_event: str = Header(..., alias="X-GitHub-Event"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Handle GitHub webhook events for a specific repository."""
    try:
        # Get raw body for signature verification
        body = await request.body()
        
        # Verify webhook signature if configured
        if x_hub_signature_256:
            repo_config = await github_service.get_repository_config(repository_name)
            if repo_config and repo_config.get("webhook_secret"):
                expected_signature = "sha256=" + hmac.new(
                    repo_config["webhook_secret"].encode(),
                    body,
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(x_hub_signature_256, expected_signature):
                    raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Add event type to payload
        payload["github_event"] = x_github_event
        payload["repository_name"] = repository_name
        
        # Process the webhook event
        result = await github_service.process_webhook_event(payload)
        
        return {
            "success": True,
            "message": "Webhook processed successfully",
            "event_type": x_github_event,
            "repository": repository_name,
            "processing_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {str(e)}")


@router.post("/repositories/{repository_name}/generate-docs", response_model=Dict[str, Any])
async def manual_docs_generation(
    repository_name: str,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Manually trigger documentation generation for a repository."""
    try:
        result = await github_service.generate_documentation(repository_name)
        
        return {
            "success": True,
            "message": "Documentation generation completed",
            "repository": repository_name,
            "generated_files": result.get("generated_files", []),
            "updated_files": result.get("updated_files", []),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate documentation: {str(e)}")


@router.post("/repositories/{repository_name}/generate-changelog", response_model=Dict[str, Any])
async def manual_changelog_generation(
    repository_name: str,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Manually trigger changelog generation for a repository."""
    try:
        result = await github_service.generate_changelog(repository_name)
        
        return {
            "success": True,
            "message": "Changelog generation completed",
            "repository": repository_name,
            "changelog_file": result.get("changelog_file"),
            "entries_added": result.get("entries_added", 0),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate changelog: {str(e)}")


@router.post("/repositories/{repository_name}/run-linting", response_model=Dict[str, Any])
async def manual_linting(
    repository_name: str,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Manually trigger linting for a repository."""
    try:
        result = await github_service.run_linting(repository_name)
        
        return {
            "success": True,
            "message": "Linting completed",
            "repository": repository_name,
            "linting_results": result.get("results", {}),
            "issues_found": result.get("issues_count", 0),
            "files_processed": result.get("files_processed", 0),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run linting: {str(e)}")


@router.get("/repositories/{repository_name}/events", response_model=List[Dict[str, Any]])
async def get_repository_events(
    repository_name: str,
    limit: int = 50,
    offset: int = 0,
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Get recent events for a repository."""
    try:
        events = await github_service.get_repository_events(
            repository=repository_name,
            limit=limit,
            offset=offset
        )
        
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repository events: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def github_integration_health(
    github_service: GitHubMonitoringService = Depends(get_github_monitoring_service)
):
    """Check GitHub integration health status."""
    try:
        status = await github_service.get_health_status()
        
        return {
            "success": True,
            "service": "GitHub Integration",
            "status": "healthy",
            "monitored_repositories": status.get("monitored_repositories", 0),
            "active_webhooks": status.get("active_webhooks", 0),
            "events_processed_today": status.get("events_today", 0),
            "last_event_processed": status.get("last_event"),
            "uptime": status.get("uptime", 0)
        }
    except Exception as e:
        return {
            "success": False,
            "service": "GitHub Integration", 
            "status": "unhealthy",
            "error": str(e)
        }


# Add the router to the main API
def setup_github_routes(app):
    """Setup GitHub integration routes in the main FastAPI app."""
    app.include_router(router)