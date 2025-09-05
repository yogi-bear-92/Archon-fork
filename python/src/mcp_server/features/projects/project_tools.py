"""
Simple project management tools for Archon MCP Server.

Provides separate, focused tools for each project operation.
No complex PRP examples - just straightforward project management.
"""

import asyncio
import json
import logging
from urllib.parse import urljoin

import httpx

from mcp.server.fastmcp import Context, FastMCP
from src.mcp_server.utils.error_handling import MCPErrorFormatter
from src.mcp_server.utils.timeout_config import (
    get_default_timeout,
    get_max_polling_attempts,
    get_polling_interval,
    get_polling_timeout,
)
from src.server.config.service_discovery import get_api_url

logger = logging.getLogger(__name__)


def register_project_tools(mcp: FastMCP):
    """Register individual project management tools with the MCP server."""

    @mcp.tool()
    async def check_duplicate_projects(
        ctx: Context,
        title: str,
        github_repo: str | None = None,
    ) -> str:
        """
        Check for existing projects with similar title or GitHub repository to avoid duplicates.

        Args:
            title: Project title to check for duplicates
            github_repo: GitHub repository URL to check for duplicates

        Returns:
            JSON with duplicate check results:
            {
                "has_duplicates": true/false,
                "similar_projects": [...],
                "recommendation": "message"
            }
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Get all projects
                response = await client.get(urljoin(api_url, "/api/projects"))
                
                if response.status_code == 200:
                    projects = response.json().get("projects", [])
                    similar_projects = []
                    
                    # Check for similar titles
                    for project in projects:
                        if title.lower() in project.get("title", "").lower():
                            similar_projects.append({
                                "id": project.get("id"),
                                "title": project.get("title"),
                                "github_repo": project.get("github_repo"),
                                "match_type": "title_similarity"
                            })
                    
                    # Check for same GitHub repo
                    if github_repo:
                        for project in projects:
                            if project.get("github_repo") == github_repo:
                                similar_projects.append({
                                    "id": project.get("id"),
                                    "title": project.get("title"),
                                    "github_repo": project.get("github_repo"),
                                    "match_type": "github_repo_exact"
                                })
                    
                    has_duplicates = len(similar_projects) > 0
                    recommendation = ""
                    
                    if has_duplicates:
                        if github_repo:
                            recommendation = f"Found {len(similar_projects)} similar projects. Consider using a more specific title or check if you meant to update an existing project."
                        else:
                            recommendation = f"Found {len(similar_projects)} similar projects. Consider providing a github_repo to better identify unique projects."
                    else:
                        recommendation = "No similar projects found. Safe to create new project."
                    
                    return json.dumps({
                        "has_duplicates": has_duplicates,
                        "similar_projects": similar_projects,
                        "recommendation": recommendation
                    })
                else:
                    return MCPErrorFormatter.format_error(
                        error_type="api_error",
                        message="Failed to fetch projects for duplicate check",
                        suggestion="Try again later or check API status",
                        http_status=response.status_code,
                    )
                    
        except Exception as e:
            logger.error(f"Error checking for duplicate projects: {e}")
            return MCPErrorFormatter.format_error(
                error_type="internal_error",
                message=f"Failed to check for duplicate projects: {str(e)}",
                suggestion="Try again later",
                http_status=500,
            )

    @mcp.tool()
    async def merge_duplicate_projects(
        ctx: Context,
        primary_project_id: str,
        duplicate_project_ids: list[str],
        merge_strategy: str = "consolidate",
    ) -> str:
        """
        Merge duplicate projects by consolidating tasks, documents, and data.

        Args:
            primary_project_id: ID of the project to keep as primary
            duplicate_project_ids: List of project IDs to merge into primary
            merge_strategy: Strategy for merging ("consolidate", "replace", "append")

        Returns:
            JSON with merge results:
            {
                "success": true,
                "merged_projects": [...],
                "consolidated_tasks": [...],
                "consolidated_docs": [...],
                "message": "Projects merged successfully"
            }
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Get primary project details
                primary_response = await client.get(
                    urljoin(api_url, f"/api/projects/{primary_project_id}")
                )
                if primary_response.status_code != 200:
                    return MCPErrorFormatter.format_error(
                        error_type="not_found",
                        message=f"Primary project {primary_project_id} not found",
                        suggestion="Check project ID and try again",
                        http_status=404,
                    )

                primary_project = primary_response.json()
                merged_projects = [primary_project_id]
                consolidated_tasks = []
                consolidated_docs = []

                # Process each duplicate project
                for duplicate_id in duplicate_project_ids:
                    # Get duplicate project details
                    dup_response = await client.get(
                        urljoin(api_url, f"/api/projects/{duplicate_id}")
                    )
                    if dup_response.status_code != 200:
                        logger.warning(f"Duplicate project {duplicate_id} not found, skipping")
                        continue

                    duplicate_project = dup_response.json()

                    # Get tasks from duplicate project
                    tasks_response = await client.get(
                        urljoin(api_url, f"/api/tasks?project_id={duplicate_id}")
                    )
                    if tasks_response.status_code == 200:
                        tasks = tasks_response.json().get("tasks", [])
                        for task in tasks:
                            # Update task to belong to primary project
                            task["project_id"] = primary_project_id
                            consolidated_tasks.append(task)

                    # Get documents from duplicate project
                    docs_response = await client.get(
                        urljoin(api_url, f"/api/documents?project_id={duplicate_id}")
                    )
                    if docs_response.status_code == 200:
                        docs = docs_response.json().get("documents", [])
                        for doc in docs:
                            # Update document to belong to primary project
                            doc["project_id"] = primary_project_id
                            consolidated_docs.append(doc)

                    # Merge project data
                    if merge_strategy == "consolidate":
                        # Merge features and data
                        primary_features = primary_project.get("features", {})
                        dup_features = duplicate_project.get("features", {})
                        primary_project["features"] = {**primary_features, **dup_features}

                        primary_data = primary_project.get("data", {})
                        dup_data = duplicate_project.get("data", {})
                        primary_project["data"] = {**primary_data, **dup_data}

                        # Merge technical and business sources
                        primary_tech_sources = primary_project.get("technical_sources", [])
                        dup_tech_sources = duplicate_project.get("technical_sources", [])
                        primary_project["technical_sources"] = list(set(primary_tech_sources + dup_tech_sources))

                        primary_business_sources = primary_project.get("business_sources", [])
                        dup_business_sources = duplicate_project.get("business_sources", [])
                        primary_project["business_sources"] = list(set(primary_business_sources + dup_business_sources))

                    merged_projects.append(duplicate_id)

                # Update primary project with merged data
                update_response = await client.put(
                    urljoin(api_url, f"/api/projects/{primary_project_id}"),
                    json={
                        "features": primary_project.get("features", {}),
                        "data": primary_project.get("data", {}),
                        "technical_sources": primary_project.get("technical_sources", []),
                        "business_sources": primary_project.get("business_sources", []),
                    }
                )

                if update_response.status_code != 200:
                    return MCPErrorFormatter.format_error(
                        error_type="update_failed",
                        message="Failed to update primary project with merged data",
                        suggestion="Try again later",
                        http_status=500,
                    )

                # Move tasks to primary project
                for task in consolidated_tasks:
                    task_response = await client.put(
                        urljoin(api_url, f"/api/tasks/{task['id']}"),
                        json={"project_id": primary_project_id}
                    )
                    if task_response.status_code != 200:
                        logger.warning(f"Failed to move task {task['id']} to primary project")

                # Move documents to primary project
                for doc in consolidated_docs:
                    doc_response = await client.put(
                        urljoin(api_url, f"/api/documents/{doc['id']}"),
                        json={"project_id": primary_project_id}
                    )
                    if doc_response.status_code != 200:
                        logger.warning(f"Failed to move document {doc['id']} to primary project")

                # Delete duplicate projects
                for duplicate_id in duplicate_project_ids:
                    delete_response = await client.delete(
                        urljoin(api_url, f"/api/projects/{duplicate_id}")
                    )
                    if delete_response.status_code != 200:
                        logger.warning(f"Failed to delete duplicate project {duplicate_id}")

                return json.dumps({
                    "success": True,
                    "merged_projects": merged_projects,
                    "consolidated_tasks": len(consolidated_tasks),
                    "consolidated_docs": len(consolidated_docs),
                    "message": f"Successfully merged {len(duplicate_project_ids)} projects into {primary_project_id}"
                })

        except Exception as e:
            logger.error(f"Error merging projects: {e}")
            return MCPErrorFormatter.format_error(
                error_type="internal_error",
                message=f"Failed to merge projects: {str(e)}",
                suggestion="Try again later",
                http_status=500,
            )

    @mcp.tool()
    async def auto_detect_github_path(
        ctx: Context,
        project_title: str,
        base_path: str = None,
    ) -> str:
        """
        Automatically detect GitHub repository path based on project structure.

        Args:
            project_title: Title of the project to detect path for
            base_path: Base path to search from (defaults to current directory)

        Returns:
            JSON with detected GitHub path:
            {
                "github_repo": "https://github.com/user/repo",
                "confidence": "high|medium|low",
                "detection_method": "git_remote|directory_structure|fallback"
            }
        """
        try:
            import os
            import subprocess
            from pathlib import Path

            if not base_path:
                base_path = os.getcwd()

            base_path = Path(base_path).resolve()
            github_repo = None
            confidence = "low"
            detection_method = "fallback"

            # Method 1: Check git remote
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=base_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    github_repo = result.stdout.strip()
                    # Convert SSH to HTTPS if needed
                    if github_repo.startswith("git@github.com:"):
                        github_repo = github_repo.replace("git@github.com:", "https://github.com/").replace(".git", "")
                    confidence = "high"
                    detection_method = "git_remote"
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Method 2: Check for .git directory and infer from directory name
            if not github_repo and (base_path / ".git").exists():
                try:
                    # Try to get user from git config
                    user_result = subprocess.run(
                        ["git", "config", "user.name"],
                        cwd=base_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if user_result.returncode == 0 and user_result.stdout.strip():
                        username = user_result.stdout.strip().lower().replace(" ", "-")
                        repo_name = base_path.name.lower().replace(" ", "-")
                        github_repo = f"https://github.com/{username}/{repo_name}"
                        confidence = "medium"
                        detection_method = "directory_structure"
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                    pass

            # Method 3: Fallback based on project title
            if not github_repo:
                # Try to extract username from common patterns
                username = "user"  # Default fallback
                repo_name = project_title.lower().replace(" ", "-").replace("_", "-")
                github_repo = f"https://github.com/{username}/{repo_name}"
                confidence = "low"
                detection_method = "fallback"

            return json.dumps({
                "github_repo": github_repo,
                "confidence": confidence,
                "detection_method": detection_method,
                "base_path": str(base_path)
            })

        except Exception as e:
            logger.error(f"Error detecting GitHub path: {e}")
            return MCPErrorFormatter.format_error(
                error_type="internal_error",
                message=f"Failed to detect GitHub path: {str(e)}",
                suggestion="Provide github_repo manually",
                http_status=500,
            )

    @mcp.tool()
    async def create_project(
        ctx: Context,
        title: str,
        description: str = "",
        github_repo: str | None = None,
    ) -> str:
        """
        Create a new project with automatic AI assistance.

        The project creation starts a background process that generates PRP documentation
        and initial tasks based on the title and description.

        Args:
            title: Project title - should be descriptive (required)
            description: Project description explaining goals and scope
            github_repo: GitHub repository URL (e.g., "https://github.com/org/repo") - RECOMMENDED to avoid duplicate projects

        Returns:
            JSON with project details:
            {
                "success": true,
                "project": {...},
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Project created successfully"
            }

        Examples:
            # Simple project (GitHub repo recommended to avoid duplicates)
            create_project(
                title="Task Management API",
                description="RESTful API for managing tasks and projects"
            )

            # Project with GitHub integration (RECOMMENDED)
            create_project(
                title="OAuth2 Authentication System",
                description="Implement secure OAuth2 authentication with multiple providers",
                github_repo="https://github.com/myorg/auth-service"
            )
        """
        try:
            # Auto-detect GitHub repo if not provided
            if not github_repo:
                logger.info(f"GitHub repository not provided for project '{title}'. Attempting auto-detection...")
                try:
                    # Call auto_detect_github_path function
                    detection_result = await auto_detect_github_path(ctx, title)
                    detection_data = json.loads(detection_result)
                    if detection_data.get("github_repo"):
                        github_repo = detection_data["github_repo"]
                        logger.info(f"Auto-detected GitHub repository: {github_repo} (confidence: {detection_data.get('confidence', 'unknown')})")
                    else:
                        logger.warning(f"Could not auto-detect GitHub repository for project '{title}'. This may lead to duplicate projects.")
                except Exception as e:
                    logger.warning(f"Auto-detection failed for project '{title}': {e}")
            
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    urljoin(api_url, "/api/projects"),
                    json={"title": title, "description": description, "github_repo": github_repo},
                )

                if response.status_code == 200:
                    result = response.json()

                    # Handle async project creation
                    if "progress_id" in result:
                        # Poll for completion with proper error handling and backoff
                        max_attempts = get_max_polling_attempts()
                        polling_timeout = get_polling_timeout()

                        for attempt in range(max_attempts):
                            try:
                                # Exponential backoff
                                sleep_interval = get_polling_interval(attempt)
                                await asyncio.sleep(sleep_interval)

                                # Create new client with polling timeout
                                async with httpx.AsyncClient(
                                    timeout=polling_timeout
                                ) as poll_client:
                                    list_response = await poll_client.get(
                                        urljoin(api_url, "/api/projects")
                                    )
                                    list_response.raise_for_status()  # Raise on HTTP errors

                                    response_data = list_response.json()
                                    # Extract projects array from response
                                    projects = response_data.get("projects", [])
                                    # Find project with matching title created recently
                                    for proj in projects:
                                        if proj.get("title") == title:
                                            return json.dumps({
                                                "success": True,
                                                "project": proj,
                                                "project_id": proj["id"],
                                                "message": f"Project created successfully with ID: {proj['id']}",
                                            })

                            except httpx.RequestError as poll_error:
                                logger.warning(
                                    f"Polling attempt {attempt + 1}/{max_attempts} failed: {poll_error}"
                                )
                                if attempt == max_attempts - 1:  # Last attempt
                                    return MCPErrorFormatter.format_error(
                                        error_type="polling_timeout",
                                        message=f"Project creation polling failed after {max_attempts} attempts",
                                        details={
                                            "progress_id": result["progress_id"],
                                            "title": title,
                                            "last_error": str(poll_error),
                                        },
                                        suggestion="The project may still be creating. Use list_projects to check status",
                                    )
                            except Exception as poll_error:
                                logger.warning(
                                    f"Unexpected error during polling attempt {attempt + 1}: {poll_error}"
                                )

                        # If we couldn't find it after polling
                        return json.dumps({
                            "success": True,
                            "progress_id": result["progress_id"],
                            "message": f"Project creation in progress after {max_attempts} checks. Use list_projects to find it once complete.",
                        })
                    else:
                        # Direct response (shouldn't happen with current API)
                        return json.dumps({"success": True, "project": result})
                else:
                    return MCPErrorFormatter.from_http_error(response, "create project")

        except httpx.ConnectError as e:
            return MCPErrorFormatter.from_exception(
                e, "create project", {"title": title, "api_url": api_url}
            )
        except httpx.TimeoutException as e:
            return MCPErrorFormatter.from_exception(
                e, "create project", {"title": title, "timeout": str(timeout)}
            )
        except Exception as e:
            logger.error(f"Error creating project: {e}", exc_info=True)
            return MCPErrorFormatter.from_exception(e, "create project", {"title": title})

    @mcp.tool()
    async def list_projects(ctx: Context) -> str:
        """
        List all projects.

        Returns:
            JSON array of all projects with their basic information

        Example:
            list_projects()
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                # CRITICAL: Pass include_content=False for lightweight response
                response = await client.get(
                    urljoin(api_url, "/api/projects"),
                    params={"include_content": False}
                )

                if response.status_code == 200:
                    response_data = response.json()
                    # Response already includes projects array and count
                    return json.dumps({
                        "success": True,
                        "projects": response_data,
                        "count": response_data.get("count", 0),
                    })
                else:
                    return MCPErrorFormatter.from_http_error(response, "list projects")

        except httpx.RequestError as e:
            return MCPErrorFormatter.from_exception(e, "list projects", {"api_url": api_url})
        except Exception as e:
            logger.error(f"Error listing projects: {e}", exc_info=True)
            return MCPErrorFormatter.from_exception(e, "list projects")

    @mcp.tool()
    async def get_project(ctx: Context, project_id: str) -> str:
        """
        Get detailed information about a specific project.

        Args:
            project_id: UUID of the project

        Returns:
            JSON with complete project details

        Example:
            get_project(project_id="550e8400-e29b-41d4-a716-446655440000")
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(urljoin(api_url, f"/api/projects/{project_id}"))

                if response.status_code == 200:
                    project = response.json()
                    return json.dumps({"success": True, "project": project})
                elif response.status_code == 404:
                    return MCPErrorFormatter.format_error(
                        error_type="not_found",
                        message=f"Project {project_id} not found",
                        suggestion="Verify the project ID is correct",
                        http_status=404,
                    )
                else:
                    return MCPErrorFormatter.from_http_error(response, "get project")

        except httpx.RequestError as e:
            return MCPErrorFormatter.from_exception(e, "get project", {"project_id": project_id})
        except Exception as e:
            logger.error(f"Error getting project: {e}", exc_info=True)
            return MCPErrorFormatter.from_exception(e, "get project")

    @mcp.tool()
    async def delete_project(ctx: Context, project_id: str) -> str:
        """
        Delete a project.

        Args:
            project_id: UUID of the project to delete

        Returns:
            JSON confirmation of deletion

        Example:
            delete_project(project_id="550e8400-e29b-41d4-a716-446655440000")
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.delete(urljoin(api_url, f"/api/projects/{project_id}"))

                if response.status_code == 200:
                    return json.dumps({
                        "success": True,
                        "message": f"Project {project_id} deleted successfully",
                    })
                elif response.status_code == 404:
                    return MCPErrorFormatter.format_error(
                        error_type="not_found",
                        message=f"Project {project_id} not found",
                        suggestion="Verify the project ID is correct",
                        http_status=404,
                    )
                else:
                    return MCPErrorFormatter.from_http_error(response, "delete project")

        except httpx.RequestError as e:
            return MCPErrorFormatter.from_exception(e, "delete project", {"project_id": project_id})
        except Exception as e:
            logger.error(f"Error deleting project: {e}", exc_info=True)
            return MCPErrorFormatter.from_exception(e, "delete project")

    @mcp.tool()
    async def update_project(
        ctx: Context,
        project_id: str,
        title: str | None = None,
        description: str | None = None,
        github_repo: str | None = None,
    ) -> str:
        """
        Update a project's basic information.

        Args:
            project_id: UUID of the project to update
            title: New title (optional)
            description: New description (optional)
            github_repo: New GitHub repository URL (optional)

        Returns:
            JSON with updated project details

        Example:
            update_project(project_id="550e8400-e29b-41d4-a716-446655440000",
                         title="Updated Project Title")
        """
        try:
            api_url = get_api_url()
            timeout = get_default_timeout()

            # Build update payload with only provided fields
            update_data = {}
            if title is not None:
                update_data["title"] = title
            if description is not None:
                update_data["description"] = description
            if github_repo is not None:
                update_data["github_repo"] = github_repo

            if not update_data:
                return MCPErrorFormatter.format_error(
                    error_type="validation_error",
                    message="No fields to update",
                    suggestion="Provide at least one field to update (title, description, or github_repo)",
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.put(
                    urljoin(api_url, f"/api/projects/{project_id}"), json=update_data
                )

                if response.status_code == 200:
                    project = response.json()
                    return json.dumps({
                        "success": True,
                        "project": project,
                        "message": "Project updated successfully",
                    })
                elif response.status_code == 404:
                    return MCPErrorFormatter.format_error(
                        error_type="not_found",
                        message=f"Project {project_id} not found",
                        suggestion="Verify the project ID is correct",
                        http_status=404,
                    )
                else:
                    return MCPErrorFormatter.from_http_error(response, "update project")

        except httpx.RequestError as e:
            return MCPErrorFormatter.from_exception(e, "update project", {"project_id": project_id})
        except Exception as e:
            logger.error(f"Error updating project: {e}", exc_info=True)
            return MCPErrorFormatter.from_exception(e, "update project")
