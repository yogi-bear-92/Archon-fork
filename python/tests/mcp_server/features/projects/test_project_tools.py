"""Unit tests for project management tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context

from src.mcp_server.features.projects.project_tools import register_project_tools


@pytest.fixture
def mock_mcp():
    """Create a mock MCP server for testing."""
    mock = MagicMock()
    # Store registered tools
    mock._tools = {}

    def tool_decorator():
        def decorator(func):
            mock._tools[func.__name__] = func
            return func

        return decorator

    mock.tool = tool_decorator
    return mock


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    return MagicMock(spec=Context)


@pytest.mark.asyncio
async def test_create_project_success(mock_mcp, mock_context):
    """Test successful project creation with polling."""
    register_project_tools(mock_mcp)

    # Get the manage_project function
    manage_project = mock_mcp._tools.get("manage_project")

    assert manage_project is not None, "manage_project tool not registered"

    # Mock initial creation response with progress_id
    mock_create_response = MagicMock()
    mock_create_response.status_code = 200
    mock_create_response.json.return_value = {
        "progress_id": "progress-123",
        "message": "Project creation started",
    }

    # Mock progress endpoint response for polling
    mock_progress_response = MagicMock()
    mock_progress_response.status_code = 200
    mock_progress_response.json.return_value = {
        "status": "completed",
        "result": {
            "project": {"id": "project-123", "title": "Test Project", "created_at": "2024-01-01"},
            "message": "Project created successfully"
        }
    }

    with patch("src.mcp_server.features.projects.project_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        # First call creates project, subsequent calls list projects
        mock_async_client.post.return_value = mock_create_response
        mock_async_client.get.return_value = mock_progress_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        # Mock sleep to speed up test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await manage_project(
                mock_context,
                action="create",
                title="Test Project",
                description="A test project",
                github_repo="https://github.com/test/repo",
            )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["project"]["id"] == "project-123"
        assert result_data["project_id"] == "project-123"
        assert "Project created successfully" in result_data["message"]


@pytest.mark.asyncio
async def test_create_project_direct_response(mock_mcp, mock_context):
    """Test project creation with direct response (no polling)."""
    register_project_tools(mock_mcp)

    manage_project = mock_mcp._tools.get("manage_project")

    # Mock direct creation response (no progress_id)
    mock_create_response = MagicMock()
    mock_create_response.status_code = 200
    mock_create_response.json.return_value = {
        "project": {"id": "project-123", "title": "Test Project"},
        "message": "Project created immediately",
    }

    with patch("src.mcp_server.features.projects.project_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_create_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await manage_project(mock_context, action="create", title="Test Project")

        result_data = json.loads(result)
        assert result_data["success"] is True
        # Direct response returns the project directly
        assert "project" in result_data


@pytest.mark.asyncio
async def test_find_projects_success(mock_mcp, mock_context):
    """Test listing projects."""
    register_project_tools(mock_mcp)

    # Get the find_projects function
    find_projects = mock_mcp._tools.get("find_projects")

    assert find_projects is not None, "find_projects tool not registered"

    # Mock HTTP response - API returns dict with projects array
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "projects": [
            {"id": "proj-1", "title": "Project 1", "created_at": "2024-01-01"},
            {"id": "proj-2", "title": "Project 2", "created_at": "2024-01-02"},
        ],
        "count": 2
    }

    with patch("src.mcp_server.features.projects.project_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await find_projects(mock_context)

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["projects"]) == 2
        assert result_data["count"] == 2


@pytest.mark.asyncio
async def test_get_project_not_found(mock_mcp, mock_context):
    """Test getting a non-existent project."""
    register_project_tools(mock_mcp)

    # Get the find_projects function (used for getting single project)
    find_projects = mock_mcp._tools.get("find_projects")

    assert find_projects is not None, "find_projects tool not registered"

    # Mock 404 response
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Project not found"

    with patch("src.mcp_server.features.projects.project_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await find_projects(mock_context, project_id="non-existent")

        result_data = json.loads(result)
        assert result_data["success"] is False
        # Error must be structured format (dict), not string
        assert "error" in result_data
        assert isinstance(result_data["error"], dict), (
            "Error should be structured format, not string"
        )
        assert result_data["error"]["type"] == "not_found"
        assert "not found" in result_data["error"]["message"].lower()
