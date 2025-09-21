"""Unit tests for task management tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context

from src.mcp_server.features.tasks.task_tools import register_task_tools


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
async def test_create_task_with_sources(mock_mcp, mock_context):
    """Test creating a task using manage_task."""
    register_task_tools(mock_mcp)

    # Get the manage_task function
    manage_task = mock_mcp._tools.get("manage_task")

    assert manage_task is not None, "manage_task tool not registered"

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "task": {"id": "task-123", "title": "Test Task"},
        "message": "Task created successfully",
    }

    with patch("src.mcp_server.features.tasks.task_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await manage_task(
            mock_context,
            action="create",
            project_id="project-123",
            title="Implement OAuth2",
            description="Add OAuth2 authentication",
            assignee="AI IDE Agent",
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["task_id"] == "task-123"

        # Verify the task was created properly
        call_args = mock_async_client.post.call_args
        sent_data = call_args[1]["json"]
        assert sent_data["title"] == "Implement OAuth2"
        assert sent_data["assignee"] == "AI IDE Agent"


@pytest.mark.asyncio
async def test_find_tasks_with_project_filter(mock_mcp, mock_context):
    """Test listing tasks with project-specific endpoint."""
    register_task_tools(mock_mcp)

    # Get the find_tasks function
    find_tasks = mock_mcp._tools.get("find_tasks")

    assert find_tasks is not None, "find_tasks tool not registered"

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "tasks": [
            {"id": "task-1", "title": "Task 1", "status": "todo"},
            {"id": "task-2", "title": "Task 2", "status": "doing"},
        ]
    }

    with patch("src.mcp_server.features.tasks.task_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await find_tasks(mock_context, filter_by="project", filter_value="project-123")

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert len(result_data["tasks"]) == 2

        # Verify project-specific endpoint was used
        call_args = mock_async_client.get.call_args
        assert "/api/projects/project-123/tasks" in call_args[0][0]


@pytest.mark.asyncio
async def test_find_tasks_with_status_filter(mock_mcp, mock_context):
    """Test listing tasks with status filter uses generic endpoint."""
    register_task_tools(mock_mcp)

    find_tasks = mock_mcp._tools.get("find_tasks")

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"id": "task-1", "title": "Task 1", "status": "todo"}]

    with patch("src.mcp_server.features.tasks.task_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await find_tasks(
            mock_context, filter_by="status", filter_value="todo", project_id="project-123"
        )

        result_data = json.loads(result)
        assert result_data["success"] is True

        # Verify generic endpoint with status param was used
        call_args = mock_async_client.get.call_args
        assert "/api/tasks" in call_args[0][0]
        assert call_args[1]["params"]["status"] == "todo"
        assert call_args[1]["params"]["project_id"] == "project-123"


@pytest.mark.asyncio
async def test_update_task_status(mock_mcp, mock_context):
    """Test updating task status."""
    register_task_tools(mock_mcp)

    # Get the manage_task function
    manage_task = mock_mcp._tools.get("manage_task")

    assert manage_task is not None, "manage_task tool not registered"

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "task": {"id": "task-123", "status": "doing"},
        "message": "Task updated successfully",
    }

    with patch("src.mcp_server.features.tasks.task_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.put.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await manage_task(
            mock_context, action="update", task_id="task-123", status="doing", assignee="User"
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "Task updated successfully" in result_data["message"]
        
        # Verify the PUT request was made with correct data
        call_args = mock_async_client.put.call_args
        sent_data = call_args[1]["json"]
        assert sent_data["status"] == "doing"
        assert sent_data["assignee"] == "User"


@pytest.mark.asyncio
async def test_update_task_no_fields(mock_mcp, mock_context):
    """Test updating task with no fields returns validation error."""
    register_task_tools(mock_mcp)

    # Get the manage_task function
    manage_task = mock_mcp._tools.get("manage_task")

    assert manage_task is not None, "manage_task tool not registered"

    # Call manage_task with update action but no fields to update
    result = await manage_task(mock_context, action="update", task_id="task-123")

    result_data = json.loads(result)
    assert result_data["success"] is False
    assert "error" in result_data
    assert isinstance(result_data["error"], dict)
    assert result_data["error"]["type"] == "validation_error"
    assert "No fields to update" in result_data["error"]["message"]


@pytest.mark.asyncio
async def test_delete_task_already_archived(mock_mcp, mock_context):
    """Test deleting an already archived task."""
    register_task_tools(mock_mcp)

    # Get the manage_task function
    manage_task = mock_mcp._tools.get("manage_task")

    assert manage_task is not None, "manage_task tool not registered"

    # Mock 400 response for already archived
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Task already archived"

    with patch("src.mcp_server.features.tasks.task_tools.httpx.AsyncClient") as mock_client:
        mock_async_client = AsyncMock()
        mock_async_client.delete.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_async_client

        result = await manage_task(mock_context, action="delete", task_id="task-123")

        result_data = json.loads(result)
        assert result_data["success"] is False
        # Error must be structured format (dict), not string
        assert "error" in result_data
        assert isinstance(result_data["error"], dict), (
            "Error should be structured format, not string"
        )
        assert result_data["error"]["type"] == "http_error"
        assert "http 400" in result_data["error"]["message"].lower()
