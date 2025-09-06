import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
                    from src.server.main import app
    import sys
    import asyncio
    import psutil
    import time
    from collections import defaultdict
        import threading
    from tests.mocks.serena.mock_serena_tools import MockClaudeFlowCoordination
"""Enhanced test configuration for Archon with Serena Claude Flow Expert Agent testing support."""

# Set test environment - always override to ensure test isolation
os.environ["TEST_MODE"] = "true"
os.environ["TESTING"] = "true"
# Set fake database credentials to prevent connection attempts
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "test-key"
# Set required port environment variables for ServiceDiscovery
os.environ["ARCHON_SERVER_PORT"] = "8181"
os.environ["ARCHON_MCP_PORT"] = "8051"
os.environ["ARCHON_AGENTS_PORT"] = "8052"

# Global patches that need to be active during module imports and app initialization
mock_client = MagicMock()
mock_table = MagicMock()
mock_select = MagicMock()
mock_execute = MagicMock()
mock_execute.data = []
mock_select.execute.return_value = mock_execute
mock_select.eq.return_value = mock_select
mock_select.order.return_value = mock_select
mock_table.select.return_value = mock_select
mock_client.table.return_value = mock_table

# Apply global patches immediately
_global_patches = [
    patch("supabase.create_client", return_value=mock_client),
    patch("src.server.services.client_manager.get_supabase_client", return_value=mock_client),
    patch("src.server.utils.get_supabase_client", return_value=mock_client),
]

for p in _global_patches:
    p.start()

@pytest.fixture(autouse=True)
def ensure_test_environment():
    """Ensure test environment is properly set for each test."""
    # Force test environment settings - this runs before each test
    os.environ["TEST_MODE"] = "true"
    os.environ["TESTING"] = "true"
    os.environ["SUPABASE_URL"] = "https://test.supabase.co"
    os.environ["SUPABASE_SERVICE_KEY"] = "test-key"
    os.environ["ARCHON_SERVER_PORT"] = "8181"
    os.environ["ARCHON_MCP_PORT"] = "8051"
    os.environ["ARCHON_AGENTS_PORT"] = "8052"
    yield

@pytest.fixture(autouse=True)
def prevent_real_db_calls():
    """Automatically prevent any real database calls in all tests."""
    # Create a mock client to use everywhere
    mock_client = MagicMock()

    # Mock table operations with chaining support
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_or = MagicMock()
    mock_execute = MagicMock()

    # Setup basic chaining
    mock_execute.data = []
    mock_or.execute.return_value = mock_execute
    mock_select.or_.return_value = mock_or
    mock_select.execute.return_value = mock_execute
    mock_select.eq.return_value = mock_select
    mock_select.order.return_value = mock_select
    mock_table.select.return_value = mock_select
    mock_table.insert.return_value.execute.return_value.data = [{"id": "test-id"}]
    mock_client.table.return_value = mock_table

    # Patch all the common ways to get a Supabase client
    with patch("supabase.create_client", return_value=mock_client):
        with patch("src.server.services.client_manager.get_supabase_client", return_value=mock_client):
            with patch("src.server.utils.get_supabase_client", return_value=mock_client):
                yield

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = MagicMock()

    # Mock table operations with chaining support
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_insert = MagicMock()
    mock_update = MagicMock()
    mock_delete = MagicMock()

    # Setup method chaining for select
    mock_select.execute.return_value.data = []
    mock_select.eq.return_value = mock_select
    mock_select.neq.return_value = mock_select
    mock_select.order.return_value = mock_select
    mock_select.limit.return_value = mock_select
    mock_table.select.return_value = mock_select

    # Setup method chaining for insert
    mock_insert.execute.return_value.data = [{"id": "test-id"}]
    mock_table.insert.return_value = mock_insert

    # Setup method chaining for update
    mock_update.execute.return_value.data = [{"id": "test-id"}]
    mock_update.eq.return_value = mock_update
    mock_table.update.return_value = mock_update

    # Setup method chaining for delete
    mock_delete.execute.return_value.data = []
    mock_delete.eq.return_value = mock_delete
    mock_table.delete.return_value = mock_delete

    # Make table() return the mock table
    mock_client.table.return_value = mock_table

    # Mock auth operations
    mock_client.auth = MagicMock()
    mock_client.auth.get_user.return_value = None

    # Mock storage operations
    mock_client.storage = MagicMock()

    return mock_client

@pytest.fixture
def client(mock_supabase_client):
    """FastAPI test client with mocked database."""
    # Patch all the ways Supabase client can be created
    with patch(
        "src.server.services.client_manager.get_supabase_client",
        return_value=mock_supabase_client,
    ):
        with patch(
            "src.server.utils.get_supabase_client",
            return_value=mock_supabase_client,
        ):
            with patch(
                "src.server.services.credential_service.create_client",
                return_value=mock_supabase_client,
            ):
                with patch("supabase.create_client", return_value=mock_supabase_client):
                    # Import app after patching to ensure mocks are used

                    return TestClient(app)

@pytest.fixture
def test_project():
    """Simple test project data."""
    return {"title": "Test Project", "description": "A test project for essential tests"}

@pytest.fixture
def test_task():
    """Simple test task data."""
    return {
        "title": "Test Task",
        "description": "A test task for essential tests",
        "status": "todo",
        "assignee": "User",
    }

@pytest.fixture
def test_knowledge_item():
    """Simple test knowledge item data."""
    return {
        "url": "https://example.com/test",
        "title": "Test Knowledge Item",
        "content": "This is test content for knowledge base",
        "source_id": "test-source",
    }

# Serena-specific test configuration and fixtures
@pytest.fixture
def serena_test_config():
    """Configuration for Serena Claude Flow Expert Agent tests."""
    return {
        "test_mode": True,
        "mock_mcp_tools": True,
        "mock_coordination": True,
        "performance_monitoring": True,
        "load_testing": False,
        "timeout_seconds": 30,
        "concurrent_operations": 10,
        "memory_limit_mb": 512
    }

@pytest.fixture(autouse=True)
def serena_test_isolation():
    """Ensure Serena tests are properly isolated."""
    # Reset any global state that might affect Serena tests

    # Clear any cached modules that might interfere
    modules_to_clear = [
        module for module in sys.modules.keys()
        if 'serena' in module.lower() or 'claude_flow' in module.lower()
    ]

    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    yield

    # Cleanup after test
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

@pytest.fixture
def performance_baseline():
    """Performance baseline expectations for Serena operations."""
    return {
        "semantic_analysis": {
            "max_response_time_ms": 500,
            "max_memory_usage_mb": 100,
            "min_success_rate": 0.95
        },
        "coordination": {
            "max_message_latency_ms": 50,
            "max_coordination_overhead_ms": 200,
            "min_success_rate": 0.98
        },
        "memory_operations": {
            "max_storage_time_ms": 100,
            "max_retrieval_time_ms": 50,
            "min_success_rate": 0.99
        }
    }

@pytest.fixture(scope="function")
def isolated_event_loop():
    """Provide an isolated event loop for async tests."""

    # Create new event loop for each test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    # Clean up
    try:
        # Cancel all remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Run loop until all tasks are cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""

    monitor_data = defaultdict(list)
    monitoring = {'active': True}

    def start_monitoring():

        def monitor():
            while monitoring['active']:
                memory_info = psutil.Process().memory_info()
                monitor_data['rss'].append(memory_info.rss)
                monitor_data['vms'].append(memory_info.vms)
                monitor_data['timestamp'].append(time.time())
                time.sleep(0.1)  # Sample every 100ms

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread

    monitoring_thread = start_monitoring()

    yield {
        'data': monitor_data,
        'get_peak_memory': lambda: max(monitor_data['rss']) if monitor_data['rss'] else 0,
        'get_memory_delta': lambda: (max(monitor_data['rss']) - min(monitor_data['rss'])) if len(monitor_data['rss']) > 1 else 0,
        'stop': lambda: monitoring.update({'active': False})
    }

    monitoring['active'] = False

@pytest.fixture
def coordination_test_environment():
    """Set up coordination testing environment."""

    # Create mock coordination system
    coordinator = MockClaudeFlowCoordination()

    yield {
        'coordinator': coordinator,
        'reset': coordinator.reset,
        'get_metrics': coordinator.get_performance_metrics,
        'get_agent_count': lambda: len(coordinator.agents),
        'get_message_count': lambda: len(coordinator.message_queue)
    }

    # Cleanup
    coordinator.reset()

# Test collection and filtering
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize Serena tests."""

    for item in items:
        # Add markers based on test location
        if "serena" in str(item.fspath):
            if "unit" in str(item.fspath):
                item.add_marker(pytest.mark.unit)
                item.add_marker(pytest.mark.serena)
            elif "integration" in str(item.fspath):
                item.add_marker(pytest.mark.integration)
                item.add_marker(pytest.mark.serena)
            elif "e2e" in str(item.fspath):
                item.add_marker(pytest.mark.e2e)
                item.add_marker(pytest.mark.serena)
            elif "performance" in str(item.fspath):
                item.add_marker(pytest.mark.performance)
                item.add_marker(pytest.mark.serena)
            elif "load" in str(item.fspath):
                item.add_marker(pytest.mark.load)
                item.add_marker(pytest.mark.serena)
                item.add_marker(pytest.mark.slow)

        # Add timeout markers for long-running tests
        if any(marker in str(item.fspath) for marker in ["load", "performance", "e2e"]):
            item.add_marker(pytest.mark.timeout(300))  # 5 minute timeout

        # Mark async tests
        if hasattr(item.function, 'pytestmark'):
            for mark in item.function.pytestmark:
                if mark.name == 'asyncio':
                    item.add_marker(pytest.mark.asyncio)

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "serena: mark test as part of Serena Claude Flow Expert Agent test suite"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (will be skipped in quick runs)"
    )

# Test reporting hooks
def pytest_runtest_logstart(nodeid, location):
    """Log test start for Serena tests."""
    if "serena" in nodeid:
        print(f"\n🧠 Starting Serena test: {nodeid}")

def pytest_runtest_logfinish(nodeid, location):
    """Log test completion for Serena tests."""
    if "serena" in nodeid:
        print(f"✅ Completed Serena test: {nodeid}")

def pytest_runtest_logreport(report):
    """Custom test reporting for Serena tests."""
    if "serena" in report.nodeid and report.when == "call":
        if report.outcome == "passed":
            duration = getattr(report, 'duration', 0)
            if duration > 5.0:  # Log slow tests
                print(f"⚠️  Slow Serena test: {report.nodeid} took {duration:.2f}s")
        elif report.outcome == "failed":
            print(f"❌ Failed Serena test: {report.nodeid}")
            if hasattr(report, 'longrepr'):
                print(f"   Error: {report.longrepr}")

# Performance test utilities
@pytest.fixture
def assert_performance():
    """Helper to assert performance requirements."""
    def _assert_performance(actual_ms, expected_ms, operation_name):
        assert actual_ms <= expected_ms, (
            f"{operation_name} took {actual_ms}ms, expected ≤{expected_ms}ms"
        )
    return _assert_performance

@pytest.fixture
def assert_memory_usage():
    """Helper to assert memory usage requirements."""
    def _assert_memory_usage(actual_mb, expected_mb, operation_name):
        assert actual_mb <= expected_mb, (
            f"{operation_name} used {actual_mb}MB, expected ≤{expected_mb}MB"
        )
    return _assert_memory_usage
