#!/usr/bin/env python3
"""
Pytest configuration for Phase 3 integration testing.
Provides fixtures and configuration for async testing.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_phase3_components():
    """Mock Phase 3 system components for testing."""
    from unittest.mock import Mock, AsyncMock
    
    class MockPhase3Components:
        def __init__(self):
            self.global_orchestrator = Mock()
            self.global_orchestrator.submit_global_task = AsyncMock(return_value="mock_task_123")
            self.global_orchestrator.register_coordinator = AsyncMock()
            self.global_orchestrator.orchestrator_id = "mock_orchestrator_001"
            self.global_orchestrator.regional_coordinators = {}
            self.global_orchestrator.global_metrics = {'total_tasks_processed': 0, 'total_coordinators': 0}
            self.global_orchestrator.configuration = {'max_task_retries': 3}
            self.global_orchestrator.active_global_tasks = {}
            self.global_orchestrator.global_task_queue = []
            
            self.load_balancer = Mock()
            self.load_balancer.route_request = AsyncMock(return_value="mock_swarm")
            self.load_balancer.register_swarm = Mock()
            self.load_balancer.report_request_completion = AsyncMock()
            
            self.communicator = Mock()
            self.communicator.send_message = AsyncMock(return_value=True)
            self.communicator.send_task_request = AsyncMock(return_value="msg_123")
            
    return MockPhase3Components()