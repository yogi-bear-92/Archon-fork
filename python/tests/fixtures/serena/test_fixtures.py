"""
Test fixtures and data for Serena Claude Flow Expert Agent testing.

Provides comprehensive test data for validating Serena MCP tools,
semantic analysis operations, and multi-agent coordination workflows.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock

import pytest


class SerenaTestData:
    """Test data factory for Serena integration tests."""
    
    @staticmethod
    def create_project_structure() -> Dict[str, Any]:
        """Create mock project structure for semantic analysis."""
        return {
            "project_id": str(uuid.uuid4()),
            "name": "Test Semantic Project",
            "description": "Project for testing semantic analysis capabilities",
            "files": [
                {
                    "path": "src/main.py",
                    "content": '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

class MathUtils:
    """Utility class for mathematical operations."""
    
    @staticmethod
    def is_prime(num: int) -> bool:
        """Check if a number is prime."""
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of a number."""
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)
''',
                    "language": "python",
                    "symbols": [
                        {
                            "name": "calculate_fibonacci",
                            "type": "function",
                            "line": 2,
                            "signature": "calculate_fibonacci(n: int) -> int"
                        },
                        {
                            "name": "MathUtils",
                            "type": "class",
                            "line": 8,
                            "methods": ["is_prime", "factorial"]
                        }
                    ]
                },
                {
                    "path": "src/utils.py",
                    "content": '''
from typing import List, Dict, Optional
import json

def parse_config(file_path: str) -> Dict[str, Any]:
    """Parse configuration from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

class DataProcessor:
    """Process and transform data structures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def process_items(self, items: List[Dict]) -> List[Dict]:
        """Process a list of items according to configuration."""
        processed = []
        for item in items:
            if self._is_valid_item(item):
                processed.append(self._transform_item(item))
        return processed
    
    def _is_valid_item(self, item: Dict) -> bool:
        """Validate item structure."""
        required_keys = self.config.get('required_keys', [])
        return all(key in item for key in required_keys)
    
    def _transform_item(self, item: Dict) -> Dict:
        """Transform item according to rules."""
        transformed = item.copy()
        transformations = self.config.get('transformations', {})
        
        for old_key, new_key in transformations.items():
            if old_key in transformed:
                transformed[new_key] = transformed.pop(old_key)
        
        return transformed
''',
                    "language": "python",
                    "symbols": [
                        {
                            "name": "parse_config",
                            "type": "function",
                            "line": 4,
                            "signature": "parse_config(file_path: str) -> Dict[str, Any]"
                        },
                        {
                            "name": "DataProcessor",
                            "type": "class",
                            "line": 11,
                            "methods": ["process_items", "_is_valid_item", "_transform_item"]
                        }
                    ]
                }
            ],
            "dependencies": {
                "calculate_fibonacci": ["MathUtils.factorial"],
                "DataProcessor.process_items": ["DataProcessor._is_valid_item", "DataProcessor._transform_item"],
                "parse_config": []
            }
        }
    
    @staticmethod
    def create_semantic_analysis_results() -> Dict[str, Any]:
        """Create expected semantic analysis results."""
        return {
            "symbols_found": 6,
            "functions": [
                {
                    "name": "calculate_fibonacci",
                    "complexity": "recursive",
                    "parameters": ["n: int"],
                    "return_type": "int",
                    "dependencies": []
                },
                {
                    "name": "parse_config",
                    "complexity": "simple",
                    "parameters": ["file_path: str"],
                    "return_type": "Dict[str, Any]",
                    "dependencies": ["json"]
                }
            ],
            "classes": [
                {
                    "name": "MathUtils",
                    "methods": ["is_prime", "factorial"],
                    "type": "utility",
                    "complexity": "medium"
                },
                {
                    "name": "DataProcessor",
                    "methods": ["process_items", "_is_valid_item", "_transform_item"],
                    "type": "processor",
                    "complexity": "high",
                    "has_constructor": True
                }
            ],
            "semantic_relationships": [
                {
                    "type": "composition",
                    "source": "DataProcessor",
                    "target": "process_items"
                },
                {
                    "type": "dependency",
                    "source": "parse_config",
                    "target": "json"
                }
            ],
            "code_quality_metrics": {
                "maintainability_index": 85,
                "cyclomatic_complexity": 12,
                "lines_of_code": 65,
                "test_coverage": 0  # No tests in fixtures
            }
        }
    
    @staticmethod
    def create_memory_context() -> Dict[str, Any]:
        """Create memory context for persistence tests."""
        return {
            "session_id": str(uuid.uuid4()),
            "agent_contexts": {
                "serena_master": {
                    "current_project": "test_project",
                    "analysis_state": "complete",
                    "symbols_cache": {
                        "functions": ["calculate_fibonacci", "parse_config"],
                        "classes": ["MathUtils", "DataProcessor"],
                        "last_updated": datetime.now().isoformat()
                    },
                    "coordination_data": {
                        "active_agents": ["coder", "reviewer", "tester"],
                        "task_assignments": {
                            "coder": ["implement_feature"],
                            "reviewer": ["code_review"],
                            "tester": ["write_tests"]
                        }
                    }
                },
                "coder_agent": {
                    "current_task": "implement_feature",
                    "semantic_context": {
                        "target_functions": ["calculate_fibonacci"],
                        "required_imports": ["typing", "math"],
                        "coding_patterns": ["recursive_optimization"]
                    }
                }
            },
            "shared_memory": {
                "project_metadata": {
                    "language": "python",
                    "framework": "fastapi",
                    "testing_framework": "pytest"
                },
                "coordination_state": {
                    "workflow_stage": "implementation",
                    "blockers": [],
                    "next_actions": ["code_generation", "test_creation"]
                }
            },
            "timestamp": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
    
    @staticmethod
    def create_coordination_messages() -> List[Dict[str, Any]]:
        """Create coordination messages between agents."""
        base_time = datetime.now()
        return [
            {
                "id": str(uuid.uuid4()),
                "timestamp": base_time.isoformat(),
                "from_agent": "serena_master",
                "to_agent": "coder_agent",
                "message_type": "semantic_context",
                "content": {
                    "action": "provide_context",
                    "context": {
                        "symbols": ["calculate_fibonacci", "MathUtils"],
                        "relationships": ["recursive_function", "utility_class"],
                        "optimization_hints": ["memoization_candidate"]
                    }
                },
                "priority": "high",
                "status": "sent"
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
                "from_agent": "coder_agent",
                "to_agent": "serena_master",
                "message_type": "context_request",
                "content": {
                    "action": "request_analysis",
                    "target": "src/advanced_math.py",
                    "analysis_type": "dependency_analysis"
                },
                "priority": "medium",
                "status": "received"
            },
            {
                "id": str(uuid.uuid4()),
                "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
                "from_agent": "serena_master",
                "to_agent": "reviewer_agent",
                "message_type": "review_request",
                "content": {
                    "action": "review_code",
                    "files": ["src/main.py", "src/utils.py"],
                    "focus_areas": ["performance", "maintainability", "security"],
                    "semantic_context": {
                        "complexity_metrics": {"cyclomatic": 12, "cognitive": 8},
                        "patterns_found": ["factory_pattern", "recursive_algorithm"]
                    }
                },
                "priority": "high",
                "status": "pending"
            }
        ]
    
    @staticmethod
    def create_performance_metrics() -> Dict[str, Any]:
        """Create performance metrics for testing."""
        return {
            "semantic_analysis": {
                "avg_response_time_ms": 145,
                "max_response_time_ms": 320,
                "min_response_time_ms": 89,
                "operations_per_second": 6.8,
                "memory_usage_mb": 45.2,
                "cpu_usage_percent": 12.5,
                "cache_hit_ratio": 0.74
            },
            "coordination_overhead": {
                "message_latency_ms": 23,
                "coordination_setup_time_ms": 156,
                "agent_spawning_time_ms": 234,
                "memory_sync_time_ms": 67,
                "total_coordination_overhead_ms": 480
            },
            "memory_operations": {
                "context_store_time_ms": 12,
                "context_retrieve_time_ms": 8,
                "context_size_kb": 23.4,
                "persistence_write_time_ms": 34,
                "persistence_read_time_ms": 19
            },
            "end_to_end_workflow": {
                "total_workflow_time_ms": 2340,
                "semantic_analysis_phase_ms": 450,
                "agent_coordination_phase_ms": 680,
                "code_generation_phase_ms": 890,
                "review_phase_ms": 320
            },
            "load_test_results": {
                "concurrent_agents": 10,
                "requests_per_second": 45,
                "average_response_time_ms": 187,
                "error_rate_percent": 0.02,
                "throughput_requests_per_minute": 2700,
                "memory_usage_under_load_mb": 156.7
            }
        }
    
    @staticmethod
    def create_error_scenarios() -> List[Dict[str, Any]]:
        """Create error scenarios for robustness testing."""
        return [
            {
                "name": "network_timeout",
                "description": "Network timeout during MCP communication",
                "trigger": {"type": "timeout", "duration_ms": 5000},
                "expected_behavior": "graceful_degradation",
                "recovery_strategy": "retry_with_backoff"
            },
            {
                "name": "memory_exhaustion",
                "description": "Memory exhaustion during large file analysis",
                "trigger": {"type": "memory_limit", "limit_mb": 100},
                "expected_behavior": "chunk_processing",
                "recovery_strategy": "split_workload"
            },
            {
                "name": "semantic_parsing_error",
                "description": "Error parsing malformed code files",
                "trigger": {"type": "invalid_syntax", "file": "malformed.py"},
                "expected_behavior": "error_reporting",
                "recovery_strategy": "skip_with_log"
            },
            {
                "name": "agent_communication_failure",
                "description": "Agent coordination message delivery failure",
                "trigger": {"type": "communication_error", "agent_id": "coder_agent"},
                "expected_behavior": "queue_message",
                "recovery_strategy": "retry_delivery"
            },
            {
                "name": "context_corruption",
                "description": "Corrupted memory context data",
                "trigger": {"type": "data_corruption", "field": "semantic_context"},
                "expected_behavior": "context_rebuild",
                "recovery_strategy": "regenerate_from_source"
            }
        ]


@pytest.fixture
def serena_project_data():
    """Fixture providing project structure for testing."""
    return SerenaTestData.create_project_structure()


@pytest.fixture
def serena_semantic_results():
    """Fixture providing expected semantic analysis results."""
    return SerenaTestData.create_semantic_analysis_results()


@pytest.fixture
def serena_memory_context():
    """Fixture providing memory context data."""
    return SerenaTestData.create_memory_context()


@pytest.fixture
def serena_coordination_messages():
    """Fixture providing coordination message examples."""
    return SerenaTestData.create_coordination_messages()


@pytest.fixture
def serena_performance_metrics():
    """Fixture providing performance benchmarking data."""
    return SerenaTestData.create_performance_metrics()


@pytest.fixture
def serena_error_scenarios():
    """Fixture providing error scenario definitions."""
    return SerenaTestData.create_error_scenarios()


@pytest.fixture
def mock_serena_mcp_client():
    """Mock MCP client for Serena tools."""
    mock_client = MagicMock()
    
    # Mock successful responses
    mock_client.list_dir.return_value = {
        "dirs": ["src", "tests"],
        "files": ["src/main.py", "src/utils.py", "tests/test_main.py"]
    }
    
    mock_client.get_symbols_overview.return_value = {
        "symbols": [
            {"name": "calculate_fibonacci", "type": "function"},
            {"name": "MathUtils", "type": "class"}
        ]
    }
    
    mock_client.find_symbol.return_value = [
        {
            "name": "calculate_fibonacci",
            "location": {"file": "src/main.py", "line": 2},
            "signature": "calculate_fibonacci(n: int) -> int"
        }
    ]
    
    mock_client.search_for_pattern.return_value = {
        "matches": [
            {
                "file": "src/main.py",
                "line": 5,
                "content": "return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)"
            }
        ]
    }
    
    return mock_client


@pytest.fixture
def mock_claude_flow_coordination():
    """Mock Claude Flow coordination system."""
    mock_coordinator = MagicMock()
    
    # Mock swarm initialization
    mock_coordinator.swarm_init.return_value = {
        "status": "initialized",
        "session_id": "test-session-123",
        "topology": "mesh",
        "max_agents": 5
    }
    
    # Mock agent spawning
    mock_coordinator.agent_spawn.return_value = {
        "status": "spawned",
        "agent_id": "test-agent-456",
        "type": "coder",
        "capabilities": ["code_generation", "semantic_analysis"]
    }
    
    # Mock memory operations
    mock_coordinator.memory_store.return_value = {"status": "stored"}
    mock_coordinator.memory_retrieve.return_value = {
        "status": "retrieved",
        "data": {"context": "test_context"}
    }
    
    return mock_coordinator


@pytest.fixture
def performance_test_environment():
    """Set up performance testing environment with monitoring."""
    import time
    import psutil
    import threading
    from collections import defaultdict
    
    metrics = defaultdict(list)
    monitoring_active = threading.Event()
    monitoring_active.set()
    
    def monitor_performance():
        """Background performance monitoring."""
        while monitoring_active.is_set():
            metrics['cpu_percent'].append(psutil.cpu_percent())
            metrics['memory_mb'].append(psutil.virtual_memory().used / 1024 / 1024)
            metrics['timestamp'].append(time.time())
            time.sleep(0.1)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
    monitor_thread.start()
    
    yield {
        'metrics': metrics,
        'stop_monitoring': lambda: monitoring_active.clear(),
        'get_average_cpu': lambda: sum(metrics['cpu_percent']) / len(metrics['cpu_percent']) if metrics['cpu_percent'] else 0,
        'get_peak_memory': lambda: max(metrics['memory_mb']) if metrics['memory_mb'] else 0
    }
    
    # Cleanup
    monitoring_active.clear()


class MockAsyncContext:
    """Mock async context manager for testing."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value or {}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def json(self):
        return self.return_value
    
    async def post(self, url, **kwargs):
        return self
    
    async def get(self, url, **kwargs):
        return self


@pytest.fixture
def mock_async_http_client():
    """Mock async HTTP client for testing MCP communications."""
    
    def create_response(status="success", data=None):
        return MockAsyncContext({
            "status": status,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        })
    
    mock_client = MagicMock()
    mock_client.post.return_value = create_response()
    mock_client.get.return_value = create_response()
    
    return mock_client