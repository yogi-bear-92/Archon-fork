"""
Test fixtures for Master Agent testing.

This module provides pytest fixtures for comprehensive Master Agent testing
including configuration, test data, and performance baselines.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.agents.master.master_agent import MasterAgentConfig, MasterAgentDependencies
from src.agents.master.capability_matrix import QueryType
from tests.mocks.master_agent_mocks import (
    MockArchonMCPClient,
    MockClaudeFlowCoordinator,
    MockFallbackManager,
    MockCapabilityMatrix,
    MockMasterAgent,
    TestDataGenerator,
    PerformanceTestHelper
)


@pytest.fixture
def master_agent_config():
    """Default Master Agent configuration for testing."""
    return MasterAgentConfig(
        model="openai:gpt-4o-mini",  # Use smaller model for testing
        max_retries=2,
        timeout=30,
        enable_rate_limiting=False,  # Disable for testing
        rag_enabled=True,
        max_rag_results=3,
        max_coordinated_agents=5,
        coordination_timeout=60,
        memory_persistence_enabled=True,
        memory_ttl=3600,
        enable_metrics=True,
        circuit_breaker_enabled=False  # Disable for testing
    )


@pytest.fixture
def master_agent_deps():
    """Default Master Agent dependencies for testing."""
    return MasterAgentDependencies(
        query_type=QueryType.GENERAL,
        coordinate_agents=True,
        metrics_callback=None,
        progress_callback=None
    )


@pytest.fixture
def mock_archon_mcp_client():
    """Mock Archon MCP client for testing."""
    return MockArchonMCPClient()


@pytest.fixture
def mock_claude_flow_coordinator():
    """Mock Claude Flow coordinator for testing."""
    return MockClaudeFlowCoordinator()


@pytest.fixture
def mock_fallback_manager():
    """Mock fallback manager for testing."""
    return MockFallbackManager()


@pytest.fixture
def mock_capability_matrix():
    """Mock capability matrix for testing."""
    return MockCapabilityMatrix()


@pytest.fixture
def mock_master_agent(master_agent_config):
    """Mock master agent with default configuration."""
    return MockMasterAgent(master_agent_config)


@pytest.fixture
def test_queries():
    """Various test query scenarios."""
    return {
        "simple_coding": TestDataGenerator.create_query_request(
            query="Create a Python function to calculate fibonacci numbers",
            query_type=QueryType.CODING,
            max_agents=1
        ),
        "complex_research": TestDataGenerator.create_query_request(
            query="Research the latest trends in microservices architecture",
            query_type=QueryType.RESEARCH,
            require_rag=True,
            max_agents=2
        ),
        "multi_agent_task": TestDataGenerator.create_query_request(
            query="Build a complete web application with user authentication",
            query_type=QueryType.COORDINATION,
            max_agents=5,
            context={"technology_stack": "Python, FastAPI, React"}
        ),
        "analysis_task": TestDataGenerator.create_query_request(
            query="Analyze the performance bottlenecks in this codebase",
            query_type=QueryType.ANALYSIS,
            context={"codebase_size": "large", "languages": ["python", "javascript"]}
        ),
        "rag_required": TestDataGenerator.create_query_request(
            query="How do I implement OAuth2 with PKCE?",
            query_type=QueryType.KNOWLEDGE,
            require_rag=True
        )
    }


@pytest.fixture
def performance_test_scenarios():
    """Performance testing scenarios with expected baselines."""
    return {
        "single_query_latency": {
            "description": "Single query processing latency",
            "request": TestDataGenerator.create_query_request("Simple test query"),
            "expected_max_time_ms": 2000,
            "expected_success_rate": 0.95
        },
        "concurrent_queries": {
            "description": "Concurrent query processing",
            "requests": PerformanceTestHelper.create_concurrent_requests(10),
            "expected_max_time_ms": 5000,
            "expected_success_rate": 0.90
        },
        "rag_query_latency": {
            "description": "RAG-enhanced query processing",
            "request": TestDataGenerator.create_query_request(
                "Research query requiring knowledge base",
                require_rag=True
            ),
            "expected_max_time_ms": 3000,
            "expected_success_rate": 0.85
        },
        "multi_agent_coordination": {
            "description": "Multi-agent coordination latency",
            "request": TestDataGenerator.create_query_request(
                "Complex task requiring coordination",
                query_type=QueryType.COORDINATION,
                max_agents=5
            ),
            "expected_max_time_ms": 5000,
            "expected_success_rate": 0.80
        }
    }


@pytest.fixture
def error_scenarios():
    """Error handling test scenarios."""
    return {
        "rag_failure": {
            "description": "RAG query failure with fallback",
            "setup_failure": lambda mock_client: setattr(mock_client, "should_fail", True),
            "request": TestDataGenerator.create_query_request(
                "Query that triggers RAG failure",
                require_rag=True
            ),
            "expected_fallback": True,
            "expected_success": True  # Should succeed via fallback
        },
        "coordination_failure": {
            "description": "Agent coordination failure",
            "setup_failure": lambda mock_coord: setattr(mock_coord, "should_fail", True),
            "request": TestDataGenerator.create_query_request(
                "Task requiring coordination",
                max_agents=3
            ),
            "expected_fallback": True,
            "expected_success": True  # Should succeed via single agent fallback
        },
        "total_system_failure": {
            "description": "Complete system failure scenario",
            "setup_failure": lambda components: [
                setattr(comp, "should_fail", True) for comp in components
            ],
            "request": TestDataGenerator.create_query_request("Any query"),
            "expected_fallback": True,
            "expected_success": False  # Should fail completely
        }
    }


@pytest.fixture
def capability_matrix_scenarios():
    """Agent capability matrix test scenarios."""
    return {
        "coding_task_routing": {
            "query": "Implement a REST API",
            "query_type": QueryType.CODING,
            "expected_agents": ["coder", "tester", "reviewer"],
            "min_relevance_score": 0.7
        },
        "research_task_routing": {
            "query": "Find information about machine learning",
            "query_type": QueryType.RESEARCH,
            "expected_agents": ["researcher", "analyst"],
            "min_relevance_score": 0.6
        },
        "coordination_task_routing": {
            "query": "Plan a software architecture",
            "query_type": QueryType.COORDINATION,
            "expected_agents": ["system-architect", "planner", "coordinator"],
            "min_relevance_score": 0.8
        }
    }


@pytest.fixture
def fallback_test_data():
    """Test data for fallback mechanism testing."""
    return {
        "wiki_queries": [
            "What is Python programming language?",
            "Explain microservices architecture",
            "How does OAuth2 work?"
        ],
        "expected_wiki_responses": {
            "Python programming": {
                "status": "success",
                "results": [
                    {
                        "title": "Python (programming language)",
                        "summary": "Python is a high-level programming language...",
                        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "score": 0.95
                    }
                ]
            }
        },
        "single_agent_fallback_scenarios": [
            {
                "objective": "Code a simple web scraper",
                "preferred_agent": "coder",
                "context": {"language": "python", "libraries": ["requests", "beautifulsoup4"]}
            },
            {
                "objective": "Research competitor analysis",
                "preferred_agent": "researcher",
                "context": {"industry": "tech", "focus": "APIs"}
            }
        ]
    }


@pytest.fixture
def integration_test_scenarios():
    """End-to-end integration test scenarios."""
    return {
        "complete_coding_workflow": {
            "description": "Full coding workflow with RAG, coordination, and testing",
            "steps": [
                {
                    "query": "Research best practices for REST API design",
                    "expected_strategy": "rag_enhanced",
                    "expected_agents": ["researcher"]
                },
                {
                    "query": "Implement a REST API based on the research",
                    "expected_strategy": "multi_agent", 
                    "expected_agents": ["coder", "reviewer", "tester"]
                },
                {
                    "query": "Create comprehensive tests for the API",
                    "expected_strategy": "single_agent",
                    "expected_agents": ["tester"]
                }
            ],
            "expected_total_time_ms": 10000,
            "expected_success_rate": 0.85
        },
        "research_and_analysis_workflow": {
            "description": "Research workflow with knowledge synthesis",
            "steps": [
                {
                    "query": "Find information about GraphQL vs REST APIs",
                    "expected_strategy": "rag_enhanced",
                    "expected_agents": ["researcher"]
                },
                {
                    "query": "Analyze the trade-offs between GraphQL and REST",
                    "expected_strategy": "single_agent",
                    "expected_agents": ["analyst"]
                }
            ],
            "expected_total_time_ms": 8000,
            "expected_success_rate": 0.90
        }
    }


@pytest.fixture
def memory_test_scenarios():
    """Memory management test scenarios."""
    return {
        "cross_session_persistence": {
            "description": "Test memory persistence across sessions",
            "initial_data": {"project_context": "web_app", "tech_stack": "FastAPI + React"},
            "query_sequence": [
                "Store project architecture decisions",
                "Retrieve previous architecture decisions",
                "Build upon previous decisions"
            ],
            "expected_memory_usage_mb": 50,
            "expected_persistence_success_rate": 0.95
        },
        "memory_optimization": {
            "description": "Memory usage optimization testing",
            "load_test_queries": 100,
            "expected_max_memory_mb": 200,
            "expected_gc_efficiency": 0.85
        }
    }


@pytest.fixture
def circuit_breaker_scenarios():
    """Circuit breaker testing scenarios."""
    return {
        "rag_circuit_breaker": {
            "description": "RAG service circuit breaker activation",
            "failure_threshold": 5,
            "failure_rate": 0.8,  # 80% failure rate to trigger breaker
            "recovery_time_seconds": 60,
            "test_queries": 20
        },
        "coordination_circuit_breaker": {
            "description": "Coordination service circuit breaker",
            "failure_threshold": 3,
            "failure_rate": 1.0,  # 100% failure rate
            "recovery_time_seconds": 30,
            "test_queries": 10
        }
    }


@pytest.fixture
def load_test_configuration():
    """Load testing configuration."""
    return {
        "concurrent_users": [1, 5, 10, 20, 50],
        "test_duration_seconds": 60,
        "ramp_up_time_seconds": 10,
        "query_mix": {
            "simple_queries": 0.4,
            "rag_queries": 0.3,
            "coordination_queries": 0.2,
            "complex_queries": 0.1
        },
        "performance_thresholds": {
            "max_response_time_ms": 5000,
            "min_throughput_qps": 10,
            "max_error_rate": 0.05,
            "max_memory_usage_mb": 500
        }
    }


# Helper fixtures for patching and mocking

@pytest.fixture
def mock_pydantic_ai_agent():
    """Mock PydanticAI Agent for testing."""
    mock_agent = MagicMock()
    mock_agent.tool = lambda func: func  # Pass-through decorator
    mock_agent.run = lambda prompt, deps: {"message": f"Mock response for: {prompt}"}
    return mock_agent


@pytest.fixture
def patched_master_agent_dependencies(
    mock_archon_mcp_client,
    mock_claude_flow_coordinator,
    mock_fallback_manager,
    mock_capability_matrix
):
    """Patch all Master Agent dependencies for isolated testing."""
    with patch("src.agents.mcp_client.get_mcp_client", return_value=mock_archon_mcp_client), \
         patch("src.agents.master.coordination_hooks.ClaudeFlowCoordinator", return_value=mock_claude_flow_coordinator), \
         patch("src.agents.master.fallback_strategies.FallbackManager", return_value=mock_fallback_manager), \
         patch("src.agents.master.capability_matrix.AgentCapabilityMatrix", return_value=mock_capability_matrix):
        yield {
            "mcp_client": mock_archon_mcp_client,
            "coordinator": mock_claude_flow_coordinator,
            "fallback": mock_fallback_manager,
            "capabilities": mock_capability_matrix
        }


@pytest.fixture
def performance_monitor():
    """Performance monitoring utilities for tests."""
    import time
    import psutil
    from collections import defaultdict
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_samples = []
            self.process = psutil.Process()
            
        def start(self):
            self.start_time = time.time()
            self.memory_samples = []
            return self
            
        def sample_memory(self):
            memory_info = self.process.memory_info()
            self.memory_samples.append({
                "timestamp": time.time(),
                "rss": memory_info.rss,
                "vms": memory_info.vms
            })
            
        def stop(self):
            self.end_time = time.time()
            return self
            
        @property
        def elapsed_time_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0
            
        @property
        def peak_memory_mb(self):
            if self.memory_samples:
                return max(sample["rss"] for sample in self.memory_samples) / (1024 * 1024)
            return 0
            
        @property
        def memory_delta_mb(self):
            if len(self.memory_samples) >= 2:
                start_mem = self.memory_samples[0]["rss"]
                end_mem = self.memory_samples[-1]["rss"]
                return (end_mem - start_mem) / (1024 * 1024)
            return 0
    
    return PerformanceMonitor


@pytest.fixture
def async_test_helper():
    """Helper for async test operations."""
    import asyncio
    
    class AsyncTestHelper:
        @staticmethod
        async def run_concurrent(coroutines, max_concurrent=10):
            """Run coroutines with concurrency limit."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_coroutine(coro):
                async with semaphore:
                    return await coro
            
            tasks = [limited_coroutine(coro) for coro in coroutines]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        @staticmethod
        async def run_with_timeout(coro, timeout_seconds=30):
            """Run coroutine with timeout."""
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise AssertionError(f"Operation timed out after {timeout_seconds} seconds")
        
        @staticmethod
        def measure_async_performance(func):
            """Decorator to measure async function performance."""
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                end_time = time.time()
                
                return {
                    "result": result,
                    "success": success,
                    "execution_time_ms": (end_time - start_time) * 1000,
                    "timestamp": start_time
                }
            return wrapper
    
    return AsyncTestHelper