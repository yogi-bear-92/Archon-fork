"""
Mock objects for Claude Flow Expert Agent testing.

This module provides comprehensive mock implementations for testing the Claude Flow Expert Agent
system including Archon MCP client, Claude Flow coordination, and fallback mechanisms.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

from src.agents.claude_flow_expert.capability_matrix import QueryType
from src.agents.claude_flow_expert.claude_flow_expert_agent import ProcessingStrategy, ClaudeFlowExpertConfig


class MockArchonMCPClient:
    """Mock implementation of Archon MCP Client for testing."""
    
    def __init__(self):
        self.rag_responses = {}
        self.code_search_responses = {}
        self.call_count = 0
        self.last_query = None
        self.should_fail = False
        self.failure_rate = 0.0
        self.response_delay = 0.0
        
    async def perform_rag_query(
        self,
        query: str,
        source: Optional[str] = None,
        match_count: int = 5
    ) -> str:
        """Mock RAG query with configurable responses."""
        self.call_count += 1
        self.last_query = query
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        if self.should_fail or (self.failure_rate > 0 and self.call_count % int(1/self.failure_rate) == 0):
            raise Exception(f"Mock RAG query failed for: {query}")
        
        # Return mock response based on query
        if query in self.rag_responses:
            return json.dumps(self.rag_responses[query])
        
        # Default mock response
        return json.dumps({
            "success": True,
            "results": [
                {
                    "content": f"Mock RAG result for query: {query}",
                    "source": source or "test-source",
                    "relevance": 0.9,
                    "metadata": {"query": query, "match_count": match_count}
                }
            ],
            "reranked": True,
            "total_results": 1
        })
    
    async def search_code_examples(
        self,
        query: str,
        source_id: Optional[str] = None,
        match_count: int = 5
    ) -> str:
        """Mock code search with configurable responses."""
        self.call_count += 1
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        if self.should_fail:
            raise Exception(f"Mock code search failed for: {query}")
        
        if query in self.code_search_responses:
            return json.dumps(self.code_search_responses[query])
        
        return json.dumps({
            "success": True,
            "results": [
                {
                    "file": "example.py",
                    "function": "example_function",
                    "code": f"def example_function():\n    # Mock code for {query}\n    pass",
                    "summary": f"Mock code example for {query}",
                    "relevance": 0.8
                }
            ],
            "total_results": 1
        })
    
    def set_rag_response(self, query: str, response: Dict[str, Any]):
        """Set mock response for a specific RAG query."""
        self.rag_responses[query] = response
    
    def set_code_search_response(self, query: str, response: Dict[str, Any]):
        """Set mock response for a specific code search query."""
        self.code_search_responses[query] = response
    
    def reset(self):
        """Reset mock state."""
        self.rag_responses = {}
        self.code_search_responses = {}
        self.call_count = 0
        self.last_query = None
        self.should_fail = False
        self.failure_rate = 0.0
        self.response_delay = 0.0


class MockClaudeFlowCoordinator:
    """Mock implementation of Claude Flow Coordinator for testing."""
    
    def __init__(self):
        self.swarms = {}
        self.agents = {}
        self.coordination_results = {}
        self.call_count = 0
        self.should_fail = False
        self.coordination_delay = 0.1
        self.current_session = None
        
    async def initialize_swarm(
        self,
        topology: str = "adaptive",
        max_agents: int = 8,
        strategy: str = "balanced",
        archon_integration: bool = True
    ) -> Dict[str, Any]:
        """Mock swarm initialization."""
        self.call_count += 1
        
        if self.should_fail:
            return {"status": "error", "error": "Mock swarm initialization failed"}
        
        session_id = f"test-swarm-{self.call_count}"
        self.current_session = session_id
        
        self.swarms[session_id] = {
            "session_id": session_id,
            "topology": topology,
            "max_agents": max_agents,
            "strategy": strategy,
            "status": "initialized",
            "agents": [],
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "topology": topology,
            "max_agents": max_agents,
            "result": "Swarm initialized successfully"
        }
    
    async def spawn_agents(
        self,
        objective: str,
        agents: List[str],
        strategy: str = "adaptive",
        archon_task_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock agent spawning."""
        if self.coordination_delay > 0:
            await asyncio.sleep(self.coordination_delay)
        
        target_session = session_id or self.current_session
        
        if self.should_fail:
            return {"status": "error", "error": "Mock agent spawning failed"}
        
        if target_session not in self.swarms:
            return {"status": "error", "error": "No active swarm session"}
        
        # Update swarm with agents
        self.swarms[target_session]["agents"].extend(agents)
        self.swarms[target_session]["status"] = "active"
        
        # Store agent information
        for agent_type in agents:
            agent_id = f"{agent_type}-{len(self.agents) + 1}"
            self.agents[agent_id] = {
                "id": agent_id,
                "type": agent_type,
                "session_id": target_session,
                "status": "active",
                "objective": objective
            }
        
        return {
            "status": "success",
            "session_id": target_session,
            "agents_spawned": agents,
            "objective": objective,
            "result": f"Spawned {len(agents)} agents successfully"
        }
    
    async def coordinate_multi_agent(
        self,
        objective: str,
        agent_types: List[str],
        max_agents: int = 5,
        strategy: str = "adaptive",
        archon_task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock multi-agent coordination."""
        coordination_start = time.time()
        
        if self.coordination_delay > 0:
            await asyncio.sleep(self.coordination_delay)
        
        if self.should_fail:
            return {
                "status": "error",
                "error": "Mock multi-agent coordination failed"
            }
        
        # Initialize swarm if needed
        if not self.current_session:
            init_result = await self.initialize_swarm()
            if init_result.get("status") != "success":
                return init_result
        
        # Spawn agents
        spawn_result = await self.spawn_agents(
            objective=objective,
            agents=agent_types[:max_agents],
            strategy=strategy,
            archon_task_id=archon_task_id
        )
        
        coordination_time = time.time() - coordination_start
        
        return {
            "status": "success",
            "coordination_time": coordination_time,
            "session_id": self.current_session,
            "agents_coordinated": agent_types[:max_agents],
            "objective": objective,
            "spawn_result": spawn_result,
            "metrics": {
                "total_agents": len(agent_types[:max_agents]),
                "coordination_overhead": coordination_time * 0.1,
                "success_rate": 1.0
            }
        }
    
    async def monitor_coordination(
        self,
        session_id: Optional[str] = None,
        duration: int = 30
    ) -> Dict[str, Any]:
        """Mock coordination monitoring."""
        target_session = session_id or self.current_session
        
        return {
            "status": "success",
            "session_id": target_session,
            "monitoring_result": {
                "agents_active": len(self.agents),
                "tasks_completed": 5,
                "average_response_time": 0.5,
                "error_rate": 0.02
            },
            "metrics": {
                "coordination_efficiency": 0.95,
                "resource_utilization": 0.75,
                "throughput": 10.5
            }
        }
    
    def reset(self):
        """Reset mock state."""
        self.swarms = {}
        self.agents = {}
        self.coordination_results = {}
        self.call_count = 0
        self.should_fail = False
        self.current_session = None


class MockFallbackManager:
    """Mock implementation of Fallback Manager for testing."""
    
    def __init__(self):
        self.wiki_responses = {}
        self.call_count = 0
        self.should_fail = False
        self.response_delay = 0.0
        
    async def wiki_search_fallback(
        self,
        query: str,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """Mock Wikipedia search fallback."""
        self.call_count += 1
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        if self.should_fail:
            return {
                "status": "error",
                "error": "Mock Wikipedia search failed",
                "fallback_type": "wiki_search",
                "query": query
            }
        
        if query in self.wiki_responses:
            return self.wiki_responses[query]
        
        return {
            "status": "success",
            "fallback_type": "wiki_search",
            "query": query,
            "results": [
                {
                    "title": f"Mock Wiki Article for {query}",
                    "summary": f"This is a mock Wikipedia summary for the query: {query}",
                    "url": f"https://en.wikipedia.org/wiki/Mock_{query.replace(' ', '_')}",
                    "score": 0.85
                }
            ],
            "total_results": 1,
            "timestamp": datetime.now().isoformat()
        }
    
    async def single_agent_fallback(
        self,
        objective: str,
        preferred_agent: str = "coder",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Mock single agent fallback."""
        if self.should_fail:
            return {
                "status": "error",
                "error": "Mock single agent fallback failed"
            }
        
        return {
            "status": "success",
            "fallback_type": "single_agent",
            "agent_used": preferred_agent,
            "objective": objective,
            "response": f"Mock single agent response for: {objective}",
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_wiki_response(self, query: str, response: Dict[str, Any]):
        """Set mock response for a specific wiki query."""
        self.wiki_responses[query] = response
    
    def reset(self):
        """Reset mock state."""
        self.wiki_responses = {}
        self.call_count = 0
        self.should_fail = False


class MockCapabilityMatrix:
    """Mock implementation of Agent Capability Matrix for testing."""
    
    def __init__(self):
        self.capabilities = self._create_mock_capabilities()
        
    def _create_mock_capabilities(self) -> Dict[str, Any]:
        """Create mock agent capabilities."""
        return {
            "coder": {
                "agent_type": "coder",
                "domains": ["programming", "software_development"],
                "strengths": ["code_generation", "debugging"],
                "performance_score": 0.9,
                "relevance_score": 0.8
            },
            "researcher": {
                "agent_type": "researcher", 
                "domains": ["research", "analysis"],
                "strengths": ["information_gathering", "analysis"],
                "performance_score": 0.87,
                "relevance_score": 0.75
            },
            "tester": {
                "agent_type": "tester",
                "domains": ["testing", "quality_assurance"],
                "strengths": ["test_generation", "validation"],
                "performance_score": 0.85,
                "relevance_score": 0.7
            }
        }
    
    def get_capabilities_for_query_type(
        self,
        query_type: QueryType,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Mock capability matching for query types."""
        # Return capabilities sorted by relevance
        capabilities = list(self.capabilities.values())
        capabilities.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return capabilities[:max_results]
    
    def get_agent_capability(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get specific agent capability."""
        return self.capabilities.get(agent_type)
    
    def update_performance_metrics(
        self,
        agent_type: str,
        success: bool,
        response_time: float
    ) -> bool:
        """Mock performance metrics update."""
        if agent_type in self.capabilities:
            capability = self.capabilities[agent_type]
            # Simple mock update
            if success:
                capability["performance_score"] = min(1.0, capability["performance_score"] + 0.01)
            else:
                capability["performance_score"] = max(0.0, capability["performance_score"] - 0.05)
            return True
        return False


class MockClaudeFlowExpertAgent:
    """Mock Claude Flow Expert Agent for integration testing."""
    
    def __init__(self, config: Optional[ClaudeFlowExpertConfig] = None):
        self.config = config or ClaudeFlowExpertConfig()
        self.mcp_client = MockArchonMCPClient()
        self.claude_flow_coordinator = MockClaudeFlowCoordinator()
        self.fallback_manager = MockFallbackManager()
        self.capability_matrix = MockCapabilityMatrix()
        
        # Metrics tracking
        self.metrics = {
            "queries_processed": 0,
            "successful_queries": 0,
            "rag_queries": 0,
            "multi_agent_coordinations": 0,
            "fallbacks_used": 0,
            "average_processing_time": 0.0
        }
        
        # Processing behavior controls
        self.should_fail = False
        self.processing_delay = 0.1
        self.failure_rate = 0.0
    
    async def process_query(self, request, deps) -> Dict[str, Any]:
        """Mock query processing."""
        start_time = time.time()
        self.metrics["queries_processed"] += 1
        
        if self.processing_delay > 0:
            await asyncio.sleep(self.processing_delay)
        
        # Simulate failure based on configuration
        if self.should_fail or (
            self.failure_rate > 0 and 
            self.metrics["queries_processed"] % int(1/self.failure_rate) == 0
        ):
            return {
                "success": False,
                "message": "Mock query processing failed",
                "processing_time": time.time() - start_time,
                "fallback_used": True
            }
        
        # Determine processing strategy
        query_type = request.query_type or QueryType.GENERAL
        processing_strategy = self._determine_mock_strategy(request, query_type)
        
        # Mock successful processing
        self.metrics["successful_queries"] += 1
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": f"Mock processing completed for: {request.query}",
            "query_type": query_type,
            "processing_strategy": processing_strategy,
            "agents_used": ["coder", "researcher"],
            "rag_sources_used": ["test-source"],
            "processing_time": processing_time,
            "confidence_score": 0.85,
            "data": {
                "result": f"Mock result for query: {request.query}",
                "metadata": {"mock": True}
            }
        }
    
    async def route_to_agent(
        self,
        query: str,
        query_type: QueryType,
        preferred_agents: Optional[List[str]] = None
    ) -> List[str]:
        """Mock agent routing."""
        capabilities = self.capability_matrix.get_capabilities_for_query_type(query_type)
        return [cap["agent_type"] for cap in capabilities[:3]]
    
    def _determine_mock_strategy(self, request, query_type) -> ProcessingStrategy:
        """Determine mock processing strategy."""
        if request.require_rag:
            return ProcessingStrategy.RAG_ENHANCED
        elif request.max_agents > 1:
            return ProcessingStrategy.MULTI_AGENT
        else:
            return ProcessingStrategy.SINGLE_AGENT
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.metrics,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.config.model,
                "max_agents": self.config.max_coordinated_agents,
                "rag_enabled": self.config.rag_enabled
            }
        }
    
    def reset(self):
        """Reset mock state."""
        self.mcp_client.reset()
        self.claude_flow_coordinator.reset()
        self.fallback_manager.reset()
        self.metrics = {
            "queries_processed": 0,
            "successful_queries": 0,
            "rag_queries": 0,
            "multi_agent_coordinations": 0,
            "fallbacks_used": 0,
            "average_processing_time": 0.0
        }
        self.should_fail = False


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_query_request(
        query: str = "Test query",
        query_type: Optional[QueryType] = None,
        context: Optional[Dict[str, Any]] = None,
        require_rag: bool = False,
        max_agents: int = 3
    ) -> Dict[str, Any]:
        """Create a mock query request."""
        return {
            "query": query,
            "query_type": query_type,
            "context": context or {},
            "require_rag": require_rag,
            "max_agents": max_agents
        }
    
    @staticmethod
    def create_complex_scenarios() -> List[Dict[str, Any]]:
        """Create complex test scenarios."""
        return [
            {
                "name": "coding_task",
                "request": TestDataGenerator.create_query_request(
                    query="Implement a REST API with authentication",
                    query_type=QueryType.CODING,
                    max_agents=3
                ),
                "expected_strategy": ProcessingStrategy.MULTI_AGENT,
                "expected_agents": ["coder", "tester", "reviewer"]
            },
            {
                "name": "research_task",
                "request": TestDataGenerator.create_query_request(
                    query="What are the best practices for microservices?",
                    query_type=QueryType.RESEARCH,
                    require_rag=True
                ),
                "expected_strategy": ProcessingStrategy.RAG_ENHANCED,
                "expected_agents": ["researcher"]
            },
            {
                "name": "coordination_task",
                "request": TestDataGenerator.create_query_request(
                    query="Plan a complete software architecture",
                    query_type=QueryType.COORDINATION,
                    max_agents=5
                ),
                "expected_strategy": ProcessingStrategy.MULTI_AGENT,
                "expected_agents": ["system-architect", "planner", "coordinator"]
            }
        ]


# Performance testing utilities
class PerformanceTestHelper:
    """Helper utilities for performance testing."""
    
    @staticmethod
    def measure_async_execution(func):
        """Decorator to measure async function execution time."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            return result, execution_time
        return wrapper
    
    @staticmethod
    def create_concurrent_requests(
        count: int,
        query_base: str = "Test concurrent query"
    ) -> List[Dict[str, Any]]:
        """Create multiple concurrent request objects."""
        return [
            TestDataGenerator.create_query_request(
                query=f"{query_base} {i}",
                query_type=QueryType.GENERAL
            )
            for i in range(count)
        ]
    
    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculate performance percentiles."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / len(sorted_values)
        }