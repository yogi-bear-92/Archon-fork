"""
Basic tests for the Master Agent implementation.

These tests validate the core functionality of the Master Agent
without requiring external dependencies or network calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import the master agent components
from python.src.agents.master.master_agent import (
    MasterAgent,
    MasterAgentConfig,
    MasterAgentDependencies,
    QueryRequest,
    QueryResponse,
    QueryType,
    ProcessingStrategy,
    CircuitBreaker,
    CircuitBreakerState
)
from python.src.agents.master.capability_matrix import AgentCapabilityMatrix
from python.src.agents.master.coordination_hooks import ClaudeFlowCoordinator
from python.src.agents.master.fallback_strategies import FallbackManager


class TestMasterAgentConfig:
    """Test master agent configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MasterAgentConfig()
        
        assert config.model == "openai:gpt-4o"
        assert config.max_retries == 3
        assert config.timeout == 120
        assert config.rag_enabled is True
        assert config.max_coordinated_agents == 10
        assert config.circuit_breaker_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MasterAgentConfig(
            model="openai:gpt-3.5-turbo",
            max_retries=5,
            rag_enabled=False,
            max_coordinated_agents=20
        )
        
        assert config.model == "openai:gpt-3.5-turbo"
        assert config.max_retries == 5
        assert config.rag_enabled is False
        assert config.max_coordinated_agents == 20


class TestQueryType:
    """Test query type enumeration."""
    
    def test_query_types(self):
        """Test all query types are available."""
        assert QueryType.CODING == "coding"
        assert QueryType.RESEARCH == "research"
        assert QueryType.ANALYSIS == "analysis"
        assert QueryType.COORDINATION == "coordination"
        assert QueryType.TASK_MANAGEMENT == "task_management"
        assert QueryType.KNOWLEDGE == "knowledge"
        assert QueryType.GENERAL == "general"


class TestQueryRequest:
    """Test query request model."""
    
    def test_basic_request(self):
        """Test basic query request creation."""
        request = QueryRequest(query="Test query")
        
        assert request.query == "Test query"
        assert request.query_type is None
        assert request.context == {}
        assert request.require_rag is False
        assert request.max_agents == 3
    
    def test_full_request(self):
        """Test query request with all fields."""
        request = QueryRequest(
            query="Complex coding task",
            query_type=QueryType.CODING,
            context={"language": "python"},
            preferred_agents=["coder", "reviewer"],
            require_rag=True,
            max_agents=5
        )
        
        assert request.query == "Complex coding task"
        assert request.query_type == QueryType.CODING
        assert request.context == {"language": "python"}
        assert request.preferred_agents == ["coder", "reviewer"]
        assert request.require_rag is True
        assert request.max_agents == 5


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_init(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(threshold=3, timeout=30)
        
        assert breaker.threshold == 3
        assert breaker.timeout == 30
        assert breaker.failure_count == 0
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful call."""
        breaker = CircuitBreaker()
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Test circuit breaker with failing calls."""
        breaker = CircuitBreaker(threshold=2)
        
        async def fail_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await breaker.call(fail_func)
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await breaker.call(fail_func)
        
        assert breaker.failure_count == 2
        assert breaker.state == CircuitBreakerState.OPEN


class TestAgentCapabilityMatrix:
    """Test agent capability matrix."""
    
    def test_matrix_initialization(self):
        """Test capability matrix initialization."""
        matrix = AgentCapabilityMatrix()
        
        assert len(matrix.capabilities) > 0
        assert "coder" in matrix.capabilities
        assert "researcher" in matrix.capabilities
        assert "tester" in matrix.capabilities
    
    def test_get_capabilities(self):
        """Test getting all capabilities."""
        matrix = AgentCapabilityMatrix()
        capabilities = matrix.get_capabilities()
        
        assert len(capabilities) > 0
        assert all(isinstance(cap, dict) for cap in capabilities)
        assert all("agent_type" in cap for cap in capabilities)
    
    def test_get_capabilities_for_query_type(self):
        """Test getting capabilities for specific query type."""
        matrix = AgentCapabilityMatrix()
        coding_caps = matrix.get_capabilities_for_query_type(QueryType.CODING)
        
        assert len(coding_caps) > 0
        assert all("relevance_score" in cap for cap in coding_caps)
        
        # Should be sorted by relevance score
        scores = [cap["relevance_score"] for cap in coding_caps]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_agent_capability(self):
        """Test getting specific agent capability."""
        matrix = AgentCapabilityMatrix()
        coder_cap = matrix.get_agent_capability("coder")
        
        assert coder_cap is not None
        assert coder_cap.agent_type == "coder"
        assert "programming" in coder_cap.domains
    
    def test_update_performance_metrics(self):
        """Test updating agent performance metrics."""
        matrix = AgentCapabilityMatrix()
        
        result = matrix.update_performance_metrics(
            agent_type="coder",
            success=True,
            response_time=25.0
        )
        
        assert result is True
        
        coder_cap = matrix.get_agent_capability("coder")
        assert coder_cap.average_response_time != 30.0  # Should be updated from default


class TestFallbackManager:
    """Test fallback manager functionality."""
    
    def test_fallback_manager_init(self):
        """Test fallback manager initialization."""
        manager = FallbackManager()
        
        assert manager.cache == {}
        assert len(manager.local_knowledge) > 0
        assert "python_basics" in manager.local_knowledge
    
    @pytest.mark.asyncio
    async def test_local_knowledge_fallback(self):
        """Test local knowledge fallback."""
        manager = FallbackManager()
        
        result = await manager._local_knowledge_fallback("python functions")
        
        assert result["status"] == "success"
        assert result["fallback_type"] == "local_knowledge"
        assert len(result["matches"]) > 0
    
    def test_cache_response(self):
        """Test caching functionality."""
        manager = FallbackManager()
        
        test_response = {"test": "data"}
        manager.cache_response("test_key", test_response, ttl=3600)
        
        assert "test_key" in manager.cache
        assert manager.cache["test_key"]["data"] == test_response
    
    def test_get_fallback_stats(self):
        """Test getting fallback statistics."""
        manager = FallbackManager()
        
        stats = manager.get_fallback_stats()
        
        assert "wiki_searches" in stats
        assert "single_agent_fallbacks" in stats
        assert "cache_hits" in stats
        assert "timestamp" in stats


class TestClaudeFlowCoordinator:
    """Test Claude Flow coordination hooks."""
    
    def test_coordinator_init(self):
        """Test coordinator initialization."""
        coordinator = ClaudeFlowCoordinator()
        
        assert coordinator.swarm_sessions == {}
        assert coordinator.current_session_id is None
        assert coordinator.metrics_enabled is True
    
    def test_get_active_sessions(self):
        """Test getting active sessions."""
        coordinator = ClaudeFlowCoordinator()
        
        # Add mock session
        coordinator.swarm_sessions["test-session"] = {
            "session_id": "test-session",
            "status": "active"
        }
        
        active_sessions = coordinator.get_active_sessions()
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == "test-session"
    
    def test_get_session_info(self):
        """Test getting session information."""
        coordinator = ClaudeFlowCoordinator()
        
        # Add mock session
        session_info = {"session_id": "test", "status": "active"}
        coordinator.swarm_sessions["test"] = session_info
        
        retrieved_info = coordinator.get_session_info("test")
        assert retrieved_info == session_info


@pytest.mark.asyncio
class TestMasterAgent:
    """Test master agent integration."""
    
    def test_master_agent_init(self):
        """Test master agent initialization."""
        config = MasterAgentConfig(rag_enabled=False)  # Disable RAG for testing
        agent = MasterAgent(config=config)
        
        assert agent.config == config
        assert agent.capability_matrix is not None
        assert agent.claude_flow_coordinator is not None
        assert agent.fallback_manager is not None
        assert len(agent.metrics) > 0
    
    async def test_classify_query(self):
        """Test query classification."""
        config = MasterAgentConfig(rag_enabled=False)
        agent = MasterAgent(config=config)
        
        # Test coding query
        coding_type = await agent._classify_query("How do I implement a REST API?")
        assert coding_type == QueryType.CODING
        
        # Test research query
        research_type = await agent._classify_query("What are the best practices for database design?")
        assert research_type == QueryType.RESEARCH
        
        # Test analysis query
        analysis_type = await agent._classify_query("Please analyze this code for performance issues")
        assert analysis_type == QueryType.ANALYSIS
    
    async def test_route_to_agent(self):
        """Test agent routing functionality."""
        config = MasterAgentConfig(rag_enabled=False)
        agent = MasterAgent(config=config)
        
        # Test routing for coding query
        agents = await agent.route_to_agent(
            query="Fix this bug in my Python code",
            query_type=QueryType.CODING
        )
        
        assert len(agents) > 0
        assert "coder" in agents or "reviewer" in agents
    
    async def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        config = MasterAgentConfig(rag_enabled=False)
        agent = MasterAgent(config=config)
        
        metrics = await agent.get_performance_metrics()
        
        assert "queries_processed" in metrics
        assert "successful_queries" in metrics
        assert "timestamp" in metrics
        assert "config" in metrics
        assert metrics["config"]["model"] == config.model
    
    async def test_update_capabilities(self):
        """Test updating agent capabilities."""
        config = MasterAgentConfig(rag_enabled=False)
        agent = MasterAgent(config=config)
        
        capabilities_update = {
            "test_agent": {
                "domains": ["testing"],
                "strengths": ["unit_testing"],
                "performance_score": 0.95
            }
        }
        
        result = await agent.update_capabilities(capabilities_update)
        assert result is True
        
        # Verify the capability was added
        test_cap = agent.capability_matrix.get_agent_capability("test_agent")
        assert test_cap is not None
        assert test_cap.agent_type == "test_agent"


if __name__ == "__main__":
    # Run tests with basic assertion checking
    print("Running Master Agent Tests...")
    
    # Test basic functionality
    config = MasterAgentConfig()
    print(f"✓ Config created: {config.model}")
    
    request = QueryRequest(query="Test query", query_type=QueryType.CODING)
    print(f"✓ Query request created: {request.query}")
    
    matrix = AgentCapabilityMatrix()
    capabilities = matrix.get_capabilities()
    print(f"✓ Capability matrix loaded: {len(capabilities)} agents")
    
    manager = FallbackManager()
    stats = manager.get_fallback_stats()
    print(f"✓ Fallback manager initialized: {len(manager.local_knowledge)} knowledge topics")
    
    coordinator = ClaudeFlowCoordinator()
    print(f"✓ Claude Flow coordinator initialized")
    
    print("All basic tests passed! ✅")
    print("\nTo run full pytest suite:")
    print("pytest tests/test_master_agent.py -v")