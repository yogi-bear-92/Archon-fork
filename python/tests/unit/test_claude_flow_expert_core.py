import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.claude_flow_expert.capability_matrix import QueryType
from tests.mocks.claude_flow_expert_agent_mocks import MockClaudeFlowExpertAgent, TestDataGenerator
"""
Unit tests for Claude Flow Expert Agent core functionality.

This module tests the core Claude Flow Expert Agent functionality including initialization,
configuration, query classification, processing strategy determination, and
basic agent operations.
"""

    ClaudeFlowExpertAgent, ClaudeFlowExpertConfig, ClaudeFlowExpertDependencies,
    QueryRequest, QueryResponse, ProcessingStrategy, CircuitBreaker
)

class TestClaudeFlowExpertAgentInitialization:
    """Test Claude Flow Expert Agent initialization and configuration."""

    def test_default_initialization(self):
        """Test Claude Flow Expert Agent initialization with default config."""
        agent = ClaudeFlowExpertAgent()

        assert agent.config is not None
        assert agent.capability_matrix is not None
        assert agent.claude_flow_coordinator is not None
        assert agent.fallback_manager is not None
        assert agent.circuit_breakers is not None
        assert len(agent.circuit_breakers) == 3  # rag, agents, coordination

        # Check default metrics
        assert agent.metrics["queries_processed"] == 0
        assert agent.metrics["successful_queries"] == 0
        assert agent.metrics["average_processing_time"] == 0.0

    def test_custom_config_initialization(self):
        """Test Claude Flow Expert Agent with custom configuration."""
        config = ClaudeFlowExpertConfig(
            model="openai:gpt-3.5-turbo",
            max_retries=5,
            timeout=60,
            rag_enabled=False,
            max_coordinated_agents=15
        )

        agent = ClaudeFlowExpertAgent(config)

        assert agent.config.model == "openai:gpt-3.5-turbo"
        assert agent.config.max_retries == 5
        assert agent.config.timeout == 60
        assert agent.config.rag_enabled is False
        assert agent.config.max_coordinated_agents == 15

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = ClaudeFlowExpertConfig(circuit_breaker_threshold=10)
        agent = ClaudeFlowExpertAgent(config)

        for breaker in agent.circuit_breakers.values():
            assert breaker.threshold == 10
            assert breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_agent_tools_registration(self):
        """Test that PydanticAI tools are properly registered."""
        agent = ClaudeFlowExpertAgent()

        # Check that the agent has the expected tools
        # Note: This would need to be adapted based on actual PydanticAI implementation
        assert hasattr(agent, '_create_agent')
        assert callable(agent._create_agent)

class TestQueryClassification:
    """Test query classification and type determination."""

    def test_classify_coding_queries(self):
        """Test classification of coding-related queries."""
        agent = ClaudeFlowExpertAgent()

        coding_queries = [
            "Write a Python function to sort arrays",
            "Debug this JavaScript code",
            "Implement a class for user authentication",
            "Fix this bug in the database connection"
        ]

        for query in coding_queries:
            # Note: _classify_query is async in the actual implementation
            # This test would need to be async and use await
            result = asyncio.run(agent._classify_query(query))
            assert result == QueryType.CODING

    def test_classify_research_queries(self):
        """Test classification of research-related queries."""
        agent = ClaudeFlowExpertAgent()

        research_queries = [
            "What are the best practices for API design?",
            "Find information about machine learning algorithms",
            "Research the latest trends in cloud computing",
            "Learn about microservices architecture"
        ]

        for query in research_queries:
            result = asyncio.run(agent._classify_query(query))
            assert result == QueryType.RESEARCH

    def test_classify_analysis_queries(self):
        """Test classification of analysis-related queries."""
        agent = ClaudeFlowExpertAgent()

        analysis_queries = [
            "Analyze the performance of this algorithm",
            "Review this code for security vulnerabilities",
            "Check the quality of this implementation",
            "Examine the architecture for scalability issues"
        ]

        for query in analysis_queries:
            result = asyncio.run(agent._classify_query(query))
            assert result == QueryType.ANALYSIS

    def test_classify_general_queries(self):
        """Test classification of general queries."""
        agent = ClaudeFlowExpertAgent()

        general_queries = [
            "Hello, how are you?",
            "Can you help me with something?",
            "I need assistance",
            "Random question about nothing specific"
        ]

        for query in general_queries:
            result = asyncio.run(agent._classify_query(query))
            assert result == QueryType.GENERAL

class TestProcessingStrategyDetermination:
    """Test processing strategy determination logic."""

    @pytest.mark.asyncio
    async def test_rag_enhanced_strategy(self):
        """Test RAG-enhanced strategy selection."""
        agent = ClaudeFlowExpertAgent()

        # RAG required explicitly
        request = QueryRequest(
            query="Research query",
            require_rag=True
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.RESEARCH)
        assert strategy == ProcessingStrategy.RAG_ENHANCED

        # RAG beneficial for research queries
        request = QueryRequest(
            query="What is machine learning?",
            require_rag=False
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.RESEARCH)
        assert strategy == ProcessingStrategy.RAG_ENHANCED

    @pytest.mark.asyncio
    async def test_multi_agent_strategy(self):
        """Test multi-agent strategy selection."""
        agent = ClaudeFlowExpertAgent()

        # Coordination query type
        request = QueryRequest(
            query="Plan software architecture",
            max_agents=1
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.COORDINATION)
        assert strategy == ProcessingStrategy.MULTI_AGENT

        # Multiple agents requested
        request = QueryRequest(
            query="Any query",
            max_agents=3
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.GENERAL)
        assert strategy == ProcessingStrategy.MULTI_AGENT

    @pytest.mark.asyncio
    async def test_hybrid_strategy(self):
        """Test hybrid strategy selection."""
        agent = ClaudeFlowExpertAgent()

        # Complex coding task
        request = QueryRequest(
            query="Implement a complete REST API with authentication, rate limiting, and comprehensive error handling using best practices",
            max_agents=1
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.CODING)
        assert strategy == ProcessingStrategy.HYBRID

    @pytest.mark.asyncio
    async def test_single_agent_strategy(self):
        """Test single agent strategy selection."""
        agent = ClaudeFlowExpertAgent()

        request = QueryRequest(
            query="Simple task",
            max_agents=1
        )
        strategy = await agent._determine_processing_strategy(request, QueryType.GENERAL)
        assert strategy == ProcessingStrategy.SINGLE_AGENT

class TestCircuitBreakerFunctionality:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state.value == "closed"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker failure counting."""
        breaker = CircuitBreaker(threshold=3)

        async def failure_func():
            raise Exception("Service failure")

        # Trigger failures but don't exceed threshold
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failure_func)

        assert breaker.failure_count == 2
        assert breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold."""
        breaker = CircuitBreaker(threshold=3, timeout=1)

        async def failure_func():
            raise Exception("Service failure")

        # Exceed threshold
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failure_func)

        assert breaker.state.value == "open"

        # Should reject calls when open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(failure_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        breaker = CircuitBreaker(threshold=2, timeout=0.1)  # Short timeout for testing

        async def failure_func():
            raise Exception("Service failure")

        async def success_func():
            return "success"

        # Open the circuit breaker
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failure_func)

        assert breaker.state.value == "open"

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should be half-open and allow one call
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state.value == "closed"
        assert breaker.failure_count == 0

class TestAgentRouting:
    """Test agent routing and selection logic."""

    @pytest.mark.asyncio
    async def test_route_to_agent_coding_task(self, claude_flow_expert_agent_config):
        """Test agent routing for coding tasks."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        agents = await agent.route_to_agent(
            query="Implement a REST API",
            query_type=QueryType.CODING
        )

        assert len(agents) > 0
        assert len(agents) <= 3  # Should return top 3 agents
        # Should include coding-related agents
        coding_agents = ["coder", "backend-dev", "reviewer", "tester"]
        assert any(agent_type in coding_agents for agent_type in agents)

    @pytest.mark.asyncio
    async def test_route_to_agent_with_preferences(self, claude_flow_expert_agent_config):
        """Test agent routing with preferred agents."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        preferred = ["researcher", "analyst"]
        agents = await agent.route_to_agent(
            query="Research machine learning",
            query_type=QueryType.RESEARCH,
            preferred_agents=preferred
        )

        # Should respect preferences when possible
        assert any(agent_type in preferred for agent_type in agents)

    @pytest.mark.asyncio
    async def test_route_to_agent_fallback(self, claude_flow_expert_agent_config):
        """Test agent routing fallback to defaults."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        # Mock capability matrix to simulate routing failure
        with patch.object(agent.capability_matrix, 'get_capabilities_for_query_type') as mock_get:
            mock_get.side_effect = Exception("Routing failed")

            agents = await agent.route_to_agent(
                query="Any query",
                query_type=QueryType.CODING
            )

            # Should fall back to default agents
            default_agents = agent._get_default_agents(QueryType.CODING)
            assert agents == default_agents

class TestMetricsAndPerformanceTracking:
    """Test metrics collection and performance tracking."""

    def test_metrics_initialization(self, claude_flow_expert_agent_config):
        """Test metrics are properly initialized."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        expected_metrics = [
            "queries_processed",
            "successful_queries",
            "rag_queries",
            "multi_agent_coordinations",
            "fallbacks_used",
            "average_processing_time"
        ]

        for metric in expected_metrics:
            assert metric in agent.metrics
            assert isinstance(agent.metrics[metric], (int, float))

    def test_average_processing_time_calculation(self, claude_flow_expert_agent_config):
        """Test average processing time calculation."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        # Simulate processing times
        agent.metrics["queries_processed"] = 0
        agent.metrics["average_processing_time"] = 0.0

        # First query
        agent.metrics["queries_processed"] = 1
        agent._update_average_processing_time(1.0)
        assert agent.metrics["average_processing_time"] == 1.0

        # Second query
        agent.metrics["queries_processed"] = 2
        agent._update_average_processing_time(2.0)
        assert agent.metrics["average_processing_time"] == 1.5

        # Third query
        agent.metrics["queries_processed"] = 3
        agent._update_average_processing_time(3.0)
        assert agent.metrics["average_processing_time"] == 2.0

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, claude_flow_expert_agent_config):
        """Test performance metrics retrieval."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        # Set some test metrics
        agent.metrics["queries_processed"] = 100
        agent.metrics["successful_queries"] = 95
        agent.metrics["rag_queries"] = 30

        metrics = await agent.get_performance_metrics()

        assert "queries_processed" in metrics
        assert "successful_queries" in metrics
        assert "rag_queries" in metrics
        assert "timestamp" in metrics
        assert "circuit_breaker_states" in metrics
        assert "config" in metrics

        # Check circuit breaker states
        assert "rag" in metrics["circuit_breaker_states"]
        assert "agents" in metrics["circuit_breaker_states"]
        assert "coordination" in metrics["circuit_breaker_states"]

        # Check config information
        assert "model" in metrics["config"]
        assert "max_agents" in metrics["config"]
        assert "rag_enabled" in metrics["config"]

class TestDefaultAgentSelection:
    """Test default agent selection logic."""

    def test_default_agents_coding(self, claude_flow_expert_agent_config):
        """Test default agents for coding tasks."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        defaults = agent._get_default_agents(QueryType.CODING)

        assert len(defaults) > 0
        assert "coder" in defaults
        # Should include complementary agents
        expected_agents = ["coder", "reviewer", "tester"]
        assert all(agent_type in defaults for agent_type in expected_agents)

    def test_default_agents_research(self, claude_flow_expert_agent_config):
        """Test default agents for research tasks."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        defaults = agent._get_default_agents(QueryType.RESEARCH)

        assert len(defaults) > 0
        assert "researcher" in defaults
        assert "analyst" in defaults

    def test_default_agents_general(self, claude_flow_expert_agent_config):
        """Test default agents for general tasks."""
        agent = ClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        defaults = agent._get_default_agents(QueryType.GENERAL)

        assert len(defaults) > 0
        # Should include versatile agents
        versatile_agents = ["coder", "researcher"]
        assert any(agent_type in defaults for agent_type in versatile_agents)

class TestConfigurationValidation:
    """Test Claude Flow Expert Agent configuration validation."""

    def test_config_with_invalid_values(self):
        """Test configuration with invalid values."""
        # Test negative values are handled appropriately
        config = ClaudeFlowExpertConfig(
            max_retries=-1,  # Should default or be corrected
            timeout=0,       # Should default or be corrected
            max_coordinated_agents=0  # Should default or be corrected
        )

        agent = ClaudeFlowExpertAgent(config)

        # Agent should still be functional with corrected values
        assert agent.config is not None
        assert agent.capability_matrix is not None

    def test_config_memory_settings(self):
        """Test memory-related configuration."""
        config = ClaudeFlowExpertConfig(
            memory_persistence_enabled=True,
            memory_ttl=7200  # 2 hours
        )

        agent = ClaudeFlowExpertAgent(config)

        assert agent.config.memory_persistence_enabled is True
        assert agent.config.memory_ttl == 7200

    def test_config_performance_settings(self):
        """Test performance-related configuration."""
        config = ClaudeFlowExpertConfig(
            enable_metrics=True,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=10
        )

        agent = ClaudeFlowExpertAgent(config)

        assert agent.config.enable_metrics is True
        assert agent.config.circuit_breaker_enabled is True
        assert agent.config.circuit_breaker_threshold == 10

@pytest.mark.asyncio
class TestClaudeFlowExpertAgentAsyncBehavior:
    """Test Claude Flow Expert Agent async behavior and concurrency."""

    async def test_concurrent_query_processing(self, claude_flow_expert_agent_config):
        """Test Claude Flow Expert Agent handles concurrent queries properly."""
        # Use mock agent for this test to avoid actual LLM calls
        mock_agent = MockClaudeFlowExpertAgent(claude_flow_expert_agent_config)

        # Create multiple concurrent requests
        requests = [
            TestDataGenerator.create_query_request(f"Query {i}")
            for i in range(5)
        ]
        deps = ClaudeFlowExpertDependencies()

        # Process concurrently
        tasks = [
            mock_agent.process_query(request, deps)
            for request in requests
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(result["success"] for result in results)

        # Metrics should be updated
        assert mock_agent.metrics["queries_processed"] == 5

    async def test_async_timeout_handling(self, claude_flow_expert_agent_config):
        """Test async timeout handling."""
        mock_agent = MockClaudeFlowExpertAgent(claude_flow_expert_agent_config)
        mock_agent.processing_delay = 5.0  # 5 second delay

        request = TestDataGenerator.create_query_request("Slow query")
        deps = ClaudeFlowExpertDependencies()

        # Should complete even with delay (using asyncio.wait_for in real implementation)
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        elapsed_time = time.time() - start_time

        assert result["success"] is True
        assert elapsed_time >= 5.0  # Should take at least the delay time

    async def test_async_error_propagation(self, claude_flow_expert_agent_config):
        """Test async error propagation and handling."""
        mock_agent = MockClaudeFlowExpertAgent(claude_flow_expert_agent_config)
        mock_agent.should_fail = True

        request = TestDataGenerator.create_query_request("Failing query")
        deps = ClaudeFlowExpertDependencies()

        # Should handle the failure gracefully
        result = await mock_agent.process_query(request, deps)

        assert result["success"] is False
        assert "fallback_used" in result
        assert result["fallback_used"] is True
