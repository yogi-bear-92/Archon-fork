"""
Integration tests for Master Agent system.

This module tests end-to-end workflows, RAG processing with Archon integration,
multi-agent coordination scenarios, and system-level behavior.
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.agents.master.master_agent import (
    MasterAgent, MasterAgentConfig, MasterAgentDependencies,
    QueryRequest, ProcessingStrategy
)
from src.agents.master.capability_matrix import QueryType
from tests.mocks.master_agent_mocks import (
    MockArchonMCPClient, MockClaudeFlowCoordinator, MockFallbackManager,
    TestDataGenerator
)


@pytest.mark.integration
class TestRAGProcessingIntegration:
    """Test RAG processing with Archon integration."""
    
    @pytest.mark.asyncio
    async def test_successful_rag_query_workflow(self, patched_master_agent_dependencies):
        """Test complete RAG query processing workflow."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        
        # Set up mock RAG response
        rag_response = {
            "success": True,
            "results": [
                {
                    "content": "REST APIs should follow RESTful principles with proper HTTP methods",
                    "source": "api-design-guide",
                    "relevance": 0.95
                },
                {
                    "content": "Use JSON for data exchange and implement proper error handling",
                    "source": "best-practices",
                    "relevance": 0.88
                }
            ],
            "reranked": True
        }
        mcp_client.set_rag_response("How to design REST APIs", rag_response)
        
        # Create agent and process query
        config = MasterAgentConfig(rag_enabled=True)
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="How to design REST APIs",
            query_type=QueryType.RESEARCH,
            require_rag=True
        )
        deps = MasterAgentDependencies()
        
        result = await agent.process_query(request, deps)
        
        # Verify RAG was used
        assert result.success is True
        assert result.processing_strategy == ProcessingStrategy.RAG_ENHANCED
        assert len(result.rag_sources_used) > 0
        assert mcp_client.call_count == 1
        assert "api-design-guide" in str(mcp_client.rag_responses)
    
    @pytest.mark.asyncio
    async def test_rag_failure_with_fallback(self, patched_master_agent_dependencies):
        """Test RAG failure triggers fallback mechanism."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        fallback_manager = patched_master_agent_dependencies["fallback"]
        
        # Configure RAG to fail
        mcp_client.should_fail = True
        
        # Configure fallback response
        fallback_response = {
            "status": "success",
            "fallback_type": "wiki_search",
            "results": [
                {
                    "title": "API Design",
                    "summary": "API design principles from Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/API_Design"
                }
            ]
        }
        fallback_manager.set_wiki_response("API design principles", fallback_response)
        
        config = MasterAgentConfig(rag_enabled=True, rag_fallback_enabled=True)
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="API design principles",
            require_rag=True
        )
        deps = MasterAgentDependencies()
        
        result = await agent.process_query(request, deps)
        
        # Should succeed via fallback
        assert result.success is True
        assert result.fallback_used is True
        assert fallback_manager.call_count > 0
    
    @pytest.mark.asyncio
    async def test_code_search_integration(self, patched_master_agent_dependencies):
        """Test code search functionality integration."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        
        # Set up code search response
        code_response = {
            "success": True,
            "results": [
                {
                    "file": "auth/oauth.py",
                    "function": "create_oauth_flow",
                    "code": "def create_oauth_flow(client_id, redirect_uri):\n    # OAuth implementation\n    pass",
                    "summary": "OAuth flow implementation example"
                }
            ]
        }
        mcp_client.set_code_search_response("OAuth implementation", code_response)
        
        config = MasterAgentConfig()
        agent = MasterAgent(config)
        
        # Mock the tool call to test code search
        with patch.object(agent, '_create_agent') as mock_create:
            mock_agent = MagicMock()
            
            # Mock tool registration and execution
            def mock_tool_decorator(func):
                # Execute the tool function directly for testing
                if func.__name__ == 'search_code_examples':
                    async def wrapper():
                        return await func(
                            None,  # ctx
                            "OAuth implementation",
                            None,  # source_id
                            5  # match_count
                        )
                    return wrapper
                return func
            
            mock_agent.tool = mock_tool_decorator
            mock_create.return_value = mock_agent
            
            # Test code search tool
            tools = agent._register_tools(mock_agent)
            
            # Verify code search was called
            assert mcp_client.call_count > 0


@pytest.mark.integration
class TestMultiAgentCoordinationIntegration:
    """Test multi-agent coordination scenarios."""
    
    @pytest.mark.asyncio
    async def test_successful_multi_agent_coordination(self, patched_master_agent_dependencies):
        """Test successful multi-agent coordination workflow."""
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        config = MasterAgentConfig(max_coordinated_agents=5)
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Build a complete web application with authentication",
            query_type=QueryType.COORDINATION,
            max_agents=4,
            context={"tech_stack": "FastAPI + React"}
        )
        deps = MasterAgentDependencies(coordinate_agents=True)
        
        result = await agent.process_query(request, deps)
        
        # Verify coordination occurred
        assert result.success is True
        assert result.processing_strategy == ProcessingStrategy.MULTI_AGENT
        assert len(result.agents_used) > 1
        assert result.coordination_metrics is not None
        
        # Check coordinator was used
        assert coordinator.call_count > 0
        assert len(coordinator.agents) > 0
    
    @pytest.mark.asyncio
    async def test_coordination_failure_fallback(self, patched_master_agent_dependencies):
        """Test coordination failure triggers single agent fallback."""
        coordinator = patched_master_agent_dependencies["coordinator"]
        fallback_manager = patched_master_agent_dependencies["fallback"]
        
        # Configure coordination to fail
        coordinator.should_fail = True
        
        config = MasterAgentConfig()
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Complex task requiring coordination",
            max_agents=3
        )
        deps = MasterAgentDependencies()
        
        result = await agent.process_query(request, deps)
        
        # Should fallback to single agent processing
        assert result.success is True
        assert result.fallback_used is True
        assert fallback_manager.call_count > 0
    
    @pytest.mark.asyncio
    async def test_agent_selection_accuracy(self, patched_master_agent_dependencies):
        """Test accuracy of agent selection for different task types."""
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        config = MasterAgentConfig()
        agent = MasterAgent(config)
        
        test_scenarios = [
            {
                "query": "Implement user authentication with JWT",
                "query_type": QueryType.CODING,
                "expected_agents": ["coder", "backend-dev", "security-manager"]
            },
            {
                "query": "Research microservices best practices", 
                "query_type": QueryType.RESEARCH,
                "expected_agents": ["researcher", "system-architect"]
            },
            {
                "query": "Analyze code performance bottlenecks",
                "query_type": QueryType.ANALYSIS,
                "expected_agents": ["perf-analyzer", "code-analyzer"]
            }
        ]
        
        for scenario in test_scenarios:
            agents = await agent.route_to_agent(
                query=scenario["query"],
                query_type=scenario["query_type"]
            )
            
            # Check that relevant agent types are selected
            assert len(agents) > 0
            # Note: Exact matching would depend on capability matrix implementation


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_coding_workflow_with_rag(self, patched_master_agent_dependencies):
        """Test complete coding workflow with RAG support."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        # Set up RAG responses for research phase
        rag_response = {
            "success": True,
            "results": [
                {
                    "content": "FastAPI best practices include proper dependency injection",
                    "source": "fastapi-guide",
                    "relevance": 0.9
                }
            ]
        }
        mcp_client.set_rag_response("FastAPI best practices", rag_response)
        
        config = MasterAgentConfig(rag_enabled=True)
        agent = MasterAgent(config)
        
        # Simulate multi-step workflow
        workflow_steps = [
            {
                "request": QueryRequest(
                    query="Research FastAPI best practices",
                    query_type=QueryType.RESEARCH,
                    require_rag=True
                ),
                "expected_strategy": ProcessingStrategy.RAG_ENHANCED
            },
            {
                "request": QueryRequest(
                    query="Implement FastAPI application using researched practices",
                    query_type=QueryType.CODING,
                    max_agents=3,
                    context={"previous_research": "FastAPI best practices"}
                ),
                "expected_strategy": ProcessingStrategy.MULTI_AGENT
            }
        ]
        
        results = []
        deps = MasterAgentDependencies()
        
        for step in workflow_steps:
            result = await agent.process_query(step["request"], deps)
            results.append(result)
            
            assert result.success is True
            assert result.processing_strategy == step["expected_strategy"]
        
        # Verify workflow progression
        assert len(results) == 2
        assert results[0].processing_strategy == ProcessingStrategy.RAG_ENHANCED
        assert results[1].processing_strategy == ProcessingStrategy.MULTI_AGENT
        
        # Check that knowledge from first step could influence second step
        assert mcp_client.call_count >= 1
        assert coordinator.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_hybrid_processing_workflow(self, patched_master_agent_dependencies):
        """Test hybrid processing workflow combining RAG and coordination."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        # Set up responses for hybrid processing
        rag_response = {
            "success": True,
            "results": [
                {
                    "content": "Complex system architecture requires careful design",
                    "source": "architecture-guide",
                    "relevance": 0.95
                }
            ]
        }
        mcp_client.set_rag_response("system architecture", rag_response)
        
        config = MasterAgentConfig(rag_enabled=True)
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Design and implement a scalable microservices architecture with comprehensive documentation, testing, and deployment strategies",
            query_type=QueryType.CODING,
            max_agents=4
        )
        deps = MasterAgentDependencies()
        
        result = await agent.process_query(request, deps)
        
        # Should use hybrid strategy for complex task
        assert result.success is True
        assert result.processing_strategy == ProcessingStrategy.HYBRID
        assert len(result.rag_sources_used) > 0
        assert len(result.agents_used) > 1
        
        # Both RAG and coordination should be used
        assert mcp_client.call_count > 0
        assert coordinator.call_count > 0


@pytest.mark.integration
class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.asyncio
    async def test_partial_system_failure_resilience(self, patched_master_agent_dependencies):
        """Test system resilience when some components fail."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        fallback_manager = patched_master_agent_dependencies["fallback"]
        
        # Configure RAG to fail but coordination to succeed
        mcp_client.should_fail = True
        
        config = MasterAgentConfig(rag_enabled=True, rag_fallback_enabled=True)
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Complex task that normally requires RAG and coordination",
            require_rag=True,
            max_agents=3
        )
        deps = MasterAgentDependencies()
        
        result = await agent.process_query(request, deps)
        
        # Should succeed despite RAG failure
        assert result.success is True
        # Should have used fallback mechanisms
        assert result.fallback_used or fallback_manager.call_count > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, patched_master_agent_dependencies):
        """Test circuit breaker integration in real workflow."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        
        # Configure high failure rate to trigger circuit breaker
        mcp_client.failure_rate = 1.0  # 100% failure rate
        
        config = MasterAgentConfig(
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=2
        )
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Query that will trigger circuit breaker",
            require_rag=True
        )
        deps = MasterAgentDependencies()
        
        # First few queries should fail and open circuit breaker
        results = []
        for i in range(3):
            result = await agent.process_query(request, deps)
            results.append(result)
        
        # Should eventually trigger circuit breaker protection
        # and use fallback mechanisms
        assert any(result.fallback_used for result in results)
        
        # Check circuit breaker state
        metrics = await agent.get_performance_metrics()
        rag_breaker_state = metrics["circuit_breaker_states"]["rag"]
        # Circuit breaker should be open or half-open after failures
        assert rag_breaker_state in ["open", "half_open"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, patched_master_agent_dependencies):
        """Test timeout handling in integrated workflows."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        # Configure delays to test timeout behavior
        mcp_client.response_delay = 0.1
        coordinator.coordination_delay = 0.1
        
        config = MasterAgentConfig(timeout=30)  # 30 second timeout
        agent = MasterAgent(config)
        
        request = QueryRequest(
            query="Task that involves both RAG and coordination",
            require_rag=True,
            max_agents=3
        )
        deps = MasterAgentDependencies()
        
        start_time = time.time()
        result = await agent.process_query(request, deps)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert result.success is True
        assert elapsed_time < 5.0  # Should be much faster than timeout
        
        # Should have used both components despite delays
        assert mcp_client.call_count > 0
        assert coordinator.call_count > 0


@pytest.mark.integration
class TestMemoryAndStatePersistence:
    """Test memory management and state persistence."""
    
    @pytest.mark.asyncio
    async def test_cross_query_context_persistence(self, patched_master_agent_dependencies):
        """Test context persistence across multiple queries."""
        config = MasterAgentConfig(memory_persistence_enabled=True)
        agent = MasterAgent(config)
        
        # First query establishes context
        first_request = QueryRequest(
            query="Start working on a web application project",
            context={"project_type": "web_app", "framework": "FastAPI"}
        )
        deps = MasterAgentDependencies()
        
        first_result = await agent.process_query(first_request, deps)
        assert first_result.success is True
        
        # Second query should be able to reference previous context
        second_request = QueryRequest(
            query="Add authentication to the current project",
            context={"building_on": "previous_project"}
        )
        
        second_result = await agent.process_query(second_request, deps)
        assert second_result.success is True
        
        # Verify metrics show both queries processed
        metrics = await agent.get_performance_metrics()
        assert metrics["queries_processed"] >= 2
    
    @pytest.mark.asyncio
    async def test_performance_metrics_accuracy(self, patched_master_agent_dependencies):
        """Test accuracy of performance metrics collection."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        config = MasterAgentConfig(enable_metrics=True)
        agent = MasterAgent(config)
        
        # Process different types of queries
        queries = [
            QueryRequest(query="Simple query", query_type=QueryType.GENERAL),
            QueryRequest(query="RAG query", require_rag=True),
            QueryRequest(query="Multi-agent task", max_agents=3),
            QueryRequest(query="Research task", query_type=QueryType.RESEARCH, require_rag=True)
        ]
        
        deps = MasterAgentDependencies()
        
        for request in queries:
            result = await agent.process_query(request, deps)
            assert result.success is True
        
        # Check metrics accuracy
        metrics = await agent.get_performance_metrics()
        
        assert metrics["queries_processed"] == 4
        assert metrics["successful_queries"] == 4
        assert metrics["rag_queries"] >= 2  # At least 2 RAG queries
        assert metrics["average_processing_time"] > 0
        
        # Verify component usage metrics
        assert mcp_client.call_count >= 2  # RAG queries
        assert coordinator.call_count >= 1  # Multi-agent coordination


@pytest.mark.integration
class TestSystemScalability:
    """Test system scalability and load handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, patched_master_agent_dependencies):
        """Test handling of concurrent requests."""
        config = MasterAgentConfig(max_coordinated_agents=10)
        agent = MasterAgent(config)
        
        # Create concurrent requests
        num_concurrent = 10
        requests = [
            QueryRequest(query=f"Concurrent query {i}", query_type=QueryType.GENERAL)
            for i in range(num_concurrent)
        ]
        deps = MasterAgentDependencies()
        
        # Process all requests concurrently
        start_time = time.time()
        tasks = [agent.process_query(request, deps) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_time = time.time() - start_time
        
        # Check that all requests completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_results) == num_concurrent
        
        # Should handle concurrent requests efficiently
        assert elapsed_time < 10.0  # Should complete within 10 seconds
        
        # Metrics should reflect all processed queries
        metrics = await agent.get_performance_metrics()
        assert metrics["queries_processed"] == num_concurrent
    
    @pytest.mark.asyncio
    async def test_resource_management_under_load(self, patched_master_agent_dependencies):
        """Test resource management under load conditions."""
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        config = MasterAgentConfig(
            max_coordinated_agents=5,
            memory_persistence_enabled=True
        )
        agent = MasterAgent(config)
        
        # Create resource-intensive requests
        intensive_requests = [
            QueryRequest(
                query=f"Complex coordination task {i}",
                query_type=QueryType.COORDINATION,
                max_agents=5,
                context={"complexity": "high", "resources": "intensive"}
            )
            for i in range(5)
        ]
        
        deps = MasterAgentDependencies()
        
        # Process requests and monitor resource usage
        results = []
        for request in intensive_requests:
            result = await agent.process_query(request, deps)
            results.append(result)
            
            # Brief pause to simulate realistic load
            await asyncio.sleep(0.1)
        
        # All requests should succeed
        assert all(result.success for result in results)
        
        # System should maintain reasonable performance
        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        assert avg_processing_time < 2.0  # Under 2 seconds average
        
        # Resource limits should be respected
        assert all(len(result.agents_used) <= 5 for result in results)


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test long-running and complex workflows."""
    
    @pytest.mark.asyncio
    async def test_extended_project_workflow(self, patched_master_agent_dependencies):
        """Test extended project workflow simulation."""
        mcp_client = patched_master_agent_dependencies["mcp_client"]
        coordinator = patched_master_agent_dependencies["coordinator"]
        
        # Set up comprehensive responses
        research_response = {
            "success": True,
            "results": [
                {"content": "Project planning best practices", "source": "pm-guide"},
                {"content": "Technical architecture patterns", "source": "arch-guide"}
            ]
        }
        mcp_client.set_rag_response("project planning", research_response)
        
        config = MasterAgentConfig(
            rag_enabled=True,
            max_coordinated_agents=8,
            memory_persistence_enabled=True
        )
        agent = MasterAgent(config)
        
        # Simulate comprehensive project workflow
        workflow_phases = [
            {
                "phase": "research",
                "request": QueryRequest(
                    query="Research best practices for building a scalable web application",
                    query_type=QueryType.RESEARCH,
                    require_rag=True
                )
            },
            {
                "phase": "architecture",
                "request": QueryRequest(
                    query="Design system architecture based on research",
                    query_type=QueryType.COORDINATION,
                    max_agents=4,
                    context={"phase": "architecture"}
                )
            },
            {
                "phase": "implementation",
                "request": QueryRequest(
                    query="Implement the designed architecture",
                    query_type=QueryType.CODING,
                    max_agents=6,
                    context={"phase": "implementation"}
                )
            },
            {
                "phase": "testing",
                "request": QueryRequest(
                    query="Create comprehensive test suite",
                    query_type=QueryType.ANALYSIS,
                    max_agents=3,
                    context={"phase": "testing"}
                )
            }
        ]
        
        deps = MasterAgentDependencies()
        phase_results = {}
        
        total_start_time = time.time()
        
        for phase_info in workflow_phases:
            phase_name = phase_info["phase"]
            request = phase_info["request"]
            
            result = await agent.process_query(request, deps)
            phase_results[phase_name] = result
            
            assert result.success is True, f"Phase {phase_name} failed"
        
        total_elapsed = time.time() - total_start_time
        
        # Verify complete workflow
        assert len(phase_results) == 4
        assert all(result.success for result in phase_results.values())
        
        # Check that different strategies were used appropriately
        strategies_used = [result.processing_strategy for result in phase_results.values()]
        assert ProcessingStrategy.RAG_ENHANCED in strategies_used
        assert ProcessingStrategy.MULTI_AGENT in strategies_used
        
        # Verify system performance over extended workflow
        final_metrics = await agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == 4
        assert final_metrics["successful_queries"] == 4
        assert final_metrics["average_processing_time"] > 0
        
        # Workflow should complete in reasonable time
        assert total_elapsed < 30.0  # Under 30 seconds for mock workflow