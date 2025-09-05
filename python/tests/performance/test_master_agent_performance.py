"""
Performance tests for Master Agent system.

This module tests query processing latency, concurrent request handling,
memory usage optimization, and system performance under various load conditions.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import patch

from src.agents.master.master_agent import MasterAgentConfig, MasterAgentDependencies, QueryRequest
from src.agents.master.capability_matrix import QueryType
from tests.mocks.master_agent_mocks import MockMasterAgent, TestDataGenerator, PerformanceTestHelper


@pytest.mark.performance
class TestQueryProcessingLatency:
    """Test query processing latency and response times."""
    
    @pytest.mark.asyncio
    async def test_single_query_latency_baseline(self, performance_monitor):
        """Test single query processing latency baseline."""
        config = MasterAgentConfig(
            timeout=30,
            enable_rate_limiting=False,
            circuit_breaker_enabled=False
        )
        mock_agent = MockMasterAgent(config)
        mock_agent.processing_delay = 0.05  # 50ms base processing time
        
        request = QueryRequest(
            query="Test query for latency measurement",
            query_type=QueryType.GENERAL
        )
        deps = MasterAgentDependencies()
        
        # Measure single query performance
        performance_monitor.start()
        performance_monitor.sample_memory()
        
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        end_time = time.time()
        
        performance_monitor.sample_memory()
        performance_monitor.stop()
        
        # Verify performance expectations
        latency_ms = (end_time - start_time) * 1000
        
        assert result["success"] is True
        assert latency_ms < 2000  # Under 2 seconds
        assert latency_ms >= 50    # At least base processing time
        
        # Memory usage should be reasonable
        assert performance_monitor.peak_memory_mb < 100  # Under 100MB
    
    @pytest.mark.asyncio
    async def test_rag_query_latency(self, performance_monitor):
        """Test RAG-enhanced query latency."""
        config = MasterAgentConfig(rag_enabled=True)
        mock_agent = MockMasterAgent(config)
        mock_agent.mcp_client.response_delay = 0.1  # 100ms RAG delay
        
        request = QueryRequest(
            query="Research query requiring knowledge base lookup",
            query_type=QueryType.RESEARCH,
            require_rag=True
        )
        deps = MasterAgentDependencies()
        
        performance_monitor.start()
        performance_monitor.sample_memory()
        
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        end_time = time.time()
        
        performance_monitor.sample_memory()
        performance_monitor.stop()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert result["success"] is True
        assert latency_ms < 3000  # Under 3 seconds for RAG queries
        assert latency_ms >= 100   # At least RAG processing time
        assert mock_agent.mcp_client.call_count == 1
        
        # RAG queries should use more memory but stay reasonable
        assert performance_monitor.peak_memory_mb < 150  # Under 150MB
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_latency(self, performance_monitor):
        """Test multi-agent coordination latency."""
        config = MasterAgentConfig(max_coordinated_agents=5)
        mock_agent = MockMasterAgent(config)
        mock_agent.claude_flow_coordinator.coordination_delay = 0.15  # 150ms coordination delay
        
        request = QueryRequest(
            query="Complex task requiring multiple agents",
            query_type=QueryType.COORDINATION,
            max_agents=4
        )
        deps = MasterAgentDependencies()
        
        performance_monitor.start()
        
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        end_time = time.time()
        
        performance_monitor.stop()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert result["success"] is True
        assert latency_ms < 5000  # Under 5 seconds for coordination
        assert latency_ms >= 150   # At least coordination time
        assert mock_agent.claude_flow_coordinator.call_count > 0
        
        # Multi-agent coordination should manage resources efficiently
        assert performance_monitor.peak_memory_mb < 200  # Under 200MB
    
    @pytest.mark.asyncio
    async def test_latency_percentiles(self, async_test_helper):
        """Test latency distribution and percentiles."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Add some variability to processing times
        mock_agent.processing_delay = 0.05
        
        # Run multiple queries to get latency distribution
        num_queries = 50
        requests = [
            QueryRequest(query=f"Test query {i}", query_type=QueryType.GENERAL)
            for i in range(num_queries)
        ]
        deps = MasterAgentDependencies()
        
        latencies = []
        
        for request in requests:
            start_time = time.time()
            result = await mock_agent.process_query(request, deps)
            end_time = time.time()
            
            assert result["success"] is True
            latencies.append((end_time - start_time) * 1000)
        
        # Calculate percentiles
        percentiles = PerformanceTestHelper.calculate_percentiles(latencies)
        
        # Performance assertions
        assert percentiles["p50"] < 1000   # 50th percentile under 1 second
        assert percentiles["p90"] < 2000   # 90th percentile under 2 seconds
        assert percentiles["p95"] < 2500   # 95th percentile under 2.5 seconds
        assert percentiles["p99"] < 3000   # 99th percentile under 3 seconds
        assert percentiles["mean"] < 1500  # Average under 1.5 seconds


@pytest.mark.performance
class TestConcurrentRequestHandling:
    """Test concurrent request handling and throughput."""
    
    @pytest.mark.asyncio
    async def test_concurrent_query_throughput(self, performance_monitor, async_test_helper):
        """Test system throughput with concurrent queries."""
        config = MasterAgentConfig(max_coordinated_agents=10)
        mock_agent = MockMasterAgent(config)
        mock_agent.processing_delay = 0.1  # 100ms base processing
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        throughput_results = {}
        
        for concurrency in concurrency_levels:
            requests = [
                QueryRequest(query=f"Concurrent query {i}", query_type=QueryType.GENERAL)
                for i in range(concurrency)
            ]
            deps = MasterAgentDependencies()
            
            performance_monitor.start()
            
            # Measure concurrent execution
            start_time = time.time()
            
            # Create coroutines for concurrent execution
            coroutines = [
                mock_agent.process_query(request, deps)
                for request in requests
            ]
            
            results = await async_test_helper.run_concurrent(coroutines, concurrency)
            end_time = time.time()
            
            performance_monitor.stop()
            
            # Calculate throughput metrics
            total_time = end_time - start_time
            successful_queries = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            throughput_qps = successful_queries / total_time
            
            throughput_results[concurrency] = {
                "total_time": total_time,
                "successful_queries": successful_queries,
                "throughput_qps": throughput_qps,
                "peak_memory_mb": performance_monitor.peak_memory_mb
            }
            
            # Basic performance assertions
            assert successful_queries == concurrency  # All queries should succeed
            assert throughput_qps > 1.0  # At least 1 query per second
            assert performance_monitor.peak_memory_mb < 300  # Memory usage under control
        
        # Throughput should scale reasonably with concurrency
        assert throughput_results[10]["throughput_qps"] > throughput_results[1]["throughput_qps"]
        
        # Higher concurrency shouldn't cause exponential performance degradation
        high_concurrency_efficiency = throughput_results[20]["throughput_qps"] / 20
        low_concurrency_efficiency = throughput_results[1]["throughput_qps"] / 1
        efficiency_ratio = high_concurrency_efficiency / low_concurrency_efficiency
        
        assert efficiency_ratio > 0.5  # Should maintain at least 50% efficiency
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, performance_monitor):
        """Test performance with mixed query types."""
        config = MasterAgentConfig(rag_enabled=True)
        mock_agent = MockMasterAgent(config)
        
        # Configure different delays for different components
        mock_agent.processing_delay = 0.05
        mock_agent.mcp_client.response_delay = 0.1
        mock_agent.claude_flow_coordinator.coordination_delay = 0.15
        
        # Create mixed workload
        mixed_requests = [
            # Simple queries (40%)
            *[QueryRequest(query=f"Simple query {i}", query_type=QueryType.GENERAL) for i in range(8)],
            # RAG queries (30%)
            *[QueryRequest(query=f"Research query {i}", require_rag=True) for i in range(6)],
            # Multi-agent tasks (20%)
            *[QueryRequest(query=f"Multi-agent task {i}", max_agents=3) for i in range(4)],
            # Complex hybrid tasks (10%)
            *[QueryRequest(query=f"Complex hybrid task {i}", query_type=QueryType.CODING, max_agents=2, require_rag=False) for i in range(2)]
        ]
        
        deps = MasterAgentDependencies()
        
        performance_monitor.start()
        start_time = time.time()
        
        # Process mixed workload concurrently
        tasks = [mock_agent.process_query(request, deps) for request in mixed_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        performance_monitor.stop()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        total_time = end_time - start_time
        throughput = len(successful_results) / total_time
        
        # Performance assertions
        assert len(successful_results) == len(mixed_requests)  # All should succeed
        assert throughput > 5.0  # At least 5 queries per second for mixed workload
        assert total_time < 10.0  # Complete mixed workload in under 10 seconds
        assert performance_monitor.peak_memory_mb < 250  # Memory under 250MB
        
        # Verify different query types were processed
        assert mock_agent.mcp_client.call_count >= 6  # RAG queries
        assert mock_agent.claude_flow_coordinator.call_count >= 4  # Multi-agent tasks
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, performance_monitor):
        """Test performance under sustained load."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        mock_agent.processing_delay = 0.05
        
        # Sustained load test parameters
        duration_seconds = 10
        queries_per_second = 5
        total_queries = duration_seconds * queries_per_second
        
        performance_monitor.start()
        start_time = time.time()
        
        # Generate sustained load
        tasks = []
        for i in range(total_queries):
            request = QueryRequest(query=f"Sustained load query {i}")
            deps = MasterAgentDependencies()
            
            task = asyncio.create_task(mock_agent.process_query(request, deps))
            tasks.append(task)
            
            # Maintain target rate
            if i < total_queries - 1:  # Don't sleep after last query
                await asyncio.sleep(1.0 / queries_per_second)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        performance_monitor.stop()
        
        # Analyze sustained load performance
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        actual_duration = end_time - start_time
        actual_throughput = len(successful_results) / actual_duration
        
        # Performance assertions for sustained load
        assert len(successful_results) == total_queries  # All queries successful
        assert actual_duration <= duration_seconds + 2  # Reasonable timing tolerance
        assert actual_throughput >= queries_per_second * 0.8  # 80% of target throughput
        assert performance_monitor.peak_memory_mb < 200  # Stable memory usage
        
        # System should remain stable throughout sustained load
        final_metrics = await mock_agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == total_queries


@pytest.mark.performance
class TestMemoryUsageOptimization:
    """Test memory usage optimization and resource management."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_single_query(self, memory_monitor):
        """Test memory usage for single query processing."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        request = QueryRequest(query="Memory test query")
        deps = MasterAgentDependencies()
        
        # Monitor memory during query processing
        memory_monitor.start()
        memory_monitor.sample_memory()
        
        result = await mock_agent.process_query(request, deps)
        
        memory_monitor.sample_memory()
        memory_monitor.stop()
        
        assert result["success"] is True
        
        # Memory usage should be reasonable for single query
        peak_memory = memory_monitor.get_peak_memory()
        memory_delta = memory_monitor.get_memory_delta()
        
        assert peak_memory < 50 * 1024 * 1024  # Under 50MB peak
        assert abs(memory_delta) < 10 * 1024 * 1024  # Delta under 10MB
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, memory_monitor):
        """Test memory usage under concurrent load."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Create concurrent requests
        num_concurrent = 20
        requests = [
            QueryRequest(query=f"Memory load test {i}")
            for i in range(num_concurrent)
        ]
        deps = MasterAgentDependencies()
        
        memory_monitor.start()
        initial_memory = memory_monitor.get_peak_memory()
        
        # Process concurrent requests
        tasks = [mock_agent.process_query(request, deps) for request in requests]
        results = await asyncio.gather(*tasks)
        
        peak_memory = memory_monitor.get_peak_memory()
        memory_monitor.stop()
        
        # All requests should succeed
        assert all(result["success"] for result in results)
        
        # Memory usage should scale reasonably with load
        memory_increase = peak_memory - initial_memory
        memory_per_query = memory_increase / num_concurrent
        
        assert peak_memory < 200 * 1024 * 1024  # Under 200MB total
        assert memory_per_query < 5 * 1024 * 1024  # Under 5MB per query
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_processing(self, memory_monitor):
        """Test memory cleanup after query processing."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        memory_monitor.start()
        baseline_memory = memory_monitor.get_peak_memory()
        
        # Process batch of queries
        batch_size = 10
        for batch in range(3):  # 3 batches
            requests = [
                QueryRequest(query=f"Cleanup test batch {batch} query {i}")
                for i in range(batch_size)
            ]
            deps = MasterAgentDependencies()
            
            tasks = [mock_agent.process_query(request, deps) for request in requests]
            results = await asyncio.gather(*tasks)
            
            assert all(result["success"] for result in results)
            
            # Allow time for cleanup
            await asyncio.sleep(0.1)
            memory_monitor.sample_memory()
        
        final_memory = memory_monitor.get_peak_memory()
        memory_monitor.stop()
        
        # Memory shouldn't grow excessively across batches
        memory_growth = final_memory - baseline_memory
        assert memory_growth < 50 * 1024 * 1024  # Under 50MB growth
    
    @pytest.mark.asyncio
    async def test_memory_usage_different_query_types(self, memory_monitor):
        """Test memory usage patterns for different query types."""
        config = MasterAgentConfig(rag_enabled=True)
        mock_agent = MockMasterAgent(config)
        
        query_types = [
            ("simple", QueryRequest(query="Simple query", query_type=QueryType.GENERAL)),
            ("rag", QueryRequest(query="RAG query", require_rag=True)),
            ("multi_agent", QueryRequest(query="Multi-agent query", max_agents=3)),
            ("hybrid", QueryRequest(query="Complex hybrid query", query_type=QueryType.CODING, max_agents=2))
        ]
        
        memory_usage = {}
        deps = MasterAgentDependencies()
        
        for query_name, request in query_types:
            memory_monitor.start()
            initial_memory = memory_monitor.get_peak_memory()
            
            result = await mock_agent.process_query(request, deps)
            
            peak_memory = memory_monitor.get_peak_memory()
            memory_monitor.stop()
            
            assert result["success"] is True
            
            memory_usage[query_name] = {
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "memory_delta_mb": (peak_memory - initial_memory) / (1024 * 1024)
            }
            
            # Reset for next test
            memory_monitor.stop()
        
        # Memory usage should be reasonable for all query types
        for query_type, usage in memory_usage.items():
            assert usage["peak_memory_mb"] < 100, f"{query_type} used {usage['peak_memory_mb']}MB"
            assert usage["memory_delta_mb"] < 30, f"{query_type} delta {usage['memory_delta_mb']}MB"


@pytest.mark.performance
class TestAgentSelectionPerformance:
    """Test performance of agent selection and routing algorithms."""
    
    @pytest.mark.asyncio
    async def test_agent_routing_performance(self, performance_monitor):
        """Test performance of agent routing algorithm."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Test routing performance for different query types
        query_scenarios = [
            (QueryType.CODING, "Implement a web application"),
            (QueryType.RESEARCH, "Research machine learning algorithms"),
            (QueryType.ANALYSIS, "Analyze system performance"),
            (QueryType.COORDINATION, "Plan project architecture"),
            (QueryType.GENERAL, "General assistance request")
        ]
        
        routing_times = []
        
        for query_type, query_text in query_scenarios:
            # Measure routing performance
            start_time = time.time()
            
            agents = await mock_agent.route_to_agent(
                query=query_text,
                query_type=query_type
            )
            
            end_time = time.time()
            routing_time_ms = (end_time - start_time) * 1000
            routing_times.append(routing_time_ms)
            
            # Verify routing results
            assert len(agents) > 0
            assert len(agents) <= 3  # Should return top 3 agents
            assert routing_time_ms < 100  # Under 100ms for routing
        
        # Overall routing performance
        avg_routing_time = statistics.mean(routing_times)
        max_routing_time = max(routing_times)
        
        assert avg_routing_time < 50   # Average under 50ms
        assert max_routing_time < 100  # Maximum under 100ms
    
    @pytest.mark.asyncio
    async def test_capability_matrix_lookup_performance(self, performance_monitor):
        """Test capability matrix lookup performance."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Test capability lookups
        num_lookups = 100
        lookup_times = []
        
        for i in range(num_lookups):
            query_type = QueryType.CODING if i % 2 == 0 else QueryType.RESEARCH
            
            start_time = time.time()
            capabilities = mock_agent.capability_matrix.get_capabilities_for_query_type(query_type)
            end_time = time.time()
            
            lookup_time_ms = (end_time - start_time) * 1000
            lookup_times.append(lookup_time_ms)
            
            assert len(capabilities) > 0
        
        # Lookup performance analysis
        avg_lookup_time = statistics.mean(lookup_times)
        p95_lookup_time = PerformanceTestHelper.calculate_percentiles(lookup_times)["p95"]
        
        assert avg_lookup_time < 10   # Average under 10ms
        assert p95_lookup_time < 20   # 95th percentile under 20ms
    
    @pytest.mark.asyncio
    async def test_agent_selection_accuracy_vs_speed(self, performance_monitor):
        """Test trade-off between agent selection accuracy and speed."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        test_queries = [
            ("Build a REST API with authentication", QueryType.CODING),
            ("Research GraphQL vs REST performance", QueryType.RESEARCH),
            ("Analyze database query performance", QueryType.ANALYSIS),
            ("Plan microservices migration", QueryType.COORDINATION)
        ]
        
        selection_results = []
        
        for query, query_type in test_queries:
            start_time = time.time()
            
            # Test agent selection with different parameters
            agents_fast = await mock_agent.route_to_agent(query, query_type)
            agents_comprehensive = await mock_agent.route_to_agent(
                query, query_type, preferred_agents=None
            )
            
            end_time = time.time()
            selection_time_ms = (end_time - start_time) * 1000
            
            selection_results.append({
                "query_type": query_type,
                "selection_time_ms": selection_time_ms,
                "agents_fast": agents_fast,
                "agents_comprehensive": agents_comprehensive,
                "accuracy_score": len(set(agents_fast) & set(agents_comprehensive)) / max(len(agents_fast), len(agents_comprehensive))
            })
            
            # Performance requirements
            assert selection_time_ms < 200  # Under 200ms for selection
        
        # Overall accuracy and performance
        avg_selection_time = statistics.mean([r["selection_time_ms"] for r in selection_results])
        avg_accuracy = statistics.mean([r["accuracy_score"] for r in selection_results])
        
        assert avg_selection_time < 100  # Average under 100ms
        assert avg_accuracy > 0.7        # At least 70% accuracy


@pytest.mark.performance
class TestFallbackPerformance:
    """Test performance of fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_fallback_activation_speed(self, performance_monitor):
        """Test speed of fallback activation when primary services fail."""
        config = MasterAgentConfig(rag_fallback_enabled=True)
        mock_agent = MockMasterAgent(config)
        
        # Configure primary service to fail
        mock_agent.mcp_client.should_fail = True
        
        request = QueryRequest(
            query="Query that will trigger fallback",
            require_rag=True
        )
        deps = MasterAgentDependencies()
        
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        end_time = time.time()
        
        fallback_time_ms = (end_time - start_time) * 1000
        
        # Fallback should activate quickly
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert fallback_time_ms < 3000  # Under 3 seconds including fallback
        assert mock_agent.fallback_manager.call_count > 0
    
    @pytest.mark.asyncio
    async def test_multiple_fallback_strategies_performance(self, performance_monitor):
        """Test performance when multiple fallback strategies are triggered."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Configure multiple components to fail in sequence
        mock_agent.mcp_client.should_fail = True
        mock_agent.claude_flow_coordinator.should_fail = True
        
        request = QueryRequest(
            query="Query requiring both RAG and coordination",
            require_rag=True,
            max_agents=3
        )
        deps = MasterAgentDependencies()
        
        start_time = time.time()
        result = await mock_agent.process_query(request, deps)
        end_time = time.time()
        
        total_fallback_time_ms = (end_time - start_time) * 1000
        
        # Should handle multiple fallbacks efficiently
        assert result["success"] is True
        assert result["fallback_used"] is True
        assert total_fallback_time_ms < 5000  # Under 5 seconds for multiple fallbacks
        
        # Both fallback systems should have been engaged
        assert mock_agent.fallback_manager.call_count > 0


@pytest.mark.performance 
@pytest.mark.slow
class TestLoadTestingScenarios:
    """Extended load testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self, performance_monitor, async_test_helper):
        """Test handling of burst load scenarios."""
        config = MasterAgentConfig(max_coordinated_agents=15)
        mock_agent = MockMasterAgent(config)
        
        # Simulate burst load: many requests in short time
        burst_size = 50
        burst_requests = [
            QueryRequest(query=f"Burst query {i}", query_type=QueryType.GENERAL)
            for i in range(burst_size)
        ]
        deps = MasterAgentDependencies()
        
        performance_monitor.start()
        start_time = time.time()
        
        # Submit all requests simultaneously (burst)
        tasks = [mock_agent.process_query(request, deps) for request in burst_requests]
        results = await async_test_helper.run_concurrent(tasks, max_concurrent=25)
        
        end_time = time.time()
        performance_monitor.stop()
        
        # Analyze burst handling
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        burst_time = end_time - start_time
        throughput = len(successful_results) / burst_time
        
        # Burst handling requirements
        assert len(successful_results) == burst_size  # All requests should succeed
        assert burst_time < 15.0  # Complete burst in under 15 seconds
        assert throughput > 3.0   # At least 3 queries per second during burst
        assert performance_monitor.peak_memory_mb < 400  # Memory under control
    
    @pytest.mark.asyncio
    async def test_gradual_load_increase(self, performance_monitor):
        """Test performance under gradually increasing load."""
        config = MasterAgentConfig()
        mock_agent = MockMasterAgent(config)
        
        # Test with increasing load levels
        load_levels = [5, 10, 15, 20, 25]  # Queries per wave
        wave_duration = 2.0  # Seconds per wave
        
        performance_results = []
        
        for load_level in load_levels:
            requests = [
                QueryRequest(query=f"Load level {load_level} query {i}")
                for i in range(load_level)
            ]
            deps = MasterAgentDependencies()
            
            performance_monitor.start()
            wave_start = time.time()
            
            # Process load level
            tasks = [mock_agent.process_query(request, deps) for request in requests]
            results = await asyncio.gather(*tasks)
            
            wave_end = time.time()
            performance_monitor.stop()
            
            # Calculate wave metrics
            wave_time = wave_end - wave_start
            successful = sum(1 for r in results if r.get("success"))
            wave_throughput = successful / wave_time
            
            performance_results.append({
                "load_level": load_level,
                "wave_time": wave_time,
                "throughput": wave_throughput,
                "success_rate": successful / load_level,
                "peak_memory_mb": performance_monitor.peak_memory_mb
            })
            
            # Allow brief recovery between waves
            await asyncio.sleep(0.5)
        
        # Analyze scalability
        for i, result in enumerate(performance_results):
            load_level = result["load_level"]
            
            # Performance should remain acceptable at all load levels
            assert result["success_rate"] >= 0.95  # 95% success rate
            assert result["throughput"] > 2.0      # At least 2 QPS
            assert result["peak_memory_mb"] < 300  # Memory under 300MB
            
            # System shouldn't degrade catastrophically
            if i > 0:
                prev_result = performance_results[i-1]
                throughput_ratio = result["throughput"] / prev_result["throughput"]
                assert throughput_ratio > 0.7  # Maintain at least 70% relative throughput
    
    @pytest.mark.asyncio
    async def test_mixed_load_patterns(self, performance_monitor):
        """Test performance under realistic mixed load patterns."""
        config = MasterAgentConfig(rag_enabled=True)
        mock_agent = MockMasterAgent(config)
        
        # Define realistic workload mix
        workload_mix = {
            "simple_queries": 0.5,    # 50% simple queries
            "rag_queries": 0.25,      # 25% RAG queries
            "coordination": 0.15,     # 15% multi-agent
            "complex": 0.10          # 10% complex tasks
        }
        
        total_queries = 100
        mixed_requests = []
        
        # Generate mixed workload
        for query_type, proportion in workload_mix.items():
            count = int(total_queries * proportion)
            
            if query_type == "simple_queries":
                mixed_requests.extend([
                    QueryRequest(query=f"Simple query {i}", query_type=QueryType.GENERAL)
                    for i in range(count)
                ])
            elif query_type == "rag_queries":
                mixed_requests.extend([
                    QueryRequest(query=f"Research query {i}", require_rag=True)
                    for i in range(count)
                ])
            elif query_type == "coordination":
                mixed_requests.extend([
                    QueryRequest(query=f"Coordination task {i}", max_agents=3)
                    for i in range(count)
                ])
            elif query_type == "complex":
                mixed_requests.extend([
                    QueryRequest(
                        query=f"Complex task {i}",
                        query_type=QueryType.CODING,
                        max_agents=2,
                        context={"complexity": "high"}
                    )
                    for i in range(count)
                ])
        
        deps = MasterAgentDependencies()
        
        performance_monitor.start()
        start_time = time.time()
        
        # Process mixed workload with realistic concurrency
        max_concurrent = 15
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(request):
            async with semaphore:
                return await mock_agent.process_query(request, deps)
        
        tasks = [process_with_limit(request) for request in mixed_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        performance_monitor.stop()
        
        # Analyze mixed workload performance
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        total_time = end_time - start_time
        overall_throughput = len(successful_results) / total_time
        
        # Mixed workload performance requirements
        assert len(successful_results) >= total_queries * 0.95  # 95% success rate
        assert overall_throughput > 5.0                        # At least 5 QPS
        assert total_time < 30.0                               # Complete in under 30 seconds
        assert performance_monitor.peak_memory_mb < 350        # Memory under 350MB
        
        # Verify component utilization
        assert mock_agent.mcp_client.call_count >= 25          # RAG queries processed
        assert mock_agent.claude_flow_coordinator.call_count >= 15  # Coordination tasks processed
        
        # System should maintain stable performance throughout
        final_metrics = await mock_agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == len(successful_results)
        assert final_metrics["average_processing_time"] > 0