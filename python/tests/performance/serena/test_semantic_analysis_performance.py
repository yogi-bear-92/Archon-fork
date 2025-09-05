"""
Performance tests for Serena Master Agent semantic analysis operations.

Tests semantic analysis performance characteristics, scalability,
memory usage, and response times under various load conditions.
"""

import pytest
import asyncio
import time
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from tests.fixtures.serena.test_fixtures import (
    SerenaTestData,
    serena_performance_metrics,
    performance_test_environment
)
from tests.mocks.serena.mock_serena_tools import (
    MockSerenaTools,
    create_performance_monitor,
    time_test
)


class TestSemanticAnalysisPerformance:
    """Test performance characteristics of semantic analysis operations."""
    
    @pytest.mark.asyncio
    async def test_single_file_analysis_performance(self, performance_test_environment):
        """Test performance of analyzing a single file."""
        mock_tools = MockSerenaTools()
        
        # Configure realistic response for medium-sized file
        file_analysis_response = {
            "symbols": [
                {
                    "name": f"function_{i}",
                    "type": "function",
                    "complexity": "medium",
                    "lines": 15 + (i % 10)
                } for i in range(50)
            ] + [
                {
                    "name": f"Class_{i}",
                    "type": "class",
                    "methods": [f"method_{j}" for j in range(5)],
                    "complexity": "high"
                } for i in range(10)
            ],
            "total_symbols": 60,
            "analysis_metadata": {
                "lines_analyzed": 1500,
                "complexity_score": 7.2,
                "maintainability_index": 65
            }
        }
        
        mock_tools.set_response('get_symbols_overview', file_analysis_response)
        mock_tools.set_delay('get_symbols_overview', 120)  # 120ms realistic processing time
        
        # Measure performance
        async with create_performance_monitor() as monitor:
            start_time = time.time()
            
            result = await mock_tools.get_symbols_overview("src/complex_module.py")
            
            end_time = time.time()
        
        # Verify results
        assert result["total_symbols"] == 60
        assert len(result["symbols"]) == 60
        
        # Check performance metrics
        execution_time = (end_time - start_time) * 1000
        performance_metrics = monitor.get_metrics()
        
        # Performance expectations
        assert execution_time >= 120  # At least the configured delay
        assert execution_time < 300   # Should complete within 300ms
        assert performance_metrics["execution_time_ms"] < 300
        assert performance_metrics["memory_delta_mb"] < 10  # Should use <10MB additional memory
        
        # Tool-specific metrics
        tool_metrics = mock_tools.get_metrics()
        assert tool_metrics["success_rate"] == 1.0
        assert tool_metrics["average_time"] >= 0.12  # At least 120ms
    
    @pytest.mark.asyncio
    async def test_large_file_analysis_scalability(self, performance_test_environment):
        """Test scalability with large files."""
        mock_tools = MockSerenaTools()
        
        # Configure response for large file (1000+ symbols)
        large_file_response = {
            "symbols": [
                {
                    "name": f"symbol_{i}",
                    "type": "function" if i % 3 == 0 else "class",
                    "location": {"line": i * 2, "column": 1},
                    "complexity": "high" if i % 7 == 0 else "medium",
                    "dependencies": [f"symbol_{j}" for j in range(max(0, i-3), i) if j != i]
                } for i in range(1000)
            ],
            "total_symbols": 1000,
            "analysis_metadata": {
                "lines_analyzed": 25000,
                "complexity_score": 9.1,
                "processing_time_ms": 450
            }
        }
        
        mock_tools.set_response('get_symbols_overview', large_file_response)
        mock_tools.set_delay('get_symbols_overview', 450)  # Simulate realistic large file processing
        
        # Monitor system resources during analysis
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu = psutil.cpu_percent()
        
        async with create_performance_monitor() as monitor:
            result = await mock_tools.get_symbols_overview("src/large_module.py")
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Verify large file handling
        assert result["total_symbols"] == 1000
        assert len(result["symbols"]) == 1000
        
        # Performance expectations for large files
        performance_metrics = monitor.get_metrics()
        assert performance_metrics["execution_time_ms"] >= 450  # Expected processing time
        assert performance_metrics["execution_time_ms"] < 800   # Should scale reasonably
        assert memory_increase < 50  # Should handle large files without excessive memory
        
        # CPU usage should be reasonable
        assert performance_metrics["avg_cpu_percent"] < 80
    
    @pytest.mark.asyncio
    async def test_concurrent_file_analysis(self, performance_test_environment):
        """Test concurrent analysis of multiple files."""
        mock_tools = MockSerenaTools()
        
        # Configure responses for different file sizes
        file_responses = {
            'small': {
                "symbols": [{"name": f"small_symbol_{i}", "type": "function"} for i in range(10)],
                "total_symbols": 10
            },
            'medium': {
                "symbols": [{"name": f"medium_symbol_{i}", "type": "function"} for i in range(50)],
                "total_symbols": 50
            },
            'large': {
                "symbols": [{"name": f"large_symbol_{i}", "type": "function"} for i in range(200)],
                "total_symbols": 200
            }
        }
        
        # Configure different processing times based on size
        mock_tools.set_delay('get_symbols_overview', 100)  # Base delay
        
        files_to_analyze = [
            ("src/small_1.py", file_responses['small']),
            ("src/small_2.py", file_responses['small']),
            ("src/medium_1.py", file_responses['medium']),
            ("src/medium_2.py", file_responses['medium']),
            ("src/large_1.py", file_responses['large'])
        ]
        
        # Set responses for each file
        for file_path, response in files_to_analyze:
            mock_tools.set_response('get_symbols_overview', response)
        
        async with create_performance_monitor() as monitor:
            start_time = time.time()
            
            # Analyze files concurrently
            tasks = [
                mock_tools.get_symbols_overview(file_path) 
                for file_path, _ in files_to_analyze
            ]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
        
        # Verify all analyses completed
        assert len(results) == 5
        
        # Check concurrent execution performance
        execution_time = (end_time - start_time) * 1000
        performance_metrics = monitor.get_metrics()
        
        # Concurrent execution should be much faster than sequential
        # Sequential would be: 5 * 100ms = 500ms+ with overhead
        # Concurrent should be closer to max individual time (100ms) + overhead
        assert execution_time < 300  # Should complete in under 300ms
        assert performance_metrics["execution_time_ms"] < 300
        
        # All requests should have been processed
        tool_metrics = mock_tools.get_metrics()
        assert tool_metrics["total_calls"] == 5
        assert tool_metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_symbol_search_performance(self, performance_test_environment):
        """Test performance of symbol search operations."""
        mock_tools = MockSerenaTools()
        
        # Configure response for symbol search in large codebase
        search_response = {
            "matches": [
                {
                    "symbol": f"calculate_fibonacci_{i}",
                    "file": f"src/module_{i // 10}.py",
                    "line": i * 3,
                    "context": f"def calculate_fibonacci_{i}(n): # Implementation {i}",
                    "relevance_score": 0.95 - (i * 0.01)
                } for i in range(100)  # 100 matching symbols
            ],
            "total_matches": 100,
            "search_metadata": {
                "files_searched": 50,
                "search_time_ms": 85,
                "indexing_used": True
            }
        }
        
        mock_tools.set_response('find_symbol', search_response)
        mock_tools.set_delay('find_symbol', 85)  # Realistic search time
        
        async with create_performance_monitor() as monitor:
            result = await mock_tools.find_symbol(
                "calculate_fibonacci",
                substring_matching=True
            )
        
        # Verify search results
        assert len(result["matches"]) == 100
        assert result["total_matches"] == 100
        
        # Performance expectations for symbol search
        performance_metrics = monitor.get_metrics()
        assert performance_metrics["execution_time_ms"] >= 85   # Expected search time
        assert performance_metrics["execution_time_ms"] < 150  # Should be efficient
        assert performance_metrics["memory_delta_mb"] < 5      # Minimal memory overhead
    
    @pytest.mark.asyncio
    async def test_pattern_search_performance(self, performance_test_environment):
        """Test performance of regex pattern search."""
        mock_tools = MockSerenaTools()
        
        # Configure response for complex pattern search
        pattern_response = {
            "matches": {
                f"src/file_{i}.py": [
                    {
                        "line_number": j * 5,
                        "content": f"async def method_{j}(self, param_{k}): # Line in file {i}",
                        "pattern_groups": [f"method_{j}", f"param_{k}"]
                    } for j in range(5) for k in range(2)
                ] for i in range(20)  # 20 files with matches
            },
            "total_matches": 200,
            "search_metadata": {
                "files_processed": 100,
                "pattern_complexity": "high",
                "processing_time_ms": 180
            }
        }
        
        mock_tools.set_response('search_for_pattern', pattern_response)
        mock_tools.set_delay('search_for_pattern', 180)  # Complex pattern search time
        
        async with create_performance_monitor() as monitor:
            result = await mock_tools.search_for_pattern(
                r"async\s+def\s+(\w+)\(.*?(\w+):",  # Complex regex pattern
                context_lines_before=2,
                context_lines_after=2
            )
        
        # Verify pattern search results
        assert len(result["matches"]) == 20  # 20 files
        assert result["total_matches"] == 200
        
        # Performance expectations for pattern search
        performance_metrics = monitor.get_metrics()
        assert performance_metrics["execution_time_ms"] >= 180   # Expected processing time
        assert performance_metrics["execution_time_ms"] < 300   # Should complete reasonably
        assert performance_metrics["memory_delta_mb"] < 15      # Pattern matching can use more memory
    
    @pytest.mark.asyncio
    async def test_dependency_analysis_performance(self, performance_test_environment):
        """Test performance of dependency analysis operations."""
        mock_tools = MockSerenaTools()
        
        # Configure response for comprehensive dependency analysis
        dependency_response = [
            {
                "symbol": f"Class_{i}",
                "dependencies": [f"dependency_{j}" for j in range(i % 8)],
                "dependents": [f"dependent_{k}" for k in range((i * 2) % 5)],
                "complexity_score": 4.2 + (i % 10) * 0.3,
                "file": f"src/module_{i // 10}.py"
            } for i in range(150)  # 150 symbols with dependencies
        ]
        
        mock_tools.set_response('find_referencing_symbols', dependency_response)
        mock_tools.set_delay('find_referencing_symbols', 220)  # Dependency analysis time
        
        async with create_performance_monitor() as monitor:
            result = await mock_tools.find_referencing_symbols(
                "BaseClass",
                "src/base.py"
            )
        
        # Verify dependency analysis
        assert len(result) == 150
        
        # Performance expectations for dependency analysis
        performance_metrics = monitor.get_metrics()
        assert performance_metrics["execution_time_ms"] >= 220   # Expected analysis time
        assert performance_metrics["execution_time_ms"] < 400   # Should be efficient
        assert performance_metrics["memory_delta_mb"] < 20      # Dependency graphs can use memory


class TestSemanticAnalysisScalability:
    """Test scalability characteristics under increasing load."""
    
    @pytest.mark.asyncio
    async def test_increasing_codebase_size_scaling(self, performance_test_environment):
        """Test how performance scales with increasing codebase size."""
        mock_tools = MockSerenaTools()
        
        # Test different codebase sizes
        codebase_sizes = [100, 500, 1000, 2000, 5000]  # Number of files
        performance_results = []
        
        for size in codebase_sizes:
            # Configure response based on codebase size
            dir_response = {
                "dirs": [f"module_{i}" for i in range(size // 50)],
                "files": [f"file_{i}.py" for i in range(size)]
            }
            
            # Processing time should scale sublinearly (with optimizations)
            processing_time = int(50 + (size * 0.05))  # Base 50ms + 0.05ms per file
            
            mock_tools.set_response('list_dir', dir_response)
            mock_tools.set_delay('list_dir', processing_time)
            
            # Measure performance for this size
            start_time = time.time()
            result = await mock_tools.list_dir(".", recursive=True)
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000
            
            performance_results.append({
                'codebase_size': size,
                'execution_time_ms': execution_time,
                'files_processed': len(result['files']),
                'throughput': len(result['files']) / (execution_time / 1000)  # files per second
            })
            
            # Verify correct processing
            assert len(result['files']) == size
            assert execution_time >= processing_time  # Should meet minimum processing time
        
        # Analyze scaling characteristics
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i-1]
            
            # Scaling should be sublinear (better than O(n))
            size_ratio = current['codebase_size'] / previous['codebase_size']
            time_ratio = current['execution_time_ms'] / previous['execution_time_ms']
            
            # Time scaling should be better than linear
            assert time_ratio < size_ratio * 1.2  # Allow 20% overhead for sublinear scaling
            
            # Throughput should remain reasonable
            assert current['throughput'] > 100  # At least 100 files/second
    
    @pytest.mark.asyncio
    async def test_concurrent_user_scaling(self, performance_test_environment):
        """Test performance under concurrent user/agent load."""
        mock_tools = MockSerenaTools()
        
        # Configure standard analysis response
        analysis_response = {
            "symbols": [{"name": f"symbol_{i}", "type": "function"} for i in range(25)],
            "total_symbols": 25
        }
        mock_tools.set_response('get_symbols_overview', analysis_response)
        mock_tools.set_delay('get_symbols_overview', 100)  # 100ms processing time
        
        # Test different levels of concurrent users
        concurrent_levels = [1, 2, 5, 10, 20]
        scaling_results = []
        
        for concurrent_users in concurrent_levels:
            async with create_performance_monitor() as monitor:
                start_time = time.time()
                
                # Simulate concurrent users/agents
                tasks = [
                    mock_tools.get_symbols_overview(f"user_{i}/file.py")
                    for i in range(concurrent_users)
                ]
                
                results = await asyncio.gather(*tasks)
                
                end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000
            performance_metrics = monitor.get_metrics()
            
            scaling_results.append({
                'concurrent_users': concurrent_users,
                'execution_time_ms': execution_time,
                'avg_response_time_ms': execution_time / concurrent_users,
                'throughput_ops_per_sec': concurrent_users / (execution_time / 1000),
                'memory_usage_mb': performance_metrics.get('peak_memory_mb', 0)
            })
            
            # Verify all requests completed successfully
            assert len(results) == concurrent_users
            for result in results:
                assert result['total_symbols'] == 25
        
        # Analyze concurrent scaling
        for result in scaling_results:
            # Average response time should remain reasonable even under load
            assert result['avg_response_time_ms'] < 200  # Should handle concurrency well
            
            # Throughput should scale with users (at least partially)
            if result['concurrent_users'] <= 10:
                assert result['throughput_ops_per_sec'] > result['concurrent_users'] * 0.8
            
            # Memory usage should scale reasonably
            assert result['memory_usage_mb'] < result['concurrent_users'] * 5  # <5MB per user
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_under_load(self):
        """Test memory efficiency under sustained load."""
        mock_tools = MockSerenaTools()
        
        # Configure responses that would normally use significant memory
        large_symbol_response = {
            "symbols": [
                {
                    "name": f"symbol_{i}",
                    "type": "class",
                    "methods": [f"method_{j}" for j in range(10)],
                    "source_code": "x" * 1000,  # 1KB per symbol
                    "dependencies": [f"dep_{k}" for k in range(5)]
                } for i in range(100)
            ],
            "total_symbols": 100
        }
        
        mock_tools.set_response('get_symbols_overview', large_symbol_response)
        mock_tools.set_delay('get_symbols_overview', 50)
        
        # Monitor memory usage over multiple operations
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_measurements = [initial_memory]
        
        # Perform multiple analysis operations
        for i in range(20):  # 20 operations
            result = await mock_tools.get_symbols_overview(f"file_{i}.py")
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
            
            # Verify operation completed
            assert result['total_symbols'] == 100
            
            # Force garbage collection periodically
            if i % 5 == 0:
                import gc
                gc.collect()
        
        final_memory = memory_measurements[-1]
        memory_increase = final_memory - initial_memory
        peak_memory = max(memory_measurements)
        
        # Memory usage should be reasonable
        assert memory_increase < 50  # Total increase should be <50MB
        assert peak_memory - initial_memory < 100  # Peak usage should be <100MB
        
        # Memory should not grow indefinitely (no major leaks)
        recent_avg = sum(memory_measurements[-5:]) / 5
        early_avg = sum(memory_measurements[1:6]) / 5
        growth_rate = (recent_avg - early_avg) / early_avg
        
        assert growth_rate < 0.5  # Memory growth should be <50% over operations


class TestSemanticAnalysisLatency:
    """Test latency characteristics and response time distribution."""
    
    @pytest.mark.asyncio
    async def test_response_time_distribution(self, performance_test_environment):
        """Test response time distribution under normal load."""
        mock_tools = MockSerenaTools()
        
        # Configure response with variable processing time
        standard_response = {
            "symbols": [{"name": f"func_{i}", "type": "function"} for i in range(30)],
            "total_symbols": 30
        }
        mock_tools.set_response('get_symbols_overview', standard_response)
        
        # Perform multiple requests to measure distribution
        response_times = []
        
        for i in range(50):  # 50 samples
            # Add some variability in processing time (simulate realistic conditions)
            base_delay = 80
            variability = (i % 10) * 5  # 0-45ms variation
            mock_tools.set_delay('get_symbols_overview', base_delay + variability)
            
            start_time = time.time()
            result = await mock_tools.get_symbols_overview(f"test_file_{i}.py")
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
            
            assert result['total_symbols'] == 30
        
        # Calculate response time statistics
        response_times.sort()
        
        p50 = response_times[len(response_times) // 2]  # Median
        p95 = response_times[int(len(response_times) * 0.95)]  # 95th percentile
        p99 = response_times[int(len(response_times) * 0.99)]  # 99th percentile
        average = sum(response_times) / len(response_times)
        
        # Response time expectations
        assert average < 150  # Average should be under 150ms
        assert p50 < 130      # Median should be under 130ms
        assert p95 < 200      # 95% should be under 200ms
        assert p99 < 250      # 99% should be under 250ms
        
        # Consistency check (low variability)
        std_dev = (sum((x - average) ** 2 for x in response_times) / len(response_times)) ** 0.5
        assert std_dev < 30  # Standard deviation should be reasonable
    
    @pytest.mark.asyncio
    async def test_cold_start_vs_warm_performance(self, performance_test_environment):
        """Test performance difference between cold start and warm execution."""
        mock_tools = MockSerenaTools()
        
        # Configure analysis response
        analysis_response = {
            "symbols": [{"name": f"symbol_{i}", "type": "function"} for i in range(40)],
            "total_symbols": 40,
            "cached": False
        }
        mock_tools.set_response('get_symbols_overview', analysis_response)
        
        # Cold start - first request
        mock_tools.set_delay('get_symbols_overview', 200)  # Cold start penalty
        
        start_time = time.time()
        cold_result = await mock_tools.get_symbols_overview("new_file.py")
        cold_time = (time.time() - start_time) * 1000
        
        # Warm requests - simulate caching/optimization
        warm_response = analysis_response.copy()
        warm_response["cached"] = True
        mock_tools.set_response('get_symbols_overview', warm_response)
        mock_tools.set_delay('get_symbols_overview', 80)  # Much faster with warm cache
        
        # Multiple warm requests
        warm_times = []
        for i in range(5):
            start_time = time.time()
            warm_result = await mock_tools.get_symbols_overview("cached_file.py")
            warm_time = (time.time() - start_time) * 1000
            warm_times.append(warm_time)
            
            assert warm_result['cached'] is True
        
        average_warm_time = sum(warm_times) / len(warm_times)
        
        # Performance comparison
        assert cold_time >= 200  # Cold start should take at least the configured delay
        assert average_warm_time < 120  # Warm requests should be much faster
        assert cold_time > average_warm_time * 1.5  # Cold start should be significantly slower
        
        # Verify both returned correct results
        assert cold_result['total_symbols'] == 40
        assert all(result['total_symbols'] == 40 for result in [warm_result])
    
    @pytest.mark.asyncio
    async def test_timeout_and_degradation_behavior(self, performance_test_environment):
        """Test behavior under timeout conditions and graceful degradation."""
        mock_tools = MockSerenaTools()
        
        # Configure normal response
        normal_response = {
            "symbols": [{"name": f"symbol_{i}", "type": "function"} for i in range(20)],
            "total_symbols": 20
        }
        mock_tools.set_response('get_symbols_overview', normal_response)
        
        # Test 1: Normal operation
        mock_tools.set_delay('get_symbols_overview', 100)  # Normal processing time
        
        start_time = time.time()
        result = await mock_tools.get_symbols_overview("normal_file.py")
        normal_time = (time.time() - start_time) * 1000
        
        assert result['total_symbols'] == 20
        assert normal_time < 150  # Should complete normally
        
        # Test 2: Slow but acceptable operation
        mock_tools.set_delay('get_symbols_overview', 400)  # Slower but within limits
        
        start_time = time.time()
        result = await mock_tools.get_symbols_overview("slow_file.py")
        slow_time = (time.time() - start_time) * 1000
        
        assert result['total_symbols'] == 20
        assert 400 <= slow_time < 500  # Should complete but slowly
        
        # Test 3: Timeout condition
        mock_tools.set_error_condition('get_symbols_overview', asyncio.TimeoutError("Analysis timeout"))
        
        with pytest.raises(asyncio.TimeoutError):
            await mock_tools.get_symbols_overview("timeout_file.py")
        
        # Verify tool still works after timeout (recovery)
        mock_tools.set_delay('get_symbols_overview', 100)  # Back to normal
        mock_tools.error_conditions.clear()  # Clear error condition
        
        start_time = time.time()
        recovery_result = await mock_tools.get_symbols_overview("recovery_file.py")
        recovery_time = (time.time() - start_time) * 1000
        
        assert recovery_result['total_symbols'] == 20
        assert recovery_time < 150  # Should recover to normal performance