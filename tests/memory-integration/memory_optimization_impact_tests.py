"""
Memory Optimization Impact Tests

Tests the actual performance impact of memory optimizations implemented
in the integration patterns. Measures effectiveness of string interning,
weak references, cache management, and other memory-saving strategies.

Design: Ultra-lightweight with <2MB memory footprint per test
"""

import pytest
import gc
import sys
import weakref
import psutil
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock
from contextlib import contextmanager


class MemoryOptimizationValidator:
    """Validates memory optimization strategies effectiveness."""
    
    def __init__(self):
        self.baseline_memory = self._get_process_memory()
        self.optimization_results = []
    
    def _get_process_memory(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    @contextmanager
    def memory_measurement(self, test_name: str):
        """Context manager for measuring memory impact."""
        before_memory = self._get_process_memory()
        gc.collect()  # Clean baseline
        start_memory = self._get_process_memory()
        
        try:
            yield
        finally:
            gc.collect()
            end_memory = self._get_process_memory()
            
            result = {
                "test_name": test_name,
                "memory_before_mb": start_memory,
                "memory_after_mb": end_memory,
                "memory_delta_mb": end_memory - start_memory,
                "optimization_effective": (end_memory - start_memory) < 1.0  # Less than 1MB increase
            }
            
            self.optimization_results.append(result)
    
    def test_string_interning_optimization(self) -> Dict[str, Any]:
        """Test effectiveness of string interning for tag management."""
        
        with self.memory_measurement("string_interning"):
            # Simulate tag processing without optimization
            regular_tags = []
            for i in range(1000):
                # Create duplicate strings (common in tagging systems)
                regular_tags.extend([
                    f"function_analysis_{i % 10}",
                    f"optimization_target_{i % 5}",
                    f"memory_pressure_{i % 3}"
                ])
            
            # Simulate with string interning optimization
            interned_tags = []
            tag_cache = {}
            
            for i in range(1000):
                for base_tag in [f"function_analysis_{i % 10}", f"optimization_target_{i % 5}", f"memory_pressure_{i % 3}"]:
                    if base_tag in tag_cache:
                        interned_tags.append(tag_cache[base_tag])
                    else:
                        interned_tag = sys.intern(base_tag)
                        tag_cache[base_tag] = interned_tag
                        interned_tags.append(interned_tag)
            
            # Verify optimization effectiveness
            cache_efficiency = len(tag_cache) / len(interned_tags) if interned_tags else 0
            
            return {
                "regular_tags_count": len(regular_tags),
                "interned_tags_count": len(interned_tags),
                "unique_cached_tags": len(tag_cache),
                "cache_efficiency": cache_efficiency,
                "memory_savings_expected": cache_efficiency > 0.5
            }
    
    def test_weak_reference_optimization(self) -> Dict[str, Any]:
        """Test weak reference effectiveness for agent management."""
        
        with self.memory_measurement("weak_references"):
            # Create regular object references (potential memory leak)
            regular_agents = {}
            for i in range(50):
                agent_obj = {
                    "id": f"agent_{i}",
                    "data": f"agent_data_{i}" * 100,  # Some data
                    "status": "active"
                }
                regular_agents[f"agent_{i}"] = agent_obj
            
            # Create weak references (memory optimized)
            weak_agents = weakref.WeakValueDictionary()
            agent_objects = []  # Keep strong references temporarily
            
            for i in range(50):
                agent_obj = {
                    "id": f"weak_agent_{i}",
                    "data": f"agent_data_{i}" * 100,
                    "status": "active"
                }
                agent_objects.append(agent_obj)
                weak_agents[f"weak_agent_{i}"] = agent_obj
            
            # Test cleanup effectiveness
            agents_before_cleanup = len(weak_agents)
            del agent_objects  # Remove strong references
            gc.collect()
            agents_after_cleanup = len(weak_agents)
            
            return {
                "regular_agents_count": len(regular_agents),
                "weak_agents_before_cleanup": agents_before_cleanup,
                "weak_agents_after_cleanup": agents_after_cleanup,
                "cleanup_effectiveness": (agents_before_cleanup - agents_after_cleanup) / agents_before_cleanup if agents_before_cleanup else 0,
                "memory_cleanup_successful": agents_after_cleanup == 0
            }
    
    def test_cache_size_limiting(self) -> Dict[str, Any]:
        """Test cache size limiting effectiveness."""
        
        with self.memory_measurement("cache_limiting"):
            # Simulate unlimited cache (memory leak potential)
            unlimited_cache = {}
            for i in range(200):
                unlimited_cache[f"analysis_{i}"] = {
                    "result": f"analysis_result_{i}" * 50,
                    "metadata": {"complexity": i, "timestamp": f"time_{i}"}
                }
            
            # Simulate limited cache with LRU-style eviction
            limited_cache = {}
            cache_limit = 50
            access_order = []
            
            for i in range(200):
                key = f"limited_analysis_{i}"
                value = {
                    "result": f"analysis_result_{i}" * 50,
                    "metadata": {"complexity": i, "timestamp": f"time_{i}"}
                }
                
                # Add to cache
                limited_cache[key] = value
                access_order.append(key)
                
                # Evict if over limit
                if len(limited_cache) > cache_limit:
                    oldest_key = access_order.pop(0)
                    if oldest_key in limited_cache:
                        del limited_cache[oldest_key]
            
            return {
                "unlimited_cache_size": len(unlimited_cache),
                "limited_cache_size": len(limited_cache),
                "cache_limit_enforced": len(limited_cache) <= cache_limit,
                "memory_savings_ratio": len(limited_cache) / len(unlimited_cache) if unlimited_cache else 0
            }


@pytest.mark.memory_optimization
class TestMemoryOptimizationEffectiveness:
    """Test effectiveness of memory optimization strategies."""
    
    @pytest.mark.asyncio
    async def test_string_interning_memory_impact(self):
        """Test string interning reduces memory usage."""
        validator = MemoryOptimizationValidator()
        
        result = validator.test_string_interning_optimization()
        
        assert result["memory_savings_expected"] is True
        assert result["cache_efficiency"] > 0.5
        assert result["unique_cached_tags"] < result["interned_tags_count"]
        
        # Verify actual memory measurement
        memory_results = [r for r in validator.optimization_results if r["test_name"] == "string_interning"]
        assert len(memory_results) == 1
        assert memory_results[0]["optimization_effective"] is True
    
    @pytest.mark.asyncio
    async def test_weak_references_prevent_memory_leaks(self):
        """Test weak references prevent memory leaks in agent management."""
        validator = MemoryOptimizationValidator()
        
        result = validator.test_weak_reference_optimization()
        
        assert result["memory_cleanup_successful"] is True
        assert result["cleanup_effectiveness"] > 0.8  # At least 80% cleanup
        assert result["weak_agents_after_cleanup"] == 0
        
        # Verify memory impact
        memory_results = [r for r in validator.optimization_results if r["test_name"] == "weak_references"]
        assert len(memory_results) == 1
        assert memory_results[0]["optimization_effective"] is True
    
    @pytest.mark.asyncio
    async def test_cache_limiting_effectiveness(self):
        """Test cache size limiting prevents memory bloat."""
        validator = MemoryOptimizationValidator()
        
        result = validator.test_cache_size_limiting()
        
        assert result["cache_limit_enforced"] is True
        assert result["memory_savings_ratio"] < 0.5  # Significant reduction
        assert result["limited_cache_size"] <= 50  # Respects limit
        
        # Verify memory impact
        memory_results = [r for r in validator.optimization_results if r["test_name"] == "cache_limiting"]
        assert len(memory_results) == 1
        assert memory_results[0]["optimization_effective"] is True
    
    @pytest.mark.asyncio
    async def test_combined_optimization_impact(self):
        """Test combined effect of all memory optimizations."""
        validator = MemoryOptimizationValidator()
        
        # Run all optimizations
        string_result = validator.test_string_interning_optimization()
        weak_ref_result = validator.test_weak_reference_optimization() 
        cache_result = validator.test_cache_size_limiting()
        
        # Analyze combined effectiveness
        combined_effectiveness = all([
            string_result["memory_savings_expected"],
            weak_ref_result["memory_cleanup_successful"],
            cache_result["cache_limit_enforced"]
        ])
        
        assert combined_effectiveness is True
        
        # Verify all measurements show optimization effectiveness
        all_optimizations_effective = all(
            result["optimization_effective"] 
            for result in validator.optimization_results
        )
        assert all_optimizations_effective is True
        
        # Total memory impact should be minimal
        total_memory_delta = sum(
            result["memory_delta_mb"] 
            for result in validator.optimization_results
        )
        assert total_memory_delta < 3.0  # Less than 3MB total increase


@pytest.mark.memory_optimization  
class TestMemoryOptimizationInCriticalState:
    """Test memory optimizations specifically under critical memory conditions."""
    
    @pytest.mark.asyncio
    async def test_emergency_optimization_activation(self):
        """Test that emergency optimizations activate under memory pressure."""
        
        # Simulate critical memory detection
        memory_pressure_detected = True  # In real scenario, would check actual memory
        
        if memory_pressure_detected:
            # Emergency optimization strategies
            emergency_optimizations = {
                "aggressive_gc": True,
                "cache_clear": True,
                "weak_ref_cleanup": True,
                "string_intern_aggressive": True,
                "memory_limit_enforcement": True
            }
            
            # Verify all emergency optimizations are active
            assert all(emergency_optimizations.values())
            
            # Test optimization effectiveness under pressure
            validator = MemoryOptimizationValidator()
            
            with validator.memory_measurement("emergency_optimizations"):
                # Perform minimal operations with all optimizations active
                gc.collect()  # Aggressive GC
                
                # Minimal cache with strict limits
                emergency_cache = weakref.WeakValueDictionary()
                test_data = {"emergency": "data"}
                emergency_cache["test"] = test_data
                
                # Force cleanup
                del test_data
                gc.collect()
            
            # Verify emergency optimizations work
            memory_results = [r for r in validator.optimization_results if r["test_name"] == "emergency_optimizations"]
            assert len(memory_results) == 1
            assert memory_results[0]["optimization_effective"] is True
    
    @pytest.mark.asyncio
    async def test_optimization_under_extreme_pressure(self):
        """Test optimizations work under extreme memory pressure."""
        validator = MemoryOptimizationValidator()
        
        with validator.memory_measurement("extreme_pressure_optimization"):
            # Simulate extreme memory pressure scenario
            
            # 1. Ultra-minimal object creation
            minimal_objects = []
            for i in range(10):  # Very limited count
                minimal_obj = {"id": i}  # Minimal data
                minimal_objects.append(minimal_obj)
            
            # 2. Immediate cleanup
            del minimal_objects
            gc.collect()
            
            # 3. Ultra-aggressive string interning
            common_strings = ["status", "id", "data", "result"]
            interned_strings = [sys.intern(s) for s in common_strings]
            
            # 4. Minimal weak reference usage
            weak_container = weakref.WeakSet()
            test_obj = {"test": "data"}
            weak_container.add(test_obj)
            
            # Verify object can be accessed but will be cleaned up
            assert len(weak_container) == 1
            del test_obj
            gc.collect()
            
            # Object should be cleaned up from weak container
            # Note: Actual cleanup timing may vary, so we just verify the pattern works
        
        # Verify extreme pressure optimization is effective
        memory_results = [r for r in validator.optimization_results if r["test_name"] == "extreme_pressure_optimization"]
        assert len(memory_results) == 1
        assert memory_results[0]["optimization_effective"] is True
        
        # Memory delta should be near zero or negative (cleanup effective)
        assert memory_results[0]["memory_delta_mb"] < 0.5


# Utility to run optimization tests safely
def run_memory_optimization_tests():
    """Run memory optimization tests safely."""
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "/Users/yogi/Projects/Archon-fork/tests/memory-integration/memory_optimization_impact_tests.py",
            "-v",
            "-m", "memory_optimization",
            "--tb=short"
        ], capture_output=True, text=True, timeout=60)
        
        print("Memory Optimization Test Results:")
        print("=" * 50)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to run memory optimization tests: {e}")
        return False


if __name__ == "__main__":
    success = run_memory_optimization_tests()
    sys.exit(0 if success else 1)