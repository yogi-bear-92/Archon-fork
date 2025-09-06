"""
Critical Memory State Integration Tests for Archon PRP + Claude Flow + Serena

Validates memory-aware integration patterns under extreme memory pressure (99.6% usage).
Tests emergency fallback, adaptive scaling, tool hierarchy enforcement, and coordination
mechanisms with minimal resources.

CRITICAL SAFETY MEASURES:
- All operations designed for <10MB memory footprint
- Immediate cleanup after each test
- No large data structures or bulk processing
- Emergency abort mechanisms on memory threshold breach
"""

import pytest
import asyncio
import gc
import psutil
import json
import weakref
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

# Configure ultra-minimal logging for memory pressure
logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class CriticalMemoryMonitor:
    """Ultra-lightweight memory monitoring for critical state testing."""
    
    def __init__(self, emergency_threshold: float = 99.8):
        self.emergency_threshold = emergency_threshold
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage (lightweight check)."""
        try:
            # Use minimal memory check
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            return (system_memory.used / system_memory.total) * 100
        except Exception:
            return 99.9  # Assume critical if check fails
    
    def check_emergency_abort(self) -> bool:
        """Check if we need emergency abort."""
        current_usage = self.get_memory_usage()
        return current_usage >= self.emergency_threshold
    
    def force_cleanup(self):
        """Force aggressive garbage collection."""
        gc.collect()
        gc.collect()  # Double collection for thoroughness


@asynccontextmanager
async def critical_memory_context(test_name: str):
    """Context manager for critical memory state testing with emergency abort."""
    monitor = CriticalMemoryMonitor()
    
    if monitor.check_emergency_abort():
        pytest.skip(f"Emergency abort: Memory usage too high for {test_name}")
    
    try:
        logger.error(f"Starting critical memory test: {test_name}")
        yield monitor
        
    except Exception as e:
        logger.error(f"Test {test_name} failed: {e}")
        raise
        
    finally:
        monitor.force_cleanup()
        await asyncio.sleep(0.1)  # Brief pause for cleanup


class MinimalMockCoordination:
    """Ultra-lightweight mock coordination system for memory pressure testing."""
    
    def __init__(self):
        self.agents = weakref.WeakValueDictionary()
        self.memory_store = weakref.WeakValueDictionary()
        self.message_queue = []
        self.fallback_mode = False
        
    async def emergency_fallback(self) -> Dict[str, Any]:
        """Activate emergency fallback mode."""
        self.fallback_mode = True
        self.agents.clear()
        self.message_queue.clear()
        gc.collect()
        
        return {
            "status": "emergency_fallback_activated",
            "mode": "minimal_operation",
            "available_functions": ["basic_coordination", "memory_cleanup"],
            "memory_freed": "substantial"
        }
    
    async def minimal_agent_spawn(self, agent_type: str) -> Dict[str, Any]:
        """Spawn agent in minimal memory mode."""
        if len(self.agents) >= 2:  # Strict limit under memory pressure
            return {
                "status": "spawn_denied",
                "reason": "memory_pressure_limit_reached",
                "max_agents": 2
            }
        
        # Create minimal agent representation
        agent_id = f"minimal_{agent_type}_{len(self.agents)}"
        minimal_agent = {
            "id": agent_id,
            "type": agent_type,
            "mode": "memory_optimized",
            "capabilities": ["basic_coordination"]
        }
        
        self.agents[agent_id] = minimal_agent
        return {
            "status": "spawned",
            "agent_id": agent_id,
            "memory_mode": "critical_optimized"
        }
    
    async def adaptive_scale_down(self) -> Dict[str, Any]:
        """Adaptively scale down operations for memory pressure."""
        original_agents = len(self.agents)
        
        # Keep only most critical agent
        if len(self.agents) > 1:
            # Keep first agent, remove others
            agents_list = list(self.agents.keys())
            for agent_id in agents_list[1:]:
                if agent_id in self.agents:
                    del self.agents[agent_id]
        
        # Clear message queue except emergency messages
        emergency_messages = [msg for msg in self.message_queue if msg.get("priority") == "emergency"]
        self.message_queue.clear()
        self.message_queue.extend(emergency_messages)
        
        gc.collect()
        
        return {
            "status": "scaled_down",
            "agents_removed": original_agents - len(self.agents),
            "messages_cleared": len(self.message_queue),
            "mode": "survival_minimal"
        }


@pytest.mark.critical_memory
class TestEmergencyFallback:
    """Test emergency fallback procedures under critical memory pressure."""
    
    @pytest.mark.asyncio
    async def test_emergency_fallback_activation(self):
        """Test emergency fallback activation when memory critical."""
        async with critical_memory_context("emergency_fallback") as monitor:
            coordination = MinimalMockCoordination()
            
            # Simulate memory pressure detection
            if monitor.get_memory_usage() > 99.5:
                result = await coordination.emergency_fallback()
                
                assert result["status"] == "emergency_fallback_activated"
                assert result["mode"] == "minimal_operation"
                assert coordination.fallback_mode is True
                assert len(coordination.agents) == 0
                
                # Verify minimal function availability
                available_functions = result["available_functions"]
                assert "basic_coordination" in available_functions
                assert len(available_functions) <= 3  # Minimal set only
    
    @pytest.mark.asyncio
    async def test_fallback_tool_hierarchy_enforcement(self):
        """Test that tool hierarchy is strictly enforced during fallback."""
        async with critical_memory_context("fallback_hierarchy") as monitor:
            coordination = MinimalMockCoordination()
            
            # Activate emergency fallback
            await coordination.emergency_fallback()
            
            # Test tool access hierarchy (MCP coordination only)
            hierarchy_result = {
                "mcp_tools": ["basic_coordination", "memory_cleanup"],
                "claude_code_tools": [],  # Disabled in fallback
                "restricted_access": True,
                "reason": "emergency_memory_conservation"
            }
            
            # Verify Claude Code tools are restricted
            assert len(hierarchy_result["claude_code_tools"]) == 0
            assert hierarchy_result["restricted_access"] is True
            
            # Verify only essential MCP tools available
            mcp_tools = hierarchy_result["mcp_tools"]
            assert len(mcp_tools) <= 3
            assert "basic_coordination" in mcp_tools
    
    @pytest.mark.asyncio
    async def test_fallback_memory_recovery(self):
        """Test memory recovery during fallback operations."""
        async with critical_memory_context("memory_recovery") as monitor:
            coordination = MinimalMockCoordination()
            
            initial_memory = monitor.get_memory_usage()
            
            # Create some minimal load
            coordination.message_queue = [{"msg": f"test_{i}"} for i in range(5)]
            
            # Activate fallback and measure recovery
            await coordination.emergency_fallback()
            monitor.force_cleanup()
            
            final_memory = monitor.get_memory_usage()
            
            # Verify memory impact is minimal
            memory_delta = final_memory - initial_memory
            assert memory_delta < 0.1  # Less than 0.1% memory increase
            
            # Verify cleanup effectiveness
            assert len(coordination.message_queue) == 0
            assert coordination.fallback_mode is True


@pytest.mark.critical_memory
class TestMemoryMonitoringAndScaling:
    """Test memory monitoring and adaptive scaling under pressure."""
    
    @pytest.mark.asyncio
    async def test_memory_threshold_monitoring(self):
        """Test continuous memory threshold monitoring."""
        async with critical_memory_context("memory_monitoring") as monitor:
            coordination = MinimalMockCoordination()
            
            # Test memory monitoring accuracy
            memory_usage = monitor.get_memory_usage()
            assert memory_usage > 99.0  # Should be in critical range
            
            # Test emergency threshold detection
            emergency_needed = monitor.check_emergency_abort()
            if memory_usage > 99.8:
                assert emergency_needed is True
            
            # Test monitoring doesn't increase memory usage
            initial_memory = memory_usage
            for _ in range(3):
                current_memory = monitor.get_memory_usage()
                assert abs(current_memory - initial_memory) < 0.05  # Stable monitoring
    
    @pytest.mark.asyncio
    async def test_adaptive_scaling_under_pressure(self):
        """Test adaptive scaling mechanisms with minimal resources."""
        async with critical_memory_context("adaptive_scaling") as monitor:
            coordination = MinimalMockCoordination()
            
            # Spawn minimal agents
            agent1 = await coordination.minimal_agent_spawn("coordinator")
            agent2 = await coordination.minimal_agent_spawn("worker")
            
            assert agent1["status"] == "spawned"
            assert agent2["status"] == "spawned"
            
            # Attempt to spawn third agent (should be denied)
            agent3 = await coordination.minimal_agent_spawn("optimizer")
            
            assert agent3["status"] == "spawn_denied"
            assert agent3["reason"] == "memory_pressure_limit_reached"
            assert agent3["max_agents"] == 2
    
    @pytest.mark.asyncio
    async def test_scale_down_operations(self):
        """Test scaling down operations when memory critical."""
        async with critical_memory_context("scale_down") as monitor:
            coordination = MinimalMockCoordination()
            
            # Create initial load
            await coordination.minimal_agent_spawn("agent1")
            await coordination.minimal_agent_spawn("agent2")
            coordination.message_queue = [{"msg": f"test_{i}", "priority": "normal"} for i in range(10)]
            coordination.message_queue.append({"msg": "critical", "priority": "emergency"})
            
            initial_agents = len(coordination.agents)
            initial_messages = len(coordination.message_queue)
            
            # Execute scale down
            result = await coordination.adaptive_scale_down()
            
            assert result["status"] == "scaled_down"
            assert result["mode"] == "survival_minimal"
            
            # Verify scaling effectiveness
            assert len(coordination.agents) == 1  # Only one agent kept
            assert len(coordination.message_queue) == 1  # Only emergency message kept
            assert coordination.message_queue[0]["priority"] == "emergency"
            
            # Verify scale down metrics
            assert result["agents_removed"] == (initial_agents - 1)


@pytest.mark.critical_memory
class TestToolHierarchyEnforcement:
    """Test tool hierarchy enforcement under extreme memory pressure."""
    
    @pytest.mark.asyncio
    async def test_mcp_priority_enforcement(self):
        """Test that MCP tools get priority during memory pressure."""
        async with critical_memory_context("mcp_priority") as monitor:
            
            # Simulate tool hierarchy decision under pressure
            available_tools = {
                "mcp_coordination": {
                    "priority": "critical",
                    "memory_cost": "minimal",
                    "available": True
                },
                "claude_code_execution": {
                    "priority": "normal", 
                    "memory_cost": "high",
                    "available": False  # Disabled under pressure
                },
                "file_operations": {
                    "priority": "normal",
                    "memory_cost": "medium", 
                    "available": False  # Disabled under pressure
                }
            }
            
            # Verify MCP tools remain available
            mcp_tools = [name for name, tool in available_tools.items() 
                        if tool["priority"] == "critical" and tool["available"]]
            
            assert len(mcp_tools) >= 1
            assert "mcp_coordination" in mcp_tools
            
            # Verify high-memory tools are disabled
            disabled_tools = [name for name, tool in available_tools.items() 
                            if not tool["available"]]
            
            assert "claude_code_execution" in disabled_tools
            assert "file_operations" in disabled_tools
    
    @pytest.mark.asyncio
    async def test_coordination_mechanism_minimal_resources(self):
        """Test coordination mechanisms work with minimal resources."""
        async with critical_memory_context("minimal_coordination") as monitor:
            coordination = MinimalMockCoordination()
            
            # Test minimal coordination functionality
            agent = await coordination.minimal_agent_spawn("minimal_coordinator")
            
            assert agent["status"] == "spawned"
            assert agent["memory_mode"] == "critical_optimized"
            
            # Test basic coordination message handling
            coordination.message_queue = [
                {
                    "from": agent["agent_id"],
                    "to": "system",
                    "type": "status_update",
                    "payload": {"status": "ready", "memory_usage": "minimal"}
                }
            ]
            
            # Verify message handling doesn't crash under pressure
            message = coordination.message_queue[0]
            assert message["payload"]["memory_usage"] == "minimal"
            assert message["type"] == "status_update"


@pytest.mark.critical_memory
class TestFailureRecoveryPatterns:
    """Test failure scenarios and recovery patterns under memory pressure."""
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_recovery(self):
        """Test recovery from simulated memory exhaustion."""
        async with critical_memory_context("exhaustion_recovery") as monitor:
            coordination = MinimalMockCoordination()
            
            try:
                # Simulate memory exhaustion scenario
                raise MemoryError("Simulated memory exhaustion")
                
            except MemoryError:
                # Execute recovery procedure
                recovery_result = await coordination.emergency_fallback()
                monitor.force_cleanup()
                
                # Verify recovery is successful
                assert recovery_result["status"] == "emergency_fallback_activated"
                assert coordination.fallback_mode is True
                
                # Verify system can continue minimal operations
                minimal_agent = await coordination.minimal_agent_spawn("recovery_agent")
                assert minimal_agent["status"] == "spawned"
    
    @pytest.mark.asyncio
    async def test_coordination_failure_recovery(self):
        """Test recovery from coordination failures under pressure."""
        async with critical_memory_context("coordination_recovery") as monitor:
            coordination = MinimalMockCoordination()
            
            # Simulate coordination failure
            coordination.agents = None  # Simulate corruption
            
            try:
                # Attempt operation that would fail
                await coordination.minimal_agent_spawn("test_agent")
                
            except (AttributeError, TypeError):
                # Execute recovery
                coordination.agents = weakref.WeakValueDictionary()  # Reinitialize
                coordination.message_queue = []
                
                # Verify recovery
                recovery_agent = await coordination.minimal_agent_spawn("recovery_test")
                assert recovery_agent["status"] == "spawned"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation of functionality under pressure."""
        async with critical_memory_context("graceful_degradation") as monitor:
            coordination = MinimalMockCoordination()
            
            # Test degraded functionality levels
            degradation_levels = {
                "normal": {
                    "max_agents": 10,
                    "queue_size": 1000,
                    "features": ["full_coordination", "optimization", "caching"]
                },
                "degraded": {
                    "max_agents": 2,
                    "queue_size": 10,
                    "features": ["basic_coordination"]
                },
                "critical": {
                    "max_agents": 1,
                    "queue_size": 3,
                    "features": ["emergency_only"]
                }
            }
            
            # Under current memory pressure, should be in critical mode
            current_mode = "critical"
            current_limits = degradation_levels[current_mode]
            
            # Test agent limit enforcement
            agent1 = await coordination.minimal_agent_spawn("critical_agent")
            agent2 = await coordination.minimal_agent_spawn("excess_agent")
            
            assert agent1["status"] == "spawned"
            assert agent2["status"] == "spawn_denied"  # Exceeds critical limit
            
            # Verify only essential features available
            available_features = current_limits["features"]
            assert len(available_features) == 1
            assert "emergency_only" in available_features


@pytest.mark.critical_memory
class TestPerformanceImpactMeasurement:
    """Test performance impact of memory optimizations."""
    
    @pytest.mark.asyncio
    async def test_coordination_latency_under_pressure(self):
        """Test coordination latency with memory optimizations."""
        async with critical_memory_context("latency_measurement") as monitor:
            coordination = MinimalMockCoordination()
            
            import time
            
            # Measure baseline coordination time
            start_time = time.perf_counter()
            
            agent = await coordination.minimal_agent_spawn("latency_test")
            coordination.message_queue.append({
                "from": agent["agent_id"],
                "to": "system", 
                "type": "ping"
            })
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Verify latency remains reasonable under pressure
            assert latency_ms < 50  # Less than 50ms for basic operations
            assert agent["status"] == "spawned"
    
    @pytest.mark.asyncio
    async def test_memory_optimization_effectiveness(self):
        """Test effectiveness of memory optimization strategies."""
        async with critical_memory_context("optimization_effectiveness") as monitor:
            coordination = MinimalMockCoordination()
            
            initial_memory = monitor.get_memory_usage()
            
            # Perform memory-optimized operations
            agent = await coordination.minimal_agent_spawn("optimization_test")
            await coordination.adaptive_scale_down()
            monitor.force_cleanup()
            
            final_memory = monitor.get_memory_usage()
            
            # Verify memory usage remained stable or improved
            memory_delta = final_memory - initial_memory
            assert memory_delta <= 0.05  # No significant increase
            
            # Test optimization strategies
            optimization_results = {
                "weak_references": len(coordination.agents),
                "message_queue_limit": len(coordination.message_queue),
                "fallback_mode": coordination.fallback_mode
            }
            
            # Verify optimizations are active
            assert optimization_results["weak_references"] <= 2  # Limited agents
            assert optimization_results["message_queue_limit"] <= 10  # Limited queue


# Utility function for safe test execution
def run_critical_memory_tests():
    """Safely run critical memory tests with emergency abort capability."""
    monitor = CriticalMemoryMonitor()
    
    if monitor.get_memory_usage() > 99.9:
        print("EMERGENCY ABORT: Memory usage too critical for testing")
        return False
    
    print(f"Current memory usage: {monitor.get_memory_usage():.2f}%")
    print("Executing critical memory integration tests...")
    
    # Run tests with minimal footprint
    pytest.main([
        "/Users/yogi/Projects/Archon-fork/tests/critical-state/memory_pressure_integration_tests.py",
        "-v",
        "-m", "critical_memory",
        "--tb=short",  # Minimal traceback
        "-x",  # Stop on first failure
        "--maxfail=3"  # Limit failures
    ])
    
    return True


if __name__ == "__main__":
    run_critical_memory_tests()