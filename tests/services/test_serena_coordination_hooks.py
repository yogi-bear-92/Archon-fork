"""
Test Suite for Serena Master Agent Coordination Hooks

Comprehensive tests for the coordination hooks system, covering:
- Pre-task semantic preparation
- Post-task knowledge persistence  
- Multi-agent workflow coordination
- Memory synchronization
- Performance monitoring
- Error handling and retry logic
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from python.src.server.services.serena_coordination_hooks import (
    SerenaCoordinationHooks,
    CoordinationLevel,
    HookPhase,
    SemanticContext,
    CoordinationState,
    HookExecutionResult
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def coordination_hooks(temp_dir):
    """Create SerenaCoordinationHooks instance with temporary directories."""
    with patch.object(SerenaCoordinationHooks, '_start_background_tasks'):
        hooks = SerenaCoordinationHooks()
        
        # Override paths to use temporary directory
        hooks.base_path = temp_dir
        hooks.hooks_path = temp_dir / "hooks"
        hooks.memory_path = temp_dir / "memory"
        hooks.metrics_path = temp_dir / "metrics"
        hooks._ensure_directories()
        
        return hooks


@pytest.fixture
def mock_claude_flow_service():
    """Mock Claude Flow service."""
    with patch('python.src.server.services.serena_coordination_hooks.claude_flow_service') as mock:
        mock.memory_operations = AsyncMock(return_value={"status": "success", "result": {}})
        yield mock


class TestPreTaskSemanticPreparation:
    """Test pre-task semantic preparation hooks."""
    
    @pytest.mark.asyncio
    async def test_successful_preparation(self, coordination_hooks, mock_claude_flow_service):
        """Test successful pre-task semantic preparation."""
        task_context = {
            "task_id": "test_task_001",
            "project_path": "/test/project",
            "target_files": ["file1.py", "file2.py"],
            "task_type": "code_analysis"
        }
        
        # Mock the internal analysis methods
        with patch.object(coordination_hooks, '_analyze_project_structure') as mock_analyze:
            mock_analyze.return_value = {"code_files": ["file1.py", "file2.py"]}
            
            result = await coordination_hooks.pre_task_semantic_preparation(
                task_context=task_context,
                coordination_level=CoordinationLevel.INDIVIDUAL
            )
            
            assert result.success
            assert result.execution_time > 0
            assert result.data is not None
            assert "semantic_context" in result.data
            assert "coordination_state" in result.data
            assert task_context["task_id"] in coordination_hooks.semantic_contexts
    
    @pytest.mark.asyncio
    async def test_preparation_with_coordination_level(self, coordination_hooks):
        """Test preparation with different coordination levels."""
        task_context = {"task_id": "test_task_002", "project_path": "/test"}
        
        for level in CoordinationLevel:
            result = await coordination_hooks.pre_task_semantic_preparation(
                task_context=task_context,
                coordination_level=level
            )
            
            assert result.success
            state = coordination_hooks.coordination_states[task_context["task_id"]]
            assert state.coordination_level == level
    
    @pytest.mark.asyncio
    async def test_preparation_error_handling(self, coordination_hooks):
        """Test error handling in pre-task preparation."""
        task_context = {"task_id": "test_task_003"}
        
        # Mock an error in project analysis
        with patch.object(coordination_hooks, '_analyze_project_structure', side_effect=Exception("Test error")):
            result = await coordination_hooks.pre_task_semantic_preparation(
                task_context=task_context
            )
            
            assert not result.success
            assert "Test error" in result.error
            assert result.execution_time > 0


class TestPostTaskKnowledgePersistence:
    """Test post-task knowledge persistence hooks."""
    
    @pytest.mark.asyncio
    async def test_successful_persistence(self, coordination_hooks, mock_claude_flow_service):
        """Test successful post-task knowledge persistence."""
        task_result = {
            "task_id": "test_task_004",
            "success": True,
            "patterns_used": ["MVC", "Repository"],
            "code_changes": ["refactored_method", "added_tests"]
        }
        
        execution_metrics = {
            "execution_time": 2.5,
            "memory_usage": 45.2,
            "cache_hit_rate": 0.85
        }
        
        result = await coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        assert result.success
        assert result.execution_time > 0
        assert result.data is not None
        assert "knowledge_artifacts" in result.data
        assert "learning_patterns" in result.data
    
    @pytest.mark.asyncio
    async def test_learning_patterns_storage(self, coordination_hooks, temp_dir):
        """Test that learning patterns are properly stored."""
        task_result = {
            "task_id": "test_task_005",
            "success": True,
            "task_type": "refactoring"
        }
        
        execution_metrics = {"execution_time": 1.5}
        
        await coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        patterns_file = coordination_hooks.memory_path / "learning_patterns.json"
        assert patterns_file.exists()
        
        with open(patterns_file, 'r') as f:
            patterns_data = json.load(f)
            
        assert "patterns" in patterns_data
        assert len(patterns_data["patterns"]) > 0
        assert patterns_data["patterns"][-1]["task_type"] == "refactoring"
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing(self, coordination_hooks, mock_claude_flow_service):
        """Test knowledge sharing with other agents."""
        task_result = {"task_id": "test_task_006", "success": True}
        execution_metrics = {"execution_time": 1.0}
        
        result = await coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        assert result.success
        # Verify Claude Flow memory operations were called
        mock_claude_flow_service.memory_operations.assert_called()


class TestMultiAgentCoordination:
    """Test multi-agent coordination hooks."""
    
    @pytest.mark.asyncio
    async def test_workflow_coordination_setup(self, coordination_hooks):
        """Test multi-agent workflow coordination setup."""
        workflow_definition = {
            "workflow_id": "test_workflow_001",
            "name": "Test Workflow",
            "description": "Test multi-agent coordination"
        }
        
        participating_agents = ["serena-master", "coder", "reviewer", "tester"]
        
        result = await coordination_hooks.coordinate_multi_agent_workflow(
            workflow_definition=workflow_definition,
            participating_agents=participating_agents
        )
        
        assert result.success
        assert result.data is not None
        assert "workflow_id" in result.data
        assert "role_assignments" in result.data
        assert len(result.data["role_assignments"]) == len(participating_agents)
    
    @pytest.mark.asyncio
    async def test_agent_role_assignment(self, coordination_hooks):
        """Test optimal agent role assignment."""
        workflow_analysis = {
            "complexity_level": "medium",
            "resource_requirements": {"cpu_intensive": True}
        }
        
        participating_agents = ["agent1", "agent2", "agent3"]
        
        role_assignments = await coordination_hooks._assign_agent_roles(
            workflow_analysis, participating_agents
        )
        
        assert "serena-master" in role_assignments
        assert role_assignments["serena-master"]["role"] == "coordinator"
        
        for agent in participating_agents:
            if agent != "serena-master":
                assert agent in role_assignments
                assert "role" in role_assignments[agent]
                assert "responsibilities" in role_assignments[agent]
    
    @pytest.mark.asyncio
    async def test_context_sharing_setup(self, coordination_hooks):
        """Test context sharing mechanism setup."""
        workflow_id = "test_workflow_002"
        role_assignments = {
            "serena-master": {"role": "coordinator", "coordination_level": "master"},
            "coder": {"role": "coder", "coordination_level": "worker"},
            "reviewer": {"role": "reviewer", "coordination_level": "worker"}
        }
        
        context_sharing = await coordination_hooks._setup_context_sharing(
            workflow_id, role_assignments
        )
        
        assert context_sharing["workflow_id"] == workflow_id
        assert "communication_channels" in context_sharing
        assert "synchronization_points" in context_sharing


class TestMemorySynchronization:
    """Test memory synchronization hooks."""
    
    @pytest.mark.asyncio
    async def test_memory_sync_execution(self, coordination_hooks, mock_claude_flow_service):
        """Test memory synchronization execution."""
        # Add some test data to synchronize
        test_context = SemanticContext(
            project_path="/test",
            file_paths=["test.py"],
            symbol_map={"test.py": {"functions": ["test_func"]}},
            architecture_patterns=[],
            complexity_metrics={"score": 2.0},
            last_updated=datetime.now(),
            context_hash="test_hash"
        )
        
        coordination_hooks.semantic_contexts["test_task"] = test_context
        
        sync_context = {"scope": "local"}
        
        result = await coordination_hooks.memory_synchronization_hook(sync_context)
        
        assert result.success
        assert result.data is not None
        assert "sync_results" in result.data
        assert "validation_results" in result.data
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, coordination_hooks):
        """Test memory conflict detection."""
        # Create test states with potential conflicts
        old_state = CoordinationState(
            agent_id="test_agent",
            task_id="test_task",
            coordination_level=CoordinationLevel.INDIVIDUAL,
            active_agents=set(),
            shared_context={},
            performance_metrics={},
            error_count=0,
            last_sync=datetime.now() - timedelta(minutes=15)
        )
        
        coordination_hooks.coordination_states["test_task"] = old_state
        
        current_states = {
            "test_task": {
                "type": "coordination_state",
                "data": {
                    "last_sync": old_state.last_sync.isoformat(),
                    "active_agents": list(old_state.active_agents)
                },
                "checksum": "test_checksum"
            }
        }
        
        conflicts = await coordination_hooks._detect_memory_conflicts(current_states)
        
        # Should detect stale state
        stale_conflicts = [c for c in conflicts if c["type"] == "stale_state"]
        assert len(stale_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, coordination_hooks):
        """Test memory conflict resolution."""
        conflicts = [
            {
                "type": "stale_state",
                "state_id": "test_task",
                "severity": "medium"
            },
            {
                "type": "empty_agent_set",
                "state_id": "test_task",
                "severity": "high"
            }
        ]
        
        # Add test state
        test_state = CoordinationState(
            agent_id="test_agent",
            task_id="test_task",
            coordination_level=CoordinationLevel.INDIVIDUAL,
            active_agents=set(),
            shared_context={},
            performance_metrics={},
            error_count=0,
            last_sync=datetime.now() - timedelta(minutes=15)
        )
        
        coordination_hooks.coordination_states["test_task"] = test_state
        
        resolution_results = await coordination_hooks._resolve_memory_conflicts(conflicts)
        
        assert len(resolution_results["resolved"]) > 0
        assert "serena-master" in coordination_hooks.coordination_states["test_task"].active_agents


class TestPerformanceMonitoring:
    """Test performance monitoring hooks."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_execution(self, coordination_hooks):
        """Test performance monitoring execution."""
        monitoring_context = {
            "scope": "comprehensive",
            "components": ["system", "coordination"]
        }
        
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0
            
            result = await coordination_hooks.performance_monitoring_hook(monitoring_context)
            
            assert result.success
            assert result.data is not None
            assert "performance_metrics" in result.data
            assert "health_analysis" in result.data
            assert "bottleneck_analysis" in result.data
    
    @pytest.mark.asyncio
    async def test_bottleneck_detection(self, coordination_hooks):
        """Test performance bottleneck detection."""
        performance_metrics = {
            "system_metrics": {
                "cpu_usage": 90.0,
                "memory_usage": 85.0
            },
            "coordination_metrics": {
                "average_coordination_latency": 0.15
            }
        }
        
        bottleneck_analysis = await coordination_hooks._detect_performance_bottlenecks(
            performance_metrics
        )
        
        assert len(bottleneck_analysis["bottlenecks"]) > 0
        
        bottleneck_types = [b["type"] for b in bottleneck_analysis["bottlenecks"]]
        assert "cpu_bottleneck" in bottleneck_types
        assert "memory_bottleneck" in bottleneck_types
        assert "coordination_latency" in bottleneck_types
    
    @pytest.mark.asyncio
    async def test_automatic_optimizations(self, coordination_hooks):
        """Test automatic optimization application."""
        recommendations = {
            "automatic_actions": [
                {
                    "action": "clear_old_caches",
                    "description": "Clear old semantic caches",
                    "automatic": True
                }
            ]
        }
        
        # Add some old contexts to clear
        old_context = SemanticContext(
            project_path="/test",
            file_paths=[],
            symbol_map={},
            architecture_patterns=[],
            complexity_metrics={},
            last_updated=datetime.now() - timedelta(hours=2),
            context_hash="old_hash"
        )
        
        coordination_hooks.semantic_contexts["old_task"] = old_context
        
        auto_results = await coordination_hooks._apply_automatic_optimizations(recommendations)
        
        assert len(auto_results["applied"]) > 0
        assert "clear_old_caches" in [a["action"] for a in auto_results["applied"]]


class TestErrorHandlingAndRetry:
    """Test error handling and retry logic."""
    
    @pytest.mark.asyncio
    async def test_hook_execution_context(self, coordination_hooks):
        """Test hook execution context manager."""
        hook_name = "test_hook"
        context = {"test": "data"}
        
        async with coordination_hooks.hook_execution_context(hook_name, context) as hook_id:
            assert hook_id in coordination_hooks.active_hooks
            assert coordination_hooks.active_hooks[hook_id]["name"] == hook_name
        
        # Should be cleaned up after context exit
        assert hook_id not in coordination_hooks.active_hooks
    
    @pytest.mark.asyncio
    async def test_error_pattern_recording(self, coordination_hooks):
        """Test error pattern recording."""
        initial_count = len(coordination_hooks.error_patterns)
        
        # Simulate an error
        await coordination_hooks._handle_hook_error(
            "test_hook_id",
            "test_hook",
            ValueError("Test error"),
            {"test": "context"}
        )
        
        assert len(coordination_hooks.error_patterns) > initial_count
        error_key = "test_hook:ValueError"
        assert error_key in coordination_hooks.error_patterns
        assert coordination_hooks.error_patterns[error_key] > 0
    
    def test_retry_decision_logic(self, coordination_hooks):
        """Test retry decision logic."""
        # Should retry transient errors
        assert coordination_hooks._should_retry_error(ConnectionError("Network error"))
        assert coordination_hooks._should_retry_error(TimeoutError("Request timeout"))
        
        # Should not retry permanent errors
        assert not coordination_hooks._should_retry_error(ValueError("Invalid input"))
        assert not coordination_hooks._should_retry_error(KeyError("Missing key"))


class TestIntegrationPatterns:
    """Test integration patterns and utilities."""
    
    @pytest.mark.asyncio
    async def test_coordination_pattern_registration(self, coordination_hooks, temp_dir):
        """Test coordination pattern registration."""
        pattern_name = "test_pattern"
        pattern_definition = {
            "type": "hierarchical",
            "agents": ["coordinator", "worker1", "worker2"],
            "communication": "hub_and_spoke"
        }
        
        result = await coordination_hooks.register_agent_coordination_pattern(
            pattern_name, pattern_definition
        )
        
        assert result["status"] == "success"
        assert result["pattern_name"] == pattern_name
        
        # Check that pattern was saved
        patterns_file = coordination_hooks.hooks_path / "coordination_patterns.json"
        assert patterns_file.exists()
        
        with open(patterns_file, 'r') as f:
            registry = json.load(f)
            
        assert pattern_name in registry["patterns"]
        assert registry["patterns"][pattern_name]["definition"] == pattern_definition
    
    @pytest.mark.asyncio
    async def test_coordination_metrics_collection(self, coordination_hooks):
        """Test coordination metrics collection."""
        # Add some test data
        test_state = CoordinationState(
            agent_id="test_agent",
            task_id="test_task",
            coordination_level=CoordinationLevel.GROUP,
            active_agents={"agent1", "agent2"},
            shared_context={},
            performance_metrics={"latency": 0.05},
            error_count=1,
            last_sync=datetime.now()
        )
        
        coordination_hooks.coordination_states["test_task"] = test_state
        coordination_hooks.hook_performance["test_hook"] = [1.0, 2.0, 1.5]
        coordination_hooks.error_patterns["test_error"] = 3
        
        metrics = await coordination_hooks.get_coordination_metrics()
        
        assert "system_status" in metrics
        assert "performance_summary" in metrics
        assert "coordination_health" in metrics
        assert metrics["system_status"]["active_coordination_states"] == 1
        assert "test_hook" in metrics["performance_summary"]["hook_performance"]
    
    @pytest.mark.asyncio
    async def test_coordination_data_export(self, coordination_hooks, temp_dir):
        """Test coordination data export functionality."""
        # Add test data
        test_context = SemanticContext(
            project_path="/test",
            file_paths=["test.py"],
            symbol_map={"test.py": {"functions": ["test_func"]}},
            architecture_patterns=[{"type": "MVC", "confidence": 0.8}],
            complexity_metrics={"score": 2.5},
            last_updated=datetime.now(),
            context_hash="test_hash"
        )
        
        coordination_hooks.semantic_contexts["test_task"] = test_context
        
        export_path = str(temp_dir / "export_test.json")
        
        result = await coordination_hooks.export_coordination_data(
            export_path=export_path,
            include_sensitive=False
        )
        
        assert result["status"] == "success"
        assert Path(result["export_path"]).exists()
        
        # Verify export content
        with open(result["export_path"], 'r') as f:
            export_data = json.load(f)
            
        assert "export_metadata" in export_data
        assert "semantic_contexts" in export_data
        assert "test_task" in export_data["semantic_contexts"]


class TestResourceCleanup:
    """Test resource cleanup and lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_task_resource_cleanup(self, coordination_hooks, temp_dir):
        """Test task resource cleanup."""
        task_id = "cleanup_test_task"
        
        # Add test resources
        test_context = SemanticContext(
            project_path="/test",
            file_paths=[],
            symbol_map={},
            architecture_patterns=[],
            complexity_metrics={},
            last_updated=datetime.now() - timedelta(hours=2),
            context_hash="cleanup_hash"
        )
        
        coordination_hooks.semantic_contexts[task_id] = test_context
        
        test_state = CoordinationState(
            agent_id="test_agent",
            task_id=task_id,
            coordination_level=CoordinationLevel.INDIVIDUAL,
            active_agents=set(),
            shared_context={},
            performance_metrics={},
            error_count=0,
            last_sync=datetime.now()
        )
        
        coordination_hooks.coordination_states[task_id] = test_state
        
        # Create test memory file
        memory_file = coordination_hooks.memory_path / f"task_{task_id}_memory.json"
        memory_file.parent.mkdir(exist_ok=True)
        with open(memory_file, 'w') as f:
            json.dump({"test": "data"}, f)
            
        # Modify file time to be old
        old_time = datetime.now() - timedelta(days=8)
        memory_file.touch(times=(old_time.timestamp(), old_time.timestamp()))
        
        cleanup_results = await coordination_hooks._cleanup_task_resources(task_id)
        
        assert cleanup_results["contexts_cleaned"] > 0
        assert task_id not in coordination_hooks.coordination_states
    
    @pytest.mark.asyncio
    async def test_memory_file_archiving(self, coordination_hooks, temp_dir):
        """Test memory file archiving for old files."""
        task_id = "archive_test_task"
        
        # Create old memory file
        memory_file = coordination_hooks.memory_path / f"task_{task_id}_memory.json"
        memory_file.parent.mkdir(exist_ok=True)
        with open(memory_file, 'w') as f:
            json.dump({"test": "archive_data"}, f)
        
        # Make file old (8 days)
        old_time = datetime.now() - timedelta(days=8)
        memory_file.touch(times=(old_time.timestamp(), old_time.timestamp()))
        
        cleanup_results = await coordination_hooks._cleanup_task_resources(task_id)
        
        # Check if file was archived
        archive_path = coordination_hooks.memory_path / "archive" / memory_file.name
        if cleanup_results["files_archived"] > 0:
            assert archive_path.exists()
            assert not memory_file.exists()


@pytest.mark.integration
class TestFullWorkflowIntegration:
    """Integration tests for complete workflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_task_lifecycle(self, coordination_hooks, mock_claude_flow_service):
        """Test complete task lifecycle from preparation to completion."""
        task_id = "integration_test_task"
        
        # Phase 1: Pre-task preparation
        task_context = {
            "task_id": task_id,
            "project_path": "/test/project",
            "target_files": ["main.py", "utils.py"],
            "task_type": "refactoring"
        }
        
        prep_result = await coordination_hooks.pre_task_semantic_preparation(
            task_context=task_context,
            coordination_level=CoordinationLevel.GROUP
        )
        
        assert prep_result.success
        assert task_id in coordination_hooks.semantic_contexts
        assert task_id in coordination_hooks.coordination_states
        
        # Phase 2: Simulate task execution
        # (This would be done by the actual task executor)
        
        # Phase 3: Post-task completion
        task_result = {
            "task_id": task_id,
            "success": True,
            "task_type": "refactoring",
            "patterns_used": ["Extract Method", "Move Class"],
            "files_modified": ["main.py", "utils.py"],
            "quality_improvements": {"complexity_reduction": 0.15}
        }
        
        execution_metrics = {
            "execution_time": 5.2,
            "memory_usage": 67.5,
            "cache_hit_rate": 0.78
        }
        
        completion_result = await coordination_hooks.post_task_knowledge_persistence(
            task_result=task_result,
            execution_metrics=execution_metrics
        )
        
        assert completion_result.success
        
        # Verify learning patterns were stored
        patterns_file = coordination_hooks.memory_path / "learning_patterns.json"
        assert patterns_file.exists()
        
        # Phase 4: Resource cleanup
        cleanup_result = await coordination_hooks._cleanup_task_resources(task_id)
        assert cleanup_result is not None
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_integration(self, coordination_hooks):
        """Test complete multi-agent coordination scenario."""
        workflow_definition = {
            "workflow_id": "integration_workflow",
            "name": "Code Analysis and Improvement Workflow",
            "description": "Analyze code, suggest improvements, and implement changes",
            "phases": [
                {"name": "analysis", "duration": 120},
                {"name": "planning", "duration": 60},
                {"name": "implementation", "duration": 300},
                {"name": "testing", "duration": 180},
                {"name": "review", "duration": 90}
            ]
        }
        
        participating_agents = [
            "serena-master",
            "code-analyzer", 
            "system-architect",
            "coder",
            "tester",
            "reviewer"
        ]
        
        # Setup coordination
        coord_result = await coordination_hooks.coordinate_multi_agent_workflow(
            workflow_definition=workflow_definition,
            participating_agents=participating_agents
        )
        
        assert coord_result.success
        assert "workflow_id" in coord_result.data
        assert len(coord_result.data["role_assignments"]) == len(participating_agents)
        
        # Verify coordination protocol was stored
        protocol_file = coordination_hooks.hooks_path / f"workflow_{workflow_definition['workflow_id']}_protocol.json"
        assert protocol_file.exists()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_and_optimization_cycle(self, coordination_hooks):
        """Test complete performance monitoring and optimization cycle."""
        # Add some load to the system
        for i in range(5):
            task_context = SemanticContext(
                project_path=f"/test/project_{i}",
                file_paths=[f"file_{i}.py"],
                symbol_map={f"file_{i}.py": {"functions": [f"func_{i}"]}},
                architecture_patterns=[],
                complexity_metrics={"score": i * 0.5},
                last_updated=datetime.now(),
                context_hash=f"hash_{i}"
            )
            coordination_hooks.semantic_contexts[f"task_{i}"] = task_context
        
        # Run performance monitoring
        monitoring_context = {
            "scope": "comprehensive",
            "components": ["system", "coordination", "semantic_analysis"],
            "auto_optimize": True
        }
        
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 80.0
            
            result = await coordination_hooks.performance_monitoring_hook(monitoring_context)
            
            assert result.success
            assert "performance_metrics" in result.data
            assert "health_analysis" in result.data
            assert "optimization_recommendations" in result.data
            
            # Check if any automatic optimizations were applied
            auto_opts = result.data.get("auto_optimization_results", {})
            if auto_opts.get("applied"):
                assert len(auto_opts["applied"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])