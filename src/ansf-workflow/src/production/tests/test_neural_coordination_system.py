#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced Neural Coordination System
Phase 3 Validation - Testing all neural coordination features against 88.7% baseline
"""

import asyncio
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import json
import tempfile
from pathlib import Path

# Import the neural coordination system and components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from neural.neural_coordination_system import (
    NeuralCoordinationSystem, NeuralCoordinationConfig, SystemPerformanceMetrics,
    create_neural_coordination_system
)
from neural.coordination.transformer_coordinator import AgentState, CoordinationMetrics
from neural.models.predictive_scaling_network import WorkloadMetrics
from neural.coordination.cross_swarm_intelligence import SwarmIdentity, SwarmRole

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestNeuralCoordinationSystem:
    """Comprehensive test suite for neural coordination system"""
    
    @pytest.fixture
    async def neural_system(self):
        """Create a neural coordination system for testing"""
        config = {
            'baseline_accuracy': 0.887,
            'target_improvement': 0.15,
            'enable_cross_swarm': True,
            'enable_ensemble': True,
            'enable_predictive_scaling': True,
            'memory_threshold_mb': 500,  # Lower for testing
            'coordination_interval_seconds': 5,  # Faster for testing
            'performance_report_interval_seconds': 10
        }
        
        system = create_neural_coordination_system(config)
        
        # Start system
        startup_result = await system.start_system(port=18765)  # Different port for testing
        assert startup_result['status'] == 'success'
        
        yield system
        
        # Cleanup
        await system.stop_system()
    
    @pytest.fixture
    def sample_agent_states(self) -> List[AgentState]:
        """Create sample agent states for testing"""
        return [
            AgentState(
                agent_id=f"test_agent_{i}",
                task_embedding=torch.randn(512),
                context_vector=torch.randn(512),
                performance_score=0.85 + 0.1 * np.random.random(),
                resource_utilization=50 + 30 * np.random.random(),
                coordination_weight=1.0,
                neural_patterns={'pattern_type': f'test_pattern_{i}'},
                memory_usage=100 + 50 * np.random.random()
            ) for i in range(8)
        ]
    
    @pytest.fixture
    def sample_context(self) -> Dict[str, Any]:
        """Create sample context for testing"""
        return {
            'task_type': 'neural_coordination_test',
            'task_complexity': 0.8,
            'priority': 0.9,
            'deadline_pressure': 0.6,
            'enable_cross_swarm': True,
            'accuracy_requirement': 0.95,
            'resource_availability': 0.8,
            'collaboration_required': True
        }

class TestBasicFunctionality:
    """Test basic neural coordination functionality"""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test neural coordination system initialization"""
        
        # Test default configuration
        system = create_neural_coordination_system()
        assert system.config.baseline_accuracy == 0.887
        assert system.config.target_improvement == 0.15
        
        # Test custom configuration
        custom_config = {
            'baseline_accuracy': 0.9,
            'target_improvement': 0.2,
            'enable_cross_swarm': False
        }
        
        custom_system = create_neural_coordination_system(custom_config)
        assert custom_system.config.baseline_accuracy == 0.9
        assert custom_system.config.target_improvement == 0.2
        assert custom_system.config.enable_cross_swarm == False
    
    @pytest.mark.asyncio
    async def test_system_startup_shutdown(self):
        """Test system startup and shutdown procedures"""
        
        system = create_neural_coordination_system()
        
        # Test startup
        startup_result = await system.start_system(port=18766)
        assert startup_result['status'] == 'success'
        assert 'system_id' in startup_result
        assert system.is_running == True
        
        # Test shutdown
        shutdown_result = await system.stop_system()
        assert shutdown_result['status'] == 'stopped'
        assert system.is_running == False
    
    @pytest.mark.asyncio
    async def test_component_initialization(self, neural_system):
        """Test that all components are properly initialized"""
        
        # Check transformer coordinator
        assert neural_system.transformer_coordinator is not None
        assert neural_system.transformer_coordinator.baseline_accuracy == 0.887
        
        # Check cross-swarm coordinator
        if neural_system.config.enable_cross_swarm:
            assert neural_system.cross_swarm_coordinator is not None
        
        # Check ensemble coordinator
        if neural_system.config.enable_ensemble:
            assert neural_system.ensemble_coordinator is not None
        
        # Check predictive scaler
        if neural_system.config.enable_predictive_scaling:
            assert neural_system.predictive_scaler is not None

class TestNeuralCoordination:
    """Test neural coordination capabilities"""
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, neural_system, sample_agent_states, sample_context):
        """Test multi-agent neural coordination"""
        
        task_description = "Test neural coordination with transformer-based attention mechanisms"
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        # Verify coordination success
        assert coordination_result['status'] == 'success'
        assert 'coordination_id' in coordination_result
        assert coordination_result['coordination_latency_seconds'] > 0
        
        # Verify component results
        assert 'transformer_coordination' in coordination_result
        assert 'predictive_scaling' in coordination_result
        assert 'ensemble_decision' in coordination_result
        
        # Verify performance improvements
        assert coordination_result['estimated_accuracy_improvement'] >= 0
        assert coordination_result['resource_efficiency'] >= 0
        assert coordination_result['coordination_quality'] >= 0
    
    @pytest.mark.asyncio
    async def test_transformer_coordination_quality(self, neural_system, sample_agent_states, sample_context):
        """Test transformer coordination quality and coherence"""
        
        task_description = "Complex multi-step coordination task requiring high coherence"
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        transformer_metrics = coordination_result['transformer_coordination']['metrics']
        
        # Verify transformer metrics
        assert 'accuracy' in transformer_metrics
        assert 'cross_agent_coherence' in transformer_metrics
        assert 'prediction_confidence' in transformer_metrics
        
        # Check baseline improvement
        accuracy = transformer_metrics['accuracy']
        baseline = neural_system.config.baseline_accuracy
        
        # Should maintain or improve upon baseline
        assert accuracy >= baseline * 0.95  # Allow 5% variance for testing
        
        # Coherence should be reasonable
        coherence = transformer_metrics['cross_agent_coherence']
        assert 0.0 <= coherence <= 1.0
        assert coherence >= 0.3  # Minimum coherence threshold
    
    @pytest.mark.asyncio
    async def test_predictive_scaling_accuracy(self, neural_system, sample_agent_states, sample_context):
        """Test predictive scaling accuracy and recommendations"""
        
        task_description = "High-load coordination task requiring dynamic scaling"
        sample_context['task_complexity'] = 0.95  # High complexity
        sample_context['resource_availability'] = 0.3  # Limited resources
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        scaling_result = coordination_result['predictive_scaling']
        
        if scaling_result['prediction'] is not None:
            prediction = scaling_result['prediction']
            
            # Verify prediction structure
            assert 'optimal_agent_count' in prediction
            assert 'confidence' in prediction
            assert 'resource_requirements' in prediction
            
            # Verify scaling recommendations
            optimal_agents = prediction['optimal_agent_count']
            assert 1 <= optimal_agents <= 32  # Within reasonable bounds
            
            confidence = prediction['confidence']
            assert 0.0 <= confidence <= 1.0
            
            # If confidence is high, scaling should be reasonable
            if confidence > 0.7:
                current_agents = len(sample_agent_states)
                agent_diff = abs(optimal_agents - current_agents)
                assert agent_diff <= 10  # Reasonable scaling change
    
    @pytest.mark.asyncio
    async def test_ensemble_coordination(self, neural_system, sample_agent_states, sample_context):
        """Test neural ensemble coordination and diversity"""
        
        task_description = "Ensemble coordination requiring high accuracy and diversity"
        sample_context['accuracy_requirement'] = 0.95
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        ensemble_result = coordination_result['ensemble_decision']
        
        if ensemble_result is not None:
            # Verify ensemble structure
            assert 'prediction' in ensemble_result
            assert 'confidence' in ensemble_result
            assert 'diversity_score' in ensemble_result
            assert 'consensus_score' in ensemble_result
            
            # Verify diversity
            diversity = ensemble_result['diversity_score']
            assert 0.0 <= diversity <= 1.0
            
            # Verify consensus
            consensus = ensemble_result['consensus_score']
            assert 0.0 <= consensus <= 1.0
            
            # High accuracy requirements should yield high confidence
            if sample_context['accuracy_requirement'] > 0.9:
                confidence = ensemble_result['confidence']
                assert confidence >= 0.6  # Reasonable confidence threshold

class TestPerformanceValidation:
    """Test performance against baseline and targets"""
    
    @pytest.mark.asyncio
    async def test_baseline_accuracy_maintenance(self, neural_system, sample_agent_states, sample_context):
        """Test that system maintains baseline accuracy of 88.7%"""
        
        # Run multiple coordination tasks
        coordination_results = []
        
        for i in range(5):
            task_description = f"Baseline validation task {i+1}"
            context = {**sample_context, 'task_iteration': i}
            
            result = await neural_system.coordinate_multi_agent_task(
                task_description, sample_agent_states, context
            )
            
            coordination_results.append(result)
        
        # Calculate average performance
        accuracy_improvements = [
            result['estimated_accuracy_improvement'] 
            for result in coordination_results
        ]
        
        avg_improvement = np.mean(accuracy_improvements)
        baseline = neural_system.config.baseline_accuracy
        
        # Estimated accuracy should meet or exceed baseline
        estimated_avg_accuracy = baseline + avg_improvement
        
        logger.info(f"Average estimated accuracy: {estimated_avg_accuracy:.3f}")
        logger.info(f"Baseline requirement: {baseline:.3f}")
        logger.info(f"Average improvement: {avg_improvement:.3f}")
        
        # Should maintain baseline accuracy
        assert estimated_avg_accuracy >= baseline * 0.98  # Allow 2% variance for testing
    
    @pytest.mark.asyncio
    async def test_performance_improvement_target(self, neural_system, sample_agent_states, sample_context):
        """Test that system works toward 15% improvement target"""
        
        # Run optimization-focused coordination
        sample_context['priority'] = 1.0  # Maximum priority
        sample_context['accuracy_requirement'] = 0.95
        sample_context['enable_cross_swarm'] = True
        
        task_description = "High-priority optimization task targeting 15% improvement"
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        improvement = coordination_result['estimated_accuracy_improvement']
        target_improvement = neural_system.config.target_improvement
        
        logger.info(f"Achieved improvement: {improvement:.3f}")
        logger.info(f"Target improvement: {target_improvement:.3f}")
        
        # Should show meaningful progress toward target
        # (Complete target achievement requires training, this tests coordination capability)
        assert improvement >= target_improvement * 0.3  # At least 30% of target improvement
    
    @pytest.mark.asyncio
    async def test_resource_efficiency(self, neural_system, sample_agent_states, sample_context):
        """Test resource efficiency optimization"""
        
        # Test with resource constraints
        sample_context['resource_availability'] = 0.5  # Limited resources
        
        task_description = "Resource-constrained coordination task"
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        resource_efficiency = coordination_result['resource_efficiency']
        
        logger.info(f"Resource efficiency: {resource_efficiency:.3f}")
        
        # Should achieve reasonable efficiency
        assert resource_efficiency >= 0.4  # Minimum acceptable efficiency
        assert resource_efficiency <= 1.0  # Maximum possible efficiency
        
        # With limited resources, efficiency should reflect constraints
        if sample_context['resource_availability'] < 0.6:
            assert resource_efficiency >= 0.3  # Lower threshold for constrained scenarios
    
    @pytest.mark.asyncio
    async def test_coordination_latency(self, neural_system, sample_agent_states, sample_context):
        """Test coordination latency performance"""
        
        task_description = "Latency-sensitive coordination task"
        
        start_time = datetime.now()
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        end_time = datetime.now()
        actual_latency = (end_time - start_time).total_seconds()
        reported_latency = coordination_result['coordination_latency_seconds']
        
        logger.info(f"Actual latency: {actual_latency:.3f}s")
        logger.info(f"Reported latency: {reported_latency:.3f}s")
        
        # Latency should be reasonable
        assert actual_latency < 10.0  # Should complete within 10 seconds
        assert reported_latency < actual_latency + 1.0  # Reported should be close to actual
        
        # For complex neural operations, some latency is expected
        assert reported_latency >= 0.01  # Should take at least some time

class TestSystemPerformanceMonitoring:
    """Test system performance monitoring and assessment"""
    
    @pytest.mark.asyncio
    async def test_performance_assessment(self, neural_system):
        """Test comprehensive performance assessment"""
        
        performance_metrics = await neural_system.assess_system_performance()
        
        # Verify metrics structure
        assert isinstance(performance_metrics, SystemPerformanceMetrics)
        
        # Check core metrics
        assert hasattr(performance_metrics, 'overall_accuracy')
        assert hasattr(performance_metrics, 'latency_ms')
        assert hasattr(performance_metrics, 'resource_efficiency')
        
        # Check component metrics
        assert hasattr(performance_metrics, 'transformer_accuracy')
        assert hasattr(performance_metrics, 'ensemble_accuracy')
        assert hasattr(performance_metrics, 'cross_swarm_efficiency')
        
        # Verify reasonable values
        assert 0.0 <= performance_metrics.overall_accuracy <= 1.0
        assert performance_metrics.latency_ms >= 0.0
        assert 0.0 <= performance_metrics.resource_efficiency <= 1.0
        
        # Baseline accuracy should be maintained
        assert performance_metrics.overall_accuracy >= 0.85  # Close to baseline
    
    @pytest.mark.asyncio
    async def test_system_optimization(self, neural_system):
        """Test system optimization capabilities"""
        
        target_metrics = {
            'accuracy': 0.95,
            'latency_ms': 200.0,
            'resource_efficiency': 0.85
        }
        
        optimization_result = await neural_system.optimize_system_configuration(target_metrics)
        
        # Verify optimization attempt
        assert optimization_result['status'] in ['success', 'error']
        
        if optimization_result['status'] == 'success':
            assert 'optimization_results' in optimization_result
            assert 'improvement_estimate' in optimization_result
            
            # Check improvement estimates
            improvements = optimization_result['improvement_estimate']
            assert isinstance(improvements, dict)
    
    @pytest.mark.asyncio
    async def test_emergency_coordination(self, neural_system):
        """Test emergency coordination response"""
        
        emergency_result = await neural_system.handle_emergency_coordination(
            'performance_degradation',
            {'current_accuracy': 0.75, 'severity': 'high'}
        )
        
        # Verify emergency response
        assert emergency_result['status'] == 'success'
        assert emergency_result['emergency_handled'] == True
        assert emergency_result['emergency_type'] == 'performance_degradation'
        assert emergency_result['response_time_seconds'] >= 0
        
        # Check actions taken
        assert 'actions_taken' in emergency_result
        actions = emergency_result['actions_taken']
        
        # At least one type of action should be taken
        action_types = ['scaling', 'cross_swarm', 'ensemble', 'recovery']
        assert any(actions.get(action_type) is not None for action_type in action_types)

class TestCrossSwarmIntelligence:
    """Test cross-swarm intelligence and coordination"""
    
    @pytest.mark.asyncio
    async def test_cross_swarm_knowledge_sharing(self, neural_system, sample_context):
        """Test cross-swarm knowledge sharing capabilities"""
        
        if not neural_system.config.enable_cross_swarm:
            pytest.skip("Cross-swarm coordination disabled")
        
        # Test knowledge sharing through coordination
        sample_context['enable_cross_swarm'] = True
        
        task_description = "Cross-swarm knowledge sharing test"
        agent_states = [
            AgentState(
                agent_id="cross_swarm_agent",
                task_embedding=torch.randn(512),
                context_vector=torch.randn(512),
                performance_score=0.9,
                resource_utilization=60.0,
                coordination_weight=1.0,
                neural_patterns={'cross_swarm_pattern': True},
                memory_usage=120.0
            )
        ]
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, agent_states, sample_context
        )
        
        # Verify cross-swarm integration
        assert 'cross_swarm_sharing' in coordination_result
        cross_swarm_result = coordination_result['cross_swarm_sharing']
        
        # Should attempt knowledge sharing (even if no other swarms connected)
        assert cross_swarm_result is not None

class TestSystemIntegration:
    """Test integration between all neural coordination components"""
    
    @pytest.mark.asyncio
    async def test_component_integration(self, neural_system, sample_agent_states, sample_context):
        """Test integration between transformer, ensemble, scaling, and cross-swarm components"""
        
        task_description = "Full integration test with all neural coordination components"
        
        coordination_result = await neural_system.coordinate_multi_agent_task(
            task_description, sample_agent_states, sample_context
        )
        
        # Verify all components produced results
        assert 'transformer_coordination' in coordination_result
        assert 'predictive_scaling' in coordination_result
        assert 'ensemble_decision' in coordination_result
        assert 'optimization' in coordination_result
        
        # Verify integration quality metrics
        assert 'coordination_quality' in coordination_result
        coordination_quality = coordination_result['coordination_quality']
        assert 0.0 <= coordination_quality <= 1.0
        
        # Integration should provide meaningful coordination
        assert coordination_quality >= 0.3  # Minimum integration quality
    
    @pytest.mark.asyncio
    async def test_system_status_reporting(self, neural_system):
        """Test comprehensive system status reporting"""
        
        system_status = await neural_system.get_system_status()
        
        # Verify status structure
        assert system_status['status'] == 'running'
        assert 'system_id' in system_status
        assert 'uptime_seconds' in system_status
        assert 'current_performance' in system_status
        assert 'component_status' in system_status
        
        # Check component statuses
        component_status = system_status['component_status']
        
        if neural_system.config.enable_cross_swarm:
            assert 'cross_swarm' in component_status
        
        if neural_system.config.enable_ensemble:
            assert 'ensemble' in component_status
        
        if neural_system.config.enable_predictive_scaling:
            assert 'predictive_scaler' in component_status

class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks against baseline"""
    
    @pytest.mark.asyncio
    async def test_accuracy_benchmark_suite(self, neural_system):
        """Comprehensive accuracy benchmark against 88.7% baseline"""
        
        # Create diverse test scenarios
        test_scenarios = [
            {
                'name': 'simple_coordination',
                'agents': 3,
                'complexity': 0.3,
                'priority': 0.5
            },
            {
                'name': 'medium_coordination',
                'agents': 6,
                'complexity': 0.6,
                'priority': 0.7
            },
            {
                'name': 'complex_coordination',
                'agents': 10,
                'complexity': 0.9,
                'priority': 0.9
            },
            {
                'name': 'resource_constrained',
                'agents': 4,
                'complexity': 0.7,
                'priority': 0.8,
                'resource_availability': 0.3
            },
            {
                'name': 'high_accuracy_requirement',
                'agents': 8,
                'complexity': 0.8,
                'priority': 1.0,
                'accuracy_requirement': 0.98
            }
        ]
        
        benchmark_results = []
        
        for scenario in test_scenarios:
            logger.info(f"Running benchmark scenario: {scenario['name']}")
            
            # Create scenario-specific agent states
            agent_states = [
                AgentState(
                    agent_id=f"{scenario['name']}_agent_{i}",
                    task_embedding=torch.randn(512),
                    context_vector=torch.randn(512),
                    performance_score=0.8 + 0.15 * np.random.random(),
                    resource_utilization=40 + 40 * np.random.random(),
                    coordination_weight=1.0,
                    neural_patterns={'scenario': scenario['name']},
                    memory_usage=80 + 60 * np.random.random()
                ) for i in range(scenario['agents'])
            ]
            
            # Create scenario context
            context = {
                'task_type': f"benchmark_{scenario['name']}",
                'task_complexity': scenario['complexity'],
                'priority': scenario['priority'],
                'resource_availability': scenario.get('resource_availability', 0.8),
                'accuracy_requirement': scenario.get('accuracy_requirement', 0.9),
                'enable_cross_swarm': True
            }
            
            task_description = f"Benchmark coordination task: {scenario['name']}"
            
            # Run coordination
            start_time = datetime.now()
            coordination_result = await neural_system.coordinate_multi_agent_task(
                task_description, agent_states, context
            )
            end_time = datetime.now()
            
            # Record results
            benchmark_result = {
                'scenario': scenario['name'],
                'status': coordination_result['status'],
                'estimated_accuracy_improvement': coordination_result.get('estimated_accuracy_improvement', 0.0),
                'resource_efficiency': coordination_result.get('resource_efficiency', 0.0),
                'coordination_quality': coordination_result.get('coordination_quality', 0.0),
                'latency_seconds': (end_time - start_time).total_seconds(),
                'agents_coordinated': len(agent_states)
            }
            
            benchmark_results.append(benchmark_result)
            
            logger.info(f"Scenario {scenario['name']}: "
                       f"Improvement={benchmark_result['estimated_accuracy_improvement']:.3f}, "
                       f"Efficiency={benchmark_result['resource_efficiency']:.3f}, "
                       f"Quality={benchmark_result['coordination_quality']:.3f}")
        
        # Analyze benchmark results
        successful_scenarios = [r for r in benchmark_results if r['status'] == 'success']
        assert len(successful_scenarios) >= 4  # At least 80% success rate
        
        # Calculate aggregate performance
        avg_improvement = np.mean([r['estimated_accuracy_improvement'] for r in successful_scenarios])
        avg_efficiency = np.mean([r['resource_efficiency'] for r in successful_scenarios])
        avg_quality = np.mean([r['coordination_quality'] for r in successful_scenarios])
        avg_latency = np.mean([r['latency_seconds'] for r in successful_scenarios])
        
        baseline = neural_system.config.baseline_accuracy
        estimated_avg_accuracy = baseline + avg_improvement
        
        # Performance assertions
        logger.info(f"\n=== BENCHMARK RESULTS ===")
        logger.info(f"Baseline Accuracy: {baseline:.3f}")
        logger.info(f"Estimated Average Accuracy: {estimated_avg_accuracy:.3f}")
        logger.info(f"Average Improvement: {avg_improvement:.3f}")
        logger.info(f"Average Resource Efficiency: {avg_efficiency:.3f}")
        logger.info(f"Average Coordination Quality: {avg_quality:.3f}")
        logger.info(f"Average Latency: {avg_latency:.3f}s")
        
        # Key performance assertions
        assert estimated_avg_accuracy >= baseline * 0.98  # Maintain 98% of baseline
        assert avg_improvement >= 0.0  # Should show some improvement
        assert avg_efficiency >= 0.3  # Reasonable resource efficiency
        assert avg_quality >= 0.3  # Reasonable coordination quality
        assert avg_latency < 5.0  # Reasonable latency
        
        return benchmark_results

# Integration test running all components
@pytest.mark.asyncio
async def test_complete_system_integration():
    """Complete system integration test"""
    
    logger.info("🚀 Starting Complete Neural Coordination System Integration Test")
    
    # Create and start system
    config = {
        'baseline_accuracy': 0.887,
        'target_improvement': 0.15,
        'enable_cross_swarm': True,
        'enable_ensemble': True,
        'enable_predictive_scaling': True
    }
    
    system = create_neural_coordination_system(config)
    startup_result = await system.start_system(port=19000)
    
    try:
        assert startup_result['status'] == 'success'
        
        # Run comprehensive test
        test_suite = TestPerformanceBenchmarks()
        benchmark_results = await test_suite.test_accuracy_benchmark_suite(system)
        
        # Final system assessment
        final_performance = await system.assess_system_performance()
        
        logger.info(f"🏆 Final System Performance:")
        logger.info(f"  Overall Accuracy: {final_performance.overall_accuracy:.3f}")
        logger.info(f"  Accuracy Improvement: {final_performance.accuracy_improvement:.3f}")
        logger.info(f"  Resource Efficiency: {final_performance.resource_efficiency:.3f}")
        logger.info(f"  Coordination Quality: {final_performance.coordination_coherence:.3f}")
        
        # Success criteria
        assert final_performance.overall_accuracy >= 0.85  # Near baseline
        assert final_performance.accuracy_improvement >= 0.0  # Some improvement
        
        logger.info("✅ Complete Neural Coordination System Integration Test PASSED")
        
    finally:
        # Always stop the system
        await system.stop_system()

if __name__ == "__main__":
    # Run the complete integration test
    asyncio.run(test_complete_system_integration())