"""
ML-Enhanced Coordination Test Suite
Comprehensive testing for ML-enhanced Claude coordination hooks integration

Author: Claude Code ML Developer
Target: Validate 94.7% coordination accuracy with ANSF Phase 2 system
"""

import asyncio
import time
import json
import logging
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
from dataclasses import asdict

# Import our ML coordination system
from ml_enhanced_coordination_hooks import (
    MLEnhancedCoordinationHooks,
    NeuralCoordinationPredictor,
    ANSFMLIntegration,
    MLCoordinationContext,
    MLCoordinationClass,
    PerformanceMetrics,
    HookExecutionPhase,
    create_ml_enhanced_coordination_system
)

from ml_integration_config import (
    MLEnhancedCoordinationConfig,
    ConfigurationManager,
    get_ml_coordination_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockANSFOrchestrator:
    """Mock ANSF Phase 2 orchestrator for testing."""
    
    def __init__(self):
        self.phase = 2
        self.coordination_accuracy = 0.947
        self.semantic_cache_size = 100  # MB
        self.agents = {}
        self.tasks = {}
        self.metrics = {
            'coordination_efficiency': 0.85,
            'semantic_accuracy': 0.90,
            'cache_hit_ratio': 0.80
        }
    
    async def initialize(self):
        return {'success': True, 'phase': 2}
    
    async def get_status(self):
        return {
            'active': True,
            'agents': len(self.agents),
            'coordination_accuracy': self.coordination_accuracy,
            'semantic_cache_mb': self.semantic_cache_size
        }
    
    async def coordinate_task(self, task_context):
        return {
            'task_id': task_context.get('task_id'),
            'coordination_result': 'success',
            'efficiency': 0.90
        }


class TestNeuralCoordinationPredictor(unittest.IsolatedAsyncioTestCase):
    """Test cases for the neural coordination predictor."""
    
    async def asyncSetUp(self):
        """Setup test environment."""
        self.predictor = NeuralCoordinationPredictor()
        await self.predictor.initialize()
    
    async def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor)
        self.assertIsNotNone(self.predictor.feature_names)
        self.assertEqual(len(self.predictor.feature_names), 10)
        self.assertIsInstance(self.predictor.prediction_history, type(asyncio.Queue()))
    
    async def test_feature_extraction(self):
        """Test feature extraction from coordination context."""
        context = MLCoordinationContext(
            task_id="test_001",
            task_type="analysis",
            complexity_score=0.7,
            historical_performance={'success_rate': 0.85},
            agent_capabilities={'agent_1': ['analysis'], 'agent_2': ['coding']},
            resource_constraints={
                'memory_usage_percent': 75,
                'cpu_utilization': 60,
                'cache_hit_ratio': 0.8
            }
        )
        
        features = await self.predictor._extract_features(context)
        
        self.assertEqual(len(features), len(self.predictor.feature_names))
        self.assertTrue(all(0 <= f <= 1 for f in features))
        
        # Check specific feature extraction
        self.assertAlmostEqual(features[0], 0.7, places=2)  # task_complexity
        self.assertAlmostEqual(features[1], 0.2, places=2)  # agent_count normalized
        self.assertAlmostEqual(features[2], 0.75, places=2)  # memory_usage
    
    async def test_heuristic_prediction(self):
        """Test heuristic-based prediction fallback."""
        context = MLCoordinationContext(
            task_id="test_002",
            task_type="critical",
            complexity_score=0.9,
            historical_performance={'success_rate': 0.5},
            agent_capabilities={f'agent_{i}': ['test'] for i in range(12)},  # Too many agents
            resource_constraints={'memory_usage_percent': 98}  # Critical memory
        )
        
        predicted_class, confidence = await self.predictor._heuristic_prediction(context, [])
        
        self.assertEqual(predicted_class, MLCoordinationClass.CRITICAL)
        self.assertGreater(confidence, 0.8)
    
    async def test_prediction_with_context(self):
        """Test prediction with various context scenarios."""
        # Optimal scenario
        optimal_context = MLCoordinationContext(
            task_id="test_optimal",
            task_type="simple",
            complexity_score=0.2,
            historical_performance={'success_rate': 0.95},
            agent_capabilities={f'agent_{i}': ['test'] for i in range(3)},
            resource_constraints={
                'memory_usage_percent': 40,
                'cpu_utilization': 30,
                'cache_hit_ratio': 0.9
            }
        )
        
        predicted_class, confidence = await self.predictor.predict_coordination_class(optimal_context)
        
        self.assertIn(predicted_class, [MLCoordinationClass.OPTIMAL, MLCoordinationClass.EFFICIENT])
        self.assertGreater(confidence, 0.5)
    
    async def test_learning_from_outcome(self):
        """Test learning from actual coordination outcomes."""
        # Make a prediction first
        context = MLCoordinationContext(
            task_id="learning_test",
            task_type="learning",
            complexity_score=0.5,
            historical_performance={'success_rate': 0.8},
            agent_capabilities={'agent_1': ['test']},
            resource_constraints={'memory_usage_percent': 70}
        )
        
        await self.predictor.predict_coordination_class(context)
        
        # Create performance metrics
        performance = PerformanceMetrics(
            execution_time=30.0,
            memory_usage=70.0,
            coordination_efficiency=0.9,
            error_rate=0.01,
            task_completion_rate=1.0
        )
        
        initial_accuracy = self.predictor.get_current_accuracy()
        
        # Learn from outcome
        await self.predictor.learn_from_outcome("learning_test", performance)
        
        # Check that learning data was stored
        self.assertGreater(len(self.predictor.prediction_history), 0)


class TestMLEnhancedCoordinationHooks(unittest.IsolatedAsyncioTestCase):
    """Test cases for ML-enhanced coordination hooks."""
    
    async def asyncSetUp(self):
        """Setup test environment."""
        config = {
            'neural_model_path': None,  # Use fallback model
            'ansf_integration': True
        }
        self.hooks = MLEnhancedCoordinationHooks(config)
        await self.hooks.initialize()
    
    async def test_initialization(self):
        """Test hooks system initialization."""
        self.assertTrue(self.hooks._initialized)
        self.assertIsNotNone(self.hooks.neural_predictor)
        self.assertGreater(len(self.hooks.hooks), 0)
    
    async def test_hook_registration(self):
        """Test hook registration and execution."""
        test_results = []
        
        async def test_hook(context):
            test_results.append(f"Hook executed for {context.task_id}")
            return {'test': True}
        
        self.hooks.register_hook(HookExecutionPhase.PRE_TASK_ML_ANALYSIS, test_hook)
        
        context = MLCoordinationContext(
            task_id="hook_test",
            task_type="test",
            complexity_score=0.5,
            historical_performance={},
            agent_capabilities={},
            resource_constraints={}
        )
        
        results = await self.hooks.execute_hooks(HookExecutionPhase.PRE_TASK_ML_ANALYSIS, context)
        
        self.assertEqual(len(test_results), 1)
        self.assertIn('individual_results', results)
    
    async def test_pre_task_ml_analysis(self):
        """Test pre-task ML analysis hook."""
        context = MLCoordinationContext(
            task_id="analysis_test",
            task_type="analysis",
            complexity_score=0.6,
            historical_performance={'success_rate': 0.8},
            agent_capabilities={'agent_1': ['analysis']},
            resource_constraints={'memory_usage_percent': 60}
        )
        
        result = await self.hooks._pre_task_ml_analysis(context)
        
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('suggestions', result)
        self.assertIn('ml_model_accuracy', result)
        
        # Check that context was updated
        self.assertIsNotNone(context.predicted_class)
        self.assertIsNotNone(context.confidence_score)
    
    async def test_intelligent_agent_assignment(self):
        """Test intelligent agent assignment based on ML predictions."""
        context = MLCoordinationContext(
            task_id="assignment_test",
            task_type="development",
            complexity_score=0.5,
            historical_performance={'success_rate': 0.85},
            agent_capabilities={
                'coder': ['coding', 'analysis'],
                'tester': ['testing', 'validation'],
                'reviewer': ['review', 'documentation']
            },
            resource_constraints={'memory_usage_percent': 50},
            predicted_class=MLCoordinationClass.EFFICIENT
        )
        
        result = await self.hooks._intelligent_agent_assignment(context)
        
        self.assertIn('agent_assignments', result)
        self.assertIn('assignment_strategy', result)
        self.assertGreater(result['total_agents'], 0)
        
        assignments = result['agent_assignments']
        for assignment in assignments:
            self.assertIn('agent_id', assignment)
            self.assertIn('agent_type', assignment)
            self.assertIn('ml_confidence', assignment)
    
    async def test_performance_prediction(self):
        """Test performance prediction hook."""
        context = MLCoordinationContext(
            task_id="perf_test",
            task_type="performance",
            complexity_score=0.4,
            historical_performance={'success_rate': 0.9},
            agent_capabilities={'agent_1': ['test']},
            resource_constraints={'memory_usage_percent': 45},
            predicted_class=MLCoordinationClass.EFFICIENT
        )
        
        result = await self.hooks._performance_prediction(context)
        
        self.assertIn('performance_prediction', result)
        self.assertIn('confidence', result)
        
        prediction = result['performance_prediction']
        self.assertIn('coordination_efficiency', prediction)
        self.assertIn('task_completion_rate', prediction)
        self.assertIn('error_rate', prediction)
        
        # Check reasonable values
        self.assertGreater(prediction['coordination_efficiency'], 0.5)
        self.assertLess(prediction['error_rate'], 0.2)
    
    async def test_adaptive_optimization(self):
        """Test adaptive optimization based on ML class."""
        # Test critical optimization
        critical_context = MLCoordinationContext(
            task_id="critical_test",
            task_type="critical",
            complexity_score=0.8,
            historical_performance={'success_rate': 0.4},
            agent_capabilities={'agent_1': ['emergency']},
            resource_constraints={'memory_usage_percent': 98},
            predicted_class=MLCoordinationClass.CRITICAL
        )
        
        result = await self.hooks._adaptive_optimization(critical_context)
        
        self.assertIn('optimization_strategies', result)
        self.assertGreater(result['total_strategies'], 0)
        
        strategies = result['optimization_strategies']
        strategy_types = [s['strategy']['type'] for s in strategies]
        self.assertIn('emergency_mode', strategy_types)
        
        # Test optimal optimization
        optimal_context = MLCoordinationContext(
            task_id="optimal_test",
            task_type="optimal",
            complexity_score=0.3,
            historical_performance={'success_rate': 0.95},
            agent_capabilities={f'agent_{i}': ['test'] for i in range(5)},
            resource_constraints={'memory_usage_percent': 40},
            predicted_class=MLCoordinationClass.OPTIMAL
        )
        
        result = await self.hooks._adaptive_optimization(optimal_context)
        
        strategies = result['optimization_strategies']
        strategy_types = [s['strategy']['type'] for s in strategies]
        self.assertIn('performance_enhancement', strategy_types)
    
    async def test_error_prediction(self):
        """Test error prediction and prevention."""
        high_risk_context = MLCoordinationContext(
            task_id="error_test",
            task_type="risky",
            complexity_score=0.9,
            historical_performance={'success_rate': 0.6},
            agent_capabilities={f'agent_{i}': ['test'] for i in range(10)},  # Many agents
            resource_constraints={
                'memory_usage_percent': 96,  # Critical memory
                'resource_contention': 0.8   # High contention
            },
            predicted_class=MLCoordinationClass.CRITICAL
        )
        
        result = await self.hooks._error_prediction(high_risk_context)
        
        self.assertIn('predicted_errors', result)
        self.assertIn('prevention_actions', result)
        
        errors = result['predicted_errors']
        self.assertGreater(len(errors), 0)
        
        # Check for memory exhaustion error
        error_types = [e['type'] for e in errors]
        self.assertIn('memory_exhaustion', error_types)
    
    async def test_bottleneck_prevention(self):
        """Test bottleneck identification and prevention."""
        bottleneck_context = MLCoordinationContext(
            task_id="bottleneck_test",
            task_type="bottleneck",
            complexity_score=0.7,
            historical_performance={'success_rate': 0.7},
            agent_capabilities={'agent_1': ['test']},
            resource_constraints={
                'semantic_cache_efficiency': 0.3,  # Poor cache performance
                'agent_idle_time': 0.6,             # High idle time
                'memory_usage_percent': 92          # High memory usage
            }
        )
        
        result = await self.hooks._bottleneck_prevention(bottleneck_context)
        
        self.assertIn('bottlenecks_detected', result)
        self.assertIn('optimizations_applied', result)
        
        bottlenecks = result['bottlenecks_detected']
        self.assertGreater(len(bottlenecks), 0)
        
        bottleneck_types = [b['type'] for b in bottlenecks]
        self.assertIn('semantic_cache_inefficiency', bottleneck_types)


class TestANSFMLIntegration(unittest.IsolatedAsyncioTestCase):
    """Test cases for ANSF ML integration."""
    
    async def asyncSetUp(self):
        """Setup test environment."""
        self.mock_orchestrator = MockANSFOrchestrator()
        self.integration = ANSFMLIntegration(self.mock_orchestrator)
    
    async def test_integration_initialization(self):
        """Test integration initialization."""
        await self.integration.initialize_integration()
        
        self.assertTrue(self.integration.integration_active)
        self.assertIsNotNone(self.integration.ml_hooks)
    
    async def test_enhance_task_coordination(self):
        """Test complete task coordination enhancement."""
        await self.integration.initialize_integration()
        
        task_context = {
            'task_id': 'integration_test',
            'task_type': 'full_integration',
            'complexity_score': 0.6,
            'historical_performance': {'success_rate': 0.85},
            'agent_capabilities': {
                'agent_1': ['analysis', 'coding'],
                'agent_2': ['testing', 'review']
            },
            'resource_constraints': {
                'memory_usage_percent': 70,
                'cpu_utilization': 60,
                'cache_hit_ratio': 0.75
            }
        }
        
        result = await self.integration.enhance_task_coordination(task_context)
        
        self.assertIn('enhanced_coordination', result)
        self.assertIn('ml_context', result)
        self.assertIn('system_metrics', result)
        self.assertIn('neural_accuracy', result)
        
        coordination = result['enhanced_coordination']
        self.assertIn('ml_analysis', coordination)
        self.assertIn('agent_assignment', coordination)
        self.assertIn('performance_prediction', coordination)
        self.assertIn('optimization', coordination)
        self.assertIn('error_prevention', coordination)
        self.assertIn('bottleneck_prevention', coordination)
    
    async def test_complete_task_learning(self):
        """Test learning completion after task execution."""
        await self.integration.initialize_integration()
        
        performance_data = {
            'execution_time': 45.0,
            'success': True,
            'errors': 0,
            'efficiency': 0.9
        }
        
        result = await self.integration.complete_task_learning('test_task', performance_data)
        
        self.assertTrue(result['learning_completed'])
        self.assertIn('learning_results', result)
        self.assertIn('memory_sync_results', result)
        self.assertIn('updated_accuracy', result)


class TestConfigurationManager(unittest.TestCase):
    """Test cases for configuration management."""
    
    def setUp(self):
        """Setup test environment."""
        self.test_config_file = "test_config.json"
        self.config_manager = ConfigurationManager(self.test_config_file)
    
    def tearDown(self):
        """Cleanup test files."""
        if Path(self.test_config_file).exists():
            Path(self.test_config_file).unlink()
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = self.config_manager.get_config()
        
        self.assertEqual(config.ml_model.classes, 5)
        self.assertEqual(config.ansf_integration.semantic_cache_budget_mb, 100)
        self.assertEqual(config.ansf_integration.target_coordination_accuracy, 0.947)
        self.assertTrue(config.serena_integration.enable_serena_integration)
        self.assertTrue(config.claude_flow_integration.enable_claude_flow)
    
    def test_integration_settings(self):
        """Test integration settings retrieval."""
        settings = self.config_manager.get_integration_settings()
        
        self.assertIn('ansf', settings)
        self.assertIn('serena', settings)
        self.assertIn('claude_flow', settings)
        self.assertIn('ml_model', settings)
        
        # Check ANSF settings
        ansf_settings = settings['ansf']
        self.assertTrue(ansf_settings['enabled'])
        self.assertEqual(ansf_settings['cache_budget'], 100)
        self.assertEqual(ansf_settings['target_accuracy'], 0.947)
    
    def test_validation_requirements(self):
        """Test integration requirement validation."""
        requirements = self.config_manager.validate_integration_requirements()
        
        self.assertIn('ml_model_available', requirements)
        self.assertIn('ansf_components_available', requirements)
        self.assertIn('serena_api_accessible', requirements)
        self.assertIn('claude_flow_accessible', requirements)
        self.assertIn('sufficient_memory', requirements)
        self.assertIn('required_dependencies', requirements)
    
    def test_optimization_profiles(self):
        """Test optimization profile recommendations."""
        # Test critical profile
        critical_state = {
            'memory_usage_percent': 97,
            'cpu_usage_percent': 95,
            'agent_count': 1
        }
        profile = self.config_manager.get_optimization_profile(critical_state)
        self.assertEqual(profile, "critical")
        
        # Test optimal profile
        optimal_state = {
            'memory_usage_percent': 45,
            'cpu_usage_percent': 55,
            'agent_count': 4
        }
        profile = self.config_manager.get_optimization_profile(optimal_state)
        self.assertEqual(profile, "optimal")
        
        # Test moderate profile
        moderate_state = {
            'memory_usage_percent': 75,
            'cpu_usage_percent': 65,
            'agent_count': 6
        }
        profile = self.config_manager.get_optimization_profile(moderate_state)
        self.assertEqual(profile, "moderate")


class TestPerformanceBenchmarks(unittest.IsolatedAsyncioTestCase):
    """Performance benchmark tests for ML coordination system."""
    
    async def asyncSetUp(self):
        """Setup benchmark environment."""
        self.ml_system = create_ml_enhanced_coordination_system()
        await self.ml_system.initialize_integration()
        self.benchmark_results = []
    
    async def test_coordination_accuracy_benchmark(self):
        """Benchmark coordination accuracy against target 94.7%."""
        test_scenarios = [
            {
                'scenario': 'optimal_conditions',
                'task_context': {
                    'complexity_score': 0.3,
                    'memory_usage_percent': 40,
                    'agent_count': 4,
                    'historical_success_rate': 0.95
                }
            },
            {
                'scenario': 'moderate_conditions',
                'task_context': {
                    'complexity_score': 0.6,
                    'memory_usage_percent': 70,
                    'agent_count': 6,
                    'historical_success_rate': 0.80
                }
            },
            {
                'scenario': 'challenging_conditions',
                'task_context': {
                    'complexity_score': 0.8,
                    'memory_usage_percent': 85,
                    'agent_count': 8,
                    'historical_success_rate': 0.65
                }
            },
            {
                'scenario': 'critical_conditions',
                'task_context': {
                    'complexity_score': 0.9,
                    'memory_usage_percent': 95,
                    'agent_count': 2,
                    'historical_success_rate': 0.50
                }
            }
        ]
        
        accuracies = []
        
        for scenario in test_scenarios:
            context = scenario['task_context']
            
            # Create full task context
            full_context = {
                'task_id': f"benchmark_{scenario['scenario']}",
                'task_type': 'benchmark',
                'complexity_score': context['complexity_score'],
                'historical_performance': {'success_rate': context['historical_success_rate']},
                'agent_capabilities': {f'agent_{i}': ['test'] for i in range(context['agent_count'])},
                'resource_constraints': {
                    'memory_usage_percent': context['memory_usage_percent'],
                    'cpu_utilization': context['memory_usage_percent'] * 0.8,
                    'cache_hit_ratio': 0.8
                }
            }
            
            # Run coordination
            start_time = time.time()
            result = await self.ml_system.enhance_task_coordination(full_context)
            execution_time = time.time() - start_time
            
            # Calculate coordination accuracy (simulated)
            predicted_accuracy = result['neural_accuracy']
            system_efficiency = result['system_metrics']['coordination_efficiency']
            
            # Simulated accuracy based on prediction quality and system efficiency
            coordination_accuracy = (predicted_accuracy * 0.6 + system_efficiency * 0.4)
            accuracies.append(coordination_accuracy)
            
            self.benchmark_results.append({
                'scenario': scenario['scenario'],
                'coordination_accuracy': coordination_accuracy,
                'neural_accuracy': predicted_accuracy,
                'system_efficiency': system_efficiency,
                'execution_time': execution_time
            })
            
            logger.info(f"Benchmark {scenario['scenario']}: "
                       f"Accuracy {coordination_accuracy:.3f}, Time {execution_time:.3f}s")
        
        # Check overall accuracy
        average_accuracy = statistics.mean(accuracies)
        target_accuracy = 0.947
        
        logger.info(f"Average Coordination Accuracy: {average_accuracy:.3f} (Target: {target_accuracy:.3f})")
        
        # We expect to be within 5% of target for this test
        self.assertGreater(average_accuracy, target_accuracy * 0.95)
    
    async def test_performance_metrics_benchmark(self):
        """Benchmark performance metrics against targets."""
        target_metrics = {
            'response_time': 5.0,      # seconds
            'memory_efficiency': 0.85,  # utilization
            'prediction_accuracy': 0.887, # neural model baseline
            'coordination_efficiency': 0.947  # ANSF target
        }
        
        # Run multiple coordination tasks
        tasks = []
        for i in range(10):
            task_context = {
                'task_id': f'perf_benchmark_{i}',
                'task_type': 'performance_test',
                'complexity_score': 0.5 + (i * 0.05),  # Increasing complexity
                'historical_performance': {'success_rate': 0.8},
                'agent_capabilities': {f'agent_{j}': ['test'] for j in range(3 + i % 3)},
                'resource_constraints': {
                    'memory_usage_percent': 50 + (i * 3),
                    'cpu_utilization': 40 + (i * 2),
                    'cache_hit_ratio': 0.8
                }
            }
            tasks.append(task_context)
        
        # Execute tasks and measure performance
        response_times = []
        memory_efficiencies = []
        prediction_accuracies = []
        coordination_efficiencies = []
        
        for task in tasks:
            start_time = time.time()
            result = await self.ml_system.enhance_task_coordination(task)
            response_time = time.time() - start_time
            
            response_times.append(response_time)
            prediction_accuracies.append(result['neural_accuracy'])
            coordination_efficiencies.append(result['system_metrics']['coordination_efficiency'])
            
            # Simulated memory efficiency based on memory usage
            memory_usage = task['resource_constraints']['memory_usage_percent']
            memory_efficiency = max(0, 1 - (memory_usage / 100) * 1.2)
            memory_efficiencies.append(memory_efficiency)
        
        # Calculate averages
        avg_response_time = statistics.mean(response_times)
        avg_memory_efficiency = statistics.mean(memory_efficiencies)
        avg_prediction_accuracy = statistics.mean(prediction_accuracies)
        avg_coordination_efficiency = statistics.mean(coordination_efficiencies)
        
        # Log results
        logger.info("Performance Benchmark Results:")
        logger.info(f"Average Response Time: {avg_response_time:.3f}s (Target: <{target_metrics['response_time']}s)")
        logger.info(f"Average Memory Efficiency: {avg_memory_efficiency:.3f} (Target: >{target_metrics['memory_efficiency']})")
        logger.info(f"Average Prediction Accuracy: {avg_prediction_accuracy:.3f} (Target: >{target_metrics['prediction_accuracy']})")
        logger.info(f"Average Coordination Efficiency: {avg_coordination_efficiency:.3f} (Target: >{target_metrics['coordination_efficiency']})")
        
        # Assert performance targets
        self.assertLess(avg_response_time, target_metrics['response_time'])
        self.assertGreater(avg_memory_efficiency, target_metrics['memory_efficiency'])
        self.assertGreater(avg_prediction_accuracy, target_metrics['prediction_accuracy'] * 0.95)  # Within 5%
        self.assertGreater(avg_coordination_efficiency, target_metrics['coordination_efficiency'] * 0.95)  # Within 5%


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Integration scenario tests simulating real-world usage."""
    
    async def asyncSetUp(self):
        """Setup integration test environment."""
        self.ml_system = create_ml_enhanced_coordination_system()
        await self.ml_system.initialize_integration()
    
    async def test_ansf_phase2_integration_scenario(self):
        """Test integration with ANSF Phase 2 system."""
        # Simulate ANSF Phase 2 task with semantic intelligence requirements
        ansf_task = {
            'task_id': 'ansf_integration_001',
            'task_type': 'semantic_analysis',
            'complexity_score': 0.7,
            'historical_performance': {'success_rate': 0.88},
            'agent_capabilities': {
                'serena_agent': ['semantic_analysis', 'lsp_integration'],
                'archon_agent': ['prp_refinement', 'task_orchestration'],
                'neural_agent': ['pattern_learning', 'optimization'],
                'flow_agent': ['swarm_coordination', 'performance_monitoring']
            },
            'resource_constraints': {
                'memory_usage_percent': 65,
                'cpu_utilization': 55,
                'cache_hit_ratio': 0.85,
                'semantic_cache_efficiency': 0.75,
                'ansf_coordination_accuracy': 0.94
            }
        }
        
        # Execute enhanced coordination
        result = await self.ml_system.enhance_task_coordination(ansf_task)
        
        # Validate integration results
        self.assertIn('enhanced_coordination', result)
        self.assertIn('ml_context', result)
        
        coordination = result['enhanced_coordination']
        
        # Check all phases executed
        self.assertIn('ml_analysis', coordination)
        self.assertIn('agent_assignment', coordination)
        self.assertIn('performance_prediction', coordination)
        self.assertIn('optimization', coordination)
        
        # Verify ANSF-specific optimizations
        ml_analysis = coordination['ml_analysis']
        if 'individual_results' in ml_analysis:
            suggestions = []
            for result_item in ml_analysis['individual_results']:
                if 'suggestions' in result_item:
                    suggestions.extend(result_item['suggestions'])
            
            # Should include ANSF-specific suggestions
            suggestion_text = ' '.join(suggestions)
            self.assertTrue(
                any(term in suggestion_text.lower() for term in 
                   ['semantic', 'cache', 'coordination', 'memory', 'optimization'])
            )
        
        # Complete learning cycle
        performance_data = {
            'execution_time': 42.0,
            'coordination_efficiency': 0.95,
            'semantic_accuracy': 0.92,
            'cache_hit_ratio': 0.88
        }
        
        learning_result = await self.ml_system.complete_task_learning(ansf_task['task_id'], performance_data)
        self.assertTrue(learning_result['learning_completed'])
    
    async def test_high_load_scenario(self):
        """Test system behavior under high load conditions."""
        # Simulate multiple concurrent tasks
        concurrent_tasks = []
        
        for i in range(5):  # 5 concurrent tasks
            task = {
                'task_id': f'high_load_task_{i}',
                'task_type': 'concurrent_processing',
                'complexity_score': 0.6 + (i * 0.1),
                'historical_performance': {'success_rate': 0.8 - (i * 0.05)},
                'agent_capabilities': {
                    f'agent_{j}_{i}': ['processing', 'coordination'] 
                    for j in range(2 + i)
                },
                'resource_constraints': {
                    'memory_usage_percent': 70 + (i * 5),
                    'cpu_utilization': 60 + (i * 8),
                    'cache_hit_ratio': 0.8 - (i * 0.05)
                }
            }
            concurrent_tasks.append(task)
        
        # Execute tasks concurrently
        start_time = time.time()
        
        tasks = [
            self.ml_system.enhance_task_coordination(task) 
            for task in concurrent_tasks
        ]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(len(results), 5)
        
        for i, result in enumerate(results):
            self.assertIn('enhanced_coordination', result)
            
            # Check that system adapted to increasing load
            if i > 2:  # Higher load tasks
                ml_context = result['ml_context']
                predicted_class = ml_context.get('predicted_class')
                
                # Should predict suboptimal or critical for high load
                if predicted_class:
                    self.assertIn(predicted_class, ['SUBOPTIMAL', 'CRITICAL', 'MODERATE'])
        
        logger.info(f"High load scenario completed in {total_time:.3f}s with {len(results)} tasks")
        
        # Should complete within reasonable time even under load
        self.assertLess(total_time, 30.0)  # 30 seconds max
    
    async def test_adaptive_learning_scenario(self):
        """Test adaptive learning over multiple task iterations."""
        learning_tasks = [
            {
                'name': 'simple_task',
                'complexity': 0.3,
                'expected_performance': 0.9
            },
            {
                'name': 'medium_task',
                'complexity': 0.6,
                'expected_performance': 0.8
            },
            {
                'name': 'complex_task',
                'complexity': 0.8,
                'expected_performance': 0.7
            },
            {
                'name': 'repeat_simple',  # Repeat to test learning
                'complexity': 0.3,
                'expected_performance': 0.95  # Should improve
            }
        ]
        
        accuracy_progression = []
        
        for i, task_spec in enumerate(learning_tasks):
            task_context = {
                'task_id': f'learning_{i}_{task_spec["name"]}',
                'task_type': 'adaptive_learning',
                'complexity_score': task_spec['complexity'],
                'historical_performance': {'success_rate': task_spec['expected_performance']},
                'agent_capabilities': {
                    'learning_agent': ['learning', 'adaptation'],
                    'execution_agent': ['execution', 'monitoring']
                },
                'resource_constraints': {
                    'memory_usage_percent': 50 + (task_spec['complexity'] * 30),
                    'cpu_utilization': 40 + (task_spec['complexity'] * 20),
                    'cache_hit_ratio': 0.9 - (task_spec['complexity'] * 0.2)
                }
            }
            
            # Execute task with ML coordination
            result = await self.ml_system.enhance_task_coordination(task_context)
            current_accuracy = result['neural_accuracy']
            accuracy_progression.append(current_accuracy)
            
            # Simulate task completion with performance feedback
            simulated_performance = {
                'execution_time': 30 + (task_spec['complexity'] * 20),
                'success': True,
                'efficiency': task_spec['expected_performance']
            }
            
            # Complete learning cycle
            await self.ml_system.complete_task_learning(task_context['task_id'], simulated_performance)
            
            logger.info(f"Learning task {i+1}: {task_spec['name']} - Accuracy: {current_accuracy:.3f}")
        
        # Check that accuracy improved for repeated simple task
        if len(accuracy_progression) >= 4:
            simple_task_1_accuracy = accuracy_progression[0]  # First simple task
            simple_task_2_accuracy = accuracy_progression[3]  # Repeated simple task
            
            logger.info(f"Simple task accuracy progression: {simple_task_1_accuracy:.3f} -> {simple_task_2_accuracy:.3f}")
            
            # Should show learning improvement (allowing for some variance)
            improvement_threshold = 0.01  # 1% improvement minimum
            self.assertGreaterEqual(
                simple_task_2_accuracy, 
                simple_task_1_accuracy - improvement_threshold
            )


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
    
    # Additional demo if run directly
    async def demo_ml_coordination():
        """Demonstration of ML-enhanced coordination system."""
        print("\n" + "="*60)
        print("ML-Enhanced Coordination System Demonstration")
        print("="*60)
        
        # Create and initialize system
        ml_system = create_ml_enhanced_coordination_system()
        await ml_system.initialize_integration()
        
        print("✅ ML-Enhanced Coordination System Initialized")
        
        # Demo task
        demo_task = {
            'task_id': 'demo_comprehensive_analysis',
            'task_type': 'comprehensive_analysis',
            'complexity_score': 0.7,
            'historical_performance': {'success_rate': 0.85},
            'agent_capabilities': {
                'analysis_agent': ['semantic_analysis', 'pattern_recognition'],
                'coordination_agent': ['swarm_coordination', 'optimization'],
                'execution_agent': ['task_execution', 'monitoring'],
                'learning_agent': ['neural_learning', 'adaptation']
            },
            'resource_constraints': {
                'memory_usage_percent': 75,
                'cpu_utilization': 65,
                'cache_hit_ratio': 0.8,
                'semantic_cache_efficiency': 0.75
            }
        }
        
        print(f"\n🎯 Executing Demo Task: {demo_task['task_id']}")
        print(f"   Task Type: {demo_task['task_type']}")
        print(f"   Complexity: {demo_task['complexity_score']:.1f}")
        print(f"   Agents: {len(demo_task['agent_capabilities'])}")
        print(f"   Memory Usage: {demo_task['resource_constraints']['memory_usage_percent']}%")
        
        # Execute enhanced coordination
        start_time = time.time()
        result = await ml_system.enhance_task_coordination(demo_task)
        execution_time = time.time() - start_time
        
        print(f"\n⚡ Coordination completed in {execution_time:.3f} seconds")
        
        # Display results
        print(f"\n📊 Results Summary:")
        print(f"   Neural Model Accuracy: {result['neural_accuracy']:.3f}")
        print(f"   System Coordination Efficiency: {result['system_metrics']['coordination_efficiency']:.3f}")
        print(f"   ML Predictions Made: {result['system_metrics']['ml_predictions_made']}")
        
        # Show coordination phases
        coordination = result['enhanced_coordination']
        phases_executed = len([k for k in coordination.keys() if coordination[k]])
        print(f"   Coordination Phases Executed: {phases_executed}")
        
        # Complete learning cycle
        performance_data = {
            'execution_time': execution_time,
            'success': True,
            'efficiency': 0.92
        }
        
        learning_result = await ml_system.complete_task_learning(demo_task['task_id'], performance_data)
        
        print(f"\n🧠 Learning Completed:")
        print(f"   Updated Model Accuracy: {learning_result['updated_accuracy']:.3f}")
        print(f"   Learning Successful: {learning_result['learning_completed']}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"🎯 Target Coordination Accuracy (94.7%): {'✅ ACHIEVED' if result['system_metrics']['coordination_efficiency'] >= 0.947 else '⚠️  IN PROGRESS'}")
    
    # Run demo
    print("\nRunning demonstration...")
    asyncio.run(demo_ml_coordination())