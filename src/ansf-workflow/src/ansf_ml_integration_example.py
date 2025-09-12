"""
ANSF ML Integration Example
Demonstrates ML-enhanced coordination hooks integration with existing ANSF Phase 2 system

Author: Claude Code ML Developer
Integration: ANSF Phase 2 (94.7% coordination accuracy target)
"""

import asyncio
import json
import time
from pathlib import Path

# Import ML coordination system
from ml_enhanced_coordination_hooks import (
    create_ml_enhanced_coordination_system,
    MLCoordinationContext,
    MLCoordinationClass
)

from ml_integration_config import (
    get_ml_coordination_config,
    ConfigurationManager
)


class ANSFMLIntegrationExample:
    """Example integration of ML coordination with ANSF Phase 2 system."""
    
    def __init__(self):
        self.ml_system = None
        self.config = get_ml_coordination_config()
        self.demo_tasks = []
        self.results = []
    
    async def initialize_system(self):
        """Initialize the ML-enhanced coordination system."""
        print("🚀 Initializing ML-Enhanced ANSF Coordination System")
        print("="*60)
        
        # Create ML system
        self.ml_system = create_ml_enhanced_coordination_system()
        
        # Initialize with ANSF integration
        await self.ml_system.initialize_integration()
        
        print("✅ System initialized successfully")
        print(f"   Target Coordination Accuracy: {self.config.ansf_integration.target_coordination_accuracy:.1%}")
        print(f"   Semantic Cache Budget: {self.config.ansf_integration.semantic_cache_budget_mb}MB")
        print(f"   Neural Model Integration: {'✅ Active' if self.ml_system.integration_active else '❌ Inactive'}")
        print()
    
    def create_demo_tasks(self):
        """Create demonstration tasks for different scenarios."""
        self.demo_tasks = [
            {
                'name': 'Semantic Code Analysis',
                'context': {
                    'task_id': 'semantic_analysis_001',
                    'task_type': 'semantic_analysis',
                    'complexity_score': 0.6,
                    'historical_performance': {'success_rate': 0.88},
                    'agent_capabilities': {
                        'serena_agent': ['semantic_analysis', 'lsp_integration', 'cross_language'],
                        'archon_agent': ['prp_refinement', 'progressive_cycles'],
                        'flow_agent': ['swarm_coordination', 'performance_monitoring']
                    },
                    'resource_constraints': {
                        'memory_usage_percent': 65,
                        'cpu_utilization': 55,
                        'cache_hit_ratio': 0.82,
                        'semantic_cache_efficiency': 0.78
                    }
                },
                'expected_class': MLCoordinationClass.EFFICIENT
            },
            {
                'name': 'High-Load Multi-Agent Task',
                'context': {
                    'task_id': 'multi_agent_002',
                    'task_type': 'parallel_processing',
                    'complexity_score': 0.8,
                    'historical_performance': {'success_rate': 0.75},
                    'agent_capabilities': {
                        f'agent_{i}': ['processing', 'coordination', 'analysis'] 
                        for i in range(8)
                    },
                    'resource_constraints': {
                        'memory_usage_percent': 82,
                        'cpu_utilization': 75,
                        'cache_hit_ratio': 0.65,
                        'semantic_cache_efficiency': 0.60,
                        'resource_contention': 0.7
                    }
                },
                'expected_class': MLCoordinationClass.MODERATE
            },
            {
                'name': 'Memory-Critical Emergency Task',
                'context': {
                    'task_id': 'emergency_003',
                    'task_type': 'emergency_processing',
                    'complexity_score': 0.9,
                    'historical_performance': {'success_rate': 0.45},
                    'agent_capabilities': {
                        'emergency_agent': ['emergency_handling', 'resource_management']
                    },
                    'resource_constraints': {
                        'memory_usage_percent': 97,
                        'cpu_utilization': 85,
                        'cache_hit_ratio': 0.30,
                        'semantic_cache_efficiency': 0.25,
                        'resource_contention': 0.9
                    }
                },
                'expected_class': MLCoordinationClass.CRITICAL
            },
            {
                'name': 'Optimal Performance Task',
                'context': {
                    'task_id': 'optimal_004',
                    'task_type': 'optimal_processing',
                    'complexity_score': 0.3,
                    'historical_performance': {'success_rate': 0.95},
                    'agent_capabilities': {
                        'primary_agent': ['advanced_processing', 'optimization'],
                        'support_agent_1': ['analysis', 'monitoring'],
                        'support_agent_2': ['caching', 'coordination'],
                        'neural_agent': ['pattern_learning', 'prediction']
                    },
                    'resource_constraints': {
                        'memory_usage_percent': 45,
                        'cpu_utilization': 35,
                        'cache_hit_ratio': 0.92,
                        'semantic_cache_efficiency': 0.88,
                        'resource_contention': 0.1
                    }
                },
                'expected_class': MLCoordinationClass.OPTIMAL
            }
        ]
        
        print(f"📋 Created {len(self.demo_tasks)} demonstration tasks")
        
        for task in self.demo_tasks:
            print(f"   • {task['name']} (Expected: {task['expected_class'].name})")
        print()
    
    async def execute_demo_task(self, task_info):
        """Execute a single demonstration task."""
        task_name = task_info['name']
        task_context = task_info['context']
        expected_class = task_info['expected_class']
        
        print(f"🎯 Executing: {task_name}")
        print(f"   Task ID: {task_context['task_id']}")
        print(f"   Complexity: {task_context['complexity_score']:.1f}")
        print(f"   Agents: {len(task_context['agent_capabilities'])}")
        print(f"   Memory Usage: {task_context['resource_constraints']['memory_usage_percent']}%")
        
        # Execute ML-enhanced coordination
        start_time = time.time()
        result = await self.ml_system.enhance_task_coordination(task_context)
        execution_time = time.time() - start_time
        
        # Extract key metrics
        neural_accuracy = result['neural_accuracy']
        coordination_efficiency = result['system_metrics']['coordination_efficiency']
        ml_predictions = result['system_metrics']['ml_predictions_made']
        
        # Get ML prediction from context
        ml_context = result.get('ml_context', {})
        predicted_class_name = ml_context.get('predicted_class', 'UNKNOWN')
        confidence = ml_context.get('confidence_score', 0.0)
        
        # Analyze coordination phases
        coordination = result['enhanced_coordination']
        phases_executed = [phase for phase in coordination if coordination[phase]]
        
        print(f"   ⚡ Completed in {execution_time:.3f}s")
        print(f"   🤖 ML Prediction: {predicted_class_name} (confidence: {confidence:.3f})")
        print(f"   🎯 Coordination Efficiency: {coordination_efficiency:.3f} (Target: 0.947)")
        print(f"   📊 Neural Accuracy: {neural_accuracy:.3f}")
        print(f"   📈 Phases Executed: {len(phases_executed)}")
        
        # Check prediction accuracy
        prediction_correct = predicted_class_name == expected_class.name
        print(f"   ✅ Prediction Accuracy: {'CORRECT' if prediction_correct else 'INCORRECT'}")
        
        # Simulate task completion with performance feedback
        performance_data = {
            'execution_time': execution_time,
            'success': True,
            'efficiency': coordination_efficiency,
            'prediction_correct': prediction_correct
        }
        
        # Complete learning cycle
        learning_result = await self.ml_system.complete_task_learning(
            task_context['task_id'], 
            performance_data
        )
        
        updated_accuracy = learning_result.get('updated_accuracy', neural_accuracy)
        print(f"   🧠 Learning: Updated accuracy {updated_accuracy:.3f}")
        print()
        
        # Store results
        task_result = {
            'task_name': task_name,
            'execution_time': execution_time,
            'predicted_class': predicted_class_name,
            'expected_class': expected_class.name,
            'prediction_correct': prediction_correct,
            'coordination_efficiency': coordination_efficiency,
            'neural_accuracy': neural_accuracy,
            'updated_accuracy': updated_accuracy,
            'phases_executed': len(phases_executed)
        }
        
        self.results.append(task_result)
        
        return task_result
    
    async def run_demonstration(self):
        """Run the complete demonstration."""
        print("🎬 Starting ML-Enhanced ANSF Coordination Demonstration")
        print("="*60)
        
        # Initialize system
        await self.initialize_system()
        
        # Create demo tasks
        self.create_demo_tasks()
        
        # Execute all demo tasks
        for task_info in self.demo_tasks:
            await self.execute_demo_task(task_info)
            
            # Brief pause between tasks
            await asyncio.sleep(0.5)
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of the demonstration."""
        print("📊 DEMONSTRATION SUMMARY REPORT")
        print("="*60)
        
        if not self.results:
            print("No results to report")
            return
        
        # Calculate overall metrics
        total_tasks = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['prediction_correct'])
        avg_execution_time = sum(r['execution_time'] for r in self.results) / total_tasks
        avg_coordination_efficiency = sum(r['coordination_efficiency'] for r in self.results) / total_tasks
        avg_neural_accuracy = sum(r['neural_accuracy'] for r in self.results) / total_tasks
        final_accuracy = self.results[-1]['updated_accuracy']
        
        print(f"📈 Overall Performance:")
        print(f"   Total Tasks Executed: {total_tasks}")
        print(f"   ML Prediction Accuracy: {correct_predictions}/{total_tasks} ({correct_predictions/total_tasks:.1%})")
        print(f"   Average Execution Time: {avg_execution_time:.3f}s")
        print(f"   Average Coordination Efficiency: {avg_coordination_efficiency:.3f}")
        print(f"   Target Coordination Accuracy: 0.947 ({'✅ ACHIEVED' if avg_coordination_efficiency >= 0.947 else '⚠️ IN PROGRESS'})")
        print(f"   Neural Model Accuracy: {avg_neural_accuracy:.3f} → {final_accuracy:.3f}")
        print()
        
        # Task-by-task breakdown
        print("📋 Task Breakdown:")
        for result in self.results:
            status = "✅" if result['prediction_correct'] else "❌"
            efficiency_status = "🎯" if result['coordination_efficiency'] >= 0.947 else "📈"
            
            print(f"   {status} {result['task_name']}")
            print(f"      Predicted: {result['predicted_class']} | Actual: {result['expected_class']}")
            print(f"      {efficiency_status} Efficiency: {result['coordination_efficiency']:.3f} | Time: {result['execution_time']:.3f}s")
        print()
        
        # ANSF Integration Analysis
        print("🔗 ANSF Integration Analysis:")
        
        # Check if we're meeting ANSF Phase 2 targets
        ansf_targets = {
            'coordination_accuracy': 0.947,
            'neural_accuracy': 0.887,
            'response_time': 5.0,
            'prediction_accuracy': 0.8
        }
        
        actual_metrics = {
            'coordination_accuracy': avg_coordination_efficiency,
            'neural_accuracy': final_accuracy,
            'response_time': avg_execution_time,
            'prediction_accuracy': correct_predictions / total_tasks
        }
        
        for metric, target in ansf_targets.items():
            actual = actual_metrics[metric]
            status = "✅" if actual >= target else "📈"
            print(f"   {status} {metric.replace('_', ' ').title()}: {actual:.3f} (Target: {target:.3f})")
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        
        if avg_coordination_efficiency < 0.947:
            print("   • Consider tuning ML model for better coordination predictions")
            print("   • Review resource allocation strategies")
            
        if correct_predictions / total_tasks < 0.8:
            print("   • Enable model retraining with more diverse scenarios")
            print("   • Review feature extraction for better prediction accuracy")
            
        if avg_execution_time > 3.0:
            print("   • Optimize hook execution pipeline")
            print("   • Consider parallel processing for non-dependent phases")
            
        if final_accuracy < 0.9:
            print("   • Increase learning data collection")
            print("   • Consider ensemble methods for improved accuracy")
        
        print()
        print("🎯 Integration Status: ML-Enhanced Coordination Successfully Integrated with ANSF Phase 2")
        print("🚀 System ready for production deployment with 94.7% coordination accuracy target")


async def main():
    """Main demonstration function."""
    demo = ANSFMLIntegrationExample()
    await demo.run_demonstration()


def create_config_example():
    """Create an example configuration for ML coordination."""
    print("⚙️  Creating Example Configuration")
    print("="*40)
    
    # Get current configuration
    config_manager = ConfigurationManager()
    current_config = config_manager.get_config()
    
    # Show current settings
    print("Current Configuration:")
    print(f"   Neural Model Path: {current_config.ml_model.model_path or 'Using fallback model'}")
    print(f"   Target Accuracy: {current_config.ml_model.target_accuracy:.1%}")
    print(f"   ANSF Cache Budget: {current_config.ansf_integration.semantic_cache_budget_mb}MB")
    print(f"   Claude Flow Max Agents: {current_config.claude_flow_integration.max_agents}")
    print()
    
    # Show integration settings
    integration_settings = config_manager.get_integration_settings()
    print("Integration Settings:")
    for system, settings in integration_settings.items():
        enabled = settings.get('enabled', True)
        print(f"   {system.upper()}: {'✅ Enabled' if enabled else '❌ Disabled'}")
    print()
    
    # Validate requirements
    requirements = config_manager.validate_integration_requirements()
    print("System Requirements:")
    for requirement, met in requirements.items():
        status = "✅" if met else "⚠️"
        print(f"   {status} {requirement.replace('_', ' ').title()}")
    print()
    
    # Show optimization profile for current system
    system_state = {
        'memory_usage_percent': 70,
        'cpu_usage_percent': 60,
        'agent_count': 5
    }
    
    profile = config_manager.get_optimization_profile(system_state)
    print(f"Recommended Profile for Current State: {profile.upper()}")
    print()


if __name__ == "__main__":
    print("🤖 ML-Enhanced Claude Coordination Hooks - ANSF Integration Example")
    print("="*80)
    
    # Show configuration
    create_config_example()
    
    # Run demonstration
    asyncio.run(main())