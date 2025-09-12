#!/usr/bin/env python3
"""
ANSF Parallel Analysis Runner
Executes parallel coordination benchmarks and generates comprehensive analysis
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ansf_parallel_benchmark import ANSFParallelBenchmark
except ImportError:
    print("Error: Could not import ANSFParallelBenchmark. Make sure ansf_parallel_benchmark.py is in the same directory.")
    sys.exit(1)

class ANSFAnalysisRunner:
    """Comprehensive analysis runner for ANSF parallel coordination benchmarks"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark = ANSFParallelBenchmark()
        
    def generate_visualizations(self, results: dict, timestamp: str) -> dict:
        """Generate performance visualizations"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
            
            detailed_results = results.get('detailed_results', [])
            if not detailed_results:
                return {}
                
            # Create DataFrame for analysis
            df = pd.DataFrame(detailed_results)
            
            # Performance comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Coordination Accuracy vs Parallel Efficiency
            ax1.scatter(df['parallel_efficiency'], df['coordination_accuracy'], 
                       s=df['parallel_workers']*20, alpha=0.7, c=df['throughput_factor'], 
                       cmap='viridis')
            ax1.axhline(y=97.3, color='r', linestyle='--', alpha=0.7, label='Target Accuracy (97.3%)')
            ax1.set_xlabel('Parallel Efficiency (%)')
            ax1.set_ylabel('Coordination Accuracy (%)')
            ax1.set_title('Coordination Accuracy vs Parallel Efficiency')
            ax1.legend()
            ax1.grid(True)
            
            # 2. Throughput Factor by Test
            test_names = [name.split(' - ')[0] for name in df['test_name']]
            ax2.bar(range(len(test_names)), df['throughput_factor'], alpha=0.7)
            ax2.set_xlabel('Test')
            ax2.set_ylabel('Throughput Factor (x)')
            ax2.set_title('Parallel Throughput Factor by Test')
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in test_names], rotation=45)
            ax2.grid(True)
            
            # 3. Latency Distribution
            latencies = df[['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']]
            ax3.boxplot([latencies['latency_p50_ms'], latencies['latency_p95_ms'], latencies['latency_p99_ms']], 
                       labels=['P50', 'P95', 'P99'])
            ax3.set_ylabel('Latency (ms)')
            ax3.set_title('Latency Distribution Across Tests')
            ax3.grid(True)
            
            # 4. Success Rate vs Coordination Overhead
            ax4.scatter(df['coordination_overhead_ms'], df['success_rate'], 
                       s=df['parallel_workers']*20, alpha=0.7, c=df['coordination_accuracy'], 
                       cmap='RdYlGn')
            ax4.set_xlabel('Coordination Overhead (ms)')
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_title('Success Rate vs Coordination Overhead')
            ax4.grid(True)
            
            plt.tight_layout()
            viz_path = self.output_dir / f'ansf_parallel_analysis_{timestamp}.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Memory scaling analysis
            memory_tests = [r for r in detailed_results if 'Memory-Aware' in r['test_name']]
            if memory_tests:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                memory_df = pd.DataFrame(memory_tests)
                workers = memory_df['parallel_workers']
                accuracies = memory_df['coordination_accuracy']
                efficiencies = memory_df['parallel_efficiency']
                
                ax1.plot(workers, accuracies, 'o-', label='Coordination Accuracy', linewidth=2, markersize=8)
                ax1.axhline(y=97.3, color='r', linestyle='--', alpha=0.7, label='Target (97.3%)')
                ax1.set_xlabel('Max Agents (Memory Constraint)')
                ax1.set_ylabel('Coordination Accuracy (%)')
                ax1.set_title('Memory-Aware Scaling: Accuracy vs Agent Count')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(workers, efficiencies, 'o-', color='orange', label='Parallel Efficiency', linewidth=2, markersize=8)
                ax2.set_xlabel('Max Agents (Memory Constraint)')
                ax2.set_ylabel('Parallel Efficiency (%)')
                ax2.set_title('Memory-Aware Scaling: Efficiency vs Agent Count')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                memory_viz_path = self.output_dir / f'memory_scaling_analysis_{timestamp}.png'
                plt.savefig(memory_viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return {
                    'main_analysis': str(viz_path),
                    'memory_scaling': str(memory_viz_path)
                }
            
            return {'main_analysis': str(viz_path)}
            
        except ImportError as e:
            print(f"Warning: Could not generate visualizations due to missing dependencies: {e}")
            return {}
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return {}
    
    def analyze_performance_trends(self, results: dict) -> dict:
        """Analyze performance trends and patterns"""
        detailed_results = results.get('detailed_results', [])
        summary = results.get('summary', {})
        
        analysis = {
            'performance_classification': {},
            'bottleneck_analysis': {},
            'scaling_patterns': {},
            'optimization_priorities': []
        }
        
        # Classify test performance
        target_accuracy = 97.3
        for result in detailed_results:
            test_name = result['test_name']
            accuracy = result['coordination_accuracy']
            efficiency = result['parallel_efficiency']
            
            if accuracy >= target_accuracy and efficiency >= 75:
                classification = "OPTIMAL"
            elif accuracy >= target_accuracy * 0.95 and efficiency >= 60:
                classification = "GOOD"
            elif accuracy >= target_accuracy * 0.90 and efficiency >= 40:
                classification = "ACCEPTABLE"
            else:
                classification = "NEEDS_IMPROVEMENT"
            
            analysis['performance_classification'][test_name] = {
                'classification': classification,
                'accuracy_score': accuracy,
                'efficiency_score': efficiency,
                'gap_to_target': target_accuracy - accuracy
            }
        
        # Bottleneck analysis
        if detailed_results:
            overhead_values = [r['coordination_overhead_ms'] for r in detailed_results]
            latency_p95_values = [r['latency_p95_ms'] for r in detailed_results]
            
            avg_overhead = sum(overhead_values) / len(overhead_values)
            avg_latency = sum(latency_p95_values) / len(latency_p95_values)
            
            analysis['bottleneck_analysis'] = {
                'average_coordination_overhead_ms': avg_overhead,
                'average_p95_latency_ms': avg_latency,
                'high_overhead_tests': [r['test_name'] for r in detailed_results if r['coordination_overhead_ms'] > avg_overhead * 1.5],
                'high_latency_tests': [r['test_name'] for r in detailed_results if r['latency_p95_ms'] > avg_latency * 1.5]
            }
        
        # Scaling pattern analysis
        memory_tests = [r for r in detailed_results if 'Memory-Aware' in r['test_name']]
        if memory_tests:
            scaling_data = {}
            for test in memory_tests:
                workers = test['parallel_workers']
                accuracy = test['coordination_accuracy']
                efficiency = test['parallel_efficiency']
                
                scenario = test['additional_metrics'].get('scenario_name', 'Unknown')
                scaling_data[scenario] = {
                    'workers': workers,
                    'accuracy': accuracy,
                    'efficiency': efficiency
                }
            
            analysis['scaling_patterns'] = {
                'memory_constrained_scaling': scaling_data,
                'optimal_agent_count': max(scaling_data.keys(), key=lambda x: scaling_data[x]['accuracy']) if scaling_data else None
            }
        
        # Generate optimization priorities
        priorities = []
        
        avg_accuracy = summary.get('overall_coordination_accuracy', 0)
        if avg_accuracy < target_accuracy:
            priorities.append({
                'priority': 'HIGH',
                'area': 'Coordination Accuracy',
                'current': f"{avg_accuracy:.2f}%",
                'target': f"{target_accuracy}%",
                'gap': f"{target_accuracy - avg_accuracy:.2f}%",
                'recommendation': 'Optimize consensus algorithms and reduce coordination latency'
            })
        
        avg_efficiency = summary.get('average_parallel_efficiency', 0)
        if avg_efficiency < 75:
            priorities.append({
                'priority': 'HIGH' if avg_efficiency < 50 else 'MEDIUM',
                'area': 'Parallel Efficiency',
                'current': f"{avg_efficiency:.2f}%",
                'target': '75%+',
                'gap': f"{75 - avg_efficiency:.2f}%",
                'recommendation': 'Reduce coordination overhead and improve load balancing'
            })
        
        if avg_overhead > 50:  # 50ms threshold
            priorities.append({
                'priority': 'MEDIUM',
                'area': 'Coordination Overhead',
                'current': f"{avg_overhead:.2f}ms",
                'target': '<50ms',
                'gap': f"{avg_overhead - 50:.2f}ms",
                'recommendation': 'Implement faster consensus protocols and optimize message passing'
            })
        
        analysis['optimization_priorities'] = sorted(priorities, key=lambda x: ['HIGH', 'MEDIUM', 'LOW'].index(x['priority']))
        
        return analysis
    
    def generate_detailed_report(self, results: dict, analysis: dict, timestamp: str) -> str:
        """Generate comprehensive analysis report"""
        summary = results.get('summary', {})
        
        report = f"""
ANSF PARALLEL COORDINATION COMPREHENSIVE ANALYSIS REPORT
======================================================
Generated: {timestamp}
Analysis Duration: {summary.get('total_benchmark_duration_seconds', 0):.2f} seconds

EXECUTIVE SUMMARY
================
Overall Performance Status: {"EXCELLENT" if summary.get('target_accuracy_achievement', 0) >= 100 else "NEEDS OPTIMIZATION"}
Target Accuracy Achievement: {summary.get('target_accuracy_achievement', 0):.1f}% (Target: 97.3%)
Average Coordination Accuracy: {summary.get('overall_coordination_accuracy', 0):.2f}%
Average Parallel Efficiency: {summary.get('average_parallel_efficiency', 0):.2f}%
Average Throughput Factor: {summary.get('average_throughput_factor', 0):.2f}x

PERFORMANCE CLASSIFICATION
=========================
"""
        
        for test_name, classification in analysis['performance_classification'].items():
            status_icon = {
                'OPTIMAL': '🟢',
                'GOOD': '🟡', 
                'ACCEPTABLE': '🟠',
                'NEEDS_IMPROVEMENT': '🔴'
            }.get(classification['classification'], '⚪')
            
            report += f"{status_icon} {test_name}: {classification['classification']}\n"
            report += f"   Accuracy: {classification['accuracy_score']:.2f}% (Gap: {classification['gap_to_target']:.2f}%)\n"
            report += f"   Efficiency: {classification['efficiency_score']:.2f}%\n\n"
        
        report += f"""
BOTTLENECK ANALYSIS
==================
Average Coordination Overhead: {analysis['bottleneck_analysis'].get('average_coordination_overhead_ms', 0):.2f}ms
Average P95 Latency: {analysis['bottleneck_analysis'].get('average_p95_latency_ms', 0):.2f}ms

High Overhead Tests:
{chr(10).join(f"  - {test}" for test in analysis['bottleneck_analysis'].get('high_overhead_tests', []))}

High Latency Tests:
{chr(10).join(f"  - {test}" for test in analysis['bottleneck_analysis'].get('high_latency_tests', []))}

SCALING PATTERN ANALYSIS
=======================
"""
        
        if analysis['scaling_patterns'].get('memory_constrained_scaling'):
            report += "Memory-Constrained Scaling Performance:\n"
            for scenario, data in analysis['scaling_patterns']['memory_constrained_scaling'].items():
                report += f"  {scenario}: {data['workers']} agents → {data['accuracy']:.2f}% accuracy, {data['efficiency']:.2f}% efficiency\n"
            
            optimal = analysis['scaling_patterns'].get('optimal_agent_count')
            if optimal:
                report += f"\nOptimal Configuration: {optimal}\n"
        
        report += f"""
OPTIMIZATION PRIORITIES
======================
"""
        
        for i, priority in enumerate(analysis['optimization_priorities'], 1):
            priority_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(priority['priority'], '⚪')
            report += f"{i}. {priority_icon} {priority['priority']} PRIORITY: {priority['area']}\n"
            report += f"   Current: {priority['current']} | Target: {priority['target']} | Gap: {priority['gap']}\n"
            report += f"   Recommendation: {priority['recommendation']}\n\n"
        
        report += f"""
DETAILED PERFORMANCE METRICS
============================
"""
        
        for result in results.get('detailed_results', []):
            report += f"""
{result['test_name']}:
  Architecture:
    - Parallel Workers: {result['parallel_workers']}
    - Total Duration: {result['total_duration_seconds']:.3f}s
    - Memory Usage: {result['memory_usage_mb']:.2f}MB
    - CPU Usage: {result['cpu_usage_percent']:.2f}%
  
  Performance:
    - Coordination Accuracy: {result['coordination_accuracy']:.2f}%
    - Parallel Efficiency: {result['parallel_efficiency']:.2f}%
    - Throughput Factor: {result['throughput_factor']:.2f}x
    - Operations/sec: {result['operations_per_second']:.2f}
  
  Latency Profile:
    - P50: {result['latency_p50_ms']:.2f}ms
    - P95: {result['latency_p95_ms']:.2f}ms  
    - P99: {result['latency_p99_ms']:.2f}ms
    - Coordination Overhead: {result['coordination_overhead_ms']:.2f}ms
  
  Reliability:
    - Success Rate: {result['success_rate']:.2f}%
    - Error Count: {result['error_count']}
  
  Additional Metrics: {json.dumps(result.get('additional_metrics', {}), indent=4)}

"""
        
        report += f"""
TECHNICAL RECOMMENDATIONS
=========================

IMMEDIATE ACTIONS (High Priority):
1. 🎯 Focus on coordination accuracy improvements for tests below 97.3%
2. ⚡ Optimize coordination overhead for tests exceeding 50ms average
3. 🧠 Implement adaptive scaling based on memory pressure patterns
4. 🔄 Enhance parallel efficiency through better load balancing

MEDIUM-TERM OPTIMIZATIONS:
1. 📊 Deploy real-time performance monitoring for production systems
2. 🤖 Implement machine learning-based resource allocation
3. 🔧 Develop automatic failover mechanisms for memory-critical scenarios
4. 📈 Create predictive scaling algorithms based on workload patterns

LONG-TERM STRATEGIC IMPROVEMENTS:
1. 🏗️ Design next-generation coordination protocols
2. 🧪 Research quantum-inspired optimization algorithms
3. 🌐 Develop distributed consensus mechanisms
4. 🔮 Implement self-healing parallel coordination systems

PERFORMANCE TARGETS FOR PRODUCTION DEPLOYMENT:
==============================================
✅ Coordination Accuracy: ≥97.3% (Current: {summary.get('overall_coordination_accuracy', 0):.2f}%)
✅ Parallel Efficiency: ≥75% (Current: {summary.get('average_parallel_efficiency', 0):.2f}%)
✅ Throughput Factor: ≥3.0x (Current: {summary.get('average_throughput_factor', 0):.2f}x)
✅ Success Rate: ≥99% (Current: {summary.get('overall_success_rate', 0):.2f}%)
✅ Coordination Overhead: <50ms
✅ Memory Efficiency: Graceful degradation under constraints

CONCLUSION
==========
{"The ANSF system demonstrates strong parallel coordination capabilities with excellent accuracy maintenance." if summary.get('target_accuracy_achievement', 0) >= 95 else "The ANSF system shows promise but requires optimization to meet production targets."}
{"Continue monitoring and fine-tuning for optimal performance." if summary.get('target_accuracy_achievement', 0) >= 95 else "Prioritize the optimization recommendations above to achieve target performance metrics."}

Report Generation Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    async def run_analysis(self):
        """Run comprehensive ANSF parallel coordination analysis"""
        print("🚀 Starting ANSF Parallel Coordination Analysis...")
        print(f"📊 System Info: {self.benchmark.get_system_metrics()}")
        
        # Run benchmarks
        print("\n⏱️  Running comprehensive benchmark suite...")
        results = await self.benchmark.run_comprehensive_benchmark()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Perform detailed analysis
        print("\n📈 Analyzing performance trends and patterns...")
        analysis = self.analyze_performance_trends(results)
        
        # Generate visualizations
        print("\n📊 Generating performance visualizations...")
        visualizations = self.generate_visualizations(results, timestamp)
        
        # Generate comprehensive report
        print("\n📝 Generating detailed analysis report...")
        detailed_report = self.generate_detailed_report(results, analysis, timestamp)
        
        # Save all outputs
        results_file = self.output_dir / f'ansf_parallel_results_{timestamp}.json'
        analysis_file = self.output_dir / f'ansf_parallel_analysis_{timestamp}.json'
        report_file = self.output_dir / f'ansf_parallel_report_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        with open(report_file, 'w') as f:
            f.write(detailed_report)
        
        # Summary output
        summary = results.get('summary', {})
        print(f"\n✅ Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Total tests: {summary.get('total_tests', 0)}")
        print(f"🎯 Target achievement: {summary.get('target_accuracy_achievement', 0):.1f}%")
        print(f"⚡ Average efficiency: {summary.get('average_parallel_efficiency', 0):.2f}%")
        print(f"🚀 Throughput factor: {summary.get('average_throughput_factor', 0):.2f}x")
        
        if visualizations:
            print(f"📈 Visualizations: {', '.join(visualizations.values())}")
        
        print(f"\n📋 Files generated:")
        print(f"  - Raw Results: {results_file.name}")
        print(f"  - Analysis Data: {analysis_file.name}")  
        print(f"  - Detailed Report: {report_file.name}")
        
        return {
            'results': results,
            'analysis': analysis,
            'report': detailed_report,
            'files': {
                'results': str(results_file),
                'analysis': str(analysis_file),
                'report': str(report_file)
            },
            'visualizations': visualizations
        }

async def main():
    """Main entry point for ANSF parallel analysis"""
    runner = ANSFAnalysisRunner()
    await runner.run_analysis()

if __name__ == "__main__":
    asyncio.run(main())