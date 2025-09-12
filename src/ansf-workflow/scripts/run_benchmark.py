#!/usr/bin/env python3
"""
Quick runner for ANSF Parallel Coordination Benchmark
Handles memory constraints and provides real-time feedback
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ansf_parallel_benchmark import ANSFParallelBenchmark
    print("✅ Successfully imported ANSFParallelBenchmark")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying to install required packages...")
    os.system("pip install psutil numpy")
    try:
        from ansf_parallel_benchmark import ANSFParallelBenchmark
        print("✅ Successfully imported after package installation")
    except ImportError as e2:
        print(f"❌ Still cannot import: {e2}")
        sys.exit(1)

async def run_quick_benchmark():
    """Run a streamlined version of the benchmark for immediate results"""
    print("🚀 Starting ANSF Parallel Coordination Benchmark (Quick Mode)")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    benchmark = ANSFParallelBenchmark(max_workers=4)  # Limit workers for memory
    
    # Get system info
    system_info = benchmark.get_system_metrics()
    print(f"\n💻 System Information:")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   Memory Total: {system_info['memory_total_gb']:.2f} GB")
    print(f"   Memory Available: {system_info['memory_available_gb']:.2f} GB")
    print(f"   Memory Usage: {system_info['memory_percent']:.1f}%")
    print(f"   CPU Usage: {system_info['cpu_usage_percent']:.1f}%")
    
    # Check memory status
    if system_info['memory_percent'] > 95:
        print("⚠️  WARNING: High memory usage detected - using emergency mode")
        max_swarms = 2
        max_heads = 4
    elif system_info['memory_percent'] > 85:
        print("⚠️  CAUTION: Moderate memory usage - using limited mode")
        max_swarms = 4
        max_heads = 6
    else:
        print("✅ Good memory availability - using normal mode")
        max_swarms = 6
        max_heads = 8
    
    results = []
    
    try:
        # Test 1: Multi-Swarm Coordination (reduced)
        print(f"\n🔄 Test 1/5: Multi-Swarm Coordination ({max_swarms} swarms)")
        result1 = await benchmark.benchmark_multi_swarm_coordination(max_swarms)
        results.append(result1)
        print(f"   ✅ Accuracy: {result1.coordination_accuracy:.2f}% | Efficiency: {result1.parallel_efficiency:.2f}%")
        
        # Test 2: Neural Processing (reduced)
        print(f"\n🧠 Test 2/5: Neural Parallel Processing ({max_heads} heads)")
        result2 = await benchmark.benchmark_neural_parallel_processing(max_heads)
        results.append(result2)
        print(f"   ✅ Accuracy: {result2.coordination_accuracy:.2f}% | Speedup: {result2.throughput_factor:.2f}x")
        
        # Test 3: Progressive Refinement
        print(f"\n⚡ Test 3/5: Progressive Refinement (3 cycles)")
        result3 = await benchmark.benchmark_progressive_refinement_parallelization(3)
        results.append(result3)
        print(f"   ✅ Quality: {result3.coordination_accuracy:.2f}% | Speedup: {result3.throughput_factor:.2f}x")
        
        # Test 4: Memory Scaling (simplified)
        print(f"\n💾 Test 4/5: Memory-Aware Scaling")
        memory_scenarios = [
            ("Emergency", 1),
            ("Limited", 2),
            ("Normal", 4)
        ]
        memory_results = await benchmark.benchmark_memory_aware_scaling(memory_scenarios)
        results.extend(memory_results)
        for mr in memory_results:
            print(f"   ✅ {mr.additional_metrics.get('scenario_name', 'Unknown')}: {mr.coordination_accuracy:.2f}% accuracy")
        
        # Test 5: Cross-System Integration
        print(f"\n🔗 Test 5/5: Cross-System Integration")
        result5 = await benchmark.benchmark_cross_system_integration()
        results.append(result5)
        print(f"   ✅ Integration: {result5.coordination_accuracy:.2f}% | Efficiency: {result5.parallel_efficiency:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return None
    
    # Calculate summary statistics
    if results:
        accuracies = [r.coordination_accuracy for r in results]
        efficiencies = [r.parallel_efficiency for r in results]
        throughputs = [r.throughput_factor for r in results]
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_efficiency = sum(efficiencies) / len(efficiencies)
        avg_throughput = sum(throughputs) / len(throughputs)
        
        target_achievement = (avg_accuracy / 97.3) * 100
        
        print(f"\n📊 BENCHMARK RESULTS SUMMARY")
        print(f"=" * 50)
        print(f"Tests Completed: {len(results)}")
        print(f"Overall Coordination Accuracy: {avg_accuracy:.2f}%")
        print(f"Target Achievement (97.3%): {target_achievement:.1f}%")
        print(f"Average Parallel Efficiency: {avg_efficiency:.2f}%")
        print(f"Average Throughput Factor: {avg_throughput:.2f}x")
        
        # Performance assessment
        if target_achievement >= 100 and avg_efficiency >= 75:
            status = "🟢 EXCELLENT"
        elif target_achievement >= 95 and avg_efficiency >= 60:
            status = "🟡 GOOD" 
        elif target_achievement >= 90 and avg_efficiency >= 40:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 NEEDS IMPROVEMENT"
            
        print(f"Overall Performance Status: {status}")
        
        # Quick recommendations
        print(f"\n🎯 QUICK RECOMMENDATIONS:")
        if avg_accuracy < 97.3:
            print(f"   • Improve coordination accuracy by {97.3 - avg_accuracy:.1f}%")
        if avg_efficiency < 75:
            print(f"   • Optimize parallel efficiency by {75 - avg_efficiency:.1f}%")
        if avg_throughput < 3.0:
            print(f"   • Enhance throughput by {3.0 - avg_throughput:.1f}x")
            
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save quick summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"ansf_quick_results_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"ANSF Parallel Coordination Quick Benchmark Results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Tests Completed: {len(results)}\n")
            f.write(f"Overall Coordination Accuracy: {avg_accuracy:.2f}%\n")
            f.write(f"Target Achievement: {target_achievement:.1f}%\n")
            f.write(f"Average Parallel Efficiency: {avg_efficiency:.2f}%\n")
            f.write(f"Average Throughput Factor: {avg_throughput:.2f}x\n")
            f.write(f"Performance Status: {status}\n")
            
        print(f"📁 Quick results saved to: {summary_file}")
        return results
    
    else:
        print("❌ No results to analyze")
        return None

if __name__ == "__main__":
    try:
        results = asyncio.run(run_quick_benchmark())
        if results:
            print("✅ Benchmark completed successfully!")
        else:
            print("❌ Benchmark failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)