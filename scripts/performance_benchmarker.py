#!/usr/bin/env python3
"""
Knowledge Base Tagging System Performance Benchmarker

This script provides comprehensive performance analysis of the tagging system operations,
including query performance, memory usage, scalability testing, and bottleneck identification.
"""

import asyncio
import json
import time
import sys
import os
import traceback
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import httpx
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetric:
    """Container for performance measurement data."""
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    status_code: Optional[int] = None
    data_size_kb: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    metrics: List[PerformanceMetric]
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    avg_memory_mb: float
    avg_cpu_percent: float
    success_rate: float
    operations_per_second: float
    total_operations: int


class KnowledgeBasePerformanceBenchmarker:
    """Comprehensive performance benchmarker for the knowledge base tagging system."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.process = psutil.Process()
        self.benchmark_results: List[BenchmarkResult] = []
        
    async def measure_operation(self, operation_name: str, operation_func) -> PerformanceMetric:
        """Measure performance of a single operation."""
        # Get initial memory state
        memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_before = self.process.cpu_percent()
        
        start_time = time.perf_counter()
        
        try:
            result = await operation_func()
            end_time = time.perf_counter()
            
            # Get final memory state
            memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
            cpu_after = self.process.cpu_percent()
            
            duration_ms = (end_time - start_time) * 1000
            memory_mb = memory_after
            cpu_percent = max(cpu_after, cpu_before)  # Use max to capture peak usage
            
            status_code = None
            data_size_kb = None
            
            if hasattr(result, 'status_code'):
                status_code = result.status_code
            
            if hasattr(result, 'content'):
                data_size_kb = len(result.content) / 1024
            elif isinstance(result, (dict, list, str)):
                data_size_kb = len(str(result).encode('utf-8')) / 1024
            
            return PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                status_code=status_code,
                data_size_kb=data_size_kb
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            return PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_mb=self.process.memory_info().rss / (1024 * 1024),
                cpu_percent=self.process.cpu_percent(),
                status_code=-1  # Error indicator
            )
    
    async def benchmark_single_tag_query(self, tag: str) -> PerformanceMetric:
        """Benchmark a single tag query operation."""
        async def query_operation():
            url = f"{self.base_url}/api/knowledge-items/search"
            params = {"query": f"tag:{tag}", "per_page": 50}
            return await self.client.get(url, params=params)
        
        return await self.measure_operation(f"single_tag_query:{tag}", query_operation)
    
    async def benchmark_multi_tag_query(self, tags: List[str]) -> PerformanceMetric:
        """Benchmark a multi-tag query operation."""
        async def query_operation():
            url = f"{self.base_url}/api/knowledge-items/search"
            tag_query = " AND ".join([f"tag:{tag}" for tag in tags])
            params = {"query": tag_query, "per_page": 50}
            return await self.client.get(url, params=params)
        
        tag_str = "+".join(tags[:3])  # Limit display length
        return await self.measure_operation(f"multi_tag_query:{tag_str}", query_operation)
    
    async def benchmark_tag_update(self, source_id: str, tags: List[str]) -> PerformanceMetric:
        """Benchmark a tag update operation."""
        async def update_operation():
            url = f"{self.base_url}/api/knowledge-items/{source_id}"
            data = {"metadata": {"tags": tags}}
            return await self.client.put(url, json=data)
        
        return await self.measure_operation(f"tag_update:{source_id[:8]}", update_operation)
    
    async def benchmark_large_dataset_query(self, tag: str, page_size: int = 100) -> PerformanceMetric:
        """Benchmark query with large result set."""
        async def large_query_operation():
            url = f"{self.base_url}/api/knowledge-items/search"
            params = {"query": f"tag:{tag}", "per_page": page_size}
            return await self.client.get(url, params=params)
        
        return await self.measure_operation(f"large_query:{tag}:{page_size}", large_query_operation)
    
    async def run_tag_query_benchmark(self, iterations: int = 10) -> BenchmarkResult:
        """Comprehensive tag query performance benchmark."""
        print(f"🔍 Running tag query benchmark ({iterations} iterations)...")
        
        # Common tags to test
        test_tags = [
            "ai-orchestration",
            "performance-optimization", 
            "multi-agent-systems",
            "claude-flow",
            "enterprise-ai",
            "python-framework",
            "infrastructure-management",
            "best-practices"
        ]
        
        metrics = []
        
        for i in range(iterations):
            for tag in test_tags:
                metric = await self.benchmark_single_tag_query(tag)
                metrics.append(metric)
                
                if i == 0:  # Only show progress on first iteration
                    status = "✅" if metric.status_code == 200 else "❌"
                    print(f"  {status} Tag '{tag}': {metric.duration_ms:.1f}ms")
        
        return self._calculate_benchmark_result("tag_query", metrics)
    
    async def run_multi_tag_benchmark(self, iterations: int = 5) -> BenchmarkResult:
        """Multi-tag query performance benchmark."""
        print(f"🔍 Running multi-tag query benchmark ({iterations} iterations)...")
        
        # Test combinations
        tag_combinations = [
            ["claude-flow", "ai-orchestration"],
            ["python-framework", "multi-agent-systems"],
            ["enterprise-ai", "performance-optimization"],
            ["infrastructure-management", "best-practices"],
            ["ai-development", "workflow-automation", "integration-platform"]
        ]
        
        metrics = []
        
        for i in range(iterations):
            for tags in tag_combinations:
                metric = await self.benchmark_multi_tag_query(tags)
                metrics.append(metric)
                
                if i == 0:
                    status = "✅" if metric.status_code == 200 else "❌"
                    print(f"  {status} Tags {tags[:2]}: {metric.duration_ms:.1f}ms")
        
        return self._calculate_benchmark_result("multi_tag_query", metrics)
    
    async def run_tag_update_benchmark(self, iterations: int = 3) -> BenchmarkResult:
        """Tag update performance benchmark."""
        print(f"🔄 Running tag update benchmark ({iterations} iterations)...")
        
        # Test source IDs (using ones from the tagging system)
        test_sources = [
            "56cb969b4f4e75d5",  # AWS Labs MCP
            "65516ba46d606b01",  # Claude Flow Wiki
            "92913be64b1ead25",  # Claude Code
            "a51526d65470cb31",  # PydanticAI
            "ccbb49fd5eb8b6a3"   # Archon Repository
        ]
        
        test_tags = [
            ["test-tag-1", "test-tag-2"],
            ["benchmark-tag", "performance-test"],
            ["update-test", "timing-analysis"]
        ]
        
        metrics = []
        
        for i in range(iterations):
            for j, source_id in enumerate(test_sources):
                tags = test_tags[j % len(test_tags)]
                metric = await self.benchmark_tag_update(source_id, tags)
                metrics.append(metric)
                
                if i == 0:
                    status = "✅" if metric.status_code == 200 else "❌"
                    print(f"  {status} Update {source_id[:8]}: {metric.duration_ms:.1f}ms")
        
        return self._calculate_benchmark_result("tag_update", metrics)
    
    async def run_scalability_benchmark(self) -> BenchmarkResult:
        """Scalability test with increasing page sizes."""
        print(f"📈 Running scalability benchmark...")
        
        page_sizes = [10, 25, 50, 100, 200, 500]
        test_tag = "performance-optimization"
        
        metrics = []
        
        for page_size in page_sizes:
            metric = await self.benchmark_large_dataset_query(test_tag, page_size)
            metrics.append(metric)
            
            status = "✅" if metric.status_code == 200 else "❌"
            print(f"  {status} Page size {page_size}: {metric.duration_ms:.1f}ms, {metric.data_size_kb:.1f}KB")
        
        return self._calculate_benchmark_result("scalability", metrics)
    
    async def run_stress_test(self, concurrent_requests: int = 10, iterations: int = 5) -> BenchmarkResult:
        """Stress test with concurrent requests."""
        print(f"⚡ Running stress test ({concurrent_requests} concurrent, {iterations} iterations)...")
        
        test_tags = ["claude-flow", "ai-orchestration", "multi-agent-systems"]
        metrics = []
        
        for i in range(iterations):
            # Create concurrent tasks
            tasks = []
            for j in range(concurrent_requests):
                tag = test_tags[j % len(test_tags)]
                task = self.benchmark_single_tag_query(tag)
                tasks.append(task)
            
            # Execute concurrently and collect results
            batch_metrics = await asyncio.gather(*tasks, return_exceptions=True)
            
            for metric in batch_metrics:
                if isinstance(metric, PerformanceMetric):
                    metrics.append(metric)
                elif isinstance(metric, Exception):
                    # Create error metric
                    error_metric = PerformanceMetric(
                        operation="stress_test_error",
                        duration_ms=0.0,
                        memory_mb=self.process.memory_info().rss / (1024 * 1024),
                        cpu_percent=self.process.cpu_percent(),
                        status_code=-1
                    )
                    metrics.append(error_metric)
            
            if i == 0:
                avg_duration = statistics.mean([m.duration_ms for m in batch_metrics if isinstance(m, PerformanceMetric)])
                print(f"  Iteration 1 avg: {avg_duration:.1f}ms per request")
        
        return self._calculate_benchmark_result("stress_test", metrics)
    
    def _calculate_benchmark_result(self, test_name: str, metrics: List[PerformanceMetric]) -> BenchmarkResult:
        """Calculate comprehensive benchmark statistics."""
        if not metrics:
            return BenchmarkResult(
                test_name=test_name,
                metrics=[],
                avg_duration_ms=0.0,
                min_duration_ms=0.0,
                max_duration_ms=0.0,
                p95_duration_ms=0.0,
                p99_duration_ms=0.0,
                avg_memory_mb=0.0,
                avg_cpu_percent=0.0,
                success_rate=0.0,
                operations_per_second=0.0,
                total_operations=0
            )
        
        durations = [m.duration_ms for m in metrics]
        memories = [m.memory_mb for m in metrics]
        cpus = [m.cpu_percent for m in metrics if m.cpu_percent > 0]
        successful = len([m for m in metrics if m.status_code == 200])
        
        # Calculate percentiles
        durations.sort()
        p95_idx = int(0.95 * len(durations))
        p99_idx = int(0.99 * len(durations))
        
        total_time_seconds = sum(durations) / 1000  # Convert to seconds
        ops_per_second = len(metrics) / total_time_seconds if total_time_seconds > 0 else 0
        
        return BenchmarkResult(
            test_name=test_name,
            metrics=metrics,
            avg_duration_ms=statistics.mean(durations),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            p95_duration_ms=durations[p95_idx] if durations else 0.0,
            p99_duration_ms=durations[p99_idx] if durations else 0.0,
            avg_memory_mb=statistics.mean(memories),
            avg_cpu_percent=statistics.mean(cpus) if cpus else 0.0,
            success_rate=(successful / len(metrics)) * 100,
            operations_per_second=ops_per_second,
            total_operations=len(metrics)
        )
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all performance benchmarks and generate comprehensive report."""
        print("🚀 Starting Comprehensive Knowledge Base Performance Analysis")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run all benchmark suites
        try:
            # Basic query performance
            tag_query_result = await self.run_tag_query_benchmark()
            self.benchmark_results.append(tag_query_result)
            
            # Multi-tag query performance
            multi_tag_result = await self.run_multi_tag_benchmark()
            self.benchmark_results.append(multi_tag_result)
            
            # Update operations performance
            tag_update_result = await self.run_tag_update_benchmark()
            self.benchmark_results.append(tag_update_result)
            
            # Scalability testing
            scalability_result = await self.run_scalability_benchmark()
            self.benchmark_results.append(scalability_result)
            
            # Stress testing
            stress_result = await self.run_stress_test()
            self.benchmark_results.append(stress_result)
            
        except Exception as e:
            print(f"❌ Error during benchmarking: {str(e)}")
            traceback.print_exc()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_performance_report(total_time)
        
        # Store results to memory via hooks
        await self._store_performance_metrics(report)
        
        return report
    
    def _generate_performance_report(self, total_benchmark_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        report = {
            "benchmark_summary": {
                "timestamp": time.time(),
                "total_benchmark_time_seconds": total_benchmark_time,
                "total_tests": len(self.benchmark_results),
                "total_operations": sum(r.total_operations for r in self.benchmark_results)
            },
            "results": {},
            "analysis": {},
            "recommendations": []
        }
        
        # Process each benchmark result
        for result in self.benchmark_results:
            report["results"][result.test_name] = {
                "total_operations": result.total_operations,
                "avg_duration_ms": result.avg_duration_ms,
                "min_duration_ms": result.min_duration_ms,
                "max_duration_ms": result.max_duration_ms,
                "p95_duration_ms": result.p95_duration_ms,
                "p99_duration_ms": result.p99_duration_ms,
                "avg_memory_mb": result.avg_memory_mb,
                "avg_cpu_percent": result.avg_cpu_percent,
                "success_rate": result.success_rate,
                "operations_per_second": result.operations_per_second
            }
        
        # Performance analysis
        if self.benchmark_results:
            avg_query_time = statistics.mean([r.avg_duration_ms for r in self.benchmark_results])
            max_query_time = max([r.max_duration_ms for r in self.benchmark_results])
            avg_success_rate = statistics.mean([r.success_rate for r in self.benchmark_results])
            total_ops_per_second = sum([r.operations_per_second for r in self.benchmark_results])
            
            report["analysis"] = {
                "overall_avg_response_time_ms": avg_query_time,
                "worst_case_response_time_ms": max_query_time,
                "overall_success_rate": avg_success_rate,
                "total_throughput_ops_per_second": total_ops_per_second,
                "performance_grade": self._calculate_performance_grade(avg_query_time, avg_success_rate),
                "bottlenecks_identified": self._identify_bottlenecks(),
                "resource_utilization": self._analyze_resource_usage()
            }
            
            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(report["analysis"])
        
        return report
    
    def _calculate_performance_grade(self, avg_response_time: float, success_rate: float) -> str:
        """Calculate overall performance grade."""
        if avg_response_time <= 100 and success_rate >= 99:
            return "A+ (Excellent)"
        elif avg_response_time <= 250 and success_rate >= 95:
            return "A (Very Good)"
        elif avg_response_time <= 500 and success_rate >= 90:
            return "B (Good)"
        elif avg_response_time <= 1000 and success_rate >= 85:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def _identify_bottlenecks(self) -> List[Dict[str, str]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for result in self.benchmark_results:
            # Response time bottlenecks
            if result.avg_duration_ms > 500:
                bottlenecks.append({
                    "type": "response_time",
                    "test": result.test_name,
                    "issue": f"High average response time: {result.avg_duration_ms:.1f}ms",
                    "severity": "high" if result.avg_duration_ms > 1000 else "medium"
                })
            
            # Variability bottlenecks
            if result.max_duration_ms > result.avg_duration_ms * 3:
                bottlenecks.append({
                    "type": "variability",
                    "test": result.test_name,
                    "issue": f"High response time variability (max: {result.max_duration_ms:.1f}ms, avg: {result.avg_duration_ms:.1f}ms)",
                    "severity": "medium"
                })
            
            # Success rate bottlenecks
            if result.success_rate < 95:
                bottlenecks.append({
                    "type": "reliability",
                    "test": result.test_name,
                    "issue": f"Low success rate: {result.success_rate:.1f}%",
                    "severity": "high" if result.success_rate < 90 else "medium"
                })
            
            # Throughput bottlenecks
            if result.operations_per_second < 10:
                bottlenecks.append({
                    "type": "throughput",
                    "test": result.test_name,
                    "issue": f"Low throughput: {result.operations_per_second:.1f} ops/sec",
                    "severity": "medium"
                })
        
        return bottlenecks
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        if not self.benchmark_results:
            return {}
        
        all_metrics = []
        for result in self.benchmark_results:
            all_metrics.extend(result.metrics)
        
        memory_usage = [m.memory_mb for m in all_metrics if m.memory_mb > 0]
        cpu_usage = [m.cpu_percent for m in all_metrics if m.cpu_percent > 0]
        
        return {
            "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0,
            "avg_cpu_percent": statistics.mean(cpu_usage) if cpu_usage else 0,
            "peak_cpu_percent": max(cpu_usage) if cpu_usage else 0,
            "memory_efficiency": "good" if (max(memory_usage) if memory_usage else 0) < 500 else "needs_monitoring",
            "cpu_efficiency": "good" if (max(cpu_usage) if cpu_usage else 0) < 50 else "needs_monitoring"
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Response time recommendations
        avg_response = analysis.get("overall_avg_response_time_ms", 0)
        if avg_response > 300:
            recommendations.append({
                "category": "response_time",
                "priority": "high",
                "recommendation": "Consider implementing database query optimization and indexing for tag-based searches",
                "expected_improvement": "30-50% reduction in query time"
            })
        
        # Caching recommendations
        if avg_response > 200:
            recommendations.append({
                "category": "caching",
                "priority": "medium", 
                "recommendation": "Implement Redis caching for frequently accessed tags and search results",
                "expected_improvement": "50-70% reduction in repeated queries"
            })
        
        # Database optimization
        bottlenecks = analysis.get("bottlenecks_identified", [])
        has_query_issues = any(b["type"] == "response_time" for b in bottlenecks)
        if has_query_issues:
            recommendations.append({
                "category": "database",
                "priority": "high",
                "recommendation": "Add database indexes on tag columns and implement query optimization",
                "expected_improvement": "40-60% faster tag queries"
            })
        
        # Scalability recommendations
        throughput = analysis.get("total_throughput_ops_per_second", 0)
        if throughput < 50:
            recommendations.append({
                "category": "scalability",
                "priority": "medium",
                "recommendation": "Consider connection pooling and async processing for better concurrent handling",
                "expected_improvement": "2-3x improvement in concurrent request handling"
            })
        
        # Memory optimization
        resource_usage = analysis.get("resource_utilization", {})
        if resource_usage.get("memory_efficiency") == "needs_monitoring":
            recommendations.append({
                "category": "memory",
                "priority": "medium",
                "recommendation": "Implement memory usage monitoring and optimize large query result handling",
                "expected_improvement": "20-30% reduction in memory usage"
            })
        
        return recommendations
    
    async def _store_performance_metrics(self, report: Dict[str, Any]):
        """Store performance metrics in memory using hooks."""
        try:
            # Store overall performance metrics
            await asyncio.create_subprocess_exec(
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", "performance/overview",
                "--file", json.dumps({
                    "avg_response_time_ms": report["analysis"].get("overall_avg_response_time_ms", 0),
                    "success_rate": report["analysis"].get("overall_success_rate", 0),
                    "throughput_ops_per_second": report["analysis"].get("total_throughput_ops_per_second", 0),
                    "performance_grade": report["analysis"].get("performance_grade", "Unknown")
                })
            )
            
            # Store benchmark results
            for test_name, results in report["results"].items():
                await asyncio.create_subprocess_exec(
                    "npx", "claude-flow@alpha", "hooks", "post-edit",
                    "--memory-key", f"performance/benchmarks/{test_name}",
                    "--file", json.dumps(results)
                )
            
            # Store bottlenecks
            bottlenecks = report["analysis"].get("bottlenecks_identified", [])
            if bottlenecks:
                await asyncio.create_subprocess_exec(
                    "npx", "claude-flow@alpha", "hooks", "post-edit",
                    "--memory-key", "performance/bottlenecks",
                    "--file", json.dumps(bottlenecks)
                )
            
            # Store recommendations
            recommendations = report.get("recommendations", [])
            if recommendations:
                await asyncio.create_subprocess_exec(
                    "npx", "claude-flow@alpha", "hooks", "post-edit", 
                    "--memory-key", "performance/recommendations",
                    "--file", json.dumps(recommendations)
                )
            
        except Exception as e:
            print(f"⚠️  Warning: Could not store metrics in memory: {str(e)}")
    
    def print_performance_report(self, report: Dict[str, Any]):
        """Print a comprehensive performance analysis report."""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        print("=" * 70)
        
        # Summary
        summary = report["benchmark_summary"]
        print(f"📈 Test Summary:")
        print(f"   • Total benchmark time: {summary['total_benchmark_time_seconds']:.1f} seconds")
        print(f"   • Tests executed: {summary['total_tests']}")
        print(f"   • Total operations: {summary['total_operations']}")
        
        # Overall Performance
        if "analysis" in report:
            analysis = report["analysis"]
            print(f"\n⚡ Overall Performance:")
            print(f"   • Average response time: {analysis.get('overall_avg_response_time_ms', 0):.1f}ms")
            print(f"   • Worst case response time: {analysis.get('worst_case_response_time_ms', 0):.1f}ms")
            print(f"   • Success rate: {analysis.get('overall_success_rate', 0):.1f}%")
            print(f"   • Total throughput: {analysis.get('total_throughput_ops_per_second', 0):.1f} ops/sec")
            print(f"   • Performance grade: {analysis.get('performance_grade', 'Unknown')}")
        
        # Detailed Results
        print(f"\n📋 Detailed Test Results:")
        for test_name, results in report["results"].items():
            print(f"\n   🔸 {test_name.replace('_', ' ').title()}:")
            print(f"      • Operations: {results['total_operations']}")
            print(f"      • Avg time: {results['avg_duration_ms']:.1f}ms")
            print(f"      • P95 time: {results['p95_duration_ms']:.1f}ms")
            print(f"      • Success rate: {results['success_rate']:.1f}%")
            print(f"      • Throughput: {results['operations_per_second']:.1f} ops/sec")
        
        # Resource Usage
        if "analysis" in report and "resource_utilization" in report["analysis"]:
            resources = report["analysis"]["resource_utilization"]
            print(f"\n💾 Resource Utilization:")
            print(f"   • Average memory: {resources.get('avg_memory_mb', 0):.1f}MB")
            print(f"   • Peak memory: {resources.get('peak_memory_mb', 0):.1f}MB")
            print(f"   • Average CPU: {resources.get('avg_cpu_percent', 0):.1f}%")
            print(f"   • Peak CPU: {resources.get('peak_cpu_percent', 0):.1f}%")
        
        # Bottlenecks
        if "analysis" in report and "bottlenecks_identified" in report["analysis"]:
            bottlenecks = report["analysis"]["bottlenecks_identified"]
            if bottlenecks:
                print(f"\n🚨 Performance Bottlenecks Identified:")
                for bottleneck in bottlenecks:
                    severity_icon = "🔴" if bottleneck["severity"] == "high" else "🟡"
                    print(f"   {severity_icon} {bottleneck['type'].title()}: {bottleneck['issue']}")
            else:
                print(f"\n✅ No significant performance bottlenecks identified")
        
        # Recommendations
        if "recommendations" in report and report["recommendations"]:
            print(f"\n🎯 Performance Optimization Recommendations:")
            for rec in report["recommendations"]:
                priority_icon = "🔥" if rec["priority"] == "high" else "⚡"
                print(f"\n   {priority_icon} {rec['category'].title()} Optimization:")
                print(f"      • Recommendation: {rec['recommendation']}")
                print(f"      • Expected improvement: {rec['expected_improvement']}")
        
        print("\n" + "=" * 70)
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


async def main():
    """Main execution function."""
    benchmarker = None
    try:
        print("🚀 Initializing Knowledge Base Performance Benchmarker...")
        benchmarker = KnowledgeBasePerformanceBenchmarker()
        
        # Run comprehensive performance analysis
        report = await benchmarker.run_comprehensive_benchmark()
        
        # Display results
        benchmarker.print_performance_report(report)
        
        # Save detailed report to file
        report_file = Path("performance_analysis_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Critical error during performance analysis: {str(e)}")
        traceback.print_exc()
        return 1
        
    finally:
        if benchmarker:
            await benchmarker.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)