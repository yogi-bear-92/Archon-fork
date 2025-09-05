#!/usr/bin/env python3
"""
Test runner for Claude Flow Expert Agent comprehensive test suite.

This script provides various testing modes and configurations for running
the Claude Flow Expert Agent test suite with proper reporting and performance monitoring.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class ClaudeFlowExpertAgentTestRunner:
    """Test runner for Claude Flow Expert Agent test suite."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        
        # Test categories and their descriptions
        self.test_categories = {
            "unit": {
                "path": "tests/unit/",
                "description": "Unit tests for core Claude Flow Expert Agent functionality",
                "estimated_time": "2-3 minutes",
                "markers": ["unit"]
            },
            "integration": {
                "path": "tests/integration/",
                "description": "Integration tests for RAG, coordination, and workflows", 
                "estimated_time": "5-8 minutes",
                "markers": ["integration"]
            },
            "performance": {
                "path": "tests/performance/",
                "description": "Performance tests for latency, throughput, and load",
                "estimated_time": "8-12 minutes", 
                "markers": ["performance"]
            },
            "e2e": {
                "path": "tests/e2e/",
                "description": "End-to-end workflow and scenario tests",
                "estimated_time": "10-15 minutes",
                "markers": ["e2e"]
            },
            "fallback": {
                "path": "tests/unit/test_fallback_mechanisms.py",
                "description": "Fallback mechanism and error handling tests",
                "estimated_time": "3-5 minutes",
                "markers": ["unit", "fallback"]
            }
        }
    
    def run_tests(
        self,
        categories: List[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        parallel: bool = False,
        performance_baseline: bool = False,
        output_format: str = "terminal",
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Run Claude Flow Expert Agent tests with specified configuration.
        
        Args:
            categories: List of test categories to run
            verbose: Enable verbose output
            coverage: Generate coverage report
            parallel: Run tests in parallel
            performance_baseline: Run performance baseline tests
            output_format: Output format (terminal, junit, html)
            max_workers: Maximum parallel workers
            
        Returns:
            Test execution results and metrics
        """
        start_time = time.time()
        
        # Default to all categories if none specified
        if not categories:
            categories = list(self.test_categories.keys())
        
        print("🧠 Claude Flow Expert Agent Test Suite")
        print("=" * 50)
        print(f"Running categories: {', '.join(categories)}")
        print(f"Estimated time: {self._estimate_total_time(categories)}")
        print()
        
        # Build pytest command
        cmd = self._build_pytest_command(
            categories, verbose, coverage, parallel, output_format, max_workers
        )
        
        # Set environment variables
        env = self._setup_test_environment()
        
        # Run tests
        print("Starting test execution...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=False,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            # Collect results
            results = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "categories_run": categories,
                "command": cmd
            }
            
            # Print summary
            self._print_test_summary(results, categories)
            
            # Generate performance baseline if requested
            if performance_baseline and "performance" in categories:
                self._generate_performance_baseline(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _build_pytest_command(
        self,
        categories: List[str],
        verbose: bool,
        coverage: bool,
        parallel: bool,
        output_format: str,
        max_workers: int
    ) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ["python3", "-m", "pytest"]
        
        # Add test paths
        test_paths = []
        for category in categories:
            if category in self.test_categories:
                test_paths.append(self.test_categories[category]["path"])
        
        cmd.extend(test_paths)
        
        # Add markers
        markers = []
        for category in categories:
            if category in self.test_categories:
                markers.extend(self.test_categories[category]["markers"])
        
        if markers:
            unique_markers = list(set(markers))
            if len(unique_markers) == 1:
                cmd.extend(["-m", unique_markers[0]])
            else:
                cmd.extend(["-m", " or ".join(unique_markers)])
        
        # Verbose output
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("--tb=short")
        
        # Coverage
        if coverage:
            cmd.extend([
                "--cov=src/agents/master",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        # Parallel execution
        if parallel:
            cmd.extend(["-n", str(max_workers)])
        
        # Output format
        if output_format == "junit":
            cmd.extend(["--junit-xml=test-results.xml"])
        elif output_format == "html":
            cmd.extend(["--html=test-report.html", "--self-contained-html"])
        
        # Performance settings (timeout plugins not available)
        # Note: Individual test timeouts can be set within test functions if needed
        
        # Additional pytest options
        cmd.extend([
            "--strict-markers",
            "--disable-warnings",
            "--color=yes"
        ])
        
        return cmd
    
    def _setup_test_environment(self) -> Dict[str, str]:
        """Setup test environment variables."""
        env = os.environ.copy()
        
        # Test mode settings
        env.update({
            "TEST_MODE": "true",
            "TESTING": "true",
            "PYTHONPATH": str(self.project_root),
            
            # Mock database settings
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-key",
            
            # Service ports
            "ARCHON_SERVER_PORT": "8181",
            "ARCHON_MCP_PORT": "8051", 
            "ARCHON_AGENTS_PORT": "8052",
            
            # Logging
            "LOG_LEVEL": "WARNING",
            "LOGFIRE_ENABLED": "false",
            
            # Performance settings
            "MAX_CONCURRENT_TESTS": "10",
            "TEST_TIMEOUT": "300"
        })
        
        return env
    
    def _estimate_total_time(self, categories: List[str]) -> str:
        """Estimate total execution time for given categories."""
        time_estimates = []
        
        for category in categories:
            if category in self.test_categories:
                estimate = self.test_categories[category]["estimated_time"]
                # Parse estimate (take max value)
                if "-" in estimate:
                    max_time = int(estimate.split("-")[1].split()[0])
                else:
                    max_time = int(estimate.split()[0])
                time_estimates.append(max_time)
        
        if not time_estimates:
            return "Unknown"
        
        total_max = sum(time_estimates)
        return f"{total_max//2}-{total_max} minutes"
    
    def _print_test_summary(self, results: Dict[str, Any], categories: List[str]):
        """Print test execution summary."""
        print("\n" + "=" * 50)
        print("🧠 Claude Flow Expert Agent Test Results")
        print("=" * 50)
        
        if results["success"]:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        print(f"Categories run: {', '.join(categories)}")
        print(f"Execution time: {results['execution_time']:.1f} seconds")
        print(f"Return code: {results['return_code']}")
        
        # Category-specific information
        print("\n📋 Test Categories:")
        for category in categories:
            if category in self.test_categories:
                info = self.test_categories[category]
                print(f"  • {category}: {info['description']}")
        
        print("\n💡 Next Steps:")
        if results["success"]:
            print("  • All Claude Flow Expert Agent functionality validated")
            print("  • System ready for integration testing")
            print("  • Consider running performance benchmarks")
        else:
            print("  • Review failed test output above")
            print("  • Check test logs for detailed error information") 
            print("  • Run specific test categories to isolate issues")
        
        print("\n📊 Reports Generated:")
        print("  • Terminal output (above)")
        if os.path.exists("htmlcov/index.html"):
            print("  • Coverage report: htmlcov/index.html")
        if os.path.exists("test-results.xml"):
            print("  • JUnit XML: test-results.xml")
        if os.path.exists("test-report.html"):
            print("  • HTML report: test-report.html")
    
    def _generate_performance_baseline(self, results: Dict[str, Any]):
        """Generate performance baseline report."""
        print("\n📈 Generating Performance Baseline...")
        
        baseline_file = self.test_dir / "performance_baseline.json"
        
        baseline_data = {
            "timestamp": time.time(),
            "execution_time": results["execution_time"],
            "categories": results["categories_run"],
            "success": results["success"],
            "baseline_metrics": {
                "max_query_latency_ms": 2000,
                "max_concurrent_throughput_qps": 10,
                "max_memory_usage_mb": 500,
                "min_success_rate": 0.95,
                "max_fallback_activation_time_ms": 3000
            },
            "system_config": {
                "python_version": sys.version,
                "platform": sys.platform,
                "test_environment": "mock"
            }
        }
        
        try:
            import json
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            print(f"  • Baseline saved to: {baseline_file}")
            print("  • Use this baseline for regression testing")
            
        except Exception as e:
            print(f"  • Failed to save baseline: {e}")
    
    def list_tests(self, category: str = None):
        """List available tests."""
        print("🧠 Claude Flow Expert Agent Test Suite - Available Tests")
        print("=" * 60)
        
        if category and category in self.test_categories:
            categories = [category]
        else:
            categories = list(self.test_categories.keys())
        
        for cat in categories:
            info = self.test_categories[cat]
            print(f"\n📁 {cat.upper()}")
            print(f"   Path: {info['path']}")
            print(f"   Description: {info['description']}")
            print(f"   Estimated time: {info['estimated_time']}")
            print(f"   Markers: {', '.join(info['markers'])}")
        
        print(f"\n📊 Total Categories: {len(self.test_categories)}")
        print(f"📋 Total estimated time: {self._estimate_total_time(categories)}")
    
    def validate_environment(self):
        """Validate test environment setup."""
        print("🔧 Validating Claude Flow Expert Agent Test Environment")
        print("=" * 50)
        
        checks = []
        
        # Check Python version
        if sys.version_info >= (3, 8):
            checks.append(("✅", f"Python version: {sys.version.split()[0]}"))
        else:
            checks.append(("❌", f"Python version too old: {sys.version.split()[0]} (need >=3.8)"))
        
        # Check required packages
        required_packages = ["pytest", "asyncio", "pydantic", "fastapi"]
        for package in required_packages:
            try:
                __import__(package)
                checks.append(("✅", f"Package available: {package}"))
            except ImportError:
                checks.append(("❌", f"Missing package: {package}"))
        
        # Check test files exist
        for category, info in self.test_categories.items():
            test_path = self.project_root / info["path"]
            if test_path.exists():
                checks.append(("✅", f"Test category: {category}"))
            else:
                checks.append(("❌", f"Missing tests: {category} ({test_path})"))
        
        # Check source code exists
        source_path = self.project_root / "src/agents/master"
        if source_path.exists():
            checks.append(("✅", f"Source code: {source_path}"))
        else:
            checks.append(("❌", f"Missing source: {source_path}"))
        
        # Print results
        for status, message in checks:
            print(f"  {status} {message}")
        
        # Summary
        passed = len([c for c in checks if c[0] == "✅"])
        total = len(checks)
        
        print(f"\n📊 Environment Check: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 Environment ready for testing!")
            return True
        else:
            print("⚠️  Please fix environment issues before running tests")
            return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Claude Flow Expert Agent Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_claude_flow_expert_agent_tests.py --all
  
  # Run specific categories
  python run_claude_flow_expert_agent_tests.py --categories unit integration
  
  # Run with coverage and parallel execution
  python run_claude_flow_expert_agent_tests.py --all --coverage --parallel
  
  # Performance testing with baseline
  python run_claude_flow_expert_agent_tests.py --categories performance --baseline
  
  # List available tests
  python run_claude_flow_expert_agent_tests.py --list
  
  # Validate environment
  python run_claude_flow_expert_agent_tests.py --validate
        """
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["unit", "integration", "performance", "e2e", "fallback"],
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test categories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Generate performance baseline"
    )
    
    parser.add_argument(
        "--output",
        choices=["terminal", "junit", "html"],
        default="terminal",
        help="Output format"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate test environment"
    )
    
    args = parser.parse_args()
    
    runner = ClaudeFlowExpertAgentTestRunner()
    
    # Handle special commands
    if args.list:
        runner.list_tests()
        return
    
    if args.validate:
        if runner.validate_environment():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Determine categories to run
    if args.all:
        categories = list(runner.test_categories.keys())
    elif args.categories:
        categories = args.categories
    else:
        print("Please specify --all or --categories")
        parser.print_help()
        sys.exit(1)
    
    # Validate environment before running tests
    if not runner.validate_environment():
        print("\n⚠️  Environment validation failed. Please fix issues before running tests.")
        sys.exit(1)
    
    # Run tests
    results = runner.run_tests(
        categories=categories,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        performance_baseline=args.baseline,
        output_format=args.output,
        max_workers=args.workers
    )
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()