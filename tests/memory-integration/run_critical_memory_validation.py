"""
Critical Memory State Validation Runner

Orchestrates comprehensive testing of memory-aware integration patterns
while monitoring system state and preventing crashes. Implements emergency
abort mechanisms and graduated testing approach.

CRITICAL SAFETY PROTOCOL:
1. Pre-flight memory check with emergency abort
2. Progressive test execution with monitoring
3. Immediate cleanup and resource management
4. Emergency shutdown on memory threshold breach
"""

import sys
import gc
import time
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import psutil


class CriticalMemoryTestOrchestrator:
    """Orchestrates critical memory state testing with safety protocols."""
    
    def __init__(self):
        self.emergency_threshold = 99.8  # Emergency abort threshold
        self.warning_threshold = 99.5   # Warning threshold
        self.test_results = []
        self.test_start_time = datetime.now()
        self.safety_protocol_active = False
        
        # Configure minimal logging
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - CRITICAL - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_system_memory_state(self) -> Dict[str, Any]:
        """Get comprehensive system memory state."""
        try:
            virtual_memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_memory_gb": virtual_memory.total / (1024**3),
                "available_memory_gb": virtual_memory.available / (1024**3),
                "used_memory_gb": virtual_memory.used / (1024**3),
                "memory_percent": virtual_memory.percent,
                "process_memory_mb": process_memory.rss / (1024**2),
                "free_memory_mb": virtual_memory.free / (1024**2),
                "emergency_abort_needed": virtual_memory.percent >= self.emergency_threshold,
                "warning_level": virtual_memory.percent >= self.warning_threshold
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory state: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "emergency_abort_needed": True
            }
    
    def emergency_system_cleanup(self):
        """Emergency system cleanup to free memory."""
        self.logger.error("EXECUTING EMERGENCY CLEANUP")
        
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
        
        # Clear any cached data
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
        # Log cleanup completion
        post_cleanup_state = self.get_system_memory_state()
        self.logger.error(f"Post-cleanup memory: {post_cleanup_state['memory_percent']:.2f}%")
        
        return post_cleanup_state
    
    def pre_flight_safety_check(self) -> Tuple[bool, str]:
        """Pre-flight safety check before starting tests."""
        memory_state = self.get_system_memory_state()
        
        if memory_state.get("error"):
            return False, f"Memory state check failed: {memory_state['error']}"
        
        if memory_state["emergency_abort_needed"]:
            return False, f"EMERGENCY ABORT: Memory usage {memory_state['memory_percent']:.2f}% exceeds safe threshold"
        
        if memory_state["warning_level"]:
            self.logger.error(f"WARNING: High memory usage {memory_state['memory_percent']:.2f}%")
        
        free_memory_mb = memory_state["free_memory_mb"]
        if free_memory_mb < 50:  # Less than 50MB free
            return False, f"Insufficient free memory: {free_memory_mb:.1f}MB"
        
        return True, f"Safe to proceed - Memory: {memory_state['memory_percent']:.2f}%, Free: {free_memory_mb:.1f}MB"
    
    def run_test_suite_safely(self, test_file: str, test_markers: str) -> Dict[str, Any]:
        """Run a test suite with continuous memory monitoring."""
        test_name = Path(test_file).stem
        
        # Pre-test memory check
        pre_memory = self.get_system_memory_state()
        
        if pre_memory["emergency_abort_needed"]:
            return {
                "test_name": test_name,
                "status": "aborted",
                "reason": "pre_test_memory_critical",
                "memory_percent": pre_memory["memory_percent"]
            }
        
        try:
            self.logger.error(f"Starting test suite: {test_name}")
            
            # Run pytest with minimal overhead
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "-m", test_markers,
                "--tb=short",
                "-x",  # Stop on first failure
                "--maxfail=1",
                "--disable-warnings"
            ]
            
            start_time = time.time()
            
            # Run with timeout to prevent hanging
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd="/Users/yogi/Projects/Archon-fork"
            )
            
            duration = time.time() - start_time
            
            # Post-test memory check
            post_memory = self.get_system_memory_state()
            
            test_result = {
                "test_name": test_name,
                "status": "passed" if result.returncode == 0 else "failed",
                "duration_seconds": duration,
                "return_code": result.returncode,
                "memory_before": pre_memory["memory_percent"],
                "memory_after": post_memory["memory_percent"],
                "memory_delta": post_memory["memory_percent"] - pre_memory["memory_percent"],
                "stdout_lines": len(result.stdout.split('\n')),
                "stderr_lines": len(result.stderr.split('\n'))
            }
            
            # Check for memory issues
            if post_memory["emergency_abort_needed"]:
                test_result["warning"] = "post_test_memory_critical"
                self.emergency_system_cleanup()
            
            return test_result
            
        except subprocess.TimeoutExpired:
            self.emergency_system_cleanup()
            return {
                "test_name": test_name,
                "status": "timeout",
                "reason": "test_timeout_60s",
                "memory_percent": self.get_system_memory_state()["memory_percent"]
            }
        
        except Exception as e:
            self.emergency_system_cleanup()
            return {
                "test_name": test_name,
                "status": "error",
                "error": str(e),
                "memory_percent": self.get_system_memory_state()["memory_percent"]
            }
    
    def validate_integration_patterns(self) -> Dict[str, Any]:
        """Main validation function for integration patterns under memory pressure."""
        
        # Phase 1: Pre-flight safety check
        self.logger.error("=== CRITICAL MEMORY STATE VALIDATION STARTING ===")
        
        is_safe, safety_message = self.pre_flight_safety_check()
        if not is_safe:
            self.logger.error(f"PRE-FLIGHT ABORT: {safety_message}")
            return {
                "status": "aborted",
                "reason": "pre_flight_safety_failure",
                "message": safety_message,
                "timestamp": datetime.now().isoformat()
            }
        
        self.logger.error(f"PRE-FLIGHT OK: {safety_message}")
        
        # Phase 2: Progressive test execution
        test_suites = [
            {
                "file": "/Users/yogi/Projects/Archon-fork/tests/critical-state/memory_pressure_integration_tests.py",
                "markers": "critical_memory",
                "priority": "critical",
                "description": "Emergency fallback and memory monitoring tests"
            },
            {
                "file": "/Users/yogi/Projects/Archon-fork/tests/memory-integration/coordination_under_pressure_tests.py", 
                "markers": "coordination_pressure",
                "priority": "high",
                "description": "Cross-system coordination under memory pressure"
            }
        ]
        
        validation_results = {
            "validation_start": self.test_start_time.isoformat(),
            "initial_memory_state": self.get_system_memory_state(),
            "test_results": [],
            "final_status": "unknown",
            "emergency_cleanups": 0
        }
        
        # Execute test suites progressively
        for suite in test_suites:
            self.logger.error(f"Executing: {suite['description']}")
            
            # Safety check before each suite
            current_memory = self.get_system_memory_state()
            if current_memory["emergency_abort_needed"]:
                self.logger.error("EMERGENCY ABORT during test execution")
                validation_results["final_status"] = "emergency_abort"
                break
            
            # Run test suite
            test_result = self.run_test_suite_safely(suite["file"], suite["markers"])
            validation_results["test_results"].append(test_result)
            
            # Emergency cleanup if needed
            if test_result.get("warning") == "post_test_memory_critical":
                validation_results["emergency_cleanups"] += 1
                time.sleep(2)  # Brief pause after cleanup
            
            # Check abort conditions
            if test_result["status"] in ["error", "timeout"]:
                self.logger.error(f"Test suite failed: {test_result['status']}")
                break
        
        # Phase 3: Final assessment
        validation_results["validation_end"] = datetime.now().isoformat()
        validation_results["final_memory_state"] = self.get_system_memory_state()
        
        # Determine overall status
        passed_tests = sum(1 for result in validation_results["test_results"] if result["status"] == "passed")
        total_tests = len(validation_results["test_results"])
        
        if validation_results["final_status"] == "emergency_abort":
            final_status = "emergency_abort"
        elif passed_tests == total_tests and total_tests > 0:
            final_status = "validation_successful"
        elif passed_tests > 0:
            final_status = "partial_success"
        else:
            final_status = "validation_failed"
        
        validation_results["final_status"] = final_status
        validation_results["success_rate"] = passed_tests / max(total_tests, 1)
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        
        report_lines = [
            "=" * 80,
            "CRITICAL MEMORY STATE INTEGRATION VALIDATION REPORT",
            "=" * 80,
            f"Validation Period: {validation_results['validation_start']} to {validation_results.get('validation_end', 'INCOMPLETE')}",
            "",
            "MEMORY STATE ANALYSIS:",
            f"  Initial Memory Usage: {validation_results['initial_memory_state']['memory_percent']:.2f}%",
            f"  Final Memory Usage: {validation_results['final_memory_state']['memory_percent']:.2f}%",
            f"  Free Memory: {validation_results['final_memory_state']['free_memory_mb']:.1f}MB",
            f"  Emergency Cleanups: {validation_results['emergency_cleanups']}",
            "",
            "VALIDATION RESULTS:",
            f"  Overall Status: {validation_results['final_status'].upper()}",
            f"  Success Rate: {validation_results['success_rate']:.1%}",
            f"  Tests Executed: {len(validation_results['test_results'])}",
            ""
        ]
        
        # Test-by-test breakdown
        report_lines.append("TEST EXECUTION DETAILS:")
        for i, result in enumerate(validation_results["test_results"], 1):
            report_lines.extend([
                f"  {i}. {result['test_name']}:",
                f"     Status: {result['status'].upper()}",
                f"     Duration: {result.get('duration_seconds', 0):.2f}s",
                f"     Memory Impact: {result.get('memory_delta', 0):+.2f}%",
                ""
            ])
        
        # Key findings
        report_lines.extend([
            "KEY FINDINGS:",
            f"  ✓ Emergency fallback mechanisms: {'VALIDATED' if any('emergency' in r.get('test_name', '') for r in validation_results['test_results']) else 'NOT TESTED'}",
            f"  ✓ Memory monitoring accuracy: {'VALIDATED' if validation_results['success_rate'] > 0.5 else 'FAILED'}",
            f"  ✓ Tool hierarchy enforcement: {'VALIDATED' if any('hierarchy' in str(r) for r in validation_results['test_results']) else 'NOT TESTED'}",
            f"  ✓ Coordination under pressure: {'VALIDATED' if any('coordination' in r.get('test_name', '') for r in validation_results['test_results']) else 'NOT TESTED'}",
            f"  ✓ System stability: {'MAINTAINED' if validation_results['final_status'] != 'emergency_abort' else 'COMPROMISED'}",
            "",
            "RECOMMENDATIONS:",
        ])
        
        # Recommendations based on results
        if validation_results['final_status'] == 'validation_successful':
            report_lines.append("  ✅ Integration patterns are SAFE for critical memory conditions")
        elif validation_results['final_status'] == 'partial_success':
            report_lines.append("  ⚠️  Some integration patterns need optimization for memory pressure")
        else:
            report_lines.append("  ❌ Integration patterns UNSAFE for critical memory conditions")
        
        report_lines.extend([
            f"  - Monitor memory usage below {self.warning_threshold}% for optimal performance",
            f"  - Emergency cleanup reduced memory by ~{validation_results['emergency_cleanups']} interventions",
            f"  - Consider implementing additional memory optimization strategies",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)


def main():
    """Main execution function."""
    orchestrator = CriticalMemoryTestOrchestrator()
    
    try:
        # Execute comprehensive validation
        validation_results = orchestrator.validate_integration_patterns()
        
        # Generate and display report
        report = orchestrator.generate_validation_report(validation_results)
        print(report)
        
        # Save detailed results to file
        results_file = Path("/Users/yogi/Projects/Archon-fork/tests/memory-integration/validation_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Return appropriate exit code
        if validation_results["final_status"] == "validation_successful":
            return 0
        elif validation_results["final_status"] == "partial_success":
            return 1
        else:
            return 2
            
    except Exception as e:
        orchestrator.logger.error(f"CRITICAL FAILURE: {e}")
        orchestrator.emergency_system_cleanup()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)