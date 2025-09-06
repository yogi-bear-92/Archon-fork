// Bug Hunter's Gauntlet Challenge Test Suite
// Comprehensive testing for advanced debugging and error resolution system

import { BugHuntersGauntlet, executeBugHuntersGauntlet } from './solution.js';

// Test utilities
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  async runTest(name, testFunction) {
    try {
      console.log(`🧪 Running test: ${name}`);
      await testFunction();
      console.log(`✅ PASSED: ${name}`);
      this.passed++;
    } catch (error) {
      console.log(`❌ FAILED: ${name} - ${error.message}`);
      this.failed++;
    }
  }

  async runAllTests() {
    console.log("🚀 Starting Bug Hunter's Gauntlet Test Suite...");
    
    // Core functionality tests
    await this.runTest("Tool Registration", this.testToolRegistration.bind(this));
    await this.runTest("Bug Report Creation", this.testBugReportCreation.bind(this));
    await this.runTest("Priority Calculation", this.testPriorityCalculation.bind(this));
    await this.runTest("Bug Analysis", this.testBugAnalysis.bind(this));
    await this.runTest("Root Cause Analysis", this.testRootCauseAnalysis.bind(this));
    await this.runTest("Solution Development", this.testSolutionDevelopment.bind(this));
    await this.runTest("Solution Validation", this.testSolutionValidation.bind(this));
    await this.runTest("Debugging Process", this.testDebuggingProcess.bind(this));
    await this.runTest("Performance Metrics", this.testPerformanceMetrics.bind(this));
    await this.runTest("Report Generation", this.testReportGeneration.bind(this));
    
    // Edge case tests
    await this.runTest("Error Handling", this.testErrorHandling.bind(this));
    await this.runTest("Complex Bug Scenarios", this.testComplexBugScenarios.bind(this));
    await this.runTest("Multiple Bug Processing", this.testMultipleBugProcessing.bind(this));
    
    // Integration tests
    await this.runTest("Full Challenge Execution", this.testFullChallengeExecution.bind(this));
    
    this.printSummary();
  }

  printSummary() {
    console.log("\n📊 Test Summary:");
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`❌ Failed: ${this.failed}`);
    console.log(`📈 Success Rate: ${Math.round((this.passed / (this.passed + this.failed)) * 100)}%`);
  }

  // Test tool registration
  async testToolRegistration() {
    const gauntlet = new BugHuntersGauntlet();
    
    const testTool = async (bug) => ({ success: true, findings: ["Test finding"] });
    const tool = gauntlet.registerDebuggingTool("Test Tool", testTool, "test");
    
    if (!tool || tool.name !== "Test Tool") {
      throw new Error("Tool registration failed");
    }
    
    if (gauntlet.debuggingTools.size !== 1) {
      throw new Error("Tool not stored in gauntlet");
    }
  }

  // Test bug report creation
  async testBugReportCreation() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "high",
      category: "test"
    };
    
    const bug = gauntlet.createBugReport(bugData);
    
    if (!bug || bug.title !== "Test Bug") {
      throw new Error("Bug report creation failed");
    }
    
    if (bug.status !== "open") {
      throw new Error("Bug status not set correctly");
    }
    
    if (bug.priority <= 0) {
      throw new Error("Priority not calculated");
    }
  }

  // Test priority calculation
  async testPriorityCalculation() {
    const gauntlet = new BugHuntersGauntlet();
    
    const criticalBug = {
      title: "Critical Bug",
      severity: "critical",
      category: "security",
      impact: 2
    };
    
    const lowBug = {
      title: "Low Bug",
      severity: "low",
      category: "ui",
      impact: 1
    };
    
    const criticalPriority = gauntlet.calculatePriority(criticalBug);
    const lowPriority = gauntlet.calculatePriority(lowBug);
    
    if (criticalPriority <= lowPriority) {
      throw new Error("Priority calculation incorrect");
    }
    
    if (criticalPriority > 10 || lowPriority < 1) {
      throw new Error("Priority out of valid range");
    }
  }

  // Test bug analysis
  async testBugAnalysis() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "high",
      category: "performance",
      stackTrace: "Error at line 1\nError at line 2",
      reproductionSteps: ["Step 1", "Step 2"],
      environment: { browser: "Chrome" }
    };
    
    const bug = gauntlet.createBugReport(bugData);
    const analysis = gauntlet.analyzeBug(bug.id);
    
    if (!analysis || analysis.bugId !== bug.id) {
      throw new Error("Bug analysis failed");
    }
    
    if (analysis.complexity <= 0 || analysis.complexity > 10) {
      throw new Error("Complexity calculation incorrect");
    }
    
    if (!analysis.suggestedTools || analysis.suggestedTools.length === 0) {
      throw new Error("No debugging tools suggested");
    }
  }

  // Test root cause analysis
  async testRootCauseAnalysis() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "medium",
      category: "functionality"
    };
    
    const bug = gauntlet.createBugReport(bugData);
    const investigation = { findings: ["Test finding"] };
    const rootCause = gauntlet.analyzeRootCause(bug, investigation);
    
    if (!rootCause || !rootCause.identified) {
      throw new Error("Root cause analysis failed");
    }
    
    if (!rootCause.category || !rootCause.description) {
      throw new Error("Root cause details missing");
    }
    
    if (rootCause.confidence < 0 || rootCause.confidence > 1) {
      throw new Error("Confidence level invalid");
    }
  }

  // Test solution development
  async testSolutionDevelopment() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "high",
      category: "performance"
    };
    
    const bug = gauntlet.createBugReport(bugData);
    const rootCause = {
      category: "Resource Constraint",
      description: "Insufficient resources"
    };
    
    const solution = gauntlet.developSolution(bug, rootCause);
    
    if (!solution || !solution.approach) {
      throw new Error("Solution development failed");
    }
    
    if (!solution.implementation || !solution.testingStrategy) {
      throw new Error("Solution details missing");
    }
    
    if (!solution.rollbackPlan || !solution.estimatedImpact) {
      throw new Error("Solution planning incomplete");
    }
  }

  // Test solution validation
  async testSolutionValidation() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "medium",
      category: "functionality"
    };
    
    const bug = gauntlet.createBugReport(bugData);
    const solution = {
      approach: "Test approach",
      implementation: { codeChanges: ["Test change"] }
    };
    
    const validation = await gauntlet.validateSolution(bug, solution);
    
    if (typeof validation.success !== "boolean") {
      throw new Error("Validation success not boolean");
    }
    
    if (validation.testsPassed < 0 || validation.totalTests <= 0) {
      throw new Error("Test counts invalid");
    }
    
    if (!validation.performanceImpact || !validation.securityImpact) {
      throw new Error("Impact assessments missing");
    }
  }

  // Test debugging process
  async testDebuggingProcess() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugData = {
      title: "Test Bug",
      description: "Test description",
      severity: "high",
      category: "functionality"
    };
    
    const bug = gauntlet.createBugReport(bugData);
    const session = await gauntlet.executeDebugging(bug.id);
    
    if (!session || session.bugId !== bug.id) {
      throw new Error("Debugging session failed");
    }
    
    if (!session.startTime || !session.endTime) {
      throw new Error("Session timing missing");
    }
    
    if (session.duration <= 0) {
      throw new Error("Session duration invalid");
    }
    
    if (!Array.isArray(session.steps)) {
      throw new Error("Session steps not array");
    }
  }

  // Test performance metrics
  async testPerformanceMetrics() {
    const gauntlet = new BugHuntersGauntlet();
    
    // Create and resolve some bugs
    const bug1 = gauntlet.createBugReport({
      title: "Bug 1",
      description: "Test",
      severity: "critical"
    });
    
    const bug2 = gauntlet.createBugReport({
      title: "Bug 2", 
      description: "Test",
      severity: "high"
    });
    
    await gauntlet.executeDebugging(bug1.id);
    await gauntlet.executeDebugging(bug2.id);
    
    const metrics = gauntlet.performanceMetrics;
    
    if (metrics.bugsResolved < 0) {
      throw new Error("Bugs resolved count invalid");
    }
    
    if (metrics.averageResolutionTime < 0) {
      throw new Error("Average resolution time invalid");
    }
    
    if (metrics.successRate < 0 || metrics.successRate > 100) {
      throw new Error("Success rate invalid");
    }
  }

  // Test report generation
  async testReportGeneration() {
    const gauntlet = new BugHuntersGauntlet();
    
    // Create some test data
    gauntlet.createBugReport({
      title: "Test Bug 1",
      description: "Test",
      severity: "high",
      category: "performance"
    });
    
    gauntlet.createBugReport({
      title: "Test Bug 2",
      description: "Test", 
      severity: "critical",
      category: "security"
    });
    
    const report = gauntlet.generateReport();
    
    if (!report || !report.summary) {
      throw new Error("Report generation failed");
    }
    
    if (report.summary.totalBugs !== 2) {
      throw new Error("Total bugs count incorrect");
    }
    
    if (!report.topTools || !report.bugCategories) {
      throw new Error("Report sections missing");
    }
  }

  // Test error handling
  async testErrorHandling() {
    const gauntlet = new BugHuntersGauntlet();
    
    // Test non-existent bug
    try {
      await gauntlet.executeDebugging("non-existent-bug");
      throw new Error("Should have thrown error for non-existent bug");
    } catch (error) {
      if (!error.message.includes("not found")) {
        throw new Error("Wrong error message for non-existent bug");
      }
    }
    
    // Test invalid bug analysis
    try {
      gauntlet.analyzeBug("non-existent-bug");
      throw new Error("Should have thrown error for non-existent bug analysis");
    } catch (error) {
      if (!error.message.includes("not found")) {
        throw new Error("Wrong error message for non-existent bug analysis");
      }
    }
  }

  // Test complex bug scenarios
  async testComplexBugScenarios() {
    const gauntlet = new BugHuntersGauntlet();
    
    const complexBug = gauntlet.createBugReport({
      title: "Complex Bug",
      description: "Very complex bug with multiple issues",
      severity: "critical",
      category: "security",
      stackTrace: "Error at line 1\nError at line 2\nError at line 3\nError at line 4\nError at line 5",
      reproductionSteps: ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
      environment: { browser: "Chrome", database: "PostgreSQL", server: "AWS" },
      tags: ["production", "security", "user-facing"]
    });
    
    const analysis = gauntlet.analyzeBug(complexBug.id);
    
    if (analysis.complexity < 5) {
      throw new Error("Complex bug not assessed correctly");
    }
    
    if (analysis.riskAssessment < 7) {
      throw new Error("Risk assessment too low for critical security bug");
    }
    
    if (analysis.suggestedTools.length < 3) {
      throw new Error("Not enough tools suggested for complex bug");
    }
  }

  // Test multiple bug processing
  async testMultipleBugProcessing() {
    const gauntlet = new BugHuntersGauntlet();
    
    const bugs = [];
    for (let i = 0; i < 5; i++) {
      const bug = gauntlet.createBugReport({
        title: `Bug ${i + 1}`,
        description: `Test bug ${i + 1}`,
        severity: i % 2 === 0 ? "high" : "medium",
        category: i % 3 === 0 ? "performance" : "functionality"
      });
      bugs.push(bug);
    }
    
    // Process all bugs
    for (const bug of bugs) {
      await gauntlet.executeDebugging(bug.id);
    }
    
    if (gauntlet.bugReports.size !== 5) {
      throw new Error("Not all bugs created");
    }
    
    if (gauntlet.resolutionHistory.length !== 5) {
      throw new Error("Not all bugs processed");
    }
    
    const report = gauntlet.generateReport();
    if (report.summary.totalBugs !== 5) {
      throw new Error("Report shows incorrect bug count");
    }
  }

  // Test full challenge execution
  async testFullChallengeExecution() {
    const result = await executeBugHuntersGauntlet();
    
    if (!result || result.status !== "completed") {
      throw new Error("Full challenge execution failed");
    }
    
    if (!result.gauntlet || !result.report) {
      throw new Error("Challenge result missing required fields");
    }
    
    if (result.gauntlet.totalBugs < 4) {
      throw new Error("Not enough bugs processed");
    }
    
    if (result.performance.bugsProcessed < 4) {
      throw new Error("Performance metrics incorrect");
    }
  }
}

// Performance benchmarks
class PerformanceBenchmark {
  constructor() {
    this.benchmarks = [];
  }

  async runBenchmark(name, testFunction, iterations = 100) {
    const times = [];
    
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await testFunction();
      const end = performance.now();
      times.push(end - start);
    }
    
    const average = times.reduce((sum, time) => sum + time, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    
    this.benchmarks.push({
      name,
      average: Math.round(average * 100) / 100,
      min: Math.round(min * 100) / 100,
      max: Math.round(max * 100) / 100,
      iterations
    });
    
    console.log(`📊 Benchmark ${name}: ${average.toFixed(2)}ms average (${min.toFixed(2)}-${max.toFixed(2)}ms)`);
  }

  printSummary() {
    console.log("\n📈 Performance Benchmarks:");
    this.benchmarks.forEach(benchmark => {
      console.log(`${benchmark.name}: ${benchmark.average}ms avg (${benchmark.min}-${benchmark.max}ms) over ${benchmark.iterations} iterations`);
    });
  }
}

// Main test execution
async function runTests() {
  const testRunner = new TestRunner();
  const benchmark = new PerformanceBenchmark();
  
  // Run all tests
  await testRunner.runAllTests();
  
  // Run performance benchmarks
  console.log("\n🏃 Running Performance Benchmarks...");
  
  const gauntlet = new BugHuntersGauntlet();
  const testBug = gauntlet.createBugReport({
    title: "Test Bug",
    description: "Test",
    severity: "medium",
    category: "test"
  });
  
  await benchmark.runBenchmark("Bug Analysis", async () => {
    gauntlet.analyzeBug(testBug.id);
  }, 1000);
  
  await benchmark.runBenchmark("Debugging Process", async () => {
    const newBug = gauntlet.createBugReport({
      title: "Benchmark Bug",
      description: "Test",
      severity: "low",
      category: "test"
    });
    await gauntlet.executeDebugging(newBug.id);
  }, 100);
  
  benchmark.printSummary();
  
  console.log("\n🎉 All tests completed!");
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(console.error);
}

export { TestRunner, PerformanceBenchmark, runTests };
