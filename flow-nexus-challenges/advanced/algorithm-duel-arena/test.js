// Algorithm Duel Arena Challenge Test Suite
// Comprehensive testing for algorithm performance comparison system

import { AlgorithmDuelArena, executeAlgorithmDuelArena } from './solution.js';

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
    console.log("🚀 Starting Algorithm Duel Arena Test Suite...");
    
    // Core functionality tests
    await this.runTest("Algorithm Registration", this.testAlgorithmRegistration.bind(this));
    await this.runTest("Algorithm Execution", this.testAlgorithmExecution.bind(this));
    await this.runTest("Performance Calculation", this.testPerformanceCalculation.bind(this));
    await this.runTest("Duel System", this.testDuelSystem.bind(this));
    await this.runTest("ELO Rating System", this.testELORatingSystem.bind(this));
    await this.runTest("Leaderboard Generation", this.testLeaderboardGeneration.bind(this));
    await this.runTest("Tournament System", this.testTournamentSystem.bind(this));
    await this.runTest("Analytics Generation", this.testAnalyticsGeneration.bind(this));
    
    // Edge case tests
    await this.runTest("Error Handling", this.testErrorHandling.bind(this));
    await this.runTest("Large Dataset Performance", this.testLargeDatasetPerformance.bind(this));
    await this.runTest("Memory Usage", this.testMemoryUsage.bind(this));
    
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

  // Test algorithm registration
  async testAlgorithmRegistration() {
    const arena = new AlgorithmDuelArena();
    
    const testAlgorithm = async (data) => data * 2;
    const algorithm = arena.registerAlgorithm("Test Algorithm", testAlgorithm, "test");
    
    if (!algorithm || algorithm.name !== "Test Algorithm") {
      throw new Error("Algorithm registration failed");
    }
    
    if (arena.algorithms.size !== 1) {
      throw new Error("Algorithm not stored in arena");
    }
  }

  // Test algorithm execution
  async testAlgorithmExecution() {
    const arena = new AlgorithmDuelArena();
    
    const testAlgorithm = async (data) => {
      await new Promise(resolve => setTimeout(resolve, 10)); // Simulate work
      return data * 2;
    };
    
    arena.registerAlgorithm("Test Algorithm", testAlgorithm, "test");
    
    const result = await arena.executeAlgorithm("Test Algorithm", 5);
    
    if (!result.success || result.result !== 10) {
      throw new Error("Algorithm execution failed");
    }
    
    if (result.executionTime < 0) {
      throw new Error("Execution time not measured");
    }
  }

  // Test performance calculation
  async testPerformanceCalculation() {
    const arena = new AlgorithmDuelArena();
    
    const executionData = {
      result: 10,
      executionTime: 50,
      memoryUsage: 1024,
      success: true
    };
    
    const score = arena.calculatePerformanceScore(executionData, 10, {});
    
    if (score.accuracy !== 100) {
      throw new Error("Accuracy calculation incorrect");
    }
    
    if (score.score <= 0) {
      throw new Error("Performance score calculation failed");
    }
  }

  // Test duel system
  async testDuelSystem() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => data * 2;
    const algorithm2 = async (data) => data * 3;
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    
    const testCases = [
      {
        description: "Test case 1",
        input: 5,
        expected: 10
      }
    ];
    
    const duelResult = await arena.duel("Algorithm 1", "Algorithm 2", testCases);
    
    if (!duelResult || !duelResult.algorithm1 || !duelResult.algorithm2) {
      throw new Error("Duel system failed");
    }
    
    if (duelResult.testCases !== 1) {
      throw new Error("Test cases not processed correctly");
    }
  }

  // Test ELO rating system
  async testELORatingSystem() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => data * 2;
    const algorithm2 = async (data) => data * 2;
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    
    const initialRating1 = arena.algorithms.get("Algorithm 1").rating;
    const initialRating2 = arena.algorithms.get("Algorithm 2").rating;
    
    const testCases = [
      {
        description: "Test case 1",
        input: 5,
        expected: 10
      }
    ];
    
    await arena.duel("Algorithm 1", "Algorithm 2", testCases);
    
    const finalRating1 = arena.algorithms.get("Algorithm 1").rating;
    const finalRating2 = arena.algorithms.get("Algorithm 2").rating;
    
    // Ratings should have changed (even if slightly)
    if (initialRating1 === finalRating1 && initialRating2 === finalRating2) {
      throw new Error("ELO ratings not updated");
    }
  }

  // Test leaderboard generation
  async testLeaderboardGeneration() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => data * 2;
    const algorithm2 = async (data) => data * 3;
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    
    const leaderboard = arena.generateLeaderboard();
    
    if (leaderboard.length !== 2) {
      throw new Error("Leaderboard not generated correctly");
    }
    
    if (leaderboard[0].rank !== 1 || leaderboard[1].rank !== 2) {
      throw new Error("Leaderboard ranking incorrect");
    }
  }

  // Test tournament system
  async testTournamentSystem() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => data * 2;
    const algorithm2 = async (data) => data * 3;
    const algorithm3 = async (data) => data * 4;
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    arena.registerAlgorithm("Algorithm 3", algorithm3, "test");
    
    const testCases = [
      {
        description: "Test case 1",
        input: 5,
        expected: 10
      }
    ];
    
    const tournament = await arena.runTournament(
      ["Algorithm 1", "Algorithm 2", "Algorithm 3"],
      testCases,
      "round-robin"
    );
    
    if (!tournament || tournament.participants.length !== 3) {
      throw new Error("Tournament system failed");
    }
    
    if (tournament.matches.length < 3) {
      throw new Error("Not all matches played in round-robin");
    }
  }

  // Test analytics generation
  async testAnalyticsGeneration() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => data * 2;
    const algorithm2 = async (data) => data * 3;
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    
    const analytics = arena.generateAnalytics();
    
    if (analytics.totalAlgorithms !== 2) {
      throw new Error("Analytics total algorithms incorrect");
    }
    
    if (!analytics.categoryBreakdown.test) {
      throw new Error("Category breakdown missing");
    }
  }

  // Test error handling
  async testErrorHandling() {
    const arena = new AlgorithmDuelArena();
    
    // Test non-existent algorithm
    try {
      await arena.executeAlgorithm("Non-existent", 5);
      throw new Error("Should have thrown error for non-existent algorithm");
    } catch (error) {
      if (!error.message.includes("not found")) {
        throw new Error("Wrong error message for non-existent algorithm");
      }
    }
    
    // Test algorithm that throws error
    const errorAlgorithm = async (data) => {
      throw new Error("Test error");
    };
    
    arena.registerAlgorithm("Error Algorithm", errorAlgorithm, "test");
    
    const result = await arena.executeAlgorithm("Error Algorithm", 5);
    
    if (result.success) {
      throw new Error("Should have caught algorithm error");
    }
    
    if (!result.error) {
      throw new Error("Error not captured");
    }
  }

  // Test large dataset performance
  async testLargeDatasetPerformance() {
    const arena = new AlgorithmDuelArena();
    
    const algorithm1 = async (data) => {
      return data.sort((a, b) => a - b);
    };
    
    const algorithm2 = async (data) => {
      return [...data].sort((a, b) => a - b);
    };
    
    arena.registerAlgorithm("Algorithm 1", algorithm1, "test");
    arena.registerAlgorithm("Algorithm 2", algorithm2, "test");
    
    const largeArray = Array.from({length: 10000}, () => Math.floor(Math.random() * 10000));
    const expected = [...largeArray].sort((a, b) => a - b);
    
    const testCases = [
      {
        description: "Large array test",
        input: largeArray,
        expected: expected
      }
    ];
    
    const startTime = Date.now();
    const duelResult = await arena.duel("Algorithm 1", "Algorithm 2", testCases);
    const endTime = Date.now();
    
    if (endTime - startTime > 10000) { // 10 seconds
      throw new Error("Large dataset test took too long");
    }
    
    if (!duelResult || duelResult.results.length !== 1) {
      throw new Error("Large dataset duel failed");
    }
  }

  // Test memory usage
  async testMemoryUsage() {
    const arena = new AlgorithmDuelArena();
    
    const memoryIntensiveAlgorithm = async (data) => {
      const largeArray = new Array(1000000).fill(0);
      return largeArray.length;
    };
    
    arena.registerAlgorithm("Memory Intensive", memoryIntensiveAlgorithm, "test");
    
    const result = await arena.executeAlgorithm("Memory Intensive", 5);
    
    if (!result.success) {
      throw new Error("Memory intensive algorithm failed");
    }
    
    if (result.memoryUsage <= 0) {
      throw new Error("Memory usage not measured");
    }
  }

  // Test full challenge execution
  async testFullChallengeExecution() {
    const result = await executeAlgorithmDuelArena();
    
    if (!result || result.status !== "completed") {
      throw new Error("Full challenge execution failed");
    }
    
    if (!result.arena || !result.tournament) {
      throw new Error("Challenge result missing required fields");
    }
    
    if (result.arena.totalAlgorithms < 4) {
      throw new Error("Not enough algorithms registered");
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
  
  const arena = new AlgorithmDuelArena();
  const testAlgorithm = async (data) => data * 2;
  arena.registerAlgorithm("Test Algorithm", testAlgorithm, "test");
  
  await benchmark.runBenchmark("Algorithm Execution", async () => {
    await arena.executeAlgorithm("Test Algorithm", 5);
  }, 1000);
  
  await benchmark.runBenchmark("Duel System", async () => {
    const testCases = [{ description: "Test", input: 5, expected: 10 }];
    await arena.duel("Test Algorithm", "Test Algorithm", testCases);
  }, 100);
  
  benchmark.printSummary();
  
  console.log("\n🎉 All tests completed!");
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(console.error);
}

export { TestRunner, PerformanceBenchmark, runTests };
