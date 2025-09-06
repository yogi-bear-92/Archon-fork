// The Neural Conductor Challenge Test Suite
// Comprehensive testing for multi-agent orchestration and emergent intelligence

import { NeuralConductor, executeNeuralConductor } from './solution.js';

class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
    this.startTime = Date.now();
  }

  addTest(name, testFn) {
    this.tests.push({ name, testFn });
  }

  async runTests() {
    console.log("🧪 Running Neural Conductor Challenge Tests...\n");
    
    for (const test of this.tests) {
      try {
        await test.testFn();
        console.log(`✅ ${test.name}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ ${test.name}: ${error.message}`);
        this.failed++;
      }
    }
    
    const duration = Date.now() - this.startTime;
    console.log(`\n📊 Test Results: ${this.passed} passed, ${this.failed} failed (${duration}ms)`);
    
    return {
      passed: this.passed,
      failed: this.failed,
      total: this.tests.length,
      duration
    };
  }
}

class PerformanceBenchmark {
  constructor() {
    this.benchmarks = new Map();
  }

  start(name) {
    this.benchmarks.set(name, { start: Date.now() });
  }

  end(name) {
    const benchmark = this.benchmarks.get(name);
    if (benchmark) {
      benchmark.end = Date.now();
      benchmark.duration = benchmark.end - benchmark.start;
    }
  }

  getResults() {
    return Object.fromEntries(this.benchmarks);
  }
}

async function runTests() {
  const testRunner = new TestRunner();
  const benchmark = new PerformanceBenchmark();

  // Test 1: Neural Conductor Initialization
  testRunner.addTest('Neural Conductor Initialization', () => {
    const conductor = new NeuralConductor();
    
    if (!conductor.agents) {
      throw new Error('Agents map not initialized');
    }
    if (!conductor.coordinationMatrix) {
      throw new Error('Coordination matrix not initialized');
    }
    if (conductor.emergentIntelligence !== 0) {
      throw new Error('Emergent intelligence should start at 0');
    }
  });

  // Test 2: Agent Creation
  testRunner.addTest('Agent Creation', () => {
    const conductor = new NeuralConductor();
    const agent = conductor.createAgent(
      'test-agent',
      'TestAgent',
      ['capability1', 'capability2'],
      'Test Specialization'
    );
    
    if (agent.id !== 'test-agent') {
      throw new Error('Agent ID not set correctly');
    }
    if (agent.type !== 'TestAgent') {
      throw new Error('Agent type not set correctly');
    }
    if (agent.capabilities.length !== 2) {
      throw new Error('Agent capabilities not set correctly');
    }
    if (agent.specialization !== 'Test Specialization') {
      throw new Error('Agent specialization not set correctly');
    }
    if (agent.status !== 'idle') {
      throw new Error('Agent status should be idle initially');
    }
  });

  // Test 3: Agent Connection Establishment
  testRunner.addTest('Agent Connection Establishment', () => {
    const conductor = new NeuralConductor();
    
    const agent1 = conductor.createAgent('agent1', 'Type1', ['cap1'], 'Spec1');
    const agent2 = conductor.createAgent('agent2', 'Type2', ['cap2'], 'Spec2');
    
    conductor.establishConnection('agent1', 'agent2', 'test-connection', 0.8);
    
    if (!agent1.connections.has('agent2')) {
      throw new Error('Agent1 should have connection to agent2');
    }
    if (!agent2.connections.has('agent1')) {
      throw new Error('Agent2 should have connection to agent1');
    }
    
    const connectionKey = 'agent1-agent2';
    const connection = conductor.coordinationMatrix.get(connectionKey);
    
    if (!connection) {
      throw new Error('Connection not stored in coordination matrix');
    }
    if (connection.strength !== 0.8) {
      throw new Error('Connection strength not set correctly');
    }
  });

  // Test 4: Climate Crisis Swarm Deployment
  testRunner.addTest('Climate Crisis Swarm Deployment', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployClimateCrisisSwarm();
    
    if (swarm.agents.length !== 5) {
      throw new Error(`Expected 5 agents, got ${swarm.agents.length}`);
    }
    
    const agentTypes = swarm.agents.map(agent => agent.type);
    const expectedTypes = ['DataAnalyst', 'ClimateModeler', 'PolicyAdvisor', 'EconomicEvaluator', 'CoordinationAgent'];
    
    for (const expectedType of expectedTypes) {
      if (!agentTypes.includes(expectedType)) {
        throw new Error(`Missing agent type: ${expectedType}`);
      }
    }
    
    if (swarm.scenario !== 'Climate Crisis Response') {
      throw new Error('Incorrect scenario name');
    }
  });

  // Test 5: Traffic Optimization Swarm Deployment
  testRunner.addTest('Traffic Optimization Swarm Deployment', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployTrafficOptimizationSwarm();
    
    if (swarm.agents.length !== 5) {
      throw new Error(`Expected 5 agents, got ${swarm.agents.length}`);
    }
    
    const agentTypes = swarm.agents.map(agent => agent.type);
    const expectedTypes = ['TrafficPattern', 'RouteOptimizer', 'PredictionEngine', 'EmergencyResponse', 'LearningAgent'];
    
    for (const expectedType of expectedTypes) {
      if (!agentTypes.includes(expectedType)) {
        throw new Error(`Missing agent type: ${expectedType}`);
      }
    }
    
    if (swarm.scenario !== 'Smart City Traffic Optimization') {
      throw new Error('Incorrect scenario name');
    }
  });

  // Test 6: Agent Output Generation
  testRunner.addTest('Agent Output Generation', () => {
    const conductor = new NeuralConductor();
    const agent = conductor.createAgent('test', 'DataAnalyst', ['data-processing'], 'Climate Analysis');
    
    const output = conductor.generateAgentOutput(agent, { temperature: 2.1 });
    
    if (!output.insights || !Array.isArray(output.insights)) {
      throw new Error('DataAnalyst should generate insights array');
    }
    if (!output.recommendations || !Array.isArray(output.recommendations)) {
      throw new Error('DataAnalyst should generate recommendations array');
    }
    if (output.insights.length === 0) {
      throw new Error('Insights should not be empty');
    }
  });

  // Test 7: Collaboration Bonus Calculation
  testRunner.addTest('Collaboration Bonus Calculation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployClimateCrisisSwarm();
    
    const bonus = conductor.calculateCollaborationBonus(swarm);
    
    if (typeof bonus !== 'number') {
      throw new Error('Collaboration bonus should be a number');
    }
    if (bonus < 0 || bonus > 0.3) {
      throw new Error('Collaboration bonus should be between 0 and 0.3');
    }
  });

  // Test 8: Swarm Efficiency Calculation
  testRunner.addTest('Swarm Efficiency Calculation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployTrafficOptimizationSwarm();
    
    // Set some performance data
    swarm.agents.forEach(agent => {
      agent.performance.tasksCompleted = 5;
      agent.performance.efficiency = 0.8;
    });
    
    const efficiency = conductor.calculateSwarmEfficiency(swarm);
    
    if (typeof efficiency !== 'number') {
      throw new Error('Swarm efficiency should be a number');
    }
    if (efficiency < 0) {
      throw new Error('Swarm efficiency should be non-negative');
    }
  });

  // Test 9: Coordination Efficiency Calculation
  testRunner.addTest('Coordination Efficiency Calculation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployClimateCrisisSwarm();
    
    const efficiency = conductor.calculateCoordinationEfficiency(swarm);
    
    if (typeof efficiency !== 'number') {
      throw new Error('Coordination efficiency should be a number');
    }
    if (efficiency < 0 || efficiency > 1) {
      throw new Error('Coordination efficiency should be between 0 and 1');
    }
  });

  // Test 10: Problem Decomposition Evaluation
  testRunner.addTest('Problem Decomposition Evaluation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployTrafficOptimizationSwarm();
    
    const score = conductor.evaluateProblemDecomposition(swarm);
    
    if (typeof score !== 'number') {
      throw new Error('Problem decomposition score should be a number');
    }
    if (score < 0 || score > 1) {
      throw new Error('Problem decomposition score should be between 0 and 1');
    }
  });

  // Test 11: Creative AI Use Evaluation
  testRunner.addTest('Creative AI Use Evaluation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployTrafficOptimizationSwarm();
    
    const score = conductor.evaluateCreativeAIUse(swarm);
    
    if (typeof score !== 'number') {
      throw new Error('Creative AI use score should be a number');
    }
    if (score < 0 || score > 1) {
      throw new Error('Creative AI use score should be between 0 and 1');
    }
  });

  // Test 12: Solution Quality Evaluation
  testRunner.addTest('Solution Quality Evaluation', () => {
    const conductor = new NeuralConductor();
    const swarm = conductor.deployClimateCrisisSwarm();
    
    const emergentResults = {
      results: swarm.agents.map(agent => ({
        performance: { accuracy: 0.8, efficiency: 0.7 }
      }))
    };
    
    const score = conductor.evaluateSolutionQuality(emergentResults);
    
    if (typeof score !== 'number') {
      throw new Error('Solution quality score should be a number');
    }
    if (score < 0 || score > 1) {
      throw new Error('Solution quality score should be between 0 and 1');
    }
  });

  // Test 13: Grade Assignment
  testRunner.addTest('Grade Assignment', () => {
    const conductor = new NeuralConductor();
    
    const grades = [
      { score: 95, expected: 'A+ (Exceptional)' },
      { score: 85, expected: 'A (Excellent)' },
      { score: 75, expected: 'B+ (Very Good)' },
      { score: 65, expected: 'B (Good)' },
      { score: 55, expected: 'C+ (Satisfactory)' },
      { score: 45, expected: 'C (Adequate)' },
      { score: 35, expected: 'D (Needs Improvement)' }
    ];
    
    for (const { score, expected } of grades) {
      const grade = conductor.getGrade(score);
      if (grade !== expected) {
        throw new Error(`Expected grade "${expected}" for score ${score}, got "${grade}"`);
      }
    }
  });

  // Test 14: Feedback Generation
  testRunner.addTest('Feedback Generation', () => {
    const conductor = new NeuralConductor();
    
    const feedbacks = [
      { score: 95, shouldContain: 'Outstanding' },
      { score: 85, shouldContain: 'Excellent' },
      { score: 75, shouldContain: 'Very good' },
      { score: 65, shouldContain: 'Good effort' },
      { score: 35, shouldContain: 'needs significant improvement' }
    ];
    
    for (const { score, shouldContain } of feedbacks) {
      const feedback = conductor.generateFeedback(score);
      if (!feedback.includes(shouldContain)) {
        throw new Error(`Feedback for score ${score} should contain "${shouldContain}"`);
      }
    }
  });

  // Test 15: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Neural Conductor Execution');
    
    const conductor = new NeuralConductor();
    const swarm = conductor.deployClimateCrisisSwarm();
    
    // Simulate a quick emergent intelligence test
    const problemData = { temperature: 2.1, co2Levels: 420 };
    const emergentResults = await conductor.simulateEmergentIntelligence(swarm, problemData);
    
    benchmark.end('Neural Conductor Execution');
    
    if (emergentResults.emergentIntelligence < 0.5) {
      throw new Error(`Emergent intelligence too low: ${emergentResults.emergentIntelligence}`);
    }
    
    if (emergentResults.results.length !== swarm.agents.length) {
      throw new Error(`Expected ${swarm.agents.length} results, got ${emergentResults.results.length}`);
    }
  });

  // Test 16: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    const result = await executeNeuralConductor();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID missing from result');
    }
    if (!result.status || result.status !== 'completed') {
      throw new Error('Challenge status should be completed');
    }
    if (!result.swarm || !result.swarm.agents) {
      throw new Error('Swarm data missing from result');
    }
    if (!result.queenSeraphinaEvaluation) {
      throw new Error('Queen Seraphina evaluation missing from result');
    }
    if (!result.performance || !result.performance.grade) {
      throw new Error('Performance evaluation missing from result');
    }
    
    if (result.swarm.agents.length < 5) {
      throw new Error(`Expected at least 5 agents, got ${result.swarm.agents.length}`);
    }
    
    if (result.queenSeraphinaEvaluation.totalScore < 0 || result.queenSeraphinaEvaluation.totalScore > 100) {
      throw new Error(`Total score should be between 0 and 100, got ${result.queenSeraphinaEvaluation.totalScore}`);
    }
  });

  // Run all tests
  const results = await testRunner.runTests();
  
  console.log("\n📈 Performance Benchmarks:");
  const benchmarkResults = benchmark.getResults();
  for (const [name, data] of Object.entries(benchmarkResults)) {
    if (data.duration) {
      console.log(`  ${name}: ${data.duration}ms`);
    }
  }
  
  console.log("\n🎯 Neural Conductor Challenge Test Summary:");
  console.log(`  Total Tests: ${results.total}`);
  console.log(`  Passed: ${results.passed}`);
  console.log(`  Failed: ${results.failed}`);
  console.log(`  Success Rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);
  console.log(`  Duration: ${results.duration}ms`);
  
  if (results.failed === 0) {
    console.log("\n🏆 All tests passed! The Neural Conductor is ready for Queen Seraphina's judgment!");
  } else {
    console.log(`\n⚠️  ${results.failed} test(s) failed. Please review and fix issues.`);
  }
  
  return results;
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests()
    .then(results => {
      process.exit(results.failed === 0 ? 0 : 1);
    })
    .catch(error => {
      console.error('Test execution failed:', error);
      process.exit(1);
    });
}

export { runTests, TestRunner, PerformanceBenchmark };
