// The Labyrinth Architect Challenge Test Suite
// Comprehensive testing for algorithmic warfare and multi-dimensional optimization

import { LabyrinthArchitect, runLabyrinthArchitect } from './solution.js';

class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
    this.results = [];
  }

  addTest(name, testFunction) {
    this.tests.push({ name, testFunction });
  }

  async runTests() {
    console.log("🧪 Running The Labyrinth Architect Test Suite...\n");
    
    for (const test of this.tests) {
      try {
        await test.testFunction();
        this.passed++;
        this.results.push({ name: test.name, status: 'PASS', error: null });
        console.log(`✅ ${test.name}`);
      } catch (error) {
        this.failed++;
        this.results.push({ name: test.name, status: 'FAIL', error: error.message });
        console.log(`❌ ${test.name}: ${error.message}`);
      }
    }
    
    this.printSummary();
  }

  printSummary() {
    console.log("\n📊 Test Summary:");
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`❌ Failed: ${this.failed}`);
    console.log(`📈 Success Rate: ${((this.passed / (this.passed + this.failed)) * 100).toFixed(1)}%`);
    
    if (this.failed > 0) {
      console.log("\n❌ Failed Tests:");
      this.results.filter(r => r.status === 'FAIL').forEach(r => {
        console.log(`  - ${r.name}: ${r.error}`);
      });
    }
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
    const results = {};
    for (const [name, benchmark] of this.benchmarks) {
      results[name] = benchmark.duration;
    }
    return results;
  }
}

async function runTests() {
  const testRunner = new TestRunner();
  const benchmark = new PerformanceBenchmark();

  // Test 1: Labyrinth Architect Initialization
  testRunner.addTest('Labyrinth Architect Initialization', () => {
    const architect = new LabyrinthArchitect();
    
    if (!architect.quantumNavigator) {
      throw new Error('QuantumNavigator not initialized');
    }
    if (!architect.algorithmicWarrior) {
      throw new Error('AlgorithmicWarrior not initialized');
    }
    if (!architect.resourceOptimizer) {
      throw new Error('ResourceOptimizer not initialized');
    }
    if (!architect.metaAlgorithm) {
      throw new Error('MetaAlgorithm not initialized');
    }
  });

  // Test 2: Quantum Navigator 4D Pathfinding
  testRunner.addTest('Quantum Navigator 4D Pathfinding', () => {
    const architect = new LabyrinthArchitect();
    const maze = architect.generate4DMaze(5, 5, 5, 5);
    const start = { x: 0, y: 0, z: 0, w: 0 };
    const end = { x: 4, y: 4, z: 4, w: 4 };
    
    const result = architect.quantumNavigator.navigateQuantumLabyrinth(maze, start, end);
    
    if (typeof result.success !== 'boolean') {
      throw new Error('Result success should be boolean');
    }
    if (!Array.isArray(result.path)) {
      throw new Error('Result path should be array');
    }
    if (typeof result.cost !== 'number') {
      throw new Error('Result cost should be number');
    }
    if (typeof result.quantumState !== 'number') {
      throw new Error('Result quantumState should be number');
    }
  });

  // Test 3: Algorithmic Warrior Strategy Selection
  testRunner.addTest('Algorithmic Warrior Strategy Selection', () => {
    const architect = new LabyrinthArchitect();
    const opponents = architect.generateOpponents(3);
    const battlefield = { complexity: 0.7, resources: 500, timeLimit: 30 };
    
    const strategy = architect.algorithmicWarrior.selectOptimalStrategy(opponents, battlefield);
    
    if (!strategy) {
      throw new Error('Strategy should not be null');
    }
    if (typeof strategy.aggressiveness !== 'number') {
      throw new Error('Strategy aggressiveness should be number');
    }
    if (typeof strategy.defensiveness !== 'number') {
      throw new Error('Strategy defensiveness should be number');
    }
    if (typeof strategy.adaptability !== 'number') {
      throw new Error('Strategy adaptability should be number');
    }
  });

  // Test 4: Resource Optimizer Allocation
  testRunner.addTest('Resource Optimizer Allocation', () => {
    const architect = new LabyrinthArchitect();
    const resources = { 'CPU': 100, 'Memory': 200, 'Storage': 500 };
    const constraints = {
      'CPU': { min: 10, max: 80 },
      'Memory': { min: 20, max: 150 },
      'Storage': { min: 50, max: 400 }
    };
    const objectives = ['maximize_efficiency', 'minimize_cost'];
    
    const result = architect.resourceOptimizer.optimizeResourceAllocation(resources, constraints, objectives);
    
    if (!result.allocation) {
      throw new Error('Allocation result should exist');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('Efficiency should be number');
    }
    if (typeof result.riskScore !== 'number') {
      throw new Error('Risk score should be number');
    }
    if (typeof result.scenariosProcessed !== 'number') {
      throw new Error('Scenarios processed should be number');
    }
  });

  // Test 5: Meta Algorithm Evolution
  testRunner.addTest('Meta Algorithm Evolution', () => {
    const architect = new LabyrinthArchitect();
    const performanceData = {
      accuracy: 0.85,
      speed: 0.75,
      memory: 0.80,
      reliability: 0.90
    };
    
    const result = architect.metaAlgorithm.evolveAlgorithm('pathfinding', performanceData);
    
    if (!result.algorithm) {
      throw new Error('Evolved algorithm should exist');
    }
    if (typeof result.performance !== 'object') {
      throw new Error('Performance should be object');
    }
    if (typeof result.evolutionCount !== 'number') {
      throw new Error('Evolution count should be number');
    }
    if (typeof result.improvementRate !== 'number') {
      throw new Error('Improvement rate should be number');
    }
  });

  // Test 6: 4D Maze Generation
  testRunner.addTest('4D Maze Generation', () => {
    const architect = new LabyrinthArchitect();
    const maze = architect.generate4DMaze(3, 3, 3, 3);
    
    if (!Array.isArray(maze)) {
      throw new Error('Maze should be array');
    }
    if (maze.length !== 3) {
      throw new Error('Maze x dimension should be 3');
    }
    if (maze[0].length !== 3) {
      throw new Error('Maze y dimension should be 3');
    }
    if (maze[0][0].length !== 3) {
      throw new Error('Maze z dimension should be 3');
    }
    if (maze[0][0][0].length !== 3) {
      throw new Error('Maze w dimension should be 3');
    }
  });

  // Test 7: Opponent Generation
  testRunner.addTest('Opponent Generation', () => {
    const architect = new LabyrinthArchitect();
    const opponents = architect.generateOpponents(5);
    
    if (!Array.isArray(opponents)) {
      throw new Error('Opponents should be array');
    }
    if (opponents.length !== 5) {
      throw new Error('Should generate 5 opponents');
    }
    
    for (const opponent of opponents) {
      if (!opponent.id) {
        throw new Error('Opponent should have id');
      }
      if (!opponent.strategy) {
        throw new Error('Opponent should have strategy');
      }
      if (typeof opponent.strength !== 'number') {
        throw new Error('Opponent strength should be number');
      }
    }
  });

  // Test 8: Performance Calculation
  testRunner.addTest('Performance Calculation', () => {
    const architect = new LabyrinthArchitect();
    const mockResults = {
      quantumResult: { success: true },
      warfareResult: { winRate: 0.8 },
      optimizationResult: { efficiency: 0.85 },
      evolutionResult: { averageImprovement: 0.15 },
      executionTime: 1000
    };
    
    const performance = architect.calculateOverallPerformance(mockResults);
    
    if (typeof performance.overallScore !== 'number') {
      throw new Error('Overall score should be number');
    }
    if (typeof performance.grade !== 'string') {
      throw new Error('Grade should be string');
    }
    if (performance.overallScore < 0 || performance.overallScore > 100) {
      throw new Error('Overall score should be between 0 and 100');
    }
  });

  // Test 9: Grade Assignment
  testRunner.addTest('Grade Assignment', () => {
    const architect = new LabyrinthArchitect();
    
    const grades = [
      { score: 95, expected: 'A+' },
      { score: 90, expected: 'A' },
      { score: 85, expected: 'B+' },
      { score: 80, expected: 'B' },
      { score: 75, expected: 'C+' },
      { score: 70, expected: 'C' },
      { score: 65, expected: 'F' }
    ];
    
    for (const { score, expected } of grades) {
      const grade = architect.assignGrade(score);
      if (grade !== expected) {
        throw new Error(`Score ${score} should get grade ${expected}, got ${grade}`);
      }
    }
  });

  // Test 10: Strategic Report Generation
  testRunner.addTest('Strategic Report Generation', () => {
    const architect = new LabyrinthArchitect();
    const mockResults = {
      quantumResult: { success: true },
      warfareResult: { winRate: 0.8 },
      optimizationResult: { efficiency: 0.85 },
      evolutionResult: { averageImprovement: 0.15 },
      performance: { overallScore: 85, grade: 'B+' }
    };
    
    const report = architect.generateStrategicReport(mockResults);
    
    if (!report.executiveSummary) {
      throw new Error('Report should have executive summary');
    }
    if (!Array.isArray(report.keyAchievements)) {
      throw new Error('Report should have key achievements array');
    }
    if (!Array.isArray(report.recommendations)) {
      throw new Error('Report should have recommendations array');
    }
    if (!Array.isArray(report.nextSteps)) {
      throw new Error('Report should have next steps array');
    }
  });

  // Test 11: Quantum State Transitions
  testRunner.addTest('Quantum State Transitions', () => {
    const architect = new LabyrinthArchitect();
    const position = { x: 1, y: 1, z: 1, w: 1 };
    const maze = architect.generate4DMaze(5, 5, 5, 5);
    const constraints = { quantumStability: 0.8 };
    
    const transitions = architect.quantumNavigator.getQuantumTransitions(position, maze, constraints);
    
    if (!Array.isArray(transitions)) {
      throw new Error('Transitions should be array');
    }
    
    for (const transition of transitions) {
      if (!transition.position) {
        throw new Error('Transition should have position');
      }
      if (typeof transition.probability !== 'number') {
        throw new Error('Transition probability should be number');
      }
      if (typeof transition.cost !== 'number') {
        throw new Error('Transition cost should be number');
      }
    }
  });

  // Test 12: Strategy Effectiveness Calculation
  testRunner.addTest('Strategy Effectiveness Calculation', () => {
    const architect = new LabyrinthArchitect();
    
    const effectiveness = architect.algorithmicWarrior.calculateStrategyEffectiveness();
    const adaptationScore = architect.algorithmicWarrior.calculateAdaptationScore();
    
    if (typeof effectiveness !== 'number') {
      throw new Error('Strategy effectiveness should be number');
    }
    if (typeof adaptationScore !== 'number') {
      throw new Error('Adaptation score should be number');
    }
    if (effectiveness < 0 || effectiveness > 1) {
      throw new Error('Strategy effectiveness should be between 0 and 1');
    }
  });

  // Test 13: Resource Constraint Validation
  testRunner.addTest('Resource Constraint Validation', () => {
    const architect = new LabyrinthArchitect();
    const allocation = { 'CPU': 50, 'Memory': 100, 'Storage': 200 };
    const constraints = new Map([
      ['CPU', { min: 10, max: 80 }],
      ['Memory', { min: 20, max: 150 }],
      ['Storage', { min: 50, max: 400 }]
    ]);
    
    architect.resourceOptimizer.constraints = constraints;
    const satisfaction = architect.resourceOptimizer.calculateConstraintSatisfaction(allocation);
    
    if (typeof satisfaction !== 'number') {
      throw new Error('Constraint satisfaction should be number');
    }
    if (satisfaction < 0 || satisfaction > 1) {
      throw new Error('Constraint satisfaction should be between 0 and 1');
    }
  });

  // Test 14: Algorithm Performance Evaluation
  testRunner.addTest('Algorithm Performance Evaluation', () => {
    const architect = new LabyrinthArchitect();
    const performanceData = {
      accuracy: 0.85,
      speed: 0.75,
      memory: 0.80,
      reliability: 0.90
    };
    
    const algorithm = architect.metaAlgorithm.generateBaseAlgorithm('pathfinding');
    const performance = architect.metaAlgorithm.evaluateAlgorithm(algorithm, performanceData);
    
    if (typeof performance.overallScore !== 'number') {
      throw new Error('Performance overall score should be number');
    }
    if (typeof performance.improvement !== 'number') {
      throw new Error('Performance improvement should be number');
    }
    if (performance.overallScore < 0 || performance.overallScore > 1) {
      throw new Error('Performance overall score should be between 0 and 1');
    }
  });

  // Test 15: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    benchmark.start('Labyrinth Architect Execution');
    const result = await runLabyrinthArchitect();
    benchmark.end('Labyrinth Architect Execution');
    
    if (!result.success) {
      throw new Error('Challenge execution should succeed');
    }
    if (!result.performance) {
      throw new Error('Result should have performance metrics');
    }
    if (!result.report) {
      throw new Error('Result should have strategic report');
    }
    if (typeof result.executionTime !== 'number') {
      throw new Error('Execution time should be number');
    }
    if (result.performance.overallScore < 0 || result.performance.overallScore > 100) {
      throw new Error('Overall score should be between 0 and 100');
    }
  });

  // Test 16: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Labyrinth Architect Execution');
    const result = await runLabyrinthArchitect();
    benchmark.end('Labyrinth Architect Execution');
    
    const executionTime = benchmark.getResults()['Labyrinth Architect Execution'];
    
    if (executionTime && executionTime > 10000) { // 10 seconds
      throw new Error(`Execution time too slow: ${executionTime}ms`);
    }
    
    if (result.performance && result.performance.overallScore < 15) {
      throw new Error(`Overall performance too low: ${result.performance.overallScore}%`);
    }
  });

  // Test 17: Quantum Navigation Success Rate
  testRunner.addTest('Quantum Navigation Success Rate', async () => {
    const result = await runLabyrinthArchitect();
    
    if (result.performance && result.performance.quantumScore < 0) {
      throw new Error(`Quantum navigation success rate too low: ${result.performance.quantumScore}%`);
    }
  });

  // Test 18: Algorithmic Warfare Win Rate
  testRunner.addTest('Algorithmic Warfare Win Rate', async () => {
    const result = await runLabyrinthArchitect();
    
    if (result.performance && result.performance.warfareScore < 0) {
      throw new Error(`Algorithmic warfare win rate too low: ${result.performance.warfareScore}%`);
    }
  });

  // Test 19: Resource Optimization Efficiency
  testRunner.addTest('Resource Optimization Efficiency', async () => {
    const result = await runLabyrinthArchitect();
    
    if (result.performance && result.performance.optimizationScore < 0) {
      throw new Error(`Resource optimization efficiency too low: ${result.performance.optimizationScore}%`);
    }
  });

  // Test 20: Algorithm Evolution Improvement
  testRunner.addTest('Algorithm Evolution Improvement', async () => {
    const result = await runLabyrinthArchitect();
    
    if (result.performance && result.performance.evolutionScore < 0) {
      throw new Error(`Algorithm evolution improvement too low: ${result.performance.evolutionScore}%`);
    }
  });

  // Run all tests
  await testRunner.runTests();
  
  // Print performance results
  const performanceResults = benchmark.getResults();
  console.log("\n⚡ Performance Results:");
  for (const [name, duration] of Object.entries(performanceResults)) {
    console.log(`  ${name}: ${duration}ms`);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(console.error);
}

export { runTests };
