// The Phantom Constructor Challenge Test Suite
// Comprehensive testing for rapid construction and deployment systems

import { PhantomConstructor, executePhantomConstructor } from './solution.js';

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
    console.log("🧪 Running Phantom Constructor Challenge Tests...\n");
    
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

  // Test 1: Phantom Constructor Initialization
  testRunner.addTest('Phantom Constructor Initialization', () => {
    const constructor = new PhantomConstructor();
    
    if (!constructor.constructionBots) {
      throw new Error('Construction bots map not initialized');
    }
    if (!constructor.resourceManager) {
      throw new Error('Resource manager not initialized');
    }
    if (!constructor.qualityAssurance) {
      throw new Error('Quality assurance not initialized');
    }
    if (!constructor.designEngine) {
      throw new Error('Design engine not initialized');
    }
    if (!constructor.monitoringSystem) {
      throw new Error('Monitoring system not initialized');
    }
  });

  // Test 2: Bot Creation
  testRunner.addTest('Bot Creation', () => {
    const constructor = new PhantomConstructor();
    const bot = constructor.createBot('test-type', 'test-id', {
      speed: 100,
      precision: 0.95,
      capacity: 50,
      specialization: 'Test Specialization',
      capabilities: ['test1', 'test2']
    });
    
    if (bot.id !== 'test-id') {
      throw new Error('Bot ID not set correctly');
    }
    if (bot.type !== 'test-type') {
      throw new Error('Bot type not set correctly');
    }
    if (bot.speed !== 100) {
      throw new Error('Bot speed not set correctly');
    }
    if (bot.precision !== 0.95) {
      throw new Error('Bot precision not set correctly');
    }
    if (bot.capacity !== 50) {
      throw new Error('Bot capacity not set correctly');
    }
    if (bot.specialization !== 'Test Specialization') {
      throw new Error('Bot specialization not set correctly');
    }
    if (bot.capabilities.length !== 2) {
      throw new Error('Bot capabilities not set correctly');
    }
    if (bot.status !== 'idle') {
      throw new Error('Bot status should be idle initially');
    }
  });

  // Test 3: Construction Bots Deployment
  testRunner.addTest('Construction Bots Deployment', () => {
    const constructor = new PhantomConstructor();
    const bots = constructor.deployConstructionBots('smart-city');
    
    if (bots.length !== 15) {
      throw new Error(`Expected 15 bots, got ${bots.length}`);
    }
    
    const botTypes = bots.map(bot => bot.type);
    const expectedTypes = ['assembly', 'inspector', 'logistics', 'safety', 'design'];
    
    for (const expectedType of expectedTypes) {
      const count = botTypes.filter(type => type === expectedType).length;
      if (count === 0) {
        throw new Error(`Missing bot type: ${expectedType}`);
      }
    }
    
    // Check specific counts
    const assemblyCount = botTypes.filter(type => type === 'assembly').length;
    if (assemblyCount !== 5) {
      throw new Error(`Expected 5 assembly bots, got ${assemblyCount}`);
    }
    
    const inspectorCount = botTypes.filter(type => type === 'inspector').length;
    if (inspectorCount !== 3) {
      throw new Error(`Expected 3 inspector bots, got ${inspectorCount}`);
    }
    
    const logisticsCount = botTypes.filter(type => type === 'logistics').length;
    if (logisticsCount !== 4) {
      throw new Error(`Expected 4 logistics bots, got ${logisticsCount}`);
    }
    
    const safetyCount = botTypes.filter(type => type === 'safety').length;
    if (safetyCount !== 2) {
      throw new Error(`Expected 2 safety bots, got ${safetyCount}`);
    }
    
    const designCount = botTypes.filter(type => type === 'design').length;
    if (designCount !== 1) {
      throw new Error(`Expected 1 design bot, got ${designCount}`);
    }
  });

  // Test 4: Project Initialization
  testRunner.addTest('Project Initialization', () => {
    const constructor = new PhantomConstructor();
    const project = constructor.initializeProject('smart-city', {
      complexity: 'high',
      timeline: 'aggressive'
    });
    
    if (!project) {
      throw new Error('Project not initialized');
    }
    if (project.type !== 'smart-city') {
      throw new Error('Project type not set correctly');
    }
    if (!project.components || project.components.length === 0) {
      throw new Error('Project components not generated');
    }
    if (!project.timeline) {
      throw new Error('Project timeline not calculated');
    }
    if (!project.budget) {
      throw new Error('Project budget not calculated');
    }
    if (!project.constraints) {
      throw new Error('Project constraints not identified');
    }
  });

  // Test 5: Project Components Generation
  testRunner.addTest('Project Components Generation', () => {
    const constructor = new PhantomConstructor();
    
    const smartCityComponents = constructor.generateProjectComponents('smart-city');
    const manufacturingComponents = constructor.generateProjectComponents('manufacturing-plant');
    const spaceStationComponents = constructor.generateProjectComponents('space-station');
    
    if (smartCityComponents.length === 0) {
      throw new Error('Smart city components not generated');
    }
    if (manufacturingComponents.length === 0) {
      throw new Error('Manufacturing plant components not generated');
    }
    if (spaceStationComponents.length === 0) {
      throw new Error('Space station components not generated');
    }
    
    // Check component structure
    const component = smartCityComponents[0];
    if (!component.id) {
      throw new Error('Component ID missing');
    }
    if (!component.type) {
      throw new Error('Component type missing');
    }
    if (!component.complexity) {
      throw new Error('Component complexity missing');
    }
    if (!component.priority) {
      throw new Error('Component priority missing');
    }
  });

  // Test 6: Project Timeline Calculation
  testRunner.addTest('Project Timeline Calculation', () => {
    const constructor = new PhantomConstructor();
    
    const smartCityTimeline = constructor.calculateProjectTimeline('smart-city');
    const manufacturingTimeline = constructor.calculateProjectTimeline('manufacturing-plant');
    const spaceStationTimeline = constructor.calculateProjectTimeline('space-station');
    
    if (smartCityTimeline !== 25 * 60 * 1000) {
      throw new Error(`Smart city timeline incorrect: ${smartCityTimeline}`);
    }
    if (manufacturingTimeline !== 20 * 60 * 1000) {
      throw new Error(`Manufacturing timeline incorrect: ${manufacturingTimeline}`);
    }
    if (spaceStationTimeline !== 30 * 60 * 1000) {
      throw new Error(`Space station timeline incorrect: ${spaceStationTimeline}`);
    }
  });

  // Test 7: Project Budget Calculation
  testRunner.addTest('Project Budget Calculation', () => {
    const constructor = new PhantomConstructor();
    
    const smartCityBudget = constructor.calculateProjectBudget('smart-city');
    const manufacturingBudget = constructor.calculateProjectBudget('manufacturing-plant');
    const spaceStationBudget = constructor.calculateProjectBudget('space-station');
    
    if (smartCityBudget !== 1000000) {
      throw new Error(`Smart city budget incorrect: ${smartCityBudget}`);
    }
    if (manufacturingBudget !== 800000) {
      throw new Error(`Manufacturing budget incorrect: ${manufacturingBudget}`);
    }
    if (spaceStationBudget !== 1500000) {
      throw new Error(`Space station budget incorrect: ${spaceStationBudget}`);
    }
  });

  // Test 8: Project Constraints Identification
  testRunner.addTest('Project Constraints Identification', () => {
    const constructor = new PhantomConstructor();
    
    const smartCityConstraints = constructor.identifyProjectConstraints('smart-city');
    const manufacturingConstraints = constructor.identifyProjectConstraints('manufacturing-plant');
    const spaceStationConstraints = constructor.identifyProjectConstraints('space-station');
    
    const requiredProps = ['environmental', 'safety', 'regulatory', 'timeline'];
    
    for (const prop of requiredProps) {
      if (smartCityConstraints[prop] === undefined) {
        throw new Error(`Smart city constraints missing property: ${prop}`);
      }
      if (manufacturingConstraints[prop] === undefined) {
        throw new Error(`Manufacturing constraints missing property: ${prop}`);
      }
      if (spaceStationConstraints[prop] === undefined) {
        throw new Error(`Space station constraints missing property: ${prop}`);
      }
    }
  });

  // Test 9: Project Analysis
  testRunner.addTest('Project Analysis', () => {
    const constructor = new PhantomConstructor();
    constructor.initializeProject('smart-city', { complexity: 'high' });
    
    const complexity = constructor.assessProjectComplexity();
    const risks = constructor.identifyProjectRisks();
    const dependencies = constructor.mapDependencies();
    const resources = constructor.calculateResourceRequirements();
    const timeline = constructor.optimizeTimeline();
    
    if (typeof complexity !== 'number') {
      throw new Error('Project complexity should be a number');
    }
    if (complexity < 0 || complexity > 100) {
      throw new Error(`Project complexity out of range: ${complexity}`);
    }
    
    if (!Array.isArray(risks)) {
      throw new Error('Project risks should be an array');
    }
    if (risks.length === 0) {
      throw new Error('Project risks should not be empty');
    }
    
    if (!(dependencies instanceof Map)) {
      throw new Error('Dependencies should be a Map');
    }
    
    if (!resources || typeof resources !== 'object') {
      throw new Error('Resource requirements should be an object');
    }
    
    if (!timeline || typeof timeline !== 'object') {
      throw new Error('Timeline optimization should be an object');
    }
  });

  // Test 10: Construction Phases Creation
  testRunner.addTest('Construction Phases Creation', () => {
    const constructor = new PhantomConstructor();
    const phases = constructor.createConstructionPhases();
    
    if (!Array.isArray(phases)) {
      throw new Error('Construction phases should be an array');
    }
    if (phases.length === 0) {
      throw new Error('Construction phases should not be empty');
    }
    
    const requiredProps = ['id', 'name', 'duration', 'priority'];
    for (const phase of phases) {
      for (const prop of requiredProps) {
        if (phase[prop] === undefined) {
          throw new Error(`Phase missing property: ${prop}`);
        }
      }
    }
  });

  // Test 11: Construction Sequences Optimization
  testRunner.addTest('Construction Sequences Optimization', () => {
    const constructor = new PhantomConstructor();
    constructor.initializeProject('smart-city', { complexity: 'high' });
    const sequences = constructor.optimizeConstructionSequences();
    
    if (!Array.isArray(sequences)) {
      throw new Error('Construction sequences should be an array');
    }
    if (sequences.length === 0) {
      throw new Error('Construction sequences should not be empty');
    }
    
    const requiredProps = ['phase', 'components', 'parallel'];
    for (const sequence of sequences) {
      for (const prop of requiredProps) {
        if (sequence[prop] === undefined) {
          throw new Error(`Sequence missing property: ${prop}`);
        }
      }
    }
  });

  // Test 12: Resource Allocation
  testRunner.addTest('Resource Allocation', () => {
    const constructor = new PhantomConstructor();
    constructor.deployConstructionBots('smart-city');
    constructor.initializeProject('smart-city', { complexity: 'high' });
    const allocation = constructor.allocateResources();
    
    if (!(allocation instanceof Map)) {
      throw new Error('Resource allocation should be a Map');
    }
    
    const bots = Array.from(constructor.constructionBots.values());
    if (allocation.size !== bots.length) {
      throw new Error(`Allocation size should match bot count: ${allocation.size} vs ${bots.length}`);
    }
  });

  // Test 13: Quality Checkpoints Definition
  testRunner.addTest('Quality Checkpoints Definition', () => {
    const constructor = new PhantomConstructor();
    const checkpoints = constructor.defineQualityCheckpoints();
    
    if (!Array.isArray(checkpoints)) {
      throw new Error('Quality checkpoints should be an array');
    }
    if (checkpoints.length === 0) {
      throw new Error('Quality checkpoints should not be empty');
    }
    
    const requiredProps = ['phase', 'tests'];
    for (const checkpoint of checkpoints) {
      for (const prop of requiredProps) {
        if (checkpoint[prop] === undefined) {
          throw new Error(`Checkpoint missing property: ${prop}`);
        }
      }
    }
  });

  // Test 14: Safety Protocols Definition
  testRunner.addTest('Safety Protocols Definition', () => {
    const constructor = new PhantomConstructor();
    const protocols = constructor.defineSafetyProtocols();
    
    if (!Array.isArray(protocols)) {
      throw new Error('Safety protocols should be an array');
    }
    if (protocols.length === 0) {
      throw new Error('Safety protocols should not be empty');
    }
    
    const requiredProps = ['protocol', 'enforcement'];
    for (const protocol of protocols) {
      for (const prop of requiredProps) {
        if (protocol[prop] === undefined) {
          throw new Error(`Protocol missing property: ${prop}`);
        }
      }
    }
  });

  // Test 15: Performance Metrics Calculation
  testRunner.addTest('Performance Metrics Calculation', () => {
    const constructor = new PhantomConstructor();
    constructor.deployConstructionBots('smart-city');
    constructor.initializeProject('smart-city', { complexity: 'high' });
    
    const resourceEfficiency = constructor.calculateResourceEfficiency();
    const innovationIndex = constructor.calculateInnovationIndex();
    const scalabilityFactor = constructor.calculateScalabilityFactor();
    const maintainabilityScore = constructor.calculateMaintainabilityScore();
    const costEfficiency = constructor.calculateCostEfficiency();
    
    if (typeof resourceEfficiency !== 'number') {
      throw new Error('Resource efficiency should be a number');
    }
    if (resourceEfficiency < 0 || resourceEfficiency > 100) {
      throw new Error(`Resource efficiency out of range: ${resourceEfficiency}`);
    }
    
    if (typeof innovationIndex !== 'number') {
      throw new Error('Innovation index should be a number');
    }
    if (innovationIndex < 0 || innovationIndex > 100) {
      throw new Error(`Innovation index out of range: ${innovationIndex}`);
    }
    
    if (typeof scalabilityFactor !== 'number') {
      throw new Error('Scalability factor should be a number');
    }
    if (scalabilityFactor < 0 || scalabilityFactor > 100) {
      throw new Error(`Scalability factor out of range: ${scalabilityFactor}`);
    }
    
    if (typeof maintainabilityScore !== 'number') {
      throw new Error('Maintainability score should be a number');
    }
    if (maintainabilityScore < 0 || maintainabilityScore > 100) {
      throw new Error(`Maintainability score out of range: ${maintainabilityScore}`);
    }
    
    if (typeof costEfficiency !== 'number') {
      throw new Error('Cost efficiency should be a number');
    }
    if (costEfficiency < 0 || costEfficiency > 100) {
      throw new Error(`Cost efficiency out of range: ${costEfficiency}`);
    }
  });

  // Test 16: Construction Report Generation
  testRunner.addTest('Construction Report Generation', () => {
    const constructor = new PhantomConstructor();
    constructor.deployConstructionBots('smart-city');
    constructor.initializeProject('smart-city', { complexity: 'high' });
    
    const report = constructor.generateConstructionReport();
    
    if (!report.challengeId) {
      throw new Error('Construction report missing challengeId');
    }
    if (!report.status) {
      throw new Error('Construction report missing status');
    }
    if (!report.project) {
      throw new Error('Construction report missing project');
    }
    if (!report.construction) {
      throw new Error('Construction report missing construction');
    }
    if (!report.performance) {
      throw new Error('Construction report missing performance');
    }
    if (!report.quality) {
      throw new Error('Construction report missing quality');
    }
    if (!report.timeline) {
      throw new Error('Construction report missing timeline');
    }
    if (!report.timestamp) {
      throw new Error('Construction report missing timestamp');
    }
    if (!report.message) {
      throw new Error('Construction report missing message');
    }
    
    if (report.challengeId !== "8a8b9ac8-a8d6-4e71-ad85-256de4d44143") {
      throw new Error('Incorrect challenge ID');
    }
    
    if (report.status !== 'completed') {
      throw new Error('Status should be completed');
    }
  });

  // Test 17: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Phantom Constructor Execution');
    
    const constructor = new PhantomConstructor();
    constructor.deployConstructionBots('smart-city');
    constructor.initializeProject('smart-city', { complexity: 'high' });
    
    // Simulate a quick construction process
    await constructor.analyzeProject();
    await constructor.performRapidDesign();
    
    benchmark.end('Phantom Constructor Execution');
    
    const bots = Array.from(constructor.constructionBots.values());
    if (bots.length < 15) {
      throw new Error(`Expected at least 15 bots, got ${bots.length}`);
    }
    
    const project = constructor.project;
    if (!project || !project.components) {
      throw new Error('Project not properly initialized');
    }
  });

  // Test 18: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    const result = await executePhantomConstructor();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID missing from result');
    }
    if (!result.status || result.status !== 'completed') {
      throw new Error('Challenge status should be completed');
    }
    if (!result.project) {
      throw new Error('Project data missing from result');
    }
    if (!result.construction) {
      throw new Error('Construction data missing from result');
    }
    if (!result.performance) {
      throw new Error('Performance data missing from result');
    }
    if (!result.quality) {
      throw new Error('Quality data missing from result');
    }
    if (!result.timeline) {
      throw new Error('Timeline data missing from result');
    }
    
    if (result.construction.totalBots < 15) {
      throw new Error(`Expected at least 15 bots, got ${result.construction.totalBots}`);
    }
    
    if (result.performance.buildTime < 0) {
      throw new Error(`Build time should be non-negative: ${result.performance.buildTime}`);
    }
    
    if (result.performance.qualityScore < 0 || result.performance.qualityScore > 100) {
      throw new Error(`Quality score out of range: ${result.performance.qualityScore}`);
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
  
  console.log("\n🎯 Phantom Constructor Challenge Test Summary:");
  console.log(`  Total Tests: ${results.total}`);
  console.log(`  Passed: ${results.passed}`);
  console.log(`  Failed: ${results.failed}`);
  console.log(`  Success Rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);
  console.log(`  Duration: ${results.duration}ms`);
  
  if (results.failed === 0) {
    console.log("\n🏆 All tests passed! The Phantom Constructor is ready to build at phantom speed!");
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
