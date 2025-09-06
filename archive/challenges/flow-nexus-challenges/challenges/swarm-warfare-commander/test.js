// Swarm Warfare Commander Challenge Test Suite
// Comprehensive testing for AI swarm coordination in warfare scenarios

import { SwarmWarfareCommander, executeSwarmWarfareCommander } from './solution.js';

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
    console.log("🧪 Running Swarm Warfare Commander Challenge Tests...\n");
    
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

  // Test 1: Swarm Warfare Commander Initialization
  testRunner.addTest('Swarm Warfare Commander Initialization', () => {
    const commander = new SwarmWarfareCommander();
    
    if (!commander.swarm) {
      throw new Error('Swarm map not initialized');
    }
    if (!commander.coordinationProtocols) {
      throw new Error('Coordination protocols not initialized');
    }
    if (!commander.resourceManager) {
      throw new Error('Resource manager not initialized');
    }
    if (!commander.threatAssessment) {
      throw new Error('Threat assessment not initialized');
    }
    if (!commander.tacticalPlanner) {
      throw new Error('Tactical planner not initialized');
    }
    if (!commander.performanceMonitor) {
      throw new Error('Performance monitor not initialized');
    }
  });

  // Test 2: Agent Creation
  testRunner.addTest('Agent Creation', () => {
    const commander = new SwarmWarfareCommander();
    const agent = commander.createAgent('test-type', 'test-id', {
      health: 100,
      damage: 25,
      speed: 80,
      accuracy: 0.85,
      range: 50,
      specialization: 'Test Specialization',
      capabilities: ['test1', 'test2']
    });
    
    if (agent.id !== 'test-id') {
      throw new Error('Agent ID not set correctly');
    }
    if (agent.type !== 'test-type') {
      throw new Error('Agent type not set correctly');
    }
    if (agent.health !== 100) {
      throw new Error('Agent health not set correctly');
    }
    if (agent.damage !== 25) {
      throw new Error('Agent damage not set correctly');
    }
    if (agent.speed !== 80) {
      throw new Error('Agent speed not set correctly');
    }
    if (agent.accuracy !== 0.85) {
      throw new Error('Agent accuracy not set correctly');
    }
    if (agent.range !== 50) {
      throw new Error('Agent range not set correctly');
    }
    if (agent.specialization !== 'Test Specialization') {
      throw new Error('Agent specialization not set correctly');
    }
    if (agent.capabilities.length !== 2) {
      throw new Error('Agent capabilities not set correctly');
    }
    if (agent.status !== 'idle') {
      throw new Error('Agent status should be idle initially');
    }
  });

  // Test 3: Combat Swarm Deployment
  testRunner.addTest('Combat Swarm Deployment', () => {
    const commander = new SwarmWarfareCommander();
    const swarm = commander.deployCombatSwarm('urban');
    
    if (swarm.length !== 10) {
      throw new Error(`Expected 10 agents, got ${swarm.length}`);
    }
    
    const agentTypes = swarm.map(agent => agent.type);
    const expectedTypes = ['assault', 'recon', 'support', 'defensive', 'command'];
    
    for (const expectedType of expectedTypes) {
      const count = agentTypes.filter(type => type === expectedType).length;
      if (count === 0) {
        throw new Error(`Missing agent type: ${expectedType}`);
      }
    }
    
    // Check specific counts
    const assaultCount = agentTypes.filter(type => type === 'assault').length;
    if (assaultCount !== 3) {
      throw new Error(`Expected 3 assault units, got ${assaultCount}`);
    }
    
    const reconCount = agentTypes.filter(type => type === 'recon').length;
    if (reconCount !== 2) {
      throw new Error(`Expected 2 recon units, got ${reconCount}`);
    }
    
    const supportCount = agentTypes.filter(type => type === 'support').length;
    if (supportCount !== 2) {
      throw new Error(`Expected 2 support units, got ${supportCount}`);
    }
    
    const defensiveCount = agentTypes.filter(type => type === 'defensive').length;
    if (defensiveCount !== 2) {
      throw new Error(`Expected 2 defensive units, got ${defensiveCount}`);
    }
    
    const commandCount = agentTypes.filter(type => type === 'command').length;
    if (commandCount !== 1) {
      throw new Error(`Expected 1 command unit, got ${commandCount}`);
    }
  });

  // Test 4: Coordination Protocols Establishment
  testRunner.addTest('Coordination Protocols Establishment', () => {
    const commander = new SwarmWarfareCommander();
    commander.deployCombatSwarm('urban');
    
    const protocols = commander.coordinationProtocols;
    
    if (protocols.size !== 3) {
      throw new Error(`Expected 3 coordination protocols, got ${protocols.size}`);
    }
    
    if (!protocols.has('command')) {
      throw new Error('Command protocol not established');
    }
    if (!protocols.has('tactical')) {
      throw new Error('Tactical protocol not established');
    }
    if (!protocols.has('support')) {
      throw new Error('Support protocol not established');
    }
    
    const commandProtocol = protocols.get('command');
    if (commandProtocol.type !== 'hierarchical') {
      throw new Error('Command protocol should be hierarchical');
    }
    
    const tacticalProtocol = protocols.get('tactical');
    if (tacticalProtocol.type !== 'peer-to-peer') {
      throw new Error('Tactical protocol should be peer-to-peer');
    }
    
    const supportProtocol = protocols.get('support');
    if (supportProtocol.type !== 'broadcast') {
      throw new Error('Support protocol should be broadcast');
    }
  });

  // Test 5: Battlefield Initialization
  testRunner.addTest('Battlefield Initialization', () => {
    const commander = new SwarmWarfareCommander();
    const battlefield = commander.initializeBattlefield('urban', 'test-mission');
    
    if (!battlefield) {
      throw new Error('Battlefield not initialized');
    }
    if (battlefield.scenario !== 'urban') {
      throw new Error('Battlefield scenario not set correctly');
    }
    if (battlefield.mission !== 'test-mission') {
      throw new Error('Battlefield mission not set correctly');
    }
    if (!battlefield.dimensions) {
      throw new Error('Battlefield dimensions not set');
    }
    if (!battlefield.objectives || battlefield.objectives.length === 0) {
      throw new Error('Battlefield objectives not generated');
    }
    if (!battlefield.threats || battlefield.threats.length === 0) {
      throw new Error('Battlefield threats not generated');
    }
    if (!battlefield.resources) {
      throw new Error('Battlefield resources not generated');
    }
    if (!battlefield.environment) {
      throw new Error('Battlefield environment not generated');
    }
  });

  // Test 6: Objective Generation
  testRunner.addTest('Objective Generation', () => {
    const commander = new SwarmWarfareCommander();
    
    const urbanObjectives = commander.generateObjectives('urban');
    const openFieldObjectives = commander.generateObjectives('open-field');
    const defensiveObjectives = commander.generateObjectives('defensive');
    
    if (urbanObjectives.length === 0) {
      throw new Error('Urban objectives not generated');
    }
    if (openFieldObjectives.length === 0) {
      throw new Error('Open field objectives not generated');
    }
    if (defensiveObjectives.length === 0) {
      throw new Error('Defensive objectives not generated');
    }
    
    // Check objective structure
    const objective = urbanObjectives[0];
    if (!objective.id) {
      throw new Error('Objective ID missing');
    }
    if (!objective.type) {
      throw new Error('Objective type missing');
    }
    if (!objective.priority) {
      throw new Error('Objective priority missing');
    }
    if (!objective.position) {
      throw new Error('Objective position missing');
    }
    if (!objective.value) {
      throw new Error('Objective value missing');
    }
  });

  // Test 7: Threat Generation
  testRunner.addTest('Threat Generation', () => {
    const commander = new SwarmWarfareCommander();
    
    const urbanThreats = commander.generateThreats('urban');
    const openFieldThreats = commander.generateThreats('open-field');
    const defensiveThreats = commander.generateThreats('defensive');
    
    if (urbanThreats.length === 0) {
      throw new Error('Urban threats not generated');
    }
    if (openFieldThreats.length === 0) {
      throw new Error('Open field threats not generated');
    }
    if (defensiveThreats.length === 0) {
      throw new Error('Defensive threats not generated');
    }
    
    // Check threat structure
    const threat = urbanThreats[0];
    if (!threat.id) {
      throw new Error('Threat ID missing');
    }
    if (!threat.type) {
      throw new Error('Threat type missing');
    }
    if (!threat.position) {
      throw new Error('Threat position missing');
    }
    if (!threat.strength) {
      throw new Error('Threat strength missing');
    }
    if (!threat.priority) {
      throw new Error('Threat priority missing');
    }
  });

  // Test 8: Environment Generation
  testRunner.addTest('Environment Generation', () => {
    const commander = new SwarmWarfareCommander();
    
    const urbanEnv = commander.generateEnvironment('urban');
    const openFieldEnv = commander.generateEnvironment('open-field');
    const defensiveEnv = commander.generateEnvironment('defensive');
    
    const requiredProps = ['visibility', 'cover', 'mobility', 'civilianRisk', 'complexity'];
    
    for (const prop of requiredProps) {
      if (urbanEnv[prop] === undefined) {
        throw new Error(`Urban environment missing property: ${prop}`);
      }
      if (openFieldEnv[prop] === undefined) {
        throw new Error(`Open field environment missing property: ${prop}`);
      }
      if (defensiveEnv[prop] === undefined) {
        throw new Error(`Defensive environment missing property: ${prop}`);
      }
    }
    
    // Check value ranges
    for (const prop of requiredProps) {
      if (urbanEnv[prop] < 0 || urbanEnv[prop] > 1) {
        throw new Error(`Urban environment ${prop} out of range: ${urbanEnv[prop]}`);
      }
    }
  });

  // Test 9: Agent Positioning
  testRunner.addTest('Agent Positioning', () => {
    const commander = new SwarmWarfareCommander();
    commander.deployCombatSwarm('urban');
    commander.initializeBattlefield('urban', 'test-mission');
    
    const agents = Array.from(commander.swarm.values());
    
    for (const agent of agents) {
      if (!agent.position) {
        throw new Error(`Agent ${agent.id} position not set`);
      }
      if (typeof agent.position.x !== 'number') {
        throw new Error(`Agent ${agent.id} position x not a number`);
      }
      if (typeof agent.position.y !== 'number') {
        throw new Error(`Agent ${agent.id} position y not a number`);
      }
    }
    
    // Check command unit is positioned centrally
    const commandAgent = agents.find(a => a.type === 'command');
    if (commandAgent) {
      if (commandAgent.position.x !== 500 || commandAgent.position.y !== 500) {
        throw new Error('Command unit not positioned centrally');
      }
    }
  });

  // Test 10: Distance Calculation
  testRunner.addTest('Distance Calculation', () => {
    const commander = new SwarmWarfareCommander();
    
    const pos1 = { x: 0, y: 0 };
    const pos2 = { x: 3, y: 4 };
    const distance = commander.calculateDistance(pos1, pos2);
    
    if (Math.abs(distance - 5) > 0.001) {
      throw new Error(`Distance calculation incorrect: expected 5, got ${distance}`);
    }
    
    const pos3 = { x: 0, y: 0 };
    const pos4 = { x: 0, y: 0 };
    const distance2 = commander.calculateDistance(pos3, pos4);
    
    if (distance2 !== 0) {
      throw new Error(`Distance calculation incorrect: expected 0, got ${distance2}`);
    }
  });

  // Test 11: Intelligence Gathering
  testRunner.addTest('Intelligence Gathering', () => {
    const commander = new SwarmWarfareCommander();
    commander.initializeBattlefield('urban', 'test-mission');
    
    const reconUnit = commander.createAgent('recon', 'test-recon', {
      health: 60,
      damage: 15,
      speed: 120,
      accuracy: 0.95,
      range: 100,
      specialization: 'Intelligence',
      capabilities: ['scouting', 'stealth', 'detection', 'mapping']
    });
    
    const intelligence = commander.gatherIntelligence(reconUnit);
    
    if (!intelligence.unitId) {
      throw new Error('Intelligence unitId missing');
    }
    if (!intelligence.threats) {
      throw new Error('Intelligence threats missing');
    }
    if (!intelligence.position) {
      throw new Error('Intelligence position missing');
    }
    if (!intelligence.timestamp) {
      throw new Error('Intelligence timestamp missing');
    }
    
    if (intelligence.unitId !== 'test-recon') {
      throw new Error('Intelligence unitId incorrect');
    }
  });

  // Test 12: Performance Metrics Calculation
  testRunner.addTest('Performance Metrics Calculation', () => {
    const commander = new SwarmWarfareCommander();
    commander.deployCombatSwarm('urban');
    commander.initializeBattlefield('urban', 'test-mission');
    
    const coordinationScore = commander.calculateCoordinationScore();
    const resourceEfficiency = commander.calculateResourceEfficiency();
    const responseTime = commander.calculateAverageResponseTime();
    const adaptabilityIndex = commander.calculateAdaptabilityIndex();
    const scalabilityFactor = commander.calculateScalabilityFactor();
    const innovationScore = commander.calculateInnovationScore();
    const robustnessRating = commander.calculateRobustnessRating();
    
    if (typeof coordinationScore !== 'number') {
      throw new Error('Coordination score should be a number');
    }
    if (coordinationScore < 0 || coordinationScore > 100) {
      throw new Error(`Coordination score out of range: ${coordinationScore}`);
    }
    
    if (typeof resourceEfficiency !== 'number') {
      throw new Error('Resource efficiency should be a number');
    }
    if (resourceEfficiency < 0 || resourceEfficiency > 100) {
      throw new Error(`Resource efficiency out of range: ${resourceEfficiency}`);
    }
    
    if (typeof responseTime !== 'number') {
      throw new Error('Response time should be a number');
    }
    if (responseTime < 0) {
      throw new Error(`Response time should be non-negative: ${responseTime}`);
    }
    
    if (typeof adaptabilityIndex !== 'number') {
      throw new Error('Adaptability index should be a number');
    }
    if (adaptabilityIndex < 0 || adaptabilityIndex > 100) {
      throw new Error(`Adaptability index out of range: ${adaptabilityIndex}`);
    }
    
    if (typeof scalabilityFactor !== 'number') {
      throw new Error('Scalability factor should be a number');
    }
    if (scalabilityFactor < 0 || scalabilityFactor > 100) {
      throw new Error(`Scalability factor out of range: ${scalabilityFactor}`);
    }
    
    if (typeof innovationScore !== 'number') {
      throw new Error('Innovation score should be a number');
    }
    if (innovationScore < 0 || innovationScore > 100) {
      throw new Error(`Innovation score out of range: ${innovationScore}`);
    }
    
    if (typeof robustnessRating !== 'number') {
      throw new Error('Robustness rating should be a number');
    }
    if (robustnessRating < 0 || robustnessRating > 100) {
      throw new Error(`Robustness rating out of range: ${robustnessRating}`);
    }
  });

  // Test 13: Mission Report Generation
  testRunner.addTest('Mission Report Generation', () => {
    const commander = new SwarmWarfareCommander();
    commander.deployCombatSwarm('urban');
    commander.initializeBattlefield('urban', 'test-mission');
    
    const report = commander.generateMissionReport();
    
    if (!report.challengeId) {
      throw new Error('Mission report missing challengeId');
    }
    if (!report.status) {
      throw new Error('Mission report missing status');
    }
    if (!report.scenario) {
      throw new Error('Mission report missing scenario');
    }
    if (!report.mission) {
      throw new Error('Mission report missing mission');
    }
    if (!report.battlefield) {
      throw new Error('Mission report missing battlefield');
    }
    if (!report.swarm) {
      throw new Error('Mission report missing swarm');
    }
    if (!report.coordination) {
      throw new Error('Mission report missing coordination');
    }
    if (!report.performance) {
      throw new Error('Mission report missing performance');
    }
    if (!report.combat) {
      throw new Error('Mission report missing combat');
    }
    if (!report.timestamp) {
      throw new Error('Mission report missing timestamp');
    }
    if (!report.message) {
      throw new Error('Mission report missing message');
    }
    
    if (report.challengeId !== "1a2e81a1-e2b6-4a37-9ac4-590761b57b1e") {
      throw new Error('Incorrect challenge ID');
    }
    
    if (report.status !== 'completed') {
      throw new Error('Status should be completed');
    }
  });

  // Test 14: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Swarm Warfare Commander Execution');
    
    const commander = new SwarmWarfareCommander();
    commander.deployCombatSwarm('urban');
    commander.initializeBattlefield('urban', 'test-mission');
    
    // Simulate a quick tactical operation
    await commander.assessBattlefield();
    await commander.developTacticalPlan();
    
    benchmark.end('Swarm Warfare Commander Execution');
    
    const agents = Array.from(commander.swarm.values());
    if (agents.length < 10) {
      throw new Error(`Expected at least 10 agents, got ${agents.length}`);
    }
    
    const battlefield = commander.battlefield;
    if (!battlefield || !battlefield.objectives) {
      throw new Error('Battlefield not properly initialized');
    }
  });

  // Test 15: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    const result = await executeSwarmWarfareCommander();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID missing from result');
    }
    if (!result.status || result.status !== 'completed') {
      throw new Error('Challenge status should be completed');
    }
    if (!result.scenario) {
      throw new Error('Scenario missing from result');
    }
    if (!result.mission) {
      throw new Error('Mission missing from result');
    }
    if (!result.battlefield) {
      throw new Error('Battlefield data missing from result');
    }
    if (!result.swarm) {
      throw new Error('Swarm data missing from result');
    }
    if (!result.coordination) {
      throw new Error('Coordination data missing from result');
    }
    if (!result.performance) {
      throw new Error('Performance data missing from result');
    }
    if (!result.combat) {
      throw new Error('Combat data missing from result');
    }
    
    if (result.swarm.totalAgents < 10) {
      throw new Error(`Expected at least 10 agents, got ${result.swarm.totalAgents}`);
    }
    
    if (result.performance.missionSuccessRate < 0 || result.performance.missionSuccessRate > 100) {
      throw new Error(`Mission success rate out of range: ${result.performance.missionSuccessRate}`);
    }
    
    if (result.performance.coordinationScore < 0 || result.performance.coordinationScore > 100) {
      throw new Error(`Coordination score out of range: ${result.performance.coordinationScore}`);
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
  
  console.log("\n🎯 Swarm Warfare Commander Challenge Test Summary:");
  console.log(`  Total Tests: ${results.total}`);
  console.log(`  Passed: ${results.passed}`);
  console.log(`  Failed: ${results.failed}`);
  console.log(`  Success Rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);
  console.log(`  Duration: ${results.duration}ms`);
  
  if (results.failed === 0) {
    console.log("\n🏆 All tests passed! The Swarm Warfare Commander is ready for battle!");
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
