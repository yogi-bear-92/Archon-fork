// Lightning Deploy Master Challenge Test Suite
// Comprehensive testing for real-time deployment orchestration and autonomous agent execution

import { 
  lightningDeploy, 
  lightningDeployWithSandbox, 
  runChallenge, 
  AutonomousAgent, 
  DEPLOYMENT_CONFIG 
} from './solution.js';

// Test utilities
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }
  
  addTest(name, testFunction) {
    this.tests.push({ name, testFunction });
  }
  
  async runTests() {
    console.log('🧪 Running Lightning Deploy Master Tests...\n');
    
    for (const test of this.tests) {
      try {
        await test.testFunction();
        console.log(`✅ ${test.name}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ ${test.name}: ${error.message}`);
        this.failed++;
      }
    }
    
    console.log(`\n📊 Test Results: ${this.passed} passed, ${this.failed} failed`);
    return this.passed === this.tests.length;
  }
}

class PerformanceBenchmark {
  constructor() {
    this.measurements = [];
  }
  
  start(label) {
    this.currentLabel = label;
    this.startTime = performance.now();
  }
  
  end() {
    if (this.currentLabel && this.startTime) {
      const duration = performance.now() - this.startTime;
      this.measurements.push({
        label: this.currentLabel,
        duration: duration
      });
    }
  }
  
  addMeasurement(label, duration) {
    this.measurements.push({ label, duration });
  }
  
  printSummary() {
    console.log('\n⏱️  Performance Summary:');
    this.measurements.forEach(measurement => {
      console.log(`  ${measurement.label}: ${measurement.duration.toFixed(2)}ms`);
    });
  }
}

// Test cases
async function runTests() {
  const testRunner = new TestRunner();
  const benchmark = new PerformanceBenchmark();
  
  // Test 1: Function Existence
  testRunner.addTest('Function Existence', () => {
    if (typeof lightningDeploy !== 'function') {
      throw new Error('lightningDeploy function not found');
    }
    if (typeof lightningDeployWithSandbox !== 'function') {
      throw new Error('lightningDeployWithSandbox function not found');
    }
    if (typeof runChallenge !== 'function') {
      throw new Error('runChallenge function not found');
    }
    if (typeof AutonomousAgent !== 'function') {
      throw new Error('AutonomousAgent class not found');
    }
  });
  
  // Test 2: Configuration
  testRunner.addTest('Configuration', () => {
    if (!DEPLOYMENT_CONFIG) {
      throw new Error('DEPLOYMENT_CONFIG not found');
    }
    if (DEPLOYMENT_CONFIG.maxDeployTime !== 30000) {
      throw new Error(`Expected maxDeployTime 30000, got ${DEPLOYMENT_CONFIG.maxDeployTime}`);
    }
    if (!Array.isArray(DEPLOYMENT_CONFIG.autonomousCapabilities)) {
      throw new Error('autonomousCapabilities should be an array');
    }
    if (DEPLOYMENT_CONFIG.autonomousCapabilities.length === 0) {
      throw new Error('autonomousCapabilities should not be empty');
    }
  });
  
  // Test 3: AutonomousAgent Creation
  testRunner.addTest('AutonomousAgent Creation', () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    
    if (!agent.id) {
      throw new Error('Agent ID not generated');
    }
    if (agent.status !== 'created') {
      throw new Error(`Expected status 'created', got '${agent.status}'`);
    }
    if (!agent.metrics) {
      throw new Error('Agent metrics not initialized');
    }
  });
  
  // Test 4: Agent ID Format
  testRunner.addTest('Agent ID Format', () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    
    if (!agent.id.startsWith('agent_')) {
      throw new Error(`Invalid agent ID format: ${agent.id}`);
    }
  });
  
  // Test 5: Agent Deploy Method
  testRunner.addTest('Agent Deploy Method', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    
    if (typeof agent.deploy !== 'function') {
      throw new Error('deploy method not found');
    }
    
    const result = await agent.deploy();
    
    if (!result.success) {
      throw new Error('Deployment should succeed');
    }
    if (typeof result.deployTime !== 'number') {
      throw new Error('deployTime should be a number');
    }
    if (result.deployTime <= 0) {
      throw new Error('deployTime should be positive');
    }
  });
  
  // Test 6: Agent Status After Deploy
  testRunner.addTest('Agent Status After Deploy', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    if (agent.status !== 'deployed') {
      throw new Error(`Expected status 'deployed', got '${agent.status}'`);
    }
  });
  
  // Test 7: Agent Execute Method
  testRunner.addTest('Agent Execute Method', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'test-task',
      type: 'data-processing',
      description: 'Test task execution',
      data: [{ id: 1, value: 100 }]
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (!result.success) {
      throw new Error('Task execution should succeed');
    }
    if (!result.autonomous) {
      throw new Error('Execution should be autonomous');
    }
    if (!result.result) {
      throw new Error('Result should be returned');
    }
  });
  
  // Test 8: Agent Status After Execute
  testRunner.addTest('Agent Status After Execute', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'test-task',
      type: 'data-processing',
      description: 'Test task execution',
      data: [{ id: 1, value: 100 }]
    };
    
    await agent.executeAutonomously(task);
    
    if (agent.status !== 'completed') {
      throw new Error(`Expected status 'completed', got '${agent.status}'`);
    }
  });
  
  // Test 9: Task Processing - Data Processing
  testRunner.addTest('Task Processing - Data Processing', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'data-task',
      type: 'data-processing',
      description: 'Process test data',
      data: Array.from({ length: 10 }, (_, i) => ({ id: i, value: Math.random() }))
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (!result.result.processed) {
      throw new Error('Data should be processed');
    }
    if (typeof result.result.records !== 'number') {
      throw new Error('Records count should be a number');
    }
    if (!Array.isArray(result.result.transformations)) {
      throw new Error('Transformations should be an array');
    }
  });
  
  // Test 10: Task Processing - API Integration
  testRunner.addTest('Task Processing - API Integration', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'api-task',
      type: 'api-integration',
      description: 'Integrate with API',
      apiConfig: { endpoint: 'https://test-api.com' }
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (!result.result.connected) {
      throw new Error('API should be connected');
    }
    if (typeof result.result.responseTime !== 'number') {
      throw new Error('Response time should be a number');
    }
  });
  
  // Test 11: Task Processing - Computation
  testRunner.addTest('Task Processing - Computation', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'computation-task',
      type: 'computation',
      description: 'Perform computation',
      computation: { algorithm: 'test-algorithm' }
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (typeof result.result.result !== 'number') {
      throw new Error('Computation result should be a number');
    }
    if (!result.result.algorithm) {
      throw new Error('Algorithm should be specified');
    }
  });
  
  // Test 12: Task Processing - Generic
  testRunner.addTest('Task Processing - Generic', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'generic-task',
      type: 'unknown-type',
      description: 'Generic task execution'
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (!result.result.executed) {
      throw new Error('Generic task should be executed');
    }
    if (!result.result.output) {
      throw new Error('Output should be generated');
    }
  });
  
  // Test 13: Error Recovery
  testRunner.addTest('Error Recovery', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    // Create a task that will cause an error
    const task = {
      name: 'error-task',
      type: 'data-processing',
      description: 'Task that causes error',
      data: null // This should cause an error
    };
    
    // Mock the processData method to throw an error
    const originalProcessData = agent.processData;
    agent.processData = async () => {
      throw new Error('Simulated processing error');
    };
    
    const result = await agent.executeAutonomously(task);
    
    // Restore original method
    agent.processData = originalProcessData;
    
    // Should either succeed with recovery or fail gracefully
    if (!result.success && !result.recoveryFailed) {
      throw new Error('Error should be handled gracefully');
    }
  });
  
  // Test 14: Metrics Collection
  testRunner.addTest('Metrics Collection', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'metrics-task',
      type: 'data-processing',
      description: 'Test metrics collection',
      data: [{ id: 1, value: 100 }]
    };
    
    await agent.executeAutonomously(task);
    const metrics = agent.getMetrics();
    
    if (typeof metrics.deployTime !== 'number') {
      throw new Error('deployTime should be a number');
    }
    if (typeof metrics.executionTime !== 'number') {
      throw new Error('executionTime should be a number');
    }
    if (typeof metrics.tasksCompleted !== 'number') {
      throw new Error('tasksCompleted should be a number');
    }
    if (typeof metrics.errors !== 'number') {
      throw new Error('errors should be a number');
    }
    if (metrics.tasksCompleted < 1) {
      throw new Error('At least one task should be completed');
    }
  });
  
  // Test 15: Lightning Deploy Function
  testRunner.addTest('Lightning Deploy Function', async () => {
    const result = await lightningDeploy();
    
    if (!result.success) {
      throw new Error('Lightning deploy should succeed');
    }
    if (!result.message.includes('Agent deployed and task completed <30s')) {
      throw new Error('Message should match expected pattern');
    }
    if (typeof result.totalTime !== 'number') {
      throw new Error('totalTime should be a number');
    }
    if (result.totalTime <= 0) {
      throw new Error('totalTime should be positive');
    }
  });
  
  // Test 16: Lightning Speed Validation
  testRunner.addTest('Lightning Speed Validation', async () => {
    const result = await lightningDeploy();
    
    if (result.totalTime >= DEPLOYMENT_CONFIG.maxDeployTime) {
      throw new Error(`Total time ${result.totalTime}ms exceeds max ${DEPLOYMENT_CONFIG.maxDeployTime}ms`);
    }
    if (!result.lightningSpeed) {
      throw new Error('Lightning speed should be true');
    }
  });
  
  // Test 17: Challenge Completion
  testRunner.addTest('Challenge Completion', async () => {
    const result = await lightningDeploy();
    
    if (!result.challengeCompleted) {
      throw new Error('Challenge should be completed');
    }
    if (!result.autonomous) {
      throw new Error('Execution should be autonomous');
    }
    if (!result.agent) {
      throw new Error('Agent information should be included');
    }
    if (!result.taskResult) {
      throw new Error('Task result should be included');
    }
  });
  
  // Test 18: Agent Capabilities
  testRunner.addTest('Agent Capabilities', async () => {
    const result = await lightningDeploy();
    
    if (!Array.isArray(result.agent.capabilities)) {
      throw new Error('Agent capabilities should be an array');
    }
    if (result.agent.capabilities.length === 0) {
      throw new Error('Agent capabilities should not be empty');
    }
    
    const expectedCapabilities = DEPLOYMENT_CONFIG.autonomousCapabilities;
    for (const capability of expectedCapabilities) {
      if (!result.agent.capabilities.includes(capability)) {
        throw new Error(`Missing capability: ${capability}`);
      }
    }
  });
  
  // Test 19: Run Challenge Function
  testRunner.addTest('Run Challenge Function', async () => {
    const result = await runChallenge();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID should be included');
    }
    if (!result.result) {
      throw new Error('Result should be included');
    }
    if (!result.validation) {
      throw new Error('Validation should be included');
    }
    
    if (typeof result.validation.isValid !== 'boolean') {
      throw new Error('isValid should be a boolean');
    }
    if (typeof result.validation.isLightning !== 'boolean') {
      throw new Error('isLightning should be a boolean');
    }
    if (typeof result.validation.passed !== 'boolean') {
      throw new Error('passed should be a boolean');
    }
  });
  
  // Test 20: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Lightning Deploy Execution');
    const result = await lightningDeploy();
    benchmark.end();
    
    if (result.totalTime > 5000) { // 5 seconds max for testing
      throw new Error(`Execution too slow: ${result.totalTime}ms`);
    }
  });
  
  // Test 21: Sandbox Integration Fallback
  testRunner.addTest('Sandbox Integration Fallback', async () => {
    const result = await lightningDeployWithSandbox();
    
    if (!result.success) {
      throw new Error('Sandbox integration should succeed with fallback');
    }
    if (!result.message.includes('Agent deployed and task completed <30s')) {
      throw new Error('Message should match expected pattern');
    }
  });
  
  // Test 22: Multiple Agent Creation
  testRunner.addTest('Multiple Agent Creation', () => {
    const agent1 = new AutonomousAgent({ capabilities: ['test1'] });
    const agent2 = new AutonomousAgent({ capabilities: ['test2'] });
    
    if (agent1.id === agent2.id) {
      throw new Error('Agent IDs should be unique');
    }
  });
  
  // Test 23: Agent Status Transitions
  testRunner.addTest('Agent Status Transitions', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    
    if (agent.status !== 'created') {
      throw new Error('Initial status should be created');
    }
    
    await agent.deploy();
    if (agent.status !== 'deployed') {
      throw new Error('Status after deploy should be deployed');
    }
    
    const task = {
      name: 'status-task',
      type: 'data-processing',
      description: 'Test status transitions',
      data: [{ id: 1, value: 100 }]
    };
    
    await agent.executeAutonomously(task);
    if (agent.status !== 'completed') {
      throw new Error('Status after execution should be completed');
    }
  });
  
  // Test 24: Task Result Structure
  testRunner.addTest('Task Result Structure', async () => {
    const agent = new AutonomousAgent({ capabilities: ['test'] });
    await agent.deploy();
    
    const task = {
      name: 'structure-task',
      type: 'data-processing',
      description: 'Test result structure',
      data: [{ id: 1, value: 100 }]
    };
    
    const result = await agent.executeAutonomously(task);
    
    if (!result.hasOwnProperty('success')) {
      throw new Error('Result should have success property');
    }
    if (!result.hasOwnProperty('result')) {
      throw new Error('Result should have result property');
    }
    if (!result.hasOwnProperty('metrics')) {
      throw new Error('Result should have metrics property');
    }
    if (!result.hasOwnProperty('autonomous')) {
      throw new Error('Result should have autonomous property');
    }
  });
  
  // Test 25: Challenge Validation
  testRunner.addTest('Challenge Validation', async () => {
    const result = await runChallenge();
    
    if (!result.validation.passed) {
      throw new Error('Challenge validation should pass');
    }
    if (!result.validation.isValid) {
      throw new Error('Result should be valid');
    }
    if (!result.validation.isLightning) {
      throw new Error('Execution should be lightning fast');
    }
  });
  
  // Run all tests
  const success = await testRunner.runTests();
  benchmark.printSummary();
  
  if (success) {
    console.log('\n🎉 All tests passed! Lightning Deploy Master implementation is working correctly.');
  } else {
    console.log('\n❌ Some tests failed. Please check the implementation.');
  }
  
  return success;
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests().catch(console.error);
}

export { TestRunner, PerformanceBenchmark, runTests };
