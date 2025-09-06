// Neural Mesh Coordinator Challenge Test Suite
// Comprehensive testing for advanced swarm coordination with neural mesh topology

import { neuralMeshCoordinator } from './solution.js';

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
    console.log('🧪 Running Neural Mesh Coordinator Tests...\n');
    
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
    if (typeof neuralMeshCoordinator !== 'function') {
      throw new Error('neuralMeshCoordinator function not found');
    }
  });
  
  // Test 2: Basic Execution
  testRunner.addTest('Basic Execution', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!result) {
      throw new Error('Function returned no result');
    }
    if (typeof result !== 'object') {
      throw new Error('Result should be an object');
    }
  });
  
  // Test 3: Result Structure
  testRunner.addTest('Result Structure', async () => {
    const result = await neuralMeshCoordinator();
    
    const requiredFields = ['meshId', 'topology', 'agents', 'connections', 'tasks', 'results', 'performance', 'status', 'message'];
    for (const field of requiredFields) {
      if (!(field in result)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
  });
  
  // Test 4: Mesh ID Format
  testRunner.addTest('Mesh ID Format', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!result.meshId.startsWith('neural-mesh-')) {
      throw new Error(`Invalid mesh ID format: ${result.meshId}`);
    }
  });
  
  // Test 5: Topology Type
  testRunner.addTest('Topology Type', async () => {
    const result = await neuralMeshCoordinator();
    
    if (result.topology !== 'neural-mesh') {
      throw new Error(`Expected topology 'neural-mesh', got '${result.topology}'`);
    }
  });
  
  // Test 6: Agents Structure
  testRunner.addTest('Agents Structure', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!Array.isArray(result.agents)) {
      throw new Error('Agents should be an array');
    }
    if (result.agents.length === 0) {
      throw new Error('Agents array should not be empty');
    }
    
    const requiredAgentFields = ['id', 'type', 'layer', 'capabilities', 'connections'];
    for (const agent of result.agents) {
      for (const field of requiredAgentFields) {
        if (!(field in agent)) {
          throw new Error(`Agent missing required field: ${field}`);
        }
      }
    }
  });
  
  // Test 7: Agent Types
  testRunner.addTest('Agent Types', async () => {
    const result = await neuralMeshCoordinator();
    
    const expectedTypes = ['coordinator', 'analyzer', 'optimizer', 'executor', 'monitor'];
    const actualTypes = result.agents.map(agent => agent.type);
    
    for (const expectedType of expectedTypes) {
      if (!actualTypes.includes(expectedType)) {
        throw new Error(`Missing agent type: ${expectedType}`);
      }
    }
  });
  
  // Test 8: Connections Structure
  testRunner.addTest('Connections Structure', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!Array.isArray(result.connections)) {
      throw new Error('Connections should be an array');
    }
    if (result.connections.length === 0) {
      throw new Error('Connections array should not be empty');
    }
    
    const requiredConnectionFields = ['from', 'to', 'weight', 'latency', 'bandwidth'];
    for (const connection of result.connections) {
      for (const field of requiredConnectionFields) {
        if (!(field in connection)) {
          throw new Error(`Connection missing required field: ${field}`);
        }
      }
    }
  });
  
  // Test 9: Connection Weights
  testRunner.addTest('Connection Weights', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const connection of result.connections) {
      if (connection.weight < 0.2 || connection.weight > 1.0) {
        throw new Error(`Connection weight out of range: ${connection.weight}`);
      }
    }
  });
  
  // Test 10: Connection Latency
  testRunner.addTest('Connection Latency', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const connection of result.connections) {
      if (connection.latency < 10 || connection.latency > 60) {
        throw new Error(`Connection latency out of range: ${connection.latency}ms`);
      }
    }
  });
  
  // Test 11: Connection Bandwidth
  testRunner.addTest('Connection Bandwidth', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const connection of result.connections) {
      if (connection.bandwidth < 100 || connection.bandwidth > 1000) {
        throw new Error(`Connection bandwidth out of range: ${connection.bandwidth}Mbps`);
      }
    }
  });
  
  // Test 12: Tasks Structure
  testRunner.addTest('Tasks Structure', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!Array.isArray(result.tasks)) {
      throw new Error('Tasks should be an array');
    }
    if (result.tasks.length === 0) {
      throw new Error('Tasks array should not be empty');
    }
    
    const requiredTaskFields = ['id', 'type', 'priority', 'assignedTo'];
    for (const task of result.tasks) {
      for (const field of requiredTaskFields) {
        if (!(field in task)) {
          throw new Error(`Task missing required field: ${field}`);
        }
      }
    }
  });
  
  // Test 13: Task Priorities
  testRunner.addTest('Task Priorities', async () => {
    const result = await neuralMeshCoordinator();
    
    const validPriorities = ['low', 'medium', 'high', 'critical'];
    for (const task of result.tasks) {
      if (!validPriorities.includes(task.priority)) {
        throw new Error(`Invalid task priority: ${task.priority}`);
      }
    }
  });
  
  // Test 14: Results Structure
  testRunner.addTest('Results Structure', async () => {
    const result = await neuralMeshCoordinator();
    
    if (!Array.isArray(result.results)) {
      throw new Error('Results should be an array');
    }
    if (result.results.length !== result.tasks.length) {
      throw new Error('Results length should match tasks length');
    }
    
    const requiredResultFields = ['taskId', 'status', 'processingTime', 'accuracy', 'neuralActivation', 'assignedAgent'];
    for (const taskResult of result.results) {
      for (const field of requiredResultFields) {
        if (!(field in taskResult)) {
          throw new Error(`Result missing required field: ${field}`);
        }
      }
    }
  });
  
  // Test 15: Result Status
  testRunner.addTest('Result Status', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const taskResult of result.results) {
      if (taskResult.status !== 'completed') {
        throw new Error(`Expected status 'completed', got '${taskResult.status}'`);
      }
    }
  });
  
  // Test 16: Processing Time Range
  testRunner.addTest('Processing Time Range', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const taskResult of result.results) {
      if (taskResult.processingTime < 50 || taskResult.processingTime > 550) {
        throw new Error(`Processing time out of range: ${taskResult.processingTime}ms`);
      }
    }
  });
  
  // Test 17: Accuracy Range
  testRunner.addTest('Accuracy Range', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const taskResult of result.results) {
      if (taskResult.accuracy < 0.85 || taskResult.accuracy > 1.0) {
        throw new Error(`Accuracy out of range: ${taskResult.accuracy}`);
      }
    }
  });
  
  // Test 18: Neural Activation Range
  testRunner.addTest('Neural Activation Range', async () => {
    const result = await neuralMeshCoordinator();
    
    for (const taskResult of result.results) {
      if (taskResult.neuralActivation < 0.7 || taskResult.neuralActivation > 1.0) {
        throw new Error(`Neural activation out of range: ${taskResult.neuralActivation}`);
      }
    }
  });
  
  // Test 19: Performance Metrics
  testRunner.addTest('Performance Metrics', async () => {
    const result = await neuralMeshCoordinator();
    
    const requiredMetrics = ['averageLatency', 'totalBandwidth', 'averageAccuracy', 'averageProcessingTime', 'meshEfficiency'];
    for (const metric of requiredMetrics) {
      if (!(metric in result.performance)) {
        throw new Error(`Missing performance metric: ${metric}`);
      }
    }
  });
  
  // Test 20: Performance Values
  testRunner.addTest('Performance Values', async () => {
    const result = await neuralMeshCoordinator();
    
    if (result.performance.averageLatency <= 0) {
      throw new Error(`Invalid average latency: ${result.performance.averageLatency}`);
    }
    if (result.performance.totalBandwidth <= 0) {
      throw new Error(`Invalid total bandwidth: ${result.performance.totalBandwidth}`);
    }
    if (result.performance.averageAccuracy < 0 || result.performance.averageAccuracy > 100) {
      throw new Error(`Invalid average accuracy: ${result.performance.averageAccuracy}`);
    }
    if (result.performance.averageProcessingTime <= 0) {
      throw new Error(`Invalid average processing time: ${result.performance.averageProcessingTime}`);
    }
    if (result.performance.meshEfficiency <= 0) {
      throw new Error(`Invalid mesh efficiency: ${result.performance.meshEfficiency}`);
    }
  });
  
  // Test 21: Status Message
  testRunner.addTest('Status Message', async () => {
    const result = await neuralMeshCoordinator();
    
    if (result.status !== 'operational') {
      throw new Error(`Expected status 'operational', got '${result.status}'`);
    }
    if (!result.message.includes('completed successfully')) {
      throw new Error(`Invalid status message: ${result.message}`);
    }
  });
  
  // Test 22: Agent Layer Distribution
  testRunner.addTest('Agent Layer Distribution', async () => {
    const result = await neuralMeshCoordinator();
    
    const layerCounts = {};
    for (const agent of result.agents) {
      layerCounts[agent.layer] = (layerCounts[agent.layer] || 0) + 1;
    }
    
    // Should have agents in layers 0, 1, and 2
    if (!layerCounts[0] || !layerCounts[1] || !layerCounts[2]) {
      throw new Error('Agents should be distributed across layers 0, 1, and 2');
    }
  });
  
  // Test 23: Connection Validity
  testRunner.addTest('Connection Validity', async () => {
    const result = await neuralMeshCoordinator();
    
    const agentIds = result.agents.map(agent => agent.id);
    for (const connection of result.connections) {
      if (!agentIds.includes(connection.from)) {
        throw new Error(`Connection from non-existent agent: ${connection.from}`);
      }
      if (!agentIds.includes(connection.to)) {
        throw new Error(`Connection to non-existent agent: ${connection.to}`);
      }
    }
  });
  
  // Test 24: Task Assignment Validity
  testRunner.addTest('Task Assignment Validity', async () => {
    const result = await neuralMeshCoordinator();
    
    const agentIds = result.agents.map(agent => agent.id);
    for (const task of result.tasks) {
      if (!agentIds.includes(task.assignedTo)) {
        throw new Error(`Task assigned to non-existent agent: ${task.assignedTo}`);
      }
    }
  });
  
  // Test 25: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('Neural Mesh Coordinator Execution');
    const result = await neuralMeshCoordinator();
    benchmark.end();
    
    if (result.performance.meshEfficiency < 20) {
      throw new Error(`Mesh efficiency too low: ${result.performance.meshEfficiency}%`);
    }
  });
  
  // Run all tests
  const success = await testRunner.runTests();
  benchmark.printSummary();
  
  if (success) {
    console.log('\n🎉 All tests passed! Neural Mesh Coordinator implementation is working correctly.');
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
