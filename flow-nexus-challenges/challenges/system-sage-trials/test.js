// The System Sage Trials Challenge Test Suite
// Comprehensive testing for distributed system architecture design

import { SystemSageTrials, executeSystemSageTrials } from './solution.js';

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
    console.log("🧪 Running System Sage Trials Challenge Tests...\n");
    
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

  // Test 1: System Sage Trials Initialization
  testRunner.addTest('System Sage Trials Initialization', () => {
    const sage = new SystemSageTrials();
    
    if (!sage.architecture) {
      throw new Error('Architecture not initialized');
    }
    if (!sage.loadBalancer) {
      throw new Error('Load balancer not initialized');
    }
    if (!sage.dataLayer) {
      throw new Error('Data layer not initialized');
    }
    if (!sage.faultTolerance) {
      throw new Error('Fault tolerance not initialized');
    }
    if (!sage.monitoring) {
      throw new Error('Monitoring system not initialized');
    }
    if (!sage.performance) {
      throw new Error('Performance optimizer not initialized');
    }
  });

  // Test 2: System Overview Design
  testRunner.addTest('System Overview Design', () => {
    const sage = new SystemSageTrials();
    const overview = sage.designSystemOverview();
    
    if (!overview.name) {
      throw new Error('System name missing');
    }
    if (!overview.description) {
      throw new Error('System description missing');
    }
    if (!overview.requirements) {
      throw new Error('Requirements missing');
    }
    if (!overview.architecture) {
      throw new Error('Architecture missing');
    }
    
    if (overview.requirements.concurrentUsers !== 1000000) {
      throw new Error(`Expected 1M concurrent users, got ${overview.requirements.concurrentUsers}`);
    }
    if (overview.requirements.uptime !== 99.99) {
      throw new Error(`Expected 99.99% uptime, got ${overview.requirements.uptime}`);
    }
    if (overview.requirements.latency !== 100) {
      throw new Error(`Expected 100ms latency, got ${overview.requirements.latency}`);
    }
    
    if (!overview.architecture.regions || overview.architecture.regions.length < 3) {
      throw new Error('Expected at least 3 regions');
    }
    if (!overview.architecture.components || overview.architecture.components.length < 5) {
      throw new Error('Expected at least 5 components');
    }
    if (!overview.architecture.patterns || overview.architecture.patterns.length < 4) {
      throw new Error('Expected at least 4 patterns');
    }
  });

  // Test 3: Load Balancing Design
  testRunner.addTest('Load Balancing Design', () => {
    const sage = new SystemSageTrials();
    const loadBalancing = sage.designLoadBalancing();
    
    if (!loadBalancing.globalLoadBalancer) {
      throw new Error('Global load balancer missing');
    }
    if (!loadBalancing.regionalLoadBalancers) {
      throw new Error('Regional load balancers missing');
    }
    if (!loadBalancing.cdn) {
      throw new Error('CDN configuration missing');
    }
    if (!loadBalancing.trafficShaping) {
      throw new Error('Traffic shaping missing');
    }
    
    if (!loadBalancing.globalLoadBalancer.type) {
      throw new Error('Global load balancer type missing');
    }
    if (!loadBalancing.regionalLoadBalancers.algorithm) {
      throw new Error('Regional load balancer algorithm missing');
    }
    if (!loadBalancing.cdn.providers || loadBalancing.cdn.providers.length < 2) {
      throw new Error('Expected at least 2 CDN providers');
    }
  });

  // Test 4: Data Layer Design
  testRunner.addTest('Data Layer Design', () => {
    const sage = new SystemSageTrials();
    const dataLayer = sage.designDataLayer();
    
    if (!dataLayer.databaseStrategy) {
      throw new Error('Database strategy missing');
    }
    if (!dataLayer.sharding) {
      throw new Error('Sharding strategy missing');
    }
    if (!dataLayer.replication) {
      throw new Error('Replication strategy missing');
    }
    if (!dataLayer.caching) {
      throw new Error('Caching strategy missing');
    }
    if (!dataLayer.dataPersistence) {
      throw new Error('Data persistence missing');
    }
    
    if (!dataLayer.databaseStrategy.primary) {
      throw new Error('Primary database missing');
    }
    if (!dataLayer.sharding.strategy) {
      throw new Error('Sharding strategy type missing');
    }
    if (dataLayer.sharding.shardCount < 100) {
      throw new Error(`Expected at least 100 shards, got ${dataLayer.sharding.shardCount}`);
    }
    if (dataLayer.caching.levels.length < 3) {
      throw new Error('Expected at least 3 cache levels');
    }
  });

  // Test 5: Fault Tolerance Design
  testRunner.addTest('Fault Tolerance Design', () => {
    const sage = new SystemSageTrials();
    const faultTolerance = sage.designFaultTolerance();
    
    if (!faultTolerance.circuitBreakers) {
      throw new Error('Circuit breakers missing');
    }
    if (!faultTolerance.bulkheadPattern) {
      throw new Error('Bulkhead pattern missing');
    }
    if (!faultTolerance.gracefulDegradation) {
      throw new Error('Graceful degradation missing');
    }
    if (!faultTolerance.disasterRecovery) {
      throw new Error('Disaster recovery missing');
    }
    
    if (!faultTolerance.circuitBreakers.implementation) {
      throw new Error('Circuit breaker implementation missing');
    }
    if (!faultTolerance.bulkheadPattern.isolation) {
      throw new Error('Bulkhead isolation missing');
    }
    if (!faultTolerance.disasterRecovery.rto) {
      throw new Error('RTO missing');
    }
    if (!faultTolerance.disasterRecovery.rpo) {
      throw new Error('RPO missing');
    }
  });

  // Test 6: Monitoring Design
  testRunner.addTest('Monitoring Design', () => {
    const sage = new SystemSageTrials();
    const monitoring = sage.designMonitoring();
    
    if (!monitoring.distributedTracing) {
      throw new Error('Distributed tracing missing');
    }
    if (!monitoring.metrics) {
      throw new Error('Metrics missing');
    }
    if (!monitoring.logging) {
      throw new Error('Logging missing');
    }
    if (!monitoring.alerting) {
      throw new Error('Alerting missing');
    }
    if (!monitoring.dashboards) {
      throw new Error('Dashboards missing');
    }
    
    if (!monitoring.distributedTracing.system) {
      throw new Error('Tracing system missing');
    }
    if (!monitoring.metrics.collection) {
      throw new Error('Metrics collection missing');
    }
    if (!monitoring.logging.system) {
      throw new Error('Logging system missing');
    }
    if (monitoring.alerting.rules.length < 3) {
      throw new Error('Expected at least 3 alerting rules');
    }
  });

  // Test 7: Capacity Planning
  testRunner.addTest('Capacity Planning', () => {
    const sage = new SystemSageTrials();
    const capacityPlanning = sage.performCapacityPlanning();
    
    if (!capacityPlanning.trafficProjections) {
      throw new Error('Traffic projections missing');
    }
    if (!capacityPlanning.resourceRequirements) {
      throw new Error('Resource requirements missing');
    }
    if (!capacityPlanning.costEstimation) {
      throw new Error('Cost estimation missing');
    }
    if (!capacityPlanning.scalingTriggers) {
      throw new Error('Scaling triggers missing');
    }
    
    if (capacityPlanning.trafficProjections.concurrentUsers !== 1000000) {
      throw new Error(`Expected 1M concurrent users, got ${capacityPlanning.trafficProjections.concurrentUsers}`);
    }
    if (capacityPlanning.trafficProjections.requestsPerSecond < 100000) {
      throw new Error(`Expected high RPS, got ${capacityPlanning.trafficProjections.requestsPerSecond}`);
    }
    if (!capacityPlanning.resourceRequirements.compute) {
      throw new Error('Compute requirements missing');
    }
    if (!capacityPlanning.resourceRequirements.database) {
      throw new Error('Database requirements missing');
    }
  });

  // Test 8: Failure Scenarios Analysis
  testRunner.addTest('Failure Scenarios Analysis', () => {
    const sage = new SystemSageTrials();
    const failureScenarios = sage.analyzeFailureScenarios();
    
    if (!failureScenarios.scenarios) {
      throw new Error('Failure scenarios missing');
    }
    if (!failureScenarios.recoveryProcedures) {
      throw new Error('Recovery procedures missing');
    }
    if (!failureScenarios.testing) {
      throw new Error('Testing strategy missing');
    }
    
    if (failureScenarios.scenarios.length < 3) {
      throw new Error(`Expected at least 3 failure scenarios, got ${failureScenarios.scenarios.length}`);
    }
    
    const requiredProps = ['type', 'probability', 'impact', 'mitigation', 'rto', 'rpo'];
    for (const scenario of failureScenarios.scenarios) {
      for (const prop of requiredProps) {
        if (scenario[prop] === undefined) {
          throw new Error(`Scenario missing property: ${prop}`);
        }
      }
    }
  });

  // Test 9: Implementation Plan
  testRunner.addTest('Implementation Plan', () => {
    const sage = new SystemSageTrials();
    const implementationPlan = sage.createImplementationPlan();
    
    if (!implementationPlan.phases) {
      throw new Error('Implementation phases missing');
    }
    if (!implementationPlan.milestones) {
      throw new Error('Milestones missing');
    }
    if (!implementationPlan.risks) {
      throw new Error('Risks missing');
    }
    if (!implementationPlan.mitigation) {
      throw new Error('Mitigation strategies missing');
    }
    
    if (implementationPlan.phases.length < 4) {
      throw new Error(`Expected at least 4 phases, got ${implementationPlan.phases.length}`);
    }
    
    const requiredProps = ['phase', 'duration', 'deliverables'];
    for (const phase of implementationPlan.phases) {
      for (const prop of requiredProps) {
        if (phase[prop] === undefined) {
          throw new Error(`Phase missing property: ${prop}`);
        }
      }
    }
  });

  // Test 10: Scalability Evaluation
  testRunner.addTest('Scalability Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const score = sage.evaluateScalability(architecture);
    
    if (typeof score !== 'number') {
      throw new Error('Scalability score should be a number');
    }
    if (score < 0 || score > 100) {
      throw new Error(`Scalability score out of range: ${score}`);
    }
    
    if (score < 80) {
      throw new Error(`Scalability score too low: ${score}`);
    }
  });

  // Test 11: Fault Tolerance Evaluation
  testRunner.addTest('Fault Tolerance Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const score = sage.evaluateFaultTolerance(architecture);
    
    if (typeof score !== 'number') {
      throw new Error('Fault tolerance score should be a number');
    }
    if (score < 0 || score > 100) {
      throw new Error(`Fault tolerance score out of range: ${score}`);
    }
    
    if (score < 70) {
      throw new Error(`Fault tolerance score too low: ${score}`);
    }
  });

  // Test 12: Performance Evaluation
  testRunner.addTest('Performance Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const score = sage.evaluatePerformance(architecture);
    
    if (typeof score !== 'number') {
      throw new Error('Performance score should be a number');
    }
    if (score < 0 || score > 100) {
      throw new Error(`Performance score out of range: ${score}`);
    }
    
    if (score < 70) {
      throw new Error(`Performance score too low: ${score}`);
    }
  });

  // Test 13: Innovation Evaluation
  testRunner.addTest('Innovation Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const score = sage.evaluateInnovation(architecture);
    
    if (typeof score !== 'number') {
      throw new Error('Innovation score should be a number');
    }
    if (score < 0 || score > 100) {
      throw new Error(`Innovation score out of range: ${score}`);
    }
  });

  // Test 14: Practicality Evaluation
  testRunner.addTest('Practicality Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const score = sage.evaluatePracticality(architecture);
    
    if (typeof score !== 'number') {
      throw new Error('Practicality score should be a number');
    }
    if (score < 0 || score > 100) {
      throw new Error(`Practicality score out of range: ${score}`);
    }
  });

  // Test 15: Grade Assignment
  testRunner.addTest('Grade Assignment', () => {
    const sage = new SystemSageTrials();
    
    const grades = [
      { score: 98, expected: 'A+ (Exceptional)' },
      { score: 92, expected: 'A (Excellent)' },
      { score: 85, expected: 'B+ (Very Good)' },
      { score: 75, expected: 'B (Good)' },
      { score: 65, expected: 'C+ (Satisfactory)' },
      { score: 55, expected: 'C (Adequate)' },
      { score: 45, expected: 'D (Needs Improvement)' }
    ];
    
    for (const { score, expected } of grades) {
      const grade = sage.getGrade(score);
      if (grade !== expected) {
        throw new Error(`Expected grade "${expected}" for score ${score}, got "${grade}"`);
      }
    }
  });

  // Test 16: Feedback Generation
  testRunner.addTest('Feedback Generation', () => {
    const sage = new SystemSageTrials();
    
    const feedbacks = [
      { score: 95, shouldContain: 'Outstanding' },
      { score: 85, shouldContain: 'Excellent' },
      { score: 75, shouldContain: 'Very good' },
      { score: 65, shouldContain: 'Good effort' },
      { score: 45, shouldContain: 'needs substantial improvement' }
    ];
    
    for (const { score, shouldContain } of feedbacks) {
      const feedback = sage.generateFeedback(score);
      if (!feedback.includes(shouldContain)) {
        throw new Error(`Feedback for score ${score} should contain "${shouldContain}"`);
      }
    }
  });

  // Test 17: System Design Evaluation
  testRunner.addTest('System Design Evaluation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const evaluation = sage.evaluateSystemDesign(architecture);
    
    if (!evaluation.scalability) {
      throw new Error('Scalability evaluation missing');
    }
    if (!evaluation.faultTolerance) {
      throw new Error('Fault tolerance evaluation missing');
    }
    if (!evaluation.performance) {
      throw new Error('Performance evaluation missing');
    }
    if (!evaluation.innovation) {
      throw new Error('Innovation evaluation missing');
    }
    if (!evaluation.practicality) {
      throw new Error('Practicality evaluation missing');
    }
    if (!evaluation.overall) {
      throw new Error('Overall evaluation missing');
    }
    
    if (evaluation.overall.score < 0 || evaluation.overall.score > 100) {
      throw new Error(`Overall score out of range: ${evaluation.overall.score}`);
    }
    
    if (!evaluation.overall.grade) {
      throw new Error('Overall grade missing');
    }
    if (!evaluation.overall.feedback) {
      throw new Error('Overall feedback missing');
    }
  });

  // Test 18: System Sage Report Generation
  testRunner.addTest('System Sage Report Generation', () => {
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const evaluation = sage.evaluateSystemDesign(architecture);
    const report = sage.generateSystemSageReport(architecture, evaluation);
    
    if (!report.challengeId) {
      throw new Error('Challenge ID missing from report');
    }
    if (!report.status) {
      throw new Error('Status missing from report');
    }
    if (!report.title) {
      throw new Error('Title missing from report');
    }
    if (!report.description) {
      throw new Error('Description missing from report');
    }
    if (!report.architecture) {
      throw new Error('Architecture missing from report');
    }
    if (!report.planning) {
      throw new Error('Planning missing from report');
    }
    if (!report.evaluation) {
      throw new Error('Evaluation missing from report');
    }
    if (!report.performance) {
      throw new Error('Performance missing from report');
    }
    if (!report.requirements) {
      throw new Error('Requirements missing from report');
    }
    if (!report.timestamp) {
      throw new Error('Timestamp missing from report');
    }
    if (!report.message) {
      throw new Error('Message missing from report');
    }
    
    if (report.challengeId !== "5afd06e6-b502-49ff-ae0c-565344899e12") {
      throw new Error('Incorrect challenge ID');
    }
    
    if (report.status !== 'completed') {
      throw new Error('Status should be completed');
    }
    
    if (report.requirements.concurrentUsers !== 1000000) {
      throw new Error('Requirements should specify 1M concurrent users');
    }
    
    if (report.performance.overallScore < 0 || report.performance.overallScore > 100) {
      throw new Error(`Overall score out of range: ${report.performance.overallScore}`);
    }
  });

  // Test 19: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('System Sage Trials Execution');
    
    const sage = new SystemSageTrials();
    const architecture = sage.designSystemArchitecture();
    const evaluation = sage.evaluateSystemDesign(architecture);
    
    benchmark.end('System Sage Trials Execution');
    
    if (!architecture.overview) {
      throw new Error('Architecture overview missing');
    }
    if (!evaluation.overall) {
      throw new Error('Overall evaluation missing');
    }
    
    if (evaluation.overall.score < 80) {
      throw new Error(`Overall score too low: ${evaluation.overall.score}`);
    }
  });

  // Test 20: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    const result = await executeSystemSageTrials();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID missing from result');
    }
    if (!result.status || result.status !== 'completed') {
      throw new Error('Challenge status should be completed');
    }
    if (!result.title) {
      throw new Error('Title missing from result');
    }
    if (!result.description) {
      throw new Error('Description missing from result');
    }
    if (!result.architecture) {
      throw new Error('Architecture missing from result');
    }
    if (!result.planning) {
      throw new Error('Planning missing from result');
    }
    if (!result.evaluation) {
      throw new Error('Evaluation missing from result');
    }
    if (!result.performance) {
      throw new Error('Performance missing from result');
    }
    if (!result.requirements) {
      throw new Error('Requirements missing from result');
    }
    
    if (result.requirements.concurrentUsers !== 1000000) {
      throw new Error(`Expected 1M concurrent users, got ${result.requirements.concurrentUsers}`);
    }
    
    if (result.performance.overallScore < 0 || result.performance.overallScore > 100) {
      throw new Error(`Overall score out of range: ${result.performance.overallScore}`);
    }
    
    if (result.performance.scalabilityScore < 80) {
      throw new Error(`Scalability score too low: ${result.performance.scalabilityScore}`);
    }
    
    if (result.performance.faultToleranceScore < 70) {
      throw new Error(`Fault tolerance score too low: ${result.performance.faultToleranceScore}`);
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
  
  console.log("\n🎯 System Sage Trials Challenge Test Summary:");
  console.log(`  Total Tests: ${results.total}`);
  console.log(`  Passed: ${results.passed}`);
  console.log(`  Failed: ${results.failed}`);
  console.log(`  Success Rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);
  console.log(`  Duration: ${results.duration}ms`);
  
  if (results.failed === 0) {
    console.log("\n🏆 All tests passed! The System Sage is ready for Queen Seraphina's judgment!");
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
