// rUv Economy Dominator Challenge Test Suite
// Comprehensive testing for economic optimization algorithms and rUv economy management

import { RuvEconomyDominator, executeRuvEconomyDominator } from './solution.js';

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
    console.log('🧪 Running rUv Economy Dominator Tests...\n');
    
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
  
  // Test 1: Class Existence
  testRunner.addTest('Class Existence', () => {
    if (typeof RuvEconomyDominator !== 'function') {
      throw new Error('RuvEconomyDominator class not found');
    }
    if (typeof executeRuvEconomyDominator !== 'function') {
      throw new Error('executeRuvEconomyDominator function not found');
    }
  });
  
  // Test 2: Instance Creation
  testRunner.addTest('Instance Creation', () => {
    const dominator = new RuvEconomyDominator();
    
    if (typeof dominator.totalRuv !== 'number') {
      throw new Error('totalRuv should be a number');
    }
    if (typeof dominator.marketEfficiency !== 'number') {
      throw new Error('marketEfficiency should be a number');
    }
    if (!Array.isArray(dominator.tradingStrategies)) {
      throw new Error('tradingStrategies should be an array');
    }
    if (typeof dominator.economicMetrics !== 'object') {
      throw new Error('economicMetrics should be an object');
    }
    if (!Array.isArray(dominator.optimizationHistory)) {
      throw new Error('optimizationHistory should be an array');
    }
  });
  
  // Test 3: Initial State
  testRunner.addTest('Initial State', () => {
    const dominator = new RuvEconomyDominator();
    
    if (dominator.totalRuv !== 0) {
      throw new Error(`Expected totalRuv to be 0, got ${dominator.totalRuv}`);
    }
    if (dominator.marketEfficiency !== 1.0) {
      throw new Error(`Expected marketEfficiency to be 1.0, got ${dominator.marketEfficiency}`);
    }
    if (dominator.tradingStrategies.length !== 0) {
      throw new Error(`Expected tradingStrategies to be empty, got ${dominator.tradingStrategies.length}`);
    }
    if (dominator.optimizationHistory.length !== 0) {
      throw new Error(`Expected optimizationHistory to be empty, got ${dominator.optimizationHistory.length}`);
    }
  });
  
  // Test 4: Economic Metrics Structure
  testRunner.addTest('Economic Metrics Structure', () => {
    const dominator = new RuvEconomyDominator();
    const metrics = dominator.economicMetrics;
    
    const requiredFields = ['liquidity', 'volatility', 'marketDepth', 'tradingVolume'];
    for (const field of requiredFields) {
      if (!(field in metrics)) {
        throw new Error(`Missing economic metric: ${field}`);
      }
    }
    
    if (typeof metrics.liquidity !== 'number') {
      throw new Error('liquidity should be a number');
    }
    if (typeof metrics.volatility !== 'number') {
      throw new Error('volatility should be a number');
    }
    if (typeof metrics.marketDepth !== 'number') {
      throw new Error('marketDepth should be a number');
    }
    if (typeof metrics.tradingVolume !== 'number') {
      throw new Error('tradingVolume should be a number');
    }
  });
  
  // Test 5: Liquidity Optimization
  testRunner.addTest('Liquidity Optimization', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const result = dominator.liquidityOptimization(marketData);
    
    if (result.type !== 'liquidity') {
      throw new Error(`Expected type 'liquidity', got '${result.type}'`);
    }
    if (typeof result.score !== 'number') {
      throw new Error('score should be a number');
    }
    if (typeof result.ruvMultiplier !== 'number') {
      throw new Error('ruvMultiplier should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
    if (result.score < 0 || result.score > 1) {
      throw new Error(`score should be between 0 and 1, got ${result.score}`);
    }
  });
  
  // Test 6: Volatility Management
  testRunner.addTest('Volatility Management', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.15, marketDepth: 1000, tradingVolume: 1000 };
    
    const result = dominator.volatilityManagement(marketData);
    
    if (result.type !== 'volatility') {
      throw new Error(`Expected type 'volatility', got '${result.type}'`);
    }
    if (typeof result.risk !== 'number') {
      throw new Error('risk should be a number');
    }
    if (typeof result.stabilityBonus !== 'number') {
      throw new Error('stabilityBonus should be a number');
    }
    if (typeof result.ruvMultiplier !== 'number') {
      throw new Error('ruvMultiplier should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
  });
  
  // Test 7: Arbitrage Detection
  testRunner.addTest('Arbitrage Detection', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const result = dominator.arbitrageDetection(marketData);
    
    if (result.type !== 'arbitrage') {
      throw new Error(`Expected type 'arbitrage', got '${result.type}'`);
    }
    if (typeof result.opportunities !== 'number') {
      throw new Error('opportunities should be a number');
    }
    if (typeof result.totalRuvPotential !== 'number') {
      throw new Error('totalRuvPotential should be a number');
    }
    if (typeof result.ruvMultiplier !== 'number') {
      throw new Error('ruvMultiplier should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
    if (result.opportunities >= 0 && result.opportunities <= 10) {
      // Valid range
    } else {
      throw new Error(`opportunities should be between 0 and 10, got ${result.opportunities}`);
    }
  });
  
  // Test 8: Market Making Strategy
  testRunner.addTest('Market Making Strategy', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1500, tradingVolume: 1000 };
    
    const result = dominator.marketMakingStrategy(marketData);
    
    if (result.type !== 'market-making') {
      throw new Error(`Expected type 'market-making', got '${result.type}'`);
    }
    if (typeof result.bidAskSpread !== 'number') {
      throw new Error('bidAskSpread should be a number');
    }
    if (typeof result.ruvPerTrade !== 'number') {
      throw new Error('ruvPerTrade should be a number');
    }
    if (typeof result.tradesPerHour !== 'number') {
      throw new Error('tradesPerHour should be a number');
    }
    if (typeof result.hourlyRuv !== 'number') {
      throw new Error('hourlyRuv should be a number');
    }
    if (typeof result.ruvMultiplier !== 'number') {
      throw new Error('ruvMultiplier should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
  });
  
  // Test 9: Risk Management
  testRunner.addTest('Risk Management', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.2, marketDepth: 1000, tradingVolume: 1000 };
    
    const result = dominator.riskManagement(marketData);
    
    if (result.type !== 'risk-management') {
      throw new Error(`Expected type 'risk-management', got '${result.type}'`);
    }
    if (typeof result.riskScore !== 'number') {
      throw new Error('riskScore should be a number');
    }
    if (typeof result.protectionLevel !== 'number') {
      throw new Error('protectionLevel should be a number');
    }
    if (typeof result.ruvProtection !== 'number') {
      throw new Error('ruvProtection should be a number');
    }
    if (typeof result.ruvMultiplier !== 'number') {
      throw new Error('ruvMultiplier should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
    if (result.riskScore < 0 || result.riskScore > 1) {
      throw new Error(`riskScore should be between 0 and 1, got ${result.riskScore}`);
    }
  });
  
  // Test 10: Risk Score Calculation
  testRunner.addTest('Risk Score Calculation', () => {
    const dominator = new RuvEconomyDominator();
    
    // Test with low risk data
    const lowRiskData = { liquidity: 1000, volatility: 0.05, marketDepth: 2000 };
    const lowRiskScore = dominator.calculateRiskScore(lowRiskData);
    
    // Test with high risk data
    const highRiskData = { liquidity: 100, volatility: 0.5, marketDepth: 500 };
    const highRiskScore = dominator.calculateRiskScore(highRiskData);
    
    if (typeof lowRiskScore !== 'number') {
      throw new Error('Risk score should be a number');
    }
    if (typeof highRiskScore !== 'number') {
      throw new Error('Risk score should be a number');
    }
    if (lowRiskScore < 0 || lowRiskScore > 1) {
      throw new Error(`Risk score should be between 0 and 1, got ${lowRiskScore}`);
    }
    if (highRiskScore < 0 || highRiskScore > 1) {
      throw new Error(`Risk score should be between 0 and 1, got ${highRiskScore}`);
    }
    if (lowRiskScore >= highRiskScore) {
      throw new Error('Low risk score should be lower than high risk score');
    }
  });
  
  // Test 11: Strategy Combination
  testRunner.addTest('Strategy Combination', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const strategies = [
      dominator.liquidityOptimization(marketData),
      dominator.volatilityManagement(marketData),
      dominator.arbitrageDetection(marketData),
      dominator.marketMakingStrategy(marketData),
      dominator.riskManagement(marketData)
    ];
    
    const combined = dominator.combineStrategies(strategies);
    
    if (!Array.isArray(combined.strategies)) {
      throw new Error('strategies should be an array');
    }
    if (combined.strategies.length !== strategies.length) {
      throw new Error(`Expected ${strategies.length} strategies, got ${combined.strategies.length}`);
    }
    if (typeof combined.totalEfficiency !== 'number') {
      throw new Error('totalEfficiency should be a number');
    }
    if (typeof combined.averageMultiplier !== 'number') {
      throw new Error('averageMultiplier should be a number');
    }
    if (typeof combined.synergyBonus !== 'number') {
      throw new Error('synergyBonus should be a number');
    }
    if (typeof combined.finalMultiplier !== 'number') {
      throw new Error('finalMultiplier should be a number');
    }
  });
  
  // Test 12: Economic Optimization
  testRunner.addTest('Economic Optimization', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const result = dominator.optimizeEconomy(marketData);
    
    if (typeof result.ruvGenerated !== 'number') {
      throw new Error('ruvGenerated should be a number');
    }
    if (typeof result.totalRuv !== 'number') {
      throw new Error('totalRuv should be a number');
    }
    if (typeof result.efficiency !== 'number') {
      throw new Error('efficiency should be a number');
    }
    if (typeof result.marketMultiplier !== 'number') {
      throw new Error('marketMultiplier should be a number');
    }
    if (typeof result.cycleBonus !== 'number') {
      throw new Error('cycleBonus should be a number');
    }
    if (!Array.isArray(result.strategies)) {
      throw new Error('strategies should be an array');
    }
    if (result.ruvGenerated <= 0) {
      throw new Error('ruvGenerated should be positive');
    }
  });
  
  // Test 13: Optimization History
  testRunner.addTest('Optimization History', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const initialHistoryLength = dominator.optimizationHistory.length;
    dominator.optimizeEconomy(marketData);
    const finalHistoryLength = dominator.optimizationHistory.length;
    
    if (finalHistoryLength !== initialHistoryLength + 1) {
      throw new Error('Optimization history should be updated');
    }
    
    const historyEntry = dominator.optimizationHistory[dominator.optimizationHistory.length - 1];
    if (typeof historyEntry.timestamp !== 'number') {
      throw new Error('History entry should have timestamp');
    }
    if (typeof historyEntry.strategies !== 'number') {
      throw new Error('History entry should have strategies count');
    }
    if (typeof historyEntry.efficiency !== 'number') {
      throw new Error('History entry should have efficiency');
    }
    if (typeof historyEntry.ruvGenerated !== 'number') {
      throw new Error('History entry should have ruvGenerated');
    }
  });
  
  // Test 14: Economic Cycle Simulation
  testRunner.addTest('Economic Cycle Simulation', () => {
    const dominator = new RuvEconomyDominator();
    const result = dominator.simulateEconomicCycles(5);
    
    if (typeof result.cycles !== 'number') {
      throw new Error('cycles should be a number');
    }
    if (result.cycles !== 5) {
      throw new Error(`Expected 5 cycles, got ${result.cycles}`);
    }
    if (!Array.isArray(result.results)) {
      throw new Error('results should be an array');
    }
    if (result.results.length !== 5) {
      throw new Error(`Expected 5 results, got ${result.results.length}`);
    }
    if (typeof result.summary !== 'object') {
      throw new Error('summary should be an object');
    }
    if (typeof result.totalRuv !== 'number') {
      throw new Error('totalRuv should be a number');
    }
    if (typeof result.finalEfficiency !== 'number') {
      throw new Error('finalEfficiency should be a number');
    }
  });
  
  // Test 15: Performance Summary
  testRunner.addTest('Performance Summary', () => {
    const dominator = new RuvEconomyDominator();
    
    // Create mock results
    const mockResults = [
      { ruvGenerated: 100, efficiency: 0.8 },
      { ruvGenerated: 120, efficiency: 0.9 },
      { ruvGenerated: 110, efficiency: 0.85 },
      { ruvGenerated: 130, efficiency: 0.95 },
      { ruvGenerated: 115, efficiency: 0.88 }
    ];
    
    const summary = dominator.generatePerformanceSummary(mockResults);
    
    if (typeof summary.totalRuvGenerated !== 'number') {
      throw new Error('totalRuvGenerated should be a number');
    }
    if (typeof summary.averageEfficiency !== 'number') {
      throw new Error('averageEfficiency should be a number');
    }
    if (typeof summary.maxRuvPerCycle !== 'number') {
      throw new Error('maxRuvPerCycle should be a number');
    }
    if (typeof summary.minRuvPerCycle !== 'number') {
      throw new Error('minRuvPerCycle should be a number');
    }
    if (typeof summary.consistencyScore !== 'number') {
      throw new Error('consistencyScore should be a number');
    }
    if (typeof summary.dominationLevel !== 'string') {
      throw new Error('dominationLevel should be a string');
    }
    
    if (summary.totalRuvGenerated !== 575) { // 100+120+110+130+115
      throw new Error(`Expected totalRuvGenerated 575, got ${summary.totalRuvGenerated}`);
    }
    if (summary.maxRuvPerCycle !== 130) {
      throw new Error(`Expected maxRuvPerCycle 130, got ${summary.maxRuvPerCycle}`);
    }
    if (summary.minRuvPerCycle !== 100) {
      throw new Error(`Expected minRuvPerCycle 100, got ${summary.minRuvPerCycle}`);
    }
  });
  
  // Test 16: Domination Level Classification
  testRunner.addTest('Domination Level Classification', () => {
    const dominator = new RuvEconomyDominator();
    
    // Test ECONOMIC DOMINATOR level
    const highResults = Array(10).fill({ ruvGenerated: 200 });
    const highSummary = dominator.generatePerformanceSummary(highResults);
    if (!highSummary.dominationLevel.includes('ECONOMIC DOMINATOR')) {
      throw new Error('High rUv should result in ECONOMIC DOMINATOR');
    }
    
    // Test MARKET LEADER level
    const mediumResults = Array(10).fill({ ruvGenerated: 80 });
    const mediumSummary = dominator.generatePerformanceSummary(mediumResults);
    if (!mediumSummary.dominationLevel.includes('MARKET LEADER')) {
      throw new Error('Medium rUv should result in MARKET LEADER');
    }
    
    // Test GROWING ECONOMY level
    const lowResults = Array(10).fill({ ruvGenerated: 30 });
    const lowSummary = dominator.generatePerformanceSummary(lowResults);
    if (!lowSummary.dominationLevel.includes('GROWING ECONOMY')) {
      throw new Error('Low rUv should result in GROWING ECONOMY');
    }
  });
  
  // Test 17: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    const result = await executeRuvEconomyDominator();
    
    if (!result.challengeId) {
      throw new Error('Challenge ID should be included');
    }
    if (result.status !== 'completed') {
      throw new Error(`Expected status 'completed', got '${result.status}'`);
    }
    if (!result.economicDomination) {
      throw new Error('Economic domination data should be included');
    }
    if (!result.performance) {
      throw new Error('Performance data should be included');
    }
    if (!result.timestamp) {
      throw new Error('Timestamp should be included');
    }
    if (!result.message) {
      throw new Error('Message should be included');
    }
  });
  
  // Test 18: Challenge Result Structure
  testRunner.addTest('Challenge Result Structure', async () => {
    const result = await executeRuvEconomyDominator();
    
    const requiredFields = ['challengeId', 'status', 'economicDomination', 'performance', 'timestamp', 'message'];
    for (const field of requiredFields) {
      if (!(field in result)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    if (typeof result.performance.totalRuvGenerated !== 'number') {
      throw new Error('totalRuvGenerated should be a number');
    }
    if (typeof result.performance.averageEfficiency !== 'number') {
      throw new Error('averageEfficiency should be a number');
    }
    if (typeof result.performance.dominationLevel !== 'string') {
      throw new Error('dominationLevel should be a string');
    }
    if (typeof result.performance.consistencyScore !== 'number') {
      throw new Error('consistencyScore should be a number');
    }
  });
  
  // Test 19: Market Data Evolution
  testRunner.addTest('Market Data Evolution', () => {
    const dominator = new RuvEconomyDominator();
    const result = dominator.simulateEconomicCycles(3);
    
    // Check that market data is included in results
    for (const cycleResult of result.results) {
      if (!cycleResult.marketData) {
        throw new Error('Market data should be included in each cycle result');
      }
      
      const marketData = cycleResult.marketData;
      if (typeof marketData.liquidity !== 'number') {
        throw new Error('Market data should have liquidity');
      }
      if (typeof marketData.volatility !== 'number') {
        throw new Error('Market data should have volatility');
      }
      if (typeof marketData.marketDepth !== 'number') {
        throw new Error('Market data should have marketDepth');
      }
      if (typeof marketData.tradingVolume !== 'number') {
        throw new Error('Market data should have tradingVolume');
      }
      if (typeof marketData.cycle !== 'number') {
        throw new Error('Market data should have cycle number');
      }
    }
  });
  
  // Test 20: Performance Benchmark
  testRunner.addTest('Performance Benchmark', async () => {
    benchmark.start('rUv Economy Dominator Execution');
    const result = await executeRuvEconomyDominator();
    benchmark.end();
    
    if (result.performance.totalRuvGenerated < 100) {
      throw new Error(`Total rUv generated too low: ${result.performance.totalRuvGenerated}`);
    }
    if (result.performance.averageEfficiency < 50) {
      throw new Error(`Average efficiency too low: ${result.performance.averageEfficiency}`);
    }
  });
  
  // Test 21: Strategy Efficiency Validation
  testRunner.addTest('Strategy Efficiency Validation', () => {
    const dominator = new RuvEconomyDominator();
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    
    const strategies = [
      dominator.liquidityOptimization(marketData),
      dominator.volatilityManagement(marketData),
      dominator.arbitrageDetection(marketData),
      dominator.marketMakingStrategy(marketData),
      dominator.riskManagement(marketData)
    ];
    
    for (const strategy of strategies) {
      if (strategy.efficiency < 0 || strategy.efficiency > 1) {
        throw new Error(`Strategy efficiency should be between 0 and 1, got ${strategy.efficiency}`);
      }
      if (strategy.ruvMultiplier < 1.0) {
        throw new Error(`Strategy rUv multiplier should be >= 1.0, got ${strategy.ruvMultiplier}`);
      }
    }
  });
  
  // Test 22: Economic Metrics Update
  testRunner.addTest('Economic Metrics Update', () => {
    const dominator = new RuvEconomyDominator();
    const initialLiquidity = dominator.economicMetrics.liquidity;
    const initialVolatility = dominator.economicMetrics.volatility;
    const initialMarketDepth = dominator.economicMetrics.marketDepth;
    const initialTradingVolume = dominator.economicMetrics.tradingVolume;
    
    dominator.simulateEconomicCycles(1);
    
    // Metrics should be updated after simulation
    if (dominator.economicMetrics.liquidity === initialLiquidity) {
      throw new Error('Liquidity should be updated after simulation');
    }
    if (dominator.economicMetrics.volatility === initialVolatility) {
      throw new Error('Volatility should be updated after simulation');
    }
    if (dominator.economicMetrics.marketDepth === initialMarketDepth) {
      throw new Error('Market depth should be updated after simulation');
    }
    if (dominator.economicMetrics.tradingVolume === initialTradingVolume) {
      throw new Error('Trading volume should be updated after simulation');
    }
  });
  
  // Test 23: rUv Accumulation
  testRunner.addTest('rUv Accumulation', () => {
    const dominator = new RuvEconomyDominator();
    const initialRuv = dominator.totalRuv;
    
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    dominator.optimizeEconomy(marketData);
    
    if (dominator.totalRuv <= initialRuv) {
      throw new Error('Total rUv should increase after optimization');
    }
  });
  
  // Test 24: Market Efficiency Update
  testRunner.addTest('Market Efficiency Update', () => {
    const dominator = new RuvEconomyDominator();
    const initialEfficiency = dominator.marketEfficiency;
    
    const marketData = { liquidity: 800, volatility: 0.1, marketDepth: 1000, tradingVolume: 1000 };
    dominator.optimizeEconomy(marketData);
    
    if (dominator.marketEfficiency === initialEfficiency) {
      throw new Error('Market efficiency should be updated after optimization');
    }
  });
  
  // Test 25: Challenge Message Validation
  testRunner.addTest('Challenge Message Validation', async () => {
    const result = await executeRuvEconomyDominator();
    
    if (!result.message.includes('rUv Economy successfully dominated')) {
      throw new Error(`Message should indicate successful domination, got: ${result.message}`);
    }
  });
  
  // Run all tests
  const success = await testRunner.runTests();
  benchmark.printSummary();
  
  if (success) {
    console.log('\n🎉 All tests passed! rUv Economy Dominator implementation is working correctly.');
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
