// Neural Trading Trials Challenge Test Suite
// Comprehensive testing for AI trading system with neural networks

import { NeuralTradingTrials, executeNeuralTradingTrials } from './solution.js';

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
    console.log('🧪 Running Neural Trading Trials Tests...\n');
    
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
  
  // Test 1: Neural Network Creation
  testRunner.addTest('Neural Network Creation', () => {
    const challenge = new NeuralTradingTrials();
    if (!challenge.neuralNetwork) {
      throw new Error('Neural network not initialized');
    }
    if (challenge.neuralNetwork.inputSize !== 6) {
      throw new Error(`Expected input size 6, got ${challenge.neuralNetwork.inputSize}`);
    }
    if (challenge.neuralNetwork.outputSize !== 3) {
      throw new Error(`Expected output size 3, got ${challenge.neuralNetwork.outputSize}`);
    }
  });
  
  // Test 2: Technical Indicators
  testRunner.addTest('Technical Indicators - RSI', () => {
    const challenge = new NeuralTradingTrials();
    const prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113];
    const rsi = challenge.prepareFeatures(prices);
    
    if (rsi.length !== 6) {
      throw new Error(`Expected 6 features, got ${rsi.length}`);
    }
    if (rsi.some(f => isNaN(f) || !isFinite(f))) {
      throw new Error('Features contain invalid values');
    }
  });
  
  // Test 3: Technical Indicators - MACD
  testRunner.addTest('Technical Indicators - MACD', () => {
    const challenge = new NeuralTradingTrials();
    const prices = Array.from({ length: 50 }, (_, i) => 100 + i * 0.5 + Math.sin(i * 0.1) * 2);
    const features = challenge.prepareFeatures(prices);
    
    if (features.length !== 6) {
      throw new Error(`Expected 6 features, got ${features.length}`);
    }
    if (features.some(f => isNaN(f) || !isFinite(f))) {
      throw new Error('Features contain invalid values');
    }
  });
  
  // Test 4: Technical Indicators - Bollinger Bands
  testRunner.addTest('Technical Indicators - Bollinger Bands', () => {
    const challenge = new NeuralTradingTrials();
    const prices = Array.from({ length: 30 }, (_, i) => 100 + i * 0.3 + Math.sin(i * 0.2) * 1.5);
    const features = challenge.prepareFeatures(prices);
    
    if (features.length !== 6) {
      throw new Error(`Expected 6 features, got ${features.length}`);
    }
    if (features.some(f => isNaN(f) || !isFinite(f))) {
      throw new Error('Features contain invalid values');
    }
  });
  
  // Test 5: Risk Manager
  testRunner.addTest('Risk Manager - Position Sizing', () => {
    const challenge = new NeuralTradingTrials();
    const positionSize = challenge.riskManager.calculatePositionSize(10000, 0.8, 0.02);
    
    if (positionSize <= 0) {
      throw new Error('Position size should be positive');
    }
    if (positionSize > 10000 * 0.1) {
      throw new Error('Position size exceeds maximum allowed');
    }
  });
  
  // Test 6: Risk Manager - Stop Loss
  testRunner.addTest('Risk Manager - Stop Loss Logic', () => {
    const challenge = new NeuralTradingTrials();
    const entryPrice = 100;
    const stopLossPrice = 98; // 2% below entry
    
    const shouldStop = challenge.riskManager.shouldStopLoss(entryPrice, stopLossPrice, 'long');
    if (!shouldStop) {
      throw new Error('Should trigger stop loss for long position');
    }
    
    const shouldNotStop = challenge.riskManager.shouldStopLoss(entryPrice, 99.5, 'long');
    if (shouldNotStop) {
      throw new Error('Should not trigger stop loss above threshold');
    }
  });
  
  // Test 7: Risk Manager - Take Profit
  testRunner.addTest('Risk Manager - Take Profit Logic', () => {
    const challenge = new NeuralTradingTrials();
    const entryPrice = 100;
    const takeProfitPrice = 104; // 4% above entry
    
    const shouldTake = challenge.riskManager.shouldTakeProfit(entryPrice, takeProfitPrice, 'long');
    if (!shouldTake) {
      throw new Error('Should trigger take profit for long position');
    }
    
    const shouldNotTake = challenge.riskManager.shouldTakeProfit(entryPrice, 103, 'long');
    if (shouldNotTake) {
      throw new Error('Should not trigger take profit below threshold');
    }
  });
  
  // Test 8: Backtester
  testRunner.addTest('Backtester - Initialization', () => {
    const challenge = new NeuralTradingTrials();
    if (challenge.backtester.initialBalance !== 10000) {
      throw new Error('Backtester initial balance incorrect');
    }
    if (challenge.backtester.balance !== 10000) {
      throw new Error('Backtester current balance incorrect');
    }
  });
  
  // Test 9: Trading Simulator
  testRunner.addTest('Trading Simulator - Initialization', () => {
    const challenge = new NeuralTradingTrials();
    if (challenge.simulator.isRunning !== false) {
      throw new Error('Simulator should not be running initially');
    }
    if (challenge.simulator.currentPrice !== 0) {
      throw new Error('Simulator initial price should be 0');
    }
  });
  
  // Test 10: Feature Preparation
  testRunner.addTest('Feature Preparation', () => {
    const challenge = new NeuralTradingTrials();
    const prices = Array.from({ length: 25 }, (_, i) => 100 + i * 0.5 + Math.sin(i * 0.1) * 2);
    const features = challenge.prepareFeatures(prices);
    
    if (features.length !== 6) {
      throw new Error(`Expected 6 features, got ${features.length}`);
    }
    if (features.some(f => isNaN(f) || !isFinite(f))) {
      throw new Error('Features contain invalid values');
    }
  });
  
  // Test 11: Training Data Generation
  testRunner.addTest('Training Data Generation', () => {
    const challenge = new NeuralTradingTrials();
    const priceData = challenge.generateSyntheticPriceData(100);
    const trainingData = challenge.generateTrainingData(priceData);
    
    if (trainingData.length === 0) {
      throw new Error('No training data generated');
    }
    if (trainingData[0].input.length !== 6) {
      throw new Error(`Expected 6 input features, got ${trainingData[0].input.length}`);
    }
    if (trainingData[0].target.length !== 3) {
      throw new Error(`Expected 3 target outputs, got ${trainingData[0].target.length}`);
    }
  });
  
  // Test 12: Neural Network Training
  testRunner.addTest('Neural Network Training', async () => {
    const challenge = new NeuralTradingTrials();
    const priceData = challenge.generateSyntheticPriceData(200);
    const trainingData = challenge.generateTrainingData(priceData);
    
    benchmark.start('Neural Network Training');
    challenge.neuralNetwork.train(trainingData, 10); // Reduced epochs for testing
    benchmark.end();
    
    // Test prediction
    const testInput = [0.01, 0.005, 0.6, 0.1, 0.05, 0.02];
    const prediction = challenge.neuralNetwork.predict(testInput);
    
    if (prediction.length !== 3) {
      throw new Error(`Expected 3 outputs, got ${prediction.length}`);
    }
    if (prediction.some(p => isNaN(p) || !isFinite(p))) {
      throw new Error('Prediction contains invalid values');
    }
  });
  
  // Test 13: Volatility Calculation
  testRunner.addTest('Volatility Calculation', () => {
    const challenge = new NeuralTradingTrials();
    const prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109];
    const volatility = challenge.calculateVolatility(prices);
    
    if (volatility < 0) {
      throw new Error('Volatility should be non-negative');
    }
    if (isNaN(volatility) || !isFinite(volatility)) {
      throw new Error('Volatility calculation returned invalid value');
    }
  });
  
  // Test 14: Accuracy Calculation
  testRunner.addTest('Accuracy Calculation', () => {
    const challenge = new NeuralTradingTrials();
    const mockTrainingData = [
      { input: [0.1, 0.05, 0.6, 0.1, 0.05, 0.02], target: [1, 0, 0] },
      { input: [0.05, 0.02, 0.4, -0.1, -0.05, 0.03], target: [0, 1, 0] },
      { input: [0.0, 0.0, 0.5, 0.0, 0.0, 0.02], target: [0, 0, 1] }
    ];
    
    const accuracy = challenge.calculateAccuracy(mockTrainingData);
    
    if (accuracy < 0 || accuracy > 1) {
      throw new Error(`Accuracy should be between 0 and 1, got ${accuracy}`);
    }
  });
  
  // Test 15: Full Challenge Execution
  testRunner.addTest('Full Challenge Execution', async () => {
    benchmark.start('Full Challenge Execution');
    const result = await executeNeuralTradingTrials();
    benchmark.end();
    
    if (result.status !== 'completed') {
      throw new Error(`Challenge not completed, status: ${result.status}`);
    }
    if (!result.neuralNetwork) {
      throw new Error('Neural network results missing');
    }
    if (!result.backtestResults) {
      throw new Error('Backtest results missing');
    }
    if (!result.simulation) {
      throw new Error('Simulation results missing');
    }
    if (result.neuralNetwork.accuracy < 0 || result.neuralNetwork.accuracy > 100) {
      throw new Error(`Invalid accuracy: ${result.neuralNetwork.accuracy}`);
    }
  });
  
  // Run all tests
  const success = await testRunner.runTests();
  benchmark.printSummary();
  
  if (success) {
    console.log('\n🎉 All tests passed! Neural Trading Trials implementation is working correctly.');
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
