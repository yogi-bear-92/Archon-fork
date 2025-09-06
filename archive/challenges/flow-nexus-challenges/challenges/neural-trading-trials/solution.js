// The Neural Trading Trials Challenge Solution
// Challenge ID: TBD
// Reward: 500 rUv
// Requirements: Implement a sophisticated AI trading system with neural networks

class NeuralNetwork {
  constructor(inputSize, hiddenSizes, outputSize, learningRate = 0.01) {
    this.inputSize = inputSize;
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    
    // Initialize weights and biases
    this.initializeWeights();
  }
  
  initializeWeights() {
    const sizes = [this.inputSize, ...this.hiddenSizes, this.outputSize];
    
    for (let i = 0; i < sizes.length - 1; i++) {
      const weightMatrix = [];
      const biasVector = [];
      
      for (let j = 0; j < sizes[i + 1]; j++) {
        const row = [];
        for (let k = 0; k < sizes[i]; k++) {
          row.push(Math.random() * 2 - 1); // Random weights between -1 and 1
        }
        weightMatrix.push(row);
        biasVector.push(Math.random() * 2 - 1);
      }
      
      this.weights.push(weightMatrix);
      this.biases.push(biasVector);
    }
  }
  
  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }
  
  sigmoidDerivative(x) {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }
  
  forward(input) {
    let current = [...input];
    this.activations = [current];
    this.zValues = [];
    
    for (let layer = 0; layer < this.weights.length; layer++) {
      const z = [];
      const activation = [];
      
      for (let i = 0; i < this.weights[layer].length; i++) {
        let sum = this.biases[layer][i];
        for (let j = 0; j < current.length; j++) {
          sum += this.weights[layer][i][j] * current[j];
        }
        z.push(sum);
        activation.push(this.sigmoid(sum));
      }
      
      this.zValues.push(z);
      this.activations.push(activation);
      current = activation;
    }
    
    return current;
  }
  
  backward(input, target) {
    const output = this.forward(input);
    const errors = [];
    
    // Calculate output layer error
    const outputError = [];
    for (let i = 0; i < output.length; i++) {
      outputError.push(target[i] - output[i]);
    }
    errors.push(outputError);
    
    // Calculate hidden layer errors
    for (let layer = this.weights.length - 1; layer > 0; layer--) {
      const layerError = [];
      for (let i = 0; i < this.weights[layer - 1].length; i++) {
        let error = 0;
        for (let j = 0; j < this.weights[layer].length; j++) {
          error += this.weights[layer][j][i] * errors[0][j];
        }
        layerError.push(error);
      }
      errors.unshift(layerError);
    }
    
    // Update weights and biases
    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let i = 0; i < this.weights[layer].length; i++) {
        for (let j = 0; j < this.weights[layer][i].length; j++) {
          const gradient = errors[layer][i] * this.sigmoidDerivative(this.zValues[layer][i]) * this.activations[layer][j];
          this.weights[layer][i][j] += this.learningRate * gradient;
        }
        const biasGradient = errors[layer][i] * this.sigmoidDerivative(this.zValues[layer][i]);
        this.biases[layer][i] += this.learningRate * biasGradient;
      }
    }
    
    return output;
  }
  
  train(trainingData, epochs = 1000) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalError = 0;
      
      for (const data of trainingData) {
        const output = this.backward(data.input, data.target);
        const error = data.target.reduce((sum, target, i) => sum + Math.pow(target - output[i], 2), 0);
        totalError += error;
      }
      
      if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Error: ${(totalError / trainingData.length).toFixed(6)}`);
      }
    }
  }
  
  predict(input) {
    return this.forward(input);
  }
}

class TechnicalIndicators {
  static calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return [];
    
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const rsi = [];
    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((sum, gain) => sum + gain, 0) / period;
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((sum, loss) => sum + loss, 0) / period;
      
      if (avgLoss === 0) {
        rsi.push(100);
      } else {
        const rs = avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    }
    
    return rsi;
  }
  
  static calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const emaFast = this.calculateEMA(prices, fastPeriod);
    const emaSlow = this.calculateEMA(prices, slowPeriod);
    
    const macdLine = [];
    for (let i = 0; i < Math.min(emaFast.length, emaSlow.length); i++) {
      macdLine.push(emaFast[i] - emaSlow[i]);
    }
    
    const signalLine = this.calculateEMA(macdLine, signalPeriod);
    const histogram = [];
    
    for (let i = 0; i < Math.min(macdLine.length, signalLine.length); i++) {
      histogram.push(macdLine[i] - signalLine[i]);
    }
    
    return { macdLine, signalLine, histogram };
  }
  
  static calculateBollingerBands(prices, period = 20, stdDev = 2) {
    const sma = this.calculateSMA(prices, period);
    const bands = [];
    
    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      const mean = slice.reduce((sum, price) => sum + price, 0) / slice.length;
      const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / slice.length;
      const standardDeviation = Math.sqrt(variance);
      
      bands.push({
        upper: mean + (stdDev * standardDeviation),
        middle: mean,
        lower: mean - (stdDev * standardDeviation)
      });
    }
    
    return bands;
  }
  
  static calculateSMA(prices, period) {
    const sma = [];
    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      sma.push(slice.reduce((sum, price) => sum + price, 0) / slice.length);
    }
    return sma;
  }
  
  static calculateEMA(prices, period) {
    const ema = [];
    const multiplier = 2 / (period + 1);
    
    ema.push(prices[0]);
    
    for (let i = 1; i < prices.length; i++) {
      ema.push((prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier)));
    }
    
    return ema;
  }
}

class RiskManager {
  constructor(maxPositionSize = 0.1, stopLoss = 0.02, takeProfit = 0.04) {
    this.maxPositionSize = maxPositionSize;
    this.stopLoss = stopLoss;
    this.takeProfit = takeProfit;
    this.positions = [];
  }
  
  calculatePositionSize(accountBalance, confidence, volatility) {
    const baseSize = accountBalance * this.maxPositionSize;
    const confidenceMultiplier = Math.min(confidence, 1.0);
    const volatilityAdjustment = Math.max(0.5, 1 - volatility);
    
    return baseSize * confidenceMultiplier * volatilityAdjustment;
  }
  
  shouldStopLoss(entryPrice, currentPrice, positionType) {
    if (positionType === 'long') {
      return currentPrice <= entryPrice * (1 - this.stopLoss);
    } else {
      return currentPrice >= entryPrice * (1 + this.stopLoss);
    }
  }
  
  shouldTakeProfit(entryPrice, currentPrice, positionType) {
    if (positionType === 'long') {
      return currentPrice >= entryPrice * (1 + this.takeProfit);
    } else {
      return currentPrice <= entryPrice * (1 - this.takeProfit);
    }
  }
  
  addPosition(entryPrice, size, positionType, confidence) {
    this.positions.push({
      entryPrice,
      size,
      positionType,
      confidence,
      timestamp: Date.now()
    });
  }
  
  closePosition(index, currentPrice) {
    if (index >= 0 && index < this.positions.length) {
      const position = this.positions[index];
      const pnl = position.positionType === 'long' 
        ? (currentPrice - position.entryPrice) * position.size
        : (position.entryPrice - currentPrice) * position.size;
      
      this.positions.splice(index, 1);
      return pnl;
    }
    return 0;
  }
}

class Backtester {
  constructor(initialBalance = 10000) {
    this.initialBalance = initialBalance;
    this.balance = initialBalance;
    this.positions = [];
    this.trades = [];
    this.equityCurve = [];
  }
  
  runBacktest(priceData, signals, riskManager) {
    this.balance = this.initialBalance;
    this.positions = [];
    this.trades = [];
    this.equityCurve = [];
    
    for (let i = 0; i < priceData.length; i++) {
      const currentPrice = priceData[i];
      const signal = signals[i];
      
      // Check for stop loss and take profit
      this.checkExitConditions(currentPrice, riskManager);
      
      // Process new signals
      if (signal && signal.action !== 'hold') {
        this.executeTrade(signal, currentPrice, riskManager);
      }
      
      // Update equity curve
      const currentEquity = this.calculateCurrentEquity(currentPrice);
      this.equityCurve.push(currentEquity);
    }
    
    return this.calculatePerformanceMetrics();
  }
  
  executeTrade(signal, price, riskManager) {
    const confidence = signal.confidence || 0.5;
    const volatility = signal.volatility || 0.02;
    const positionSize = riskManager.calculatePositionSize(this.balance, confidence, volatility);
    
    if (positionSize > 0) {
      riskManager.addPosition(price, positionSize, signal.action, confidence);
      this.balance -= positionSize * price;
    }
  }
  
  checkExitConditions(currentPrice, riskManager) {
    for (let i = riskManager.positions.length - 1; i >= 0; i--) {
      const position = riskManager.positions[i];
      
      if (riskManager.shouldStopLoss(position.entryPrice, currentPrice, position.positionType) ||
          riskManager.shouldTakeProfit(position.entryPrice, currentPrice, position.positionType)) {
        
        const pnl = riskManager.closePosition(i, currentPrice);
        this.balance += pnl;
        
        this.trades.push({
          entryPrice: position.entryPrice,
          exitPrice: currentPrice,
          size: position.size,
          positionType: position.positionType,
          pnl,
          timestamp: position.timestamp
        });
      }
    }
  }
  
  calculateCurrentEquity(currentPrice) {
    let equity = this.balance;
    
    for (const position of this.positions) {
      const pnl = position.positionType === 'long'
        ? (currentPrice - position.entryPrice) * position.size
        : (position.entryPrice - currentPrice) * position.size;
      equity += pnl;
    }
    
    return equity;
  }
  
  calculatePerformanceMetrics() {
    const returns = [];
    for (let i = 1; i < this.equityCurve.length; i++) {
      returns.push((this.equityCurve[i] - this.equityCurve[i - 1]) / this.equityCurve[i - 1]);
    }
    
    const totalReturn = (this.equityCurve[this.equityCurve.length - 1] - this.initialBalance) / this.initialBalance;
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = volatility > 0 ? avgReturn / volatility : 0;
    
    const maxDrawdown = this.calculateMaxDrawdown();
    const winRate = this.trades.filter(trade => trade.pnl > 0).length / this.trades.length;
    const profitFactor = this.calculateProfitFactor();
    
    return {
      totalReturn: totalReturn * 100,
      sharpeRatio,
      maxDrawdown: maxDrawdown * 100,
      winRate: winRate * 100,
      profitFactor,
      totalTrades: this.trades.length,
      finalBalance: this.equityCurve[this.equityCurve.length - 1]
    };
  }
  
  calculateMaxDrawdown() {
    let maxDrawdown = 0;
    let peak = this.equityCurve[0];
    
    for (const equity of this.equityCurve) {
      if (equity > peak) {
        peak = equity;
      }
      const drawdown = (peak - equity) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown;
  }
  
  calculateProfitFactor() {
    const grossProfit = this.trades.filter(trade => trade.pnl > 0)
      .reduce((sum, trade) => sum + trade.pnl, 0);
    const grossLoss = Math.abs(this.trades.filter(trade => trade.pnl < 0)
      .reduce((sum, trade) => sum + trade.pnl, 0));
    
    return grossLoss > 0 ? grossProfit / grossLoss : 0;
  }
}

class TradingSimulator {
  constructor(neuralNetwork, riskManager) {
    this.neuralNetwork = neuralNetwork;
    this.riskManager = riskManager;
    this.isRunning = false;
    this.currentPrice = 0;
    this.priceHistory = [];
  }
  
  start(initialPrice = 100) {
    this.isRunning = true;
    this.currentPrice = initialPrice;
    this.priceHistory = [initialPrice];
    
    console.log('🚀 Trading Simulator Started');
    console.log(`Initial Price: $${initialPrice.toFixed(2)}`);
  }
  
  updatePrice(newPrice) {
    this.currentPrice = newPrice;
    this.priceHistory.push(newPrice);
    
    if (this.priceHistory.length > 100) {
      this.priceHistory.shift();
    }
  }
  
  generateSignal() {
    if (this.priceHistory.length < 20) return null;
    
    // Prepare input features
    const features = this.prepareFeatures();
    const prediction = this.neuralNetwork.predict(features);
    
    const confidence = Math.max(...prediction);
    const actionIndex = prediction.indexOf(confidence);
    
    let action = 'hold';
    if (actionIndex === 0) action = 'buy';
    else if (actionIndex === 1) action = 'sell';
    
    return {
      action,
      confidence,
      volatility: this.calculateVolatility(),
      prediction
    };
  }
  
  prepareFeatures() {
    const prices = this.priceHistory.slice(-20);
    const rsi = TechnicalIndicators.calculateRSI(prices);
    const macd = TechnicalIndicators.calculateMACD(prices);
    const bb = TechnicalIndicators.calculateBollingerBands(prices);
    
    const features = [];
    
    // Price features
    features.push((prices[prices.length - 1] - prices[0]) / prices[0]); // Price change
    features.push((prices[prices.length - 1] - prices[prices.length - 5]) / prices[prices.length - 5]); // 5-period change
    
    // RSI features
    if (rsi.length > 0) {
      features.push(rsi[rsi.length - 1] / 100); // Normalized RSI
    } else {
      features.push(0.5);
    }
    
    // MACD features
    if (macd.macdLine.length > 0) {
      features.push(macd.macdLine[macd.macdLine.length - 1]);
      features.push(macd.histogram[macd.histogram.length - 1]);
    } else {
      features.push(0, 0);
    }
    
    // Bollinger Bands features
    if (bb.length > 0) {
      const currentBB = bb[bb.length - 1];
      const currentPrice = prices[prices.length - 1];
      features.push((currentPrice - currentBB.lower) / (currentBB.upper - currentBB.lower)); // BB position
    } else {
      features.push(0.5);
    }
    
    // Volatility
    features.push(this.calculateVolatility());
    
    return features;
  }
  
  calculateVolatility() {
    if (this.priceHistory.length < 10) return 0.02;
    
    const returns = [];
    for (let i = 1; i < this.priceHistory.length; i++) {
      returns.push((this.priceHistory[i] - this.priceHistory[i - 1]) / this.priceHistory[i - 1]);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }
  
  stop() {
    this.isRunning = false;
    console.log('🛑 Trading Simulator Stopped');
  }
}

class NeuralTradingTrials {
  constructor() {
    this.neuralNetwork = new NeuralNetwork(6, [12, 8], 3, 0.01);
    this.riskManager = new RiskManager(0.1, 0.02, 0.04);
    this.backtester = new Backtester(10000);
    this.simulator = new TradingSimulator(this.neuralNetwork, this.riskManager);
  }
  
  generateTrainingData(priceData, lookback = 20) {
    const trainingData = [];
    
    for (let i = lookback; i < priceData.length - 1; i++) {
      const features = this.prepareFeatures(priceData.slice(i - lookback, i));
      const futurePrice = priceData[i + 1];
      const currentPrice = priceData[i];
      
      // Create target based on price movement
      const priceChange = (futurePrice - currentPrice) / currentPrice;
      let target = [0, 0, 1]; // Default to hold
      
      if (priceChange > 0.01) { // 1% increase
        target = [1, 0, 0]; // Buy
      } else if (priceChange < -0.01) { // 1% decrease
        target = [0, 1, 0]; // Sell
      }
      
      trainingData.push({ input: features, target });
    }
    
    return trainingData;
  }
  
  prepareFeatures(prices) {
    const rsi = TechnicalIndicators.calculateRSI(prices);
    const macd = TechnicalIndicators.calculateMACD(prices);
    const bb = TechnicalIndicators.calculateBollingerBands(prices);
    
    const features = [];
    
    // Price features
    features.push((prices[prices.length - 1] - prices[0]) / prices[0]);
    features.push((prices[prices.length - 1] - prices[prices.length - 5]) / prices[prices.length - 5]);
    
    // RSI
    if (rsi.length > 0) {
      features.push(rsi[rsi.length - 1] / 100);
    } else {
      features.push(0.5);
    }
    
    // MACD
    if (macd.macdLine.length > 0) {
      features.push(macd.macdLine[macd.macdLine.length - 1]);
      features.push(macd.histogram[macd.histogram.length - 1]);
    } else {
      features.push(0, 0);
    }
    
    // Bollinger Bands
    if (bb.length > 0) {
      const currentBB = bb[bb.length - 1];
      const currentPrice = prices[prices.length - 1];
      features.push((currentPrice - currentBB.lower) / (currentBB.upper - currentBB.lower));
    } else {
      features.push(0.5);
    }
    
    return features;
  }
  
  async executeNeuralTradingTrials() {
    console.log('🧠 Starting Neural Trading Trials Challenge...');
    
    // Generate synthetic price data
    const priceData = this.generateSyntheticPriceData(1000);
    console.log(`📊 Generated ${priceData.length} data points for training`);
    
    // Prepare training data
    const trainingData = this.generateTrainingData(priceData);
    console.log(`🎯 Prepared ${trainingData.length} training samples`);
    
    // Train neural network
    console.log('🤖 Training neural network...');
    this.neuralNetwork.train(trainingData, 500);
    console.log('✅ Neural network training completed');
    
    // Generate trading signals
    const signals = [];
    for (let i = 20; i < priceData.length; i++) {
      const features = this.prepareFeatures(priceData.slice(i - 20, i));
      const prediction = this.neuralNetwork.predict(features);
      
      const confidence = Math.max(...prediction);
      const actionIndex = prediction.indexOf(confidence);
      
      let action = 'hold';
      if (actionIndex === 0) action = 'buy';
      else if (actionIndex === 1) action = 'sell';
      
      signals.push({
        action,
        confidence,
        volatility: this.calculateVolatility(priceData.slice(i - 10, i)),
        prediction
      });
    }
    
    console.log(`📈 Generated ${signals.length} trading signals`);
    
    // Run backtest
    console.log('📊 Running backtest...');
    const backtestResults = this.backtester.runBacktest(priceData.slice(20), signals, this.riskManager);
    console.log('✅ Backtest completed');
    
    // Run simulation
    console.log('🎮 Starting trading simulation...');
    this.simulator.start(priceData[priceData.length - 1]);
    
    // Simulate some trading
    for (let i = 0; i < 50; i++) {
      const newPrice = priceData[priceData.length - 1] * (1 + (Math.random() - 0.5) * 0.02);
      this.simulator.updatePrice(newPrice);
      
      const signal = this.simulator.generateSignal();
      if (signal && signal.action !== 'hold') {
        console.log(`📊 Signal: ${signal.action.toUpperCase()} (Confidence: ${(signal.confidence * 100).toFixed(1)}%)`);
      }
    }
    
    this.simulator.stop();
    
    // Calculate neural network accuracy
    const accuracy = this.calculateAccuracy(trainingData);
    
    return {
      challenge: 'Neural Trading Trials',
      status: 'completed',
      neuralNetwork: {
        inputSize: this.neuralNetwork.inputSize,
        hiddenSizes: this.neuralNetwork.hiddenSizes,
        outputSize: this.neuralNetwork.outputSize,
        accuracy: accuracy * 100
      },
      backtestResults,
      simulation: {
        totalSignals: signals.length,
        buySignals: signals.filter(s => s.action === 'buy').length,
        sellSignals: signals.filter(s => s.action === 'sell').length,
        holdSignals: signals.filter(s => s.action === 'hold').length
      },
      technicalIndicators: {
        rsi: 'Implemented',
        macd: 'Implemented',
        bollingerBands: 'Implemented',
        movingAverages: 'Implemented'
      },
      riskManagement: {
        maxPositionSize: this.riskManager.maxPositionSize,
        stopLoss: this.riskManager.stopLoss,
        takeProfit: this.riskManager.takeProfit
      },
      performance: {
        totalReturn: backtestResults.totalReturn,
        sharpeRatio: backtestResults.sharpeRatio,
        maxDrawdown: backtestResults.maxDrawdown,
        winRate: backtestResults.winRate,
        profitFactor: backtestResults.profitFactor
      }
    };
  }
  
  generateSyntheticPriceData(length) {
    const prices = [100];
    let currentPrice = 100;
    
    for (let i = 1; i < length; i++) {
      // Generate realistic price movement with trend and volatility
      const trend = Math.sin(i / 50) * 0.001; // Long-term trend
      const volatility = 0.01 + Math.random() * 0.02; // Variable volatility
      const randomWalk = (Math.random() - 0.5) * volatility;
      
      currentPrice *= (1 + trend + randomWalk);
      prices.push(currentPrice);
    }
    
    return prices;
  }
  
  calculateVolatility(prices) {
    if (prices.length < 2) return 0.02;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }
  
  calculateAccuracy(trainingData) {
    let correct = 0;
    
    for (const data of trainingData) {
      const prediction = this.neuralNetwork.predict(data.input);
      const predictedIndex = prediction.indexOf(Math.max(...prediction));
      const actualIndex = data.target.indexOf(Math.max(...data.target));
      
      if (predictedIndex === actualIndex) {
        correct++;
      }
    }
    
    return correct / trainingData.length;
  }
}

// Execute the challenge
async function executeNeuralTradingTrials() {
  const challenge = new NeuralTradingTrials();
  return await challenge.executeNeuralTradingTrials();
}

// Run the challenge
executeNeuralTradingTrials()
  .then(result => {
    console.log('🏆 Neural Trading Trials Challenge Result:');
    console.log(JSON.stringify(result, null, 2));
  })
  .catch(error => {
    console.error('💥 Challenge execution failed:', error);
  });

export { NeuralTradingTrials, executeNeuralTradingTrials };
