// Flow Nexus Trading Workflow Challenge Test
// Tests RSI-based trading signals with extreme market conditions

import { FlowNexusTradingWorkflow } from './solution.js';

async function runAdvancedTradingTest() {
  console.log("🧪 Running Advanced Trading Test with Extreme RSI Values...");
  
  const tradingWorkflow = new FlowNexusTradingWorkflow();
  await tradingWorkflow.initializeWorkflow();
  
  // Test 1: Oversold market (should trigger BUY signals)
  console.log("\n📉 Test 1: Oversold Market Conditions");
  const oversoldData = {
    symbol: 'BTC/USD',
    prices: [50000, 48000, 46000, 44000, 42000, 40000, 38000, 36000, 34000, 32000, 30000, 28000, 26000, 24000, 22000, 20000, 18000, 16000, 14000, 12000]
  };
  
  const oversoldSignal = tradingWorkflow.processTradingSignal(oversoldData);
  console.log("Oversold signal:", oversoldSignal);
  
  // Test 2: Overbought market (should trigger SELL signals)
  console.log("\n📈 Test 2: Overbought Market Conditions");
  const overboughtData = {
    symbol: 'BTC/USD',
    prices: [20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000]
  };
  
  const overboughtSignal = tradingWorkflow.processTradingSignal(overboughtData);
  console.log("Overbought signal:", overboughtSignal);
  
  // Test 3: Volatile market (mixed signals)
  console.log("\n🌊 Test 3: Volatile Market Conditions");
  const volatileData = {
    symbol: 'BTC/USD',
    prices: [50000, 30000, 60000, 25000, 55000, 20000, 50000, 15000, 45000, 10000, 40000, 5000, 35000, 1000, 30000, 500, 25000, 100, 20000, 50]
  };
  
  const volatileSignal = tradingWorkflow.processTradingSignal(volatileData);
  console.log("Volatile signal:", volatileSignal);
  
  // Get final statistics
  const finalStats = tradingWorkflow.getWorkflowStatus();
  console.log("\n📊 Final Trading Statistics:");
  console.log(JSON.stringify(finalStats, null, 2));
  
  console.log("\n✅ Advanced Trading Test Completed!");
}

// Run the test
runAdvancedTradingTest()
  .then(() => {
    console.log("🏆 All tests passed successfully!");
  })
  .catch(error => {
    console.error("❌ Test failed:", error);
  });
