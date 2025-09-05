# Flow Nexus Trading Workflow Challenge

**Difficulty:** Beginner  
**Reward:** 1,000 rUv + 50 rUv participation  
**Challenge ID:** `flow-nexus-trading-workflow-001`  
**Archon Project ID:** `d9ca46a5-dbb5-4a37-960a-6d40a63869af`

## 🎯 Challenge Description

Build a trading bot using Flow Nexus pgmq queues and workflow system. Implement RSI-based trading logic with queue processing for automated trading decisions.

## 📋 Requirements

1. **Queue System Setup**: Initialize pgmq queues for trading signals
2. **RSI Trading Logic**: Implement RSI < 30 = BUY, RSI > 70 = SELL
3. **Workflow Orchestration**: Process trading signals through workflow system
4. **Real-time Monitoring**: Monitor queue status and trading performance

## ✅ Test Cases

- Initialize trading workflow with pgmq queues
- Process RSI signals and generate trading decisions
- Expected: "Trading workflow operational, queues processing signals"
- Should handle BUY/SELL signals based on RSI thresholds

## 🛠️ Tools Required

- Flow Nexus MCP tools for workflow orchestration
- pgmq queue management
- RSI calculation algorithms
- Trading signal processing

## 📁 Project Structure

```
flow-nexus-trading-workflow/
├── README.md              # This file
├── challenge-info.json    # Challenge metadata
├── starter-code.js        # Original starter template
├── solution.js           # Your solution
└── test.js              # Test runner
