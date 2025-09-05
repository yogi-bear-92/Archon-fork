# Neural Trading Bot Challenge

**Difficulty:** Beginner  
**Reward:** 250 rUv + 10 rUv participation  
**Challenge ID:** `c94777b9-6af5-4b15-8411-8391aa640864`  
**Archon Project ID:** `c9b134f9-6bb8-4ca5-b2a6-7db461f163cb`

## 🎯 Challenge Description

Build a simple trading bot that uses RSI indicators to make buy/sell decisions. Basic algorithmic trading with RSI < 30 = BUY, RSI > 70 = SELL logic.

## 📋 Requirements

Create a function that takes market data and returns BUY, SELL, or HOLD based on RSI values:
- RSI < 30 means oversold (BUY)
- RSI > 70 means overbought (SELL)
- RSI 30-70 means neutral (HOLD)

## ✅ Test Cases

1. `{"rsi": 25}` → `"BUY"`
2. `{"rsi": 75}` → `"SELL"`
3. `{"rsi": 50}` → `"HOLD"`

## 🚀 Status

- **Status**: ✅ **COMPLETED** with 100% Score
- **Submission ID**: `sub_1757087742161`
- **Solution**: See `solution.py`

## 📁 Project Structure

```
neural-trading-bot-challenge/
├── README.md              # This file
├── challenge-info.json    # Challenge metadata
├── starter-code.py        # Original starter template
├── solution.py           # Completed solution
└── test.py              # Test runner
```

## 🏆 Results

Successfully completed this challenge as the first Flow Nexus challenge attempt!