#!/usr/bin/env python3

from solution import trading_bot

def test_trading_bot():
    """Test the trading bot solution"""
    
    # Test Case 1: RSI = 25 (should return BUY)
    result1 = trading_bot({"rsi": 25})
    print(f"Test 1 - RSI 25: {result1} (Expected: BUY)")
    assert result1 == "BUY", f"Test 1 failed: expected BUY, got {result1}"
    
    # Test Case 2: RSI = 75 (should return SELL)  
    result2 = trading_bot({"rsi": 75})
    print(f"Test 2 - RSI 75: {result2} (Expected: SELL)")
    assert result2 == "SELL", f"Test 2 failed: expected SELL, got {result2}"
    
    # Test Case 3: RSI = 50 (should return HOLD)
    result3 = trading_bot({"rsi": 50})
    print(f"Test 3 - RSI 50: {result3} (Expected: HOLD)")
    assert result3 == "HOLD", f"Test 3 failed: expected HOLD, got {result3}"
    
    # Additional edge cases
    result4 = trading_bot({"rsi": 30})
    print(f"Test 4 - RSI 30: {result4} (Expected: HOLD)")
    assert result4 == "HOLD", f"Test 4 failed: expected HOLD, got {result4}"
    
    result5 = trading_bot({"rsi": 70}) 
    print(f"Test 5 - RSI 70: {result5} (Expected: HOLD)")
    assert result5 == "HOLD", f"Test 5 failed: expected HOLD, got {result5}"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_trading_bot()