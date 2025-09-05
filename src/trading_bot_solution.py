# Neural Trading Bot Challenge Solution
# Challenge ID: c94777b9-6af5-4b15-8411-8391aa640864

def trading_bot(price_data):
    """
    Analyze price data and return trading signal
    Input: dict with current_price, rsi, volume, trend
    Output: "BUY", "SELL", or "HOLD"
    """
    rsi = price_data.get("rsi", 50)
    
    # Trading logic based on RSI values
    if rsi < 30:
        return "BUY"    # Oversold condition - good time to buy
    elif rsi > 70:
        return "SELL"   # Overbought condition - good time to sell
    else:
        return "HOLD"   # Neutral condition - wait for better opportunity

# Test the solution with provided test cases
if __name__ == "__main__":
    # Test Case 1: RSI = 25 (should return BUY)
    result1 = trading_bot({"rsi": 25})
    print(f"Test 1 - RSI 25: {result1} (Expected: BUY)")
    
    # Test Case 2: RSI = 75 (should return SELL)  
    result2 = trading_bot({"rsi": 75})
    print(f"Test 2 - RSI 75: {result2} (Expected: SELL)")
    
    # Test Case 3: RSI = 50 (should return HOLD)
    result3 = trading_bot({"rsi": 50})
    print(f"Test 3 - RSI 50: {result3} (Expected: HOLD)")
    
    # Verify all tests pass
    all_passed = (result1 == "BUY" and result2 == "SELL" and result3 == "HOLD")
    print(f"\nAll tests passed: {all_passed}")