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