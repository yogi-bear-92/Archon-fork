# Build Your Trading Bot
def trading_bot(price_data):
    """
    Analyze price data and return trading signal
    Input: dict with current_price, rsi, volume, trend
    Output: "BUY", "SELL", or "HOLD"
    """
    rsi = price_data.get("rsi", 50)
    
    # Add your logic here
    # RSI < 30 = oversold (BUY)
    # RSI > 70 = overbought (SELL)
    
    return "HOLD"