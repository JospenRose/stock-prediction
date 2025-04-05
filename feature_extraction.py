import pandas as pd


# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['High'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Function to calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data):
    ema_12 = data['High'].ewm(span=12, adjust=False).mean()
    ema_26 = data['High'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    return macd, signal_line


# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['High'].rolling(window=window).mean()
    rolling_std = data['High'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    return upper_band, lower_band


# Function to calculate OBV (On-Balance Volume)
def calculate_obv(data):
    obv = [0]  # Start with 0
    for i in range(1, len(data)):
        if data['High'].iloc[i] > data['High'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['High'].iloc[i] < data['High'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, name='OBV')


