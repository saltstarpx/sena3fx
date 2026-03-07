import pandas as pd
import numpy as np

def analyze_volatility(csv_path, label):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 15分足のATRを計算
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    high_low = data_15m["high"] - data_15m["low"]
    high_close = abs(data_15m["high"] - data_15m["close"].shift())
    low_close = abs(data_15m["low"] - data_15m["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = ranges.max(axis=1).mean()
    
    # スプレッド 1.0pips = 0.0001 (EURUSD等の4桁/5桁表示)
    spread_pips = 0.0001
    
    print(f"--- {label} ---")
    print(f"Average 15m ATR: {atr:.6f}")
    print(f"Average 15m Range (pips): {atr / 0.0001:.2f} pips")
    print(f"Max 15m Range (pips): {ranges.max(axis=1).max() / 0.0001:.2f} pips")
    print(f"Spread 1.0pips impact: {spread_pips / atr * 100:.1f}% of avg 15m range")

if __name__ == "__main__":
    analyze_volatility("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Jan.csv", "January")
    analyze_volatility("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Feb.csv", "February")
