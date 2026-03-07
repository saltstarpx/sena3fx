"""
髭閾値（ATR倍率）の感度分析
ATR × N の N を変化させたとき、エントリー候補数がどう変わるかを計測する。
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/ubuntu/sena3fx/strategies')

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df

def count_candidates(data_15m, data_1h, data_4h, m15_atr, wick_mult, h1_filter=True):
    data_4h = data_4h.copy()
    data_1h = data_1h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)
    data_1h["ema20"] = data_1h["close"].ewm(span=20, adjust=False).mean()

    cnt = 0
    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        if pd.isna(atr_val):
            continue

        h4_time = bar.name.floor("4h")
        if h4_time not in data_4h.index:
            continue
        trend = data_4h.loc[h4_time]["trend"]

        if h1_filter:
            h1_time = bar.name.floor("1h")
            if h1_time not in data_1h.index:
                continue
            h1_close = data_1h.loc[h1_time]["close"]
            h1_ema   = data_1h.loc[h1_time]["ema20"]
            if trend == 1 and h1_close <= h1_ema:
                continue
            if trend == -1 and h1_close >= h1_ema:
                continue

        body_high = max(bar["open"], bar["close"])
        body_low  = min(bar["open"], bar["close"])
        lower_wick = body_low - bar["low"]
        upper_wick = bar["high"] - body_high
        threshold  = atr_val * wick_mult

        if trend == 1 and lower_wick > threshold:
            cnt += 1
        elif trend == -1 and upper_wick > threshold:
            cnt += 1
    return cnt

symbol = "USDJPY"
start  = "2024-07-01"
end    = "2024-08-06"

data_15m = load_data(symbol.lower(), '15m', start, end)
data_1h  = load_data(symbol.lower(), '1h',  start, end)
data_4h  = load_data(symbol.lower(), '4h',  start, end)
m15_atr  = calculate_atr(data_15m)

print("\n" + "="*70)
print("  髭閾値（ATR倍率）感度分析  |  1時間足フィルター あり / なし")
print("="*70)
print(f"  {'ATR倍率':>8}  {'候補数(1H有)':>12}  {'候補数(1H無)':>12}  {'1Hフィルター除外':>14}")
print("-"*70)

for mult in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
    with_h1    = count_candidates(data_15m, data_1h, data_4h, m15_atr, mult, h1_filter=True)
    without_h1 = count_candidates(data_15m, data_1h, data_4h, m15_atr, mult, h1_filter=False)
    diff = without_h1 - with_h1
    marker = " ← 現在設定" if mult == 0.5 else ""
    print(f"  ATR × {mult:.1f}   {with_h1:>12}  {without_h1:>12}  {diff:>14}{marker}")

print("="*70)
print("\n  ※ 候補数はエントリー候補（全フィルター通過）の15分足バー数")
print("  ※ 実際の取引数 = 候補数 × (1分足で50%戻し到達率)")
