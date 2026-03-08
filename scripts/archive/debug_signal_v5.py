#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def main():
    data_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m_2026_Jan.csv')
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    # 4H ATR
    df_4h['tr'] = np.maximum(df_4h['high'] - df_4h['low'], np.maximum(abs(df_4h['high'] - df_4h['close'].shift(1)), abs(df_4h['low'] - df_4h['close'].shift(1))))
    df_4h['atr'] = df_4h['tr'].rolling(14).mean()
    
    # 15M Pattern Check
    df_15m['lower_wick'] = df_15m[['open', 'close']].min(axis=1) - df_15m['low']
    df_15m['upper_wick'] = df_15m['high'] - df_15m[['open', 'close']].max(axis=1)
    
    print(f"--- 15M Pattern Stats ---")
    print(f"Total 15M bars: {len(df_15m)}")
    print(f"Lower Wick > 0.005: {(df_15m['lower_wick'] >= 0.005).sum()}")
    print(f"Upper Wick > 0.005: {(df_15m['upper_wick'] >= 0.005).sum()}")
    
    # Check a specific 15M window for 1M execution
    m15_with_wick = df_15m[df_15m['lower_wick'] >= 0.005].index
    if len(m15_with_wick) > 0:
        target_time = m15_with_wick[0]
        m1_slice = df.loc[target_time : target_time + pd.Timedelta(minutes=15)]
        print(f"\n--- 1M Execution Debug (Target: {target_time}) ---")
        print(f"1M bars in slice: {len(m1_slice)}")
        
        m1_slice = m1_slice.copy()
        m1_slice['range'] = m1_slice['high'] - m1_slice['low']
        m1_slice['vol_ma'] = m1_slice['range'].rolling(5).mean()
        
        print(f"1M Vol MA stats:\n{m1_slice['vol_ma'].describe()}")
        
        for i in range(5, len(m1_slice)):
            curr_bar = m1_slice.iloc[i]
            recent_m1 = m1_slice.iloc[i-5:i]
            
            # Debug conditions
            cond1 = curr_bar['vol_ma'] <= 0.05
            cond2 = curr_bar['close'] > recent_m1['high'].max()
            
            sl_price = recent_m1['low'].min() - 0.01 * 1.2
            entry_price = curr_bar['close']
            cond3 = (entry_price - sl_price) > 0.01 * 1.5
            
            if i < 10: # Print first few
                print(f"Step {i}: VolMA={curr_bar['vol_ma']:.4f}, Cond1={cond1}, Cond2={cond2}, SL_Dist={entry_price-sl_price:.4f}, Cond3={cond3}")

if __name__ == '__main__':
    main()
