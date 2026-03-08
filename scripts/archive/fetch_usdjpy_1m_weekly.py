#!/usr/bin/env python3
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# プロジェクトルートとscriptsディレクトリをパスに追加
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPTS_DIR)

# scripts/fetch_data.py からインポート
try:
    from fetch_data import fetch_ticks
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_data", os.path.join(SCRIPTS_DIR, "fetch_data.py"))
    fetch_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_data)
    fetch_ticks = fetch_data.fetch_ticks

def ticks_to_ohlc_safe(ticks, freq):
    if ticks.empty: return None
    df = ticks.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            return None
    ohlc = df['bidPrice'].resample(freq).ohlc()
    return ohlc

def main():
    symbol = "USDJPY"
    # 検証期間: 2026-01-01 から 2026-01-31 まで
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 1, 31)
    
    output_dir = os.path.join(BASE_DIR, 'data', 'ohlc')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'USDJPY_1m_2026_Jan.csv')
    
    print(f"Fetching {symbol} 1m data weekly for January 2026...")
    
    all_ohlc = []
    current_start = start_date
    
    # 7日ごとに分割
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=7), end_date)
        print(f"  Range: {current_start} to {current_end}")
        
        try:
            ticks = fetch_ticks(symbol, current_start, current_end)
            if ticks is not None and not ticks.empty:
                ohlc = ticks_to_ohlc_safe(ticks, '1min')
                if ohlc is not None:
                    all_ohlc.append(ohlc)
                    print(f"    Success: {len(ohlc)} bars")
            else:
                print(f"    No data for this range.")
        except Exception as e:
            print(f"    Error: {e}")
        
        current_start = current_end
        time.sleep(1)
        
    if all_ohlc:
        final_df = pd.concat(all_ohlc)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)
        final_df.to_csv(output_path)
        print(f"\nSaved {len(final_df)} bars to {output_path}")
    else:
        print("\nNo data fetched.")

if __name__ == '__main__':
    main()
