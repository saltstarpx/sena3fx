"""
Polygon.io から USD/JPY の1時間足データを取得して CSV に保存するスクリプト。
ページネーションなし版（各クォーターを個別に取得）。
"""
import os
import time
import requests
import json
import pandas as pd

API_KEY = os.environ['POLYGON_API_KEY']
TICKER = 'C:USDJPY'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_single(ticker, start, end, limit=5000):
    """単一期間のOHLCデータを取得（ページネーションなし）。"""
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start}/{end}'
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': limit, 'apiKey': API_KEY}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                print(f"  Rate limited, waiting 20s...")
                time.sleep(20)
                continue
            if r.status_code == 403:
                print(f"  403 Forbidden for {start}~{end}, skipping")
                return []
            r.raise_for_status()
            data = r.json()
            return data.get('results', [])
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []


if __name__ == '__main__':
    print("USD/JPY 1時間足データを取得中...")

    # 週単位で分割して取得（ページネーション問題を回避）
    import datetime
    
    all_results = []
    
    # 2024年7月〜2025年3月を週単位で取得
    start_date = datetime.date(2024, 7, 1)
    end_date = datetime.date(2025, 3, 4)
    
    current = start_date
    while current < end_date:
        week_end = min(current + datetime.timedelta(days=30), end_date)
        s = current.strftime('%Y-%m-%d')
        e = week_end.strftime('%Y-%m-%d')
        
        results = fetch_single(TICKER, s, e)
        if results:
            all_results.extend(results)
            print(f"  {s}~{e}: {len(results)} bars (累計: {len(all_results)})")
        else:
            print(f"  {s}~{e}: 0 bars")
        
        current = week_end + datetime.timedelta(days=1)
        time.sleep(1)
    
    if not all_results:
        print("データが取得できませんでした。")
        exit(1)
    
    # DataFrameに変換
    df = pd.DataFrame(all_results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'tick_count'})
    df = df.set_index('timestamp').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume', 'tick_count']]
    
    # 重複除去
    df = df[~df.index.duplicated(keep='first')]
    
    output_path = os.path.join(OUTPUT_DIR, 'usdjpy_1h.csv')
    df.to_csv(output_path)
    print(f"\n保存完了: {output_path}")
    print(f"総バー数: {len(df)}")
    print(f"期間: {df.index[0]} 〜 {df.index[-1]}")
    print(df.tail(5))
