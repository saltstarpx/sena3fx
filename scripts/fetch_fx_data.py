"""
Polygon.io から USD/JPY の1時間足データを取得して CSV に保存するスクリプト。
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.environ['POLYGON_API_KEY']
TICKER = 'C:USDJPY'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_aggs(ticker, multiplier, timespan, start, end, limit=5000):
    """Polygon.io の /v2/aggs エンドポイントからOHLCデータを取得。ページネーション対応。"""
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}'
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': limit, 'apiKey': API_KEY}
    results = []
    while url:
        r = requests.get(url, params=params)
        if r.status_code == 429:
            print("Rate limited, waiting 15s...")
            time.sleep(15)
            continue
        r.raise_for_status()
        data = r.json()
        if 'results' in data:
            results.extend(data['results'])
        next_url = data.get('next_url')
        if next_url:
            url = next_url + f'&apiKey={API_KEY}'
            params = {}
            time.sleep(0.5)
        else:
            break
    return results


def build_dataframe(results):
    """APIレスポンスをDataFrameに変換。"""
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'tick_count'})
    df = df.set_index('timestamp').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume', 'tick_count']]
    return df


if __name__ == '__main__':
    print("USD/JPY 1時間足データを取得中...")

    # 2023年〜2024年のデータを取得
    periods = [
        ('2023-01-01', '2023-06-30'),
        ('2023-07-01', '2023-12-31'),
        ('2024-01-01', '2024-06-30'),
        ('2024-07-01', '2024-12-31'),
        ('2025-01-01', '2025-12-31'),
    ]

    all_results = []
    for start, end in periods:
        print(f"  取得中: {start} 〜 {end}")
        results = fetch_aggs(TICKER, 1, 'hour', start, end)
        print(f"  → {len(results)} bars")
        all_results.extend(results)
        time.sleep(1)

    df = build_dataframe(all_results)
    # 重複除去
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    output_path = os.path.join(OUTPUT_DIR, 'usdjpy_1h.csv')
    df.to_csv(output_path)
    print(f"\n保存完了: {output_path}")
    print(f"総バー数: {len(df)}")
    print(f"期間: {df.index[0]} 〜 {df.index[-1]}")
    print(df.tail())
