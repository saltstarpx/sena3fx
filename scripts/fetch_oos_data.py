"""
アウト・オブ・サンプル検証用データ取得スクリプト
- USDJPY: 2025年3月1日〜2026年2月28日（OOS期間）
- EURJPY: 2024年7月1日〜2026年2月28日（IS+OOS）
- GBPJPY: 2024年7月1日〜2026年2月28日（IS+OOS）
"""
import os
import time
import pandas as pd
from polygon import RESTClient

API_KEY = os.environ["POLYGON_API_KEY"]
client = RESTClient(api_key=API_KEY)

DATA_DIR = "/home/ubuntu/sena3fx/data"
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS = {
    "usdjpy_oos": ("USD", "JPY", "2025-03-01", "2026-02-28"),
    "eurjpy":     ("EUR", "JPY", "2024-07-01", "2026-02-28"),
    "gbpjpy":     ("GBP", "JPY", "2024-07-01", "2026-02-28"),
}

TIMEFRAMES = {
    "1m":  ("minute", 1),
    "15m": ("minute", 15),
    "1h":  ("hour",   1),
    "4h":  ("hour",   4),
}

def fetch_forex(from_cur, to_cur, multiplier, timespan, start, end, label):
    ticker = f"C:{from_cur}{to_cur}"
    print(f"  取得中: {ticker} {multiplier}{timespan} {start}〜{end}")
    rows = []
    try:
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=True,
            sort="asc",
            limit=50000,
        ):
            rows.append({
                "timestamp": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC"),
                "open":  agg.open,
                "high":  agg.high,
                "low":   agg.low,
                "close": agg.close,
                "volume": agg.volume,
            })
    except Exception as e:
        print(f"  エラー: {e}")
        return None
    if not rows:
        print(f"  データなし")
        return None
    df = pd.DataFrame(rows).set_index("timestamp")
    print(f"  {len(df)}行取得")
    return df

for pair_key, (fc, tc, start, end) in PAIRS.items():
    print(f"\n=== {pair_key} ({fc}/{tc}) ===")
    for tf_key, (timespan, mult) in TIMEFRAMES.items():
        out_path = f"{DATA_DIR}/{pair_key}_{tf_key}.csv"
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            print(f"  {tf_key}: 既存ファイルあり ({len(existing)}行) → スキップ")
            continue
        df = fetch_forex(fc, tc, mult, timespan, start, end, f"{pair_key}_{tf_key}")
        if df is not None:
            df.to_csv(out_path)
            print(f"  保存: {out_path}")
        time.sleep(0.5)

print("\n完了")
