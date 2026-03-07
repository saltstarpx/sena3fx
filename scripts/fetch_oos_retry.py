"""
レート制限対策付き残データ取得スクリプト
- 各リクエスト間に12秒待機（Polygon無料プランは5req/min）
"""
import os
import time
import pandas as pd
from polygon import RESTClient

API_KEY = os.environ["POLYGON_API_KEY"]
client = RESTClient(api_key=API_KEY)
DATA_DIR = "/home/ubuntu/sena3fx/data"

TASKS = [
    # (pair_key, from_cur, to_cur, timespan, multiplier, start, end)
    ("usdjpy_oos", "USD", "JPY", "minute", 15,  "2025-03-01", "2026-02-28"),
    ("usdjpy_oos", "USD", "JPY", "hour",   1,   "2025-03-01", "2026-02-28"),
    ("usdjpy_oos", "USD", "JPY", "hour",   4,   "2025-03-01", "2026-02-28"),
    ("eurjpy",     "EUR", "JPY", "hour",   1,   "2024-07-01", "2026-02-28"),
    ("gbpjpy",     "GBP", "JPY", "minute", 1,   "2024-07-01", "2026-02-28"),
    ("gbpjpy",     "GBP", "JPY", "minute", 15,  "2024-07-01", "2026-02-28"),
    ("gbpjpy",     "GBP", "JPY", "hour",   1,   "2024-07-01", "2026-02-28"),
    ("gbpjpy",     "GBP", "JPY", "hour",   4,   "2024-07-01", "2026-02-28"),
]

TF_MAP = {("minute", 1): "1m", ("minute", 15): "15m", ("hour", 1): "1h", ("hour", 4): "4h"}

for pair_key, fc, tc, timespan, mult, start, end in TASKS:
    tf_key = TF_MAP[(timespan, mult)]
    out_path = f"{DATA_DIR}/{pair_key}_{tf_key}.csv"
    if os.path.exists(out_path):
        df_ex = pd.read_csv(out_path)
        print(f"スキップ: {out_path} ({len(df_ex)}行)")
        continue

    ticker = f"C:{fc}{tc}"
    print(f"取得中: {ticker} {mult}{timespan} {start}〜{end} ...", flush=True)
    rows = []
    try:
        for agg in client.list_aggs(
            ticker=ticker, multiplier=mult, timespan=timespan,
            from_=start, to=end, adjusted=True, sort="asc", limit=50000,
        ):
            rows.append({
                "timestamp": pd.Timestamp(agg.timestamp, unit="ms", tz="UTC"),
                "open": agg.open, "high": agg.high,
                "low": agg.low, "close": agg.close, "volume": agg.volume,
            })
    except Exception as e:
        print(f"  エラー: {e}")
        time.sleep(15)
        continue

    if rows:
        df = pd.DataFrame(rows).set_index("timestamp")
        df.to_csv(out_path)
        print(f"  保存: {len(df)}行 → {out_path}")
    else:
        print(f"  データなし")

    print("  12秒待機...", flush=True)
    time.sleep(12)

print("\n全タスク完了")
