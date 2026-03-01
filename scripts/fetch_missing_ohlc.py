"""
不足しているOHLCデータをPolygon APIで取得してdata/ohlcに保存するスクリプト
対象: NAS100(NDX), US30(DJIA), EURUSD, USDJPY, GBPUSD, AUDUSD
"""
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient

API_KEY = os.environ.get("POLYGON_API_KEY")
client = RESTClient(api_key=API_KEY)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/ohlc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Polygon APIのティッカーシンボルとファイル名のマッピング
# 株価指数はI:、FXはC:プレフィックス
TARGETS = [
    # (polygon_ticker, file_prefix, multiplier, timespan)
    ("I:NDX",   "NAS100",  1, "day"),
    ("I:NDX",   "NAS100",  4, "hour"),
    ("I:DJI",   "US30",    1, "day"),
    ("I:DJI",   "US30",    4, "hour"),
    ("C:EURUSD","EURUSD",  1, "day"),
    ("C:EURUSD","EURUSD",  4, "hour"),
    ("C:USDJPY","USDJPY",  1, "day"),
    ("C:USDJPY","USDJPY",  4, "hour"),
    ("C:GBPUSD","GBPUSD",  1, "day"),
    ("C:GBPUSD","GBPUSD",  4, "hour"),
    ("C:AUDUSD","AUDUSD",  1, "day"),
    ("C:AUDUSD","AUDUSD",  4, "hour"),
]

FROM_DATE = "2019-01-01"
TO_DATE   = "2026-02-28"

def fetch_and_save(ticker, file_prefix, multiplier, timespan):
    suffix = "1d" if timespan == "day" else f"{multiplier}h"
    out_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_{suffix}.csv")

    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path} already exists")
        return

    print(f"  Fetching {ticker} ({multiplier}{timespan[0]}) ...")
    rows = []
    try:
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=FROM_DATE,
            to=TO_DATE,
            limit=50000,
        ):
            dt = datetime.utcfromtimestamp(agg.timestamp / 1000)
            rows.append({
                "datetime": dt.strftime("%Y-%m-%d") if timespan == "day" else dt.strftime("%Y-%m-%d %H:%M:%S"),
                "open":   agg.open,
                "high":   agg.high,
                "low":    agg.low,
                "close":  agg.close,
                "volume": agg.volume if agg.volume else 0,
            })
        time.sleep(0.3)  # レートリミット対策
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return

    if not rows:
        print(f"  [WARN] No data for {ticker}")
        return

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  [OK] {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    print(f"=== OHLCデータ取得開始 ===")
    print(f"期間: {FROM_DATE} ~ {TO_DATE}")
    print(f"保存先: {OUTPUT_DIR}")
    print()

    for ticker, file_prefix, multiplier, timespan in TARGETS:
        fetch_and_save(ticker, file_prefix, multiplier, timespan)

    print()
    print("=== 完了 ===")
    print("取得済みファイル一覧:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if any(p in f for p in ["NAS100", "US30"]):
            path = os.path.join(OUTPUT_DIR, f)
            df = pd.read_csv(path)
            print(f"  {f}: {len(df)}行")
