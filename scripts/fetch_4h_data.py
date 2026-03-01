"""
4時間足OHLCデータをPolygon APIで取得（レートリミット対策付き）
"""
import os
import time
import pandas as pd
from datetime import datetime

API_KEY = os.environ.get("POLYGON_API_KEY")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/ohlc")

TARGETS_4H = [
    ("I:NDX",   "NAS100"),
    ("C:EURUSD","EURUSD"),
    ("C:USDJPY","USDJPY"),
    ("C:GBPUSD","GBPUSD"),
    ("C:AUDUSD","AUDUSD"),
]

FROM_DATE = "2019-01-01"
TO_DATE   = "2026-02-28"

def fetch_4h(ticker, file_prefix):
    out_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_4h.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print(f"  [SKIP] {out_path} ({len(df)}行)")
        return

    print(f"  Fetching {ticker} 4h ...")
    import requests

    rows = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{FROM_DATE}/{TO_DATE}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print("    Rate limit hit, waiting 60s...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                print(f"    [ERROR] {resp.status_code}: {resp.text[:200]}")
                return

            data = resp.json()
            results = data.get("results", [])
            for r in results:
                dt = datetime.utcfromtimestamp(r["t"] / 1000)
                rows.append({
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "open":   r["o"],
                    "high":   r["h"],
                    "low":    r["l"],
                    "close":  r["c"],
                    "volume": r.get("v", 0),
                })

            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": API_KEY}
                time.sleep(1)  # レートリミット対策
            else:
                break

        except Exception as e:
            print(f"    [ERROR] {e}")
            time.sleep(5)
            break

    if not rows:
        print(f"  [WARN] No data for {ticker}")
        return

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  [OK] {out_path} ({len(df)}行)")

if __name__ == "__main__":
    print("=== 4時間足データ取得 ===")
    for ticker, prefix in TARGETS_4H:
        fetch_4h(ticker, prefix)
        time.sleep(2)
    print("=== 完了 ===")
