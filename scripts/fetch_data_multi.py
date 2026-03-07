"""
fetch_data_multi.py
===================
EURJPY・GBPJPYの1m・15m・4hデータをPolygon.io APIから取得する
"""
import os, time
import pandas as pd
from polygon import RESTClient

API_KEY = os.environ.get("POLYGON_API_KEY")
client = RESTClient(api_key=API_KEY)

START = "2024-07-01"
END   = "2025-02-06"
OUT   = "/home/ubuntu/sena3fx/data"
os.makedirs(OUT, exist_ok=True)

PAIRS = {
    "EURJPY": "C:EURJPY",
    "GBPJPY": "C:GBPJPY",
}

TIMEFRAMES = {
    "1m":  {"multiplier": 1,  "timespan": "minute"},
    "15m": {"multiplier": 15, "timespan": "minute"},
    "4h":  {"multiplier": 4,  "timespan": "hour"},
}

def fetch_ohlc(ticker, multiplier, timespan, start, end):
    rows = []
    try:
        aggs = client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=False,
            sort="asc",
            limit=50000,
        )
        for a in aggs:
            rows.append({
                "timestamp": pd.Timestamp(a.timestamp, unit="ms", tz="UTC"),
                "open":   a.open,
                "high":   a.high,
                "low":    a.low,
                "close":  a.close,
                "volume": a.volume if a.volume else 0,
            })
    except Exception as e:
        print(f"  ERROR: {e}")
    return rows

for pair_name, ticker in PAIRS.items():
    print(f"\n=== {pair_name} ({ticker}) ===")
    for tf_name, tf_params in TIMEFRAMES.items():
        out_path = f"{OUT}/{pair_name.lower()}_{tf_name}.csv"
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            print(f"  [{tf_name}] 既存ファイルあり ({len(df_existing)}本) - スキップ")
            continue

        print(f"  [{tf_name}] 取得中...", end="", flush=True)
        rows = fetch_ohlc(ticker, tf_params["multiplier"], tf_params["timespan"], START, END)
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.set_index("timestamp").sort_index()
            # 重複除去
            df = df[~df.index.duplicated(keep="first")]
            df.to_csv(out_path)
            print(f" {len(df)}本 → {out_path}")
        else:
            print(f" データなし")
        time.sleep(0.5)

print("\n完了")
