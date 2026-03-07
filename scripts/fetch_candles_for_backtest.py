"""
fetch_candles_for_backtest.py
OANDA APIから1分足データを取得してCSVに保存する
"""
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import os

OANDA_TOKEN = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
BASE_URL    = "https://api-fxpractice.oanda.com"
HEADERS     = {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 取得対象
TARGETS = [
    {"pair": "CADJPY", "instrument": "CAD_JPY", "filename": "cadjpy_1m.csv"},
    {"pair": "CHFJPY", "instrument": "CHF_JPY", "filename": "chfjpy_1m.csv"},
    {"pair": "NZDUSD", "instrument": "NZD_USD", "filename": "nzdusd_1m.csv"},
]

START = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2026, 2, 28, 23, 59, 0, tzinfo=timezone.utc)
COUNT_PER_REQ = 5000  # OANDAの1リクエスト上限


def fetch_candles(instrument, from_dt, to_dt):
    """指定期間の1分足を全て取得して DataFrame で返す"""
    all_rows = []
    current = from_dt

    while current < to_dt:
        params = {
            "granularity": "M1",
            "price": "M",
            "from": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": COUNT_PER_REQ,
        }
        try:
            r = requests.get(
                f"{BASE_URL}/v3/instruments/{instrument}/candles",
                headers=HEADERS, params=params, timeout=30
            )
            if r.status_code != 200:
                print(f"  ERROR {r.status_code}: {r.text[:100]}")
                break

            candles = r.json().get("candles", [])
            if not candles:
                break

            for c in candles:
                if not c.get("complete", True):
                    continue
                ts = c["time"][:19].replace("T", " ")  # "2025-12-01T00:00:00"
                m  = c["mid"]
                all_rows.append({
                    "timestamp": ts,
                    "open":  float(m["o"]),
                    "high":  float(m["h"]),
                    "low":   float(m["l"]),
                    "close": float(m["c"]),
                    "volume": int(c.get("volume", 0)),
                })

            # 次のリクエストの開始時刻
            last_ts = candles[-1]["time"]
            last_dt = datetime.strptime(last_ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            current = last_dt + timedelta(minutes=1)

            print(f"  取得中: {last_dt.strftime('%Y-%m-%d %H:%M')} | 累計{len(all_rows)}本", end="\r")
            time.sleep(0.3)  # レート制限対策

        except Exception as e:
            print(f"  例外: {e}")
            time.sleep(5)
            continue

    print()
    return pd.DataFrame(all_rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for target in TARGETS:
        pair       = target["pair"]
        instrument = target["instrument"]
        filename   = target["filename"]
        out_path   = os.path.join(DATA_DIR, filename)

        print(f"\n[{pair}] 取得開始: {START.date()} → {END.date()}")

        df = fetch_candles(instrument, START, END)

        if df.empty:
            print(f"[{pair}] データなし、スキップ")
            continue

        # 重複除去・ソート
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        df.to_csv(out_path, index=False)
        print(f"[{pair}] 保存完了: {len(df)}行 → {out_path}")
        print(f"  期間: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()