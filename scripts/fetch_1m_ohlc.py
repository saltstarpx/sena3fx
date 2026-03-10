"""
OANDA API から1分足OHLCデータを取得してCSVに保存する
対象: XAGUSD / EURJPY / GBPJPY / NZDUSD
期間: 2025-01-01 〜 2026-02-28
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api-fxpractice.oanda.com"
TOKEN    = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
HEADERS  = {"Authorization": f"Bearer {TOKEN}"}

INSTRUMENTS = [
    ("XAG_USD", "XAGUSD"),
    ("EUR_JPY", "EURJPY"),
    ("GBP_JPY", "GBPJPY"),
    ("NZD_USD", "NZDUSD"),
]

START = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2026, 2, 28, 23, 59, 0, tzinfo=timezone.utc)
OUT_DIR = "/home/ubuntu/sena3fx/data/ohlc"


def fetch_candles(instrument, from_dt, count=5000, max_retry=3):
    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": "M1",
        "price": "M",
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": count,
    }
    for attempt in range(max_retry):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("candles", [])
            elif resp.status_code == 429:
                print(f"  [429 Rate Limit] {attempt+1}/{max_retry} - 10秒待機...")
                time.sleep(10)
            elif resp.status_code in (400, 404):
                print(f"  [ERROR {resp.status_code}] {instrument}: {resp.text[:200]}")
                return None
            else:
                print(f"  [ERROR {resp.status_code}] {resp.text[:200]}")
                time.sleep(5)
        except requests.exceptions.ConnectionError as e:
            print(f"  [ConnectionError] {attempt+1}/{max_retry} - 5秒待機... {e}")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  [Timeout] {attempt+1}/{max_retry} - 5秒待機...")
            time.sleep(5)
    return None


def fetch_instrument(instrument, name):
    print(f"\n{'='*50}")
    print(f"[{name}] 取得開始: {START} 〜 {END}")
    out_path = os.path.join(OUT_DIR, f"{name}_1m.csv")

    rows = []
    current = START
    request_count = 0

    while current < END:
        candles = fetch_candles(instrument, current, count=5000)

        if candles is None:
            print(f"  [{name}] エラーによりスキップ")
            return

        if len(candles) == 0:
            print(f"  [{name}] データなし（current={current}）→ break")
            break

        added = 0
        last_time = None
        for c in candles:
            if not c.get("complete", False):
                continue
            ts_str = c["time"][:19].replace("T", " ")
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if ts > END:
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": ts_str,
                "open":   float(mid["o"]),
                "high":   float(mid["h"]),
                "low":    float(mid["l"]),
                "close":  float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
            last_time = ts
            added += 1

        request_count += 1
        if request_count % 50 == 0:
            print(f"  [{name}] {request_count}リクエスト完了 / 累計{len(rows)}行 / current={current}")

        if last_time is None:
            current = current + timedelta(minutes=5000)
        else:
            current = last_time + timedelta(minutes=1)

        time.sleep(0.2)

    if not rows:
        print(f"  [{name}] データなし")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] <= END.strftime("%Y-%m-%d %H:%M:%S")]
    df.to_csv(out_path, index=False)
    print(f"\n{name}: 取得完了 -> {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for instrument, name in INSTRUMENTS:
        fetch_instrument(instrument, name)
    print("\n\n全銘柄取得完了")
