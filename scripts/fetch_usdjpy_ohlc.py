"""
OANDA APIからUSDJPYの4時間足OHLCデータを取得してCSVに保存する。
2023-10-01〜2026-02-28 の期間を対象とする。
"""

import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

API_KEY = os.environ.get("OANDA_API_KEY")
ENVIRONMENT = os.environ.get("OANDA_ENVIRONMENT", "practice")

client = oandapyV20.API(access_token=API_KEY, environment=ENVIRONMENT)

def fetch_candles(instrument, granularity, from_dt, to_dt, batch_hours=2000):
    """指定期間のローソク足データを取得する（バッチ分割）"""
    from datetime import timedelta
    all_candles = []
    current_from = from_dt
    # H4足の場合、2000本 = 2000*4時間
    batch_delta = timedelta(hours=batch_hours)

    while current_from < to_dt:
        current_to = min(current_from + batch_delta, to_dt)
        params = {
            "granularity": granularity,
            "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": current_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M"  # Mid価格
        }
        r = instruments.InstrumentsCandles(instrument, params=params)
        response = client.request(r)
        candles = response.get("candles", [])

        if not candles:
            current_from = current_to
            continue

        all_candles.extend(candles)
        print(f"  取得済み: {len(all_candles)}本 (最新: {candles[-1]['time'][:10]})")

        current_from = current_to
        # API制限対策
        time.sleep(0.3)

    return all_candles

def candles_to_df(candles):
    """ローソク足データをDataFrameに変換する"""
    rows = []
    for c in candles:
        if c.get("complete", False):
            mid = c.get("mid", {})
            rows.append({
                "datetime": c["time"][:19],
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0))
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
    return df

if __name__ == "__main__":
    print("USDJPY 4時間足OHLCデータ取得開始...")

    from_dt = datetime(2023, 10, 1, tzinfo=timezone.utc)
    to_dt = datetime(2026, 2, 28, tzinfo=timezone.utc)

    candles = fetch_candles("USD_JPY", "H4", from_dt, to_dt)
    df = candles_to_df(candles)

    print(f"\n取得完了: {len(df)}本")
    print(f"期間: {df.index[0]} 〜 {df.index[-1]}")
    print(df.tail(3))

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ohlc', 'USDJPY_H4.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    print(f"\n保存完了: {out_path}")
