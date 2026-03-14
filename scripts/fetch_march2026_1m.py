"""
OANDA APIで7銘柄の2026年3月1分足データを取得してdata/ohlc/に追記
- from + 1日単位でループ（from+toの組み合わせ）
- 既存CSVの末尾以降のみ取得（重複なし）
- LFSなし（通常gitオブジェクト）
- タイムゾーン: UTC
"""
import os
import sys
import time
import datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
API_KEY = os.environ["OANDA_API_KEY"]
ENV = os.environ.get("OANDA_ENVIRONMENT", "practice")

client = oandapyV20.API(access_token=API_KEY, environment=ENV)

REPO = Path(__file__).parent.parent
OHLC_DIR = REPO / "data" / "ohlc"

# 対象銘柄: OANDAシンボル → ファイル名
SYMBOLS = {
    "USD_JPY": "USDJPY_1m.csv",
    "EUR_USD": "EURUSD_1m.csv",
    "GBP_USD": "GBPUSD_1m.csv",
    "USD_CAD": "USDCAD_1m.csv",
    "NZD_USD": "NZDUSD_1m.csv",
    "XAU_USD": "XAUUSD_1m.csv",
    "AUD_USD": "AUDUSD_1m.csv",
}

FETCH_START = datetime.datetime(2026, 3, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
FETCH_END = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)


def fetch_candles(instrument, from_dt, to_dt):
    """OANDA APIからM1 OHLCVを1日単位で取得してDataFrameで返す"""
    all_rows = []
    current = from_dt

    while current < to_dt:
        next_day = current + datetime.timedelta(days=1)
        chunk_end = min(next_day, to_dt)

        params = {
            "granularity": "M1",
            "from": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }

        for attempt in range(3):
            try:
                r = instruments.InstrumentsCandles(instrument, params=params)
                client.request(r)
                candles = r.response.get("candles", [])
                break
            except Exception as e:
                print(f"  Error (attempt {attempt+1}): {e}")
                time.sleep(5)
                candles = []

        for c in candles:
            if not c.get("complete", True):
                continue
            ts = c["time"][:19].replace("T", " ")
            mid = c["mid"]
            all_rows.append({
                "timestamp": ts,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })

        if candles:
            print(f"    {current.date()}: {len(candles)} bars")

        current = chunk_end
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df


for oanda_sym, filename in SYMBOLS.items():
    csv_path = OHLC_DIR / filename
    print(f"\n{'='*50}")
    print(f"Processing {oanda_sym} -> {filename}")

    # 既存CSVの末尾を確認
    fetch_from = FETCH_START
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
            last_ts = existing["timestamp"].max()
            if pd.notna(last_ts):
                fetch_from = max(FETCH_START, (last_ts + pd.Timedelta(minutes=1)).to_pydatetime().replace(tzinfo=datetime.timezone.utc))
                print(f"  Existing last: {last_ts} -> fetch from {fetch_from}")
        except Exception as e:
            print(f"  Warning: could not read existing CSV: {e}")

    if fetch_from >= FETCH_END:
        print(f"  Already up to date, skipping.")
        continue

    print(f"  Fetching {fetch_from} -> {FETCH_END}")
    df = fetch_candles(oanda_sym, fetch_from, FETCH_END)

    if df.empty:
        print(f"  No data returned.")
        continue

    print(f"  Total fetched: {len(df):,} rows")

    # 追記
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if csv_path.exists():
        with open(csv_path, "a") as f:
            df.to_csv(f, index=False, header=False, float_format="%.5f")
        print(f"  Appended to {csv_path}")
    else:
        df.to_csv(csv_path, index=False, float_format="%.5f")
        print(f"  Created {csv_path}")

print("\nDone.")
