"""
OANDA APIから低スプレッド銘柄のOHLCデータを取得してCSVに保存する。

対象銘柄: AUDUSD / USDCAD / USDCHF / NZDUSD
対象時間軸: 1分足(M1) / 15分足(M15) / 4時間足(H4)
対象期間:
  IS  (In-Sample):     2024-07-01 〜 2025-02-28 UTC
  OOS (Out-of-Sample): 2025-03-03 〜 2026-02-27 UTC

保存先: /home/ubuntu/sena3fx/data/
  {pair}_is_1m.csv  / {pair}_oos_1m.csv
  {pair}_is_15m.csv / {pair}_oos_15m.csv
  {pair}_is_4h.csv  / {pair}_oos_4h.csv
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ── 設定 ──────────────────────────────────────────────
API_KEY     = os.environ.get("OANDA_API_KEY", "")
ENVIRONMENT = os.environ.get("OANDA_ENVIRONMENT", "practice")
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = (
    "https://api-fxpractice.oanda.com"
    if ENVIRONMENT == "practice"
    else "https://api-fxtrade.oanda.com"
)
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

IS_START  = datetime(2024, 7,  1, tzinfo=timezone.utc)
IS_END    = datetime(2025, 3,  1, tzinfo=timezone.utc)
OOS_START = datetime(2025, 3,  3, tzinfo=timezone.utc)
OOS_END   = datetime(2026, 2, 28, tzinfo=timezone.utc)

# 銘柄定義: (OANDA instrument, ファイル名prefix)
INSTRUMENTS = [
    ("AUD_USD", "audusd"),
    ("USD_CAD", "usdcad"),
    ("USD_CHF", "usdchf"),
    ("NZD_USD", "nzdusd"),
]

GRANULARITIES = ["M1", "M15", "H4"]
TF_SUFFIX = {"M1": "1m", "M15": "15m", "H4": "4h"}

CHUNK = 5000
SLEEP = 0.3


def fetch_period(instrument: str, granularity: str, start: datetime, end: datetime, label: str) -> pd.DataFrame:
    url      = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    all_rows = []
    cursor   = start
    total    = 0

    print(f"\n[{label}] 取得開始: {start.date()} 〜 {end.date()} ({granularity})")

    while cursor < end:
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": CHUNK,
        }
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  [{label}] HTTP {resp.status_code}: {resp.text[:200]}")
            break

        candles = resp.json().get("candles", [])
        if not candles:
            break

        for c in candles:
            if not c.get("complete", True):
                continue
            ts_str = c["time"][:19]
            ts_dt  = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            if ts_dt >= end:
                continue
            mid = c.get("mid", {})
            all_rows.append({
                "timestamp": ts_str,
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        last_ts = candles[-1]["time"][:19]
        last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        cursor  = last_dt + timedelta(minutes=1)

        total += len(candles)
        print(f"  [{label}] 累計: {total:,}本 | 最新: {last_ts}", flush=True)

        if len(candles) < CHUNK or last_dt >= end:
            break

        time.sleep(SLEEP)

    if not all_rows:
        print(f"  [{label}] データ取得失敗")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def quality_check(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n[{label}] 品質チェック ({len(df):,}行)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        print(f"  [修正] タイムスタンプ重複 {dup}件 → 除去")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    df = df.sort_values("timestamp").reset_index(drop=True)

    ohlc_errors = (
        (df["high"] < df["low"]).sum() +
        (df["open"] > df["high"]).sum() +
        (df["open"] < df["low"]).sum() +
        (df["close"] > df["high"]).sum() +
        (df["close"] < df["low"]).sum()
    )
    if ohlc_errors > 0:
        print(f"  [ERROR] OHLC整合性エラー: {ohlc_errors}件")
        sys.exit(1)

    if (df["close"] == 0).sum() > 0 or df[["open","high","low","close"]].isnull().sum().sum() > 0:
        print("  [ERROR] ゼロ値または欠損値あり")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex) or len(df.columns) > 7:
        print(f"  [ERROR] カラム数異常: {len(df.columns)}個")
        sys.exit(1)

    print(f"  [OK] {len(df):,}行 | {df['timestamp'].min()} 〜 {df['timestamp'].max()} | カラム={len(df.columns)}")
    return df


def main():
    if not API_KEY:
        print("[ERROR] OANDA_API_KEY が設定されていません")
        sys.exit(1)

    print("=" * 60)
    print("低スプレッド銘柄 OHLCデータ取得（IS/OOS × 3時間軸）")
    print(f"対象: AUDUSD / USDCAD / USDCHF / NZDUSD")
    print(f"環境: {ENVIRONMENT.upper()}")
    print("=" * 60)

    for instrument, prefix in INSTRUMENTS:
        print(f"\n{'='*60}")
        print(f"銘柄: {instrument}")
        print(f"{'='*60}")

        for gran in GRANULARITIES:
            suffix = TF_SUFFIX[gran]

            for period_label, start, end in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
                label   = f"{prefix.upper()}-{period_label}-{suffix}"
                fname   = f"{prefix}_{period_label.lower()}_{suffix}.csv"
                out     = os.path.join(DATA_DIR, fname)

                df = fetch_period(instrument, gran, start, end, label)
                if df.empty:
                    print(f"[{label}] スキップ（データなし）")
                    continue

                df = quality_check(df, label)
                df.to_csv(out, index=False)
                print(f"[{label}] 保存完了: {fname} ({len(df):,}行)")

    print("\n" + "=" * 60)
    print("全銘柄・全ファイル取得・保存完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
