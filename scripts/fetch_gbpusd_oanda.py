"""
OANDA APIからGBPUSD OHLCデータを取得してCSVに保存する。

対象時間軸: 1分足(M1) / 15分足(M15) / 4時間足(H4)
対象期間:
  IS  (In-Sample):     2024-07-01 〜 2025-02-28 UTC
  OOS (Out-of-Sample): 2025-03-03 〜 2026-02-27 UTC

保存先: /home/ubuntu/sena3fx/data/
  gbpusd_is_1m.csv  / gbpusd_oos_1m.csv
  gbpusd_is_15m.csv / gbpusd_oos_15m.csv
  gbpusd_is_4h.csv  / gbpusd_oos_4h.csv
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
INSTRUMENT  = "GBP_USD"
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

# 時間軸ごとの設定: (granularity, チャンクサイズ, 1チャンク分の時間delta)
TF_CONFIG = {
    "M1":  {"chunk": 5000, "delta_min": 5000},
    "M15": {"chunk": 5000, "delta_min": 5000 * 15},
    "H4":  {"chunk": 5000, "delta_min": 5000 * 240},
}

IS_START  = datetime(2024, 7,  1, tzinfo=timezone.utc)
IS_END    = datetime(2025, 3,  1, tzinfo=timezone.utc)   # 2025-02-28末まで
OOS_START = datetime(2025, 3,  3, tzinfo=timezone.utc)
OOS_END   = datetime(2026, 2, 28, tzinfo=timezone.utc)

TARGETS = [
    # (label, granularity, start, end, filename)
    ("IS-1m",   "M1",  IS_START,  IS_END,   "gbpusd_is_1m.csv"),
    ("IS-15m",  "M15", IS_START,  IS_END,   "gbpusd_is_15m.csv"),
    ("IS-4h",   "H4",  IS_START,  IS_END,   "gbpusd_is_4h.csv"),
    ("OOS-1m",  "M1",  OOS_START, OOS_END,  "gbpusd_oos_1m.csv"),
    ("OOS-15m", "M15", OOS_START, OOS_END,  "gbpusd_oos_15m.csv"),
    ("OOS-4h",  "H4",  OOS_START, OOS_END,  "gbpusd_oos_4h.csv"),
]

SLEEP = 0.3


# ── データ取得 ─────────────────────────────────────────
def fetch_period(granularity: str, start: datetime, end: datetime, label: str) -> pd.DataFrame:
    chunk = TF_CONFIG[granularity]["chunk"]
    url   = f"{BASE_URL}/v3/instruments/{INSTRUMENT}/candles"
    all_rows = []
    cursor   = start
    total    = 0

    print(f"\n[{label}] 取得開始: {start.date()} 〜 {end.date()} ({granularity})")

    while cursor < end:
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": chunk,
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

        if len(candles) < chunk or last_dt >= end:
            break

        time.sleep(SLEEP)

    if not all_rows:
        print(f"  [{label}] データ取得失敗")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ── 品質チェック ───────────────────────────────────────
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


# ── メイン ─────────────────────────────────────────────
def main():
    if not API_KEY:
        print("[ERROR] OANDA_API_KEY が設定されていません")
        sys.exit(1)

    print("=" * 60)
    print("GBPUSD OHLCデータ取得スクリプト（IS/OOS × 3時間軸）")
    print(f"環境: {ENVIRONMENT.upper()}")
    print("=" * 60)

    for label, gran, start, end, fname in TARGETS:
        out = os.path.join(DATA_DIR, fname)

        df = fetch_period(gran, start, end, label)
        if df.empty:
            print(f"[{label}] スキップ（データなし）")
            continue

        df = quality_check(df, label)
        df.to_csv(out, index=False)

        print(f"[{label}] 保存完了: {fname} ({len(df):,}行)")

    print("\n" + "=" * 60)
    print("全ファイル取得・保存完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
