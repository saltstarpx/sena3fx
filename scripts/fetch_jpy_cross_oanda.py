"""
EURJPY / GBPJPY / EURGBP の1分足・15分足 IS/OOS データ取得スクリプト
OANDA v20 REST API (practice環境)

IS:  2024-07-01 〜 2025-02-28
OOS: 2025-03-03 〜 2026-02-27
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

API_KEY  = os.environ.get("OANDA_API_KEY", "")
BASE_URL = "https://api-fxpractice.oanda.com/v3"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

DATA_DIR = "/home/ubuntu/sena3fx/data"

PERIODS = {
    "is":  ("2024-07-01T00:00:00Z", "2025-03-01T00:00:00Z"),
    "oos": ("2025-03-03T00:00:00Z", "2026-02-28T00:00:00Z"),
}

PAIRS = {
    "eurjpy": "EUR_JPY",
    "gbpjpy": "GBP_JPY",
    "eurgbp": "EUR_GBP",
}

GRANULARITIES = ["M1", "M15"]
TF_MAP = {"M1": "1m", "M15": "15m"}

def fetch_candles(instrument, granularity, from_dt, to_dt, count=5000):
    """OANDA APIから最大count本ずつ分割取得"""
    url = f"{BASE_URL}/instruments/{instrument}/candles"
    all_rows = []
    current = from_dt

    while current < to_dt:
        params = {
            "granularity": granularity,
            "from": current,
            "count": count,
            "price": "M",
        }
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                if r.status_code == 200:
                    break
                elif r.status_code == 429:
                    time.sleep(5)
                else:
                    print(f"    [ERROR] {r.status_code}: {r.text[:100]}")
                    return pd.DataFrame()
            except Exception as e:
                print(f"    [RETRY] {e}")
                time.sleep(3)
        else:
            print("    [FAIL] 3回リトライ失敗")
            return pd.DataFrame()

        candles = r.json().get("candles", [])
        if not candles:
            break

        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            all_rows.append({
                "timestamp": c["time"],
                "open":  float(mid.get("o", 0)),
                "high":  float(mid.get("h", 0)),
                "low":   float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        last_time = candles[-1]["time"]
        # 次の開始時刻を最後の足の次に設定
        last_dt = pd.Timestamp(last_time)
        if granularity == "M1":
            next_dt = last_dt + pd.Timedelta(minutes=1)
        elif granularity == "M15":
            next_dt = last_dt + pd.Timedelta(minutes=15)
        else:
            next_dt = last_dt + pd.Timedelta(hours=1)

        current = next_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if len(candles) < count:
            break

        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def quality_check(df, label):
    errors = 0
    if len(df) == 0:
        print(f"    [ERROR] {label}: データが空")
        return False
    if df[["open","high","low","close"]].isnull().any().any():
        print(f"    [ERROR] {label}: 欠損値あり")
        errors += 1
    if not (df["high"] >= df["low"]).all():
        print(f"    [ERROR] {label}: high < low の行あり")
        errors += 1
    if not (df["close"] > 0).all():
        print(f"    [ERROR] {label}: close <= 0 の行あり")
        errors += 1
    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        print(f"    [WARN]  {label}: 重複 {dup}件（自動除去済み）")
    if errors == 0:
        print(f"    [OK]    品質チェック通過")
    return errors == 0

# ============================================================
# メイン処理
# ============================================================
print("=" * 60)
print("EURJPY / GBPJPY / EURGBP  1分足・15分足 IS/OOS 取得")
print("=" * 60)

for sym, instrument in PAIRS.items():
    print(f"\n--- {sym.upper()} ({instrument}) ---")
    for gran in GRANULARITIES:
        tf = TF_MAP[gran]
        for period, (from_dt, to_dt) in PERIODS.items():
            out_path = os.path.join(DATA_DIR, f"{sym}_{period}_{tf}.csv")
            if os.path.exists(out_path):
                rows = len(pd.read_csv(out_path))
                print(f"  [SKIP] {sym}_{period}_{tf}.csv 既存 ({rows:,}行)")
                continue

            print(f"  [{period.upper()}] {gran} 取得中... ({from_dt[:10]} 〜 {to_dt[:10]})")
            df = fetch_candles(instrument, gran, from_dt, to_dt)

            if df.empty:
                print(f"    [FAIL] データ取得失敗")
                continue

            quality_check(df, f"{sym}_{period}_{tf}")
            df.to_csv(out_path, index=False)
            print(f"    保存: {out_path} ({len(df):,}行)")

print("\n完了。")
