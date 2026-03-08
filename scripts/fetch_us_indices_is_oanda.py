"""
US30 / SPX500 の IS期間データを OANDA API で補完取得するスクリプト
IS期間を 2024-07-01 〜 2025-02-28 に延長する

既存の is ファイルは 2024-12-31 始まりのため、
2024-07-01 〜 2024-12-30 を追加取得して結合する
"""

import os
import time
import requests
import pandas as pd

API_KEY  = os.environ.get("OANDA_API_KEY", "")
BASE_URL = "https://api-fxpractice.oanda.com/v3"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

DATA_DIR = "/home/ubuntu/sena3fx/data"

# 追加取得期間（既存ISの前の期間）
ADD_FROM = "2024-07-01T00:00:00Z"
ADD_TO   = "2024-12-31T00:00:00Z"

# IS全期間
IS_FROM = "2024-07-01T00:00:00Z"
IS_TO   = "2025-03-01T00:00:00Z"

PAIRS = {
    "us30":   "US30_USD",
    "spx500": "SPX500_USD",
}

GRANULARITIES = {
    "M1":  "1m",
    "M15": "15m",
    "H4":  "4h",
}

def fetch_candles(instrument, granularity, from_dt, to_dt, count=5000):
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
        last_dt = pd.Timestamp(last_time)
        if granularity == "M1":
            next_dt = last_dt + pd.Timedelta(minutes=1)
        elif granularity == "M15":
            next_dt = last_dt + pd.Timedelta(minutes=15)
        elif granularity == "H4":
            next_dt = last_dt + pd.Timedelta(hours=4)
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

print("=" * 60)
print("US30 / SPX500  IS期間延長 (2024-07-01 〜 2025-02-28)")
print("=" * 60)

for sym, instrument in PAIRS.items():
    print(f"\n--- {sym.upper()} ({instrument}) ---")

    for gran, tf in GRANULARITIES.items():
        is_path = os.path.join(DATA_DIR, f"{sym}_is_{tf}.csv")

        # 既存ISファイルを読み込む
        if os.path.exists(is_path):
            df_existing = pd.read_csv(is_path)
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
            existing_start = df_existing["timestamp"].min()
            print(f"  [{tf}] 既存IS: {existing_start.date()} 〜 {df_existing['timestamp'].max().date()} ({len(df_existing):,}行)")

            # 既存が既に2024-07-01から始まっていればスキップ
            if existing_start <= pd.Timestamp("2024-07-02", tz="UTC"):
                print(f"  [SKIP] 既に2024-07始まり")
                continue

            # 追加取得: 2024-07-01 〜 既存の開始日の前日
            add_to = existing_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            df_existing = pd.DataFrame()
            add_to = IS_TO
            print(f"  [{tf}] 既存ISなし → 全期間取得")

        print(f"  [{tf}] 追加取得: {ADD_FROM[:10]} 〜 {add_to[:10]}")
        df_add = fetch_candles(instrument, gran, ADD_FROM, add_to)

        if df_add.empty:
            print(f"    [FAIL] 追加データ取得失敗")
            continue

        print(f"    追加取得: {len(df_add):,}行")

        # 既存データと結合
        if not df_existing.empty:
            df_merged = pd.concat([df_add, df_existing], ignore_index=True)
        else:
            df_merged = df_add

        df_merged = df_merged.drop_duplicates(subset=["timestamp"], keep="first")
        df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)

        # IS期間でフィルタ
        df_merged = df_merged[
            (df_merged["timestamp"] >= "2024-07-01") &
            (df_merged["timestamp"] <= "2025-02-28 23:59:59")
        ]

        # 品質チェック
        assert len(df_merged) > 0
        assert (df_merged["high"] >= df_merged["low"]).all()
        assert (df_merged["close"] > 0).all()

        df_merged.to_csv(is_path, index=False)
        print(f"    [OK] 保存: {is_path}")
        print(f"         期間: {df_merged['timestamp'].min().date()} 〜 {df_merged['timestamp'].max().date()} ({len(df_merged):,}行)")

print("\n完了。")
