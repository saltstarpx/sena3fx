"""
fetch_multi_asset_data.py
=========================
OANDAから多アセットのOHLCデータを取得する。
対象: EURUSD, GBPUSD, AUDUSD, XAUUSD (金), SPX500, US30 (ダウ), NAS100 (ナスダック)
期間: 2025/1/1 〜 2026/2/28
足種: 1分足, 15分足, 4時間足

OANDA API仕様:
- from/to を同時に count と使えない
- from + count のみ使用してページング
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

OANDA_TOKEN = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
ACCOUNT_ID  = "101-009-38652105-001"
BASE_URL    = "https://api-fxpractice.oanda.com"
HEADERS = {
    "Authorization": f"Bearer {OANDA_TOKEN}",
    "Content-Type": "application/json",
}

OUT_DIR = "/home/ubuntu/sena3fx/data"
os.makedirs(OUT_DIR, exist_ok=True)

# 取得対象アセット（OANDAの銘柄コード）
ASSETS = {
    "EURUSD": {"oanda": "EUR_USD", "spread": 0.2, "pip_factor": 100},
    "GBPUSD": {"oanda": "GBP_USD", "spread": 0.3, "pip_factor": 100},
    "AUDUSD": {"oanda": "AUD_USD", "spread": 0.3, "pip_factor": 100},
    "XAUUSD": {"oanda": "XAU_USD", "spread": 0.5, "pip_factor": 1},   # 金: 1pips=1USD
    "SPX500": {"oanda": "SPX500_USD", "spread": 1.0, "pip_factor": 1},
    "US30":   {"oanda": "US30_USD",   "spread": 2.0, "pip_factor": 1},
    "NAS100": {"oanda": "NAS100_USD", "spread": 2.0, "pip_factor": 1},
}

GRANULARITIES = {
    "1m":  "M1",
    "15m": "M15",
    "4h":  "H4",
}

START_DATE = "2025-01-01T00:00:00Z"
END_DATE   = "2026-02-28T23:59:00Z"


def fetch_candles(instrument, granularity, from_dt, to_dt, max_count=5000):
    """OANDAから指定期間のローソク足データを取得（ページング対応）
    
    OANDA仕様: from + count を使用（from + to + count は不可）
    """
    all_candles = []
    current_from = from_dt
    to_ts = pd.Timestamp(to_dt)

    while True:
        # from + count のみ指定（to は使わない）
        params = {
            "granularity": granularity,
            "from": current_from,
            "count": max_count,
            "price": "M",  # Mid price
        }
        try:
            r = requests.get(
                f"{BASE_URL}/v3/instruments/{instrument}/candles",
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            if r.status_code != 200:
                print(f"  ERROR {r.status_code}: {r.text[:200]}")
                break

            data = r.json()
            candles = data.get("candles", [])
            if not candles:
                break

            # 終了日を超えたものをフィルタ
            filtered = [c for c in candles if pd.Timestamp(c["time"]) <= to_ts]
            all_candles.extend(filtered)

            # 終了日を超えたか、最後のページなら終了
            if len(filtered) < len(candles) or len(candles) < max_count:
                break

            # 最後のローソク足の時刻を次の開始点に
            last_time = candles[-1]["time"]
            last_dt = pd.Timestamp(last_time)
            if last_dt >= to_ts:
                break
            next_from = (last_dt + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            current_from = next_from
            time.sleep(0.2)  # レート制限対策

        except Exception as e:
            print(f"  Exception: {e}")
            break

    return all_candles


def candles_to_df(candles):
    """ローソク足リストをDataFrameに変換"""
    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        mid = c.get("mid", {})
        rows.append({
            "timestamp": pd.Timestamp(c["time"]).tz_localize(None),
            "open":   float(mid.get("o", 0)),
            "high":   float(mid.get("h", 0)),
            "low":    float(mid.get("l", 0)),
            "close":  float(mid.get("c", 0)),
            "volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.set_index("timestamp").sort_index()
        # 重複除去
        df = df[~df.index.duplicated(keep="first")]
    return df


# ─── メイン処理 ─────────────────────────────────────────────────────────────
print(f"OANDAから多アセットデータ取得開始")
print(f"期間: {START_DATE} 〜 {END_DATE}")
print(f"対象: {list(ASSETS.keys())}")
print()

results = {}

for asset_name, cfg in ASSETS.items():
    instrument = cfg["oanda"]
    print(f"\n{'='*50}")
    print(f"[{asset_name}] ({instrument})")

    asset_results = {}
    for tf_name, granularity in GRANULARITIES.items():
        out_path = f"{OUT_DIR}/{asset_name.lower()}_{tf_name}.csv"

        # 既存ファイルがあればスキップ
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            print(f"  {tf_name}: 既存ファイル ({len(existing)}行) → スキップ")
            asset_results[tf_name] = len(existing)
            continue

        print(f"  {tf_name} ({granularity}) 取得中...", end="", flush=True)
        candles = fetch_candles(instrument, granularity, START_DATE, END_DATE)
        df = candles_to_df(candles)

        if len(df) > 0:
            df.reset_index().to_csv(out_path, index=False)
            print(f" {len(df)}行 → {out_path}")
            asset_results[tf_name] = len(df)
        else:
            print(f" データなし（{instrument}はOANDAで取引不可の可能性）")
            asset_results[tf_name] = 0

        time.sleep(0.5)

    results[asset_name] = asset_results

print("\n\n=== 取得結果サマリー ===")
for asset, tf_results in results.items():
    print(f"{asset}: {tf_results}")

print("\n完了")
