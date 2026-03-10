"""
OANDA API から1分足データを取得して data/ohlc/ に保存するスクリプト。
対象: EURJPY / GBPJPY / NZDUSD / XAGUSD
期間: 2024-07-01 〜 2026-02-28（IS+OOS全期間）
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api-fxpractice.oanda.com"
TOKEN    = "fef47c0fc461c23c9816317ec545f65e-7596c4a3b28374bf0655d046a3b60289"
HEADERS  = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")

# 取得対象（OANDA instrument名: 大文字_区切り）
INSTRUMENTS = {
    "EURJPY": "EUR_JPY",
    "GBPJPY": "GBP_JPY",
    "NZDUSD": "NZD_USD",
    "XAGUSD": "XAG_USD",
}

START_UTC = datetime(2024, 7, 1, tzinfo=timezone.utc)
END_UTC   = datetime(2026, 2, 28, 23, 59, tzinfo=timezone.utc)
CHUNK_BARS = 5000  # OANDAの1リクエスト上限


def fetch_candles(instrument, from_dt, to_dt):
    """指定期間の1分足を全取得（5000本ずつページネーション）"""
    rows = []
    current = from_dt

    while current < to_dt:
        params = {
            "granularity": "M1",
            "from": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": CHUNK_BARS,
            "price": "M",
        }
        url = f"{BASE_URL}/v3/instruments/{instrument}/candles"

        for attempt in range(4):
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                if r.status_code == 200:
                    break
                elif r.status_code == 429:
                    print(f"  Rate limited, wait 10s...")
                    time.sleep(10)
                else:
                    print(f"  HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(5)
            except Exception as e:
                print(f"  Error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  Failed after 4 attempts at {current}")
            break

        candles = r.json().get("candles", [])
        if not candles:
            break

        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "timestamp": c["time"][:19] + "+00:00",
                "open":  float(mid["o"]),
                "high":  float(mid["h"]),
                "low":   float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })

        # 次のチャンク開始点（最後のcandle時刻 + 1分）
        last_time = candles[-1]["time"]
        last_dt = datetime.strptime(last_time[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        next_dt = last_dt + timedelta(minutes=1)

        print(f"  {instrument}: {current.date()} 〜 {last_dt.date()} (+{len(candles)}本, 累計{len(rows)}本)")

        if next_dt >= to_dt or len(candles) < CHUNK_BARS:
            break
        current = next_dt
        time.sleep(0.3)  # レート制限対策

    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for label, instrument in INSTRUMENTS.items():
        out_path = os.path.join(OUT_DIR, f"{label}_1m.csv")
        print(f"\n{'='*50}")
        print(f"{label} ({instrument}) 取得開始")
        print(f"期間: {START_UTC.date()} 〜 {END_UTC.date()}")
        print(f"出力: {out_path}")

        rows = fetch_candles(instrument, START_UTC, END_UTC)

        if not rows:
            print(f"  ⚠️  データ取得失敗")
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        df.to_csv(out_path)
        print(f"  ✅ {len(df)}行 → {out_path}")
        print(f"  期間: {df.index[0]} 〜 {df.index[-1]}")

    print("\n完了!")


if __name__ == "__main__":
    main()
