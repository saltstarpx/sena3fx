"""
fetch_all_pairs_is_oos.py
=========================
全通貨ペアのIS/OOS期間 × 1分足/15分足/4時間足データをOANDA APIから取得する。

対象ペア: USDJPY, EURUSD, EURJPY, AUDJPY, GBPUSD, AUDUSD,
          USDCAD, USDCHF, EURGBP, NZDUSD, CADJPY, CHFJPY

期間:
  IS  : 2024-07-01 00:00 UTC 〜 2025-02-28 23:59 UTC
  OOS : 2025-03-03 00:00 UTC 〜 2026-02-27 23:59 UTC

OANDA API仕様:
  - from + count でページング（from + to + count は不可）
  - 1リクエスト上限: 5000本
  - price=M（ミッドポイント）
  - complete=False の未確定足は除外

CSVカラム: timestamp, open, high, low, close, volume（6カラム固定）
"""

import requests
import pandas as pd
import time
import os
import sys

# ── 設定 ────────────────────────────────────────────────────────────────
OANDA_TOKEN = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
BASE_URL = "https://api-fxpractice.oanda.com"
HEADERS = {
    "Authorization": f"Bearer {OANDA_TOKEN}",
    "Content-Type": "application/json",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── 対象ペア ────────────────────────────────────────────────────────────
PAIRS = {
    "usdjpy": "USD_JPY",
    "eurusd": "EUR_USD",
    "eurjpy": "EUR_JPY",
    "audjpy": "AUD_JPY",
    "gbpusd": "GBP_USD",
    "audusd": "AUD_USD",
    "usdcad": "USD_CAD",
    "usdchf": "USD_CHF",
    "eurgbp": "EUR_GBP",
    "nzdusd": "NZD_USD",
    "cadjpy": "CAD_JPY",
    "chfjpy": "CHF_JPY",
}

# ── 期間 ────────────────────────────────────────────────────────────────
PERIODS = {
    "is":  ("2024-07-01T00:00:00Z", "2025-02-28T23:59:00Z"),
    "oos": ("2025-03-03T00:00:00Z", "2026-02-27T23:59:00Z"),
}

# ── 時間軸 ──────────────────────────────────────────────────────────────
GRANULARITIES = {
    "1m":  "M1",
    "15m": "M15",
    "4h":  "H4",
}

MAX_COUNT = 5000
RATE_LIMIT_SLEEP = 0.25  # リクエスト間隔（秒）


# ── API取得関数 ─────────────────────────────────────────────────────────
def fetch_candles(instrument, granularity, from_str, to_str):
    """OANDA APIから指定期間のローソク足を全取得（ページング対応）"""
    all_rows = []
    current_from = from_str
    to_ts = pd.Timestamp(to_str)

    while True:
        params = {
            "granularity": granularity,
            "from": current_from,
            "count": MAX_COUNT,
            "price": "M",
        }
        try:
            r = requests.get(
                f"{BASE_URL}/v3/instruments/{instrument}/candles",
                headers=HEADERS, params=params, timeout=30,
            )
            if r.status_code != 200:
                print(f"\n  ERROR {r.status_code}: {r.text[:200]}")
                break

            candles = r.json().get("candles", [])
            if not candles:
                break

            for c in candles:
                # 未確定足をスキップ
                if not c.get("complete", True):
                    continue
                ts = pd.Timestamp(c["time"])
                if ts > to_ts:
                    continue
                mid = c["mid"]
                all_rows.append({
                    "timestamp": ts.tz_localize(None) if ts.tzinfo else ts,
                    "open":  float(mid["o"]),
                    "high":  float(mid["h"]),
                    "low":   float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume", 0)),
                })

            # 最後のローソク足の時刻
            last_ts = pd.Timestamp(candles[-1]["time"])
            if last_ts >= to_ts or len(candles) < MAX_COUNT:
                break

            # 次のページ開始点
            current_from = (last_ts + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            n = len(all_rows)
            print(f"\r    累計 {n:,} 本 ({last_ts.strftime('%Y-%m-%d %H:%M')})", end="", flush=True)
            time.sleep(RATE_LIMIT_SLEEP)

        except requests.exceptions.RequestException as e:
            if not hasattr(fetch_candles, '_retry_count'):
                fetch_candles._retry_count = 0
            fetch_candles._retry_count += 1
            if fetch_candles._retry_count > 5:
                print(f"\n  リトライ上限到達、取得中断")
                fetch_candles._retry_count = 0
                break
            wait = min(2 ** fetch_candles._retry_count, 16)
            print(f"\n  ネットワークエラー（{fetch_candles._retry_count}/5）: {e}")
            print(f"  {wait}秒後にリトライ...")
            time.sleep(wait)
            continue

    return all_rows


# ── 品質チェック関数 ────────────────────────────────────────────────────
def quality_check(df, label):
    """保存前の品質チェック（仕様書の5項目）"""
    errors = []

    # ① タイムスタンプ重複チェック（除去は呼び出し側で実施済み）
    # ② OHLC整合性
    if not (df["high"] >= df["low"]).all():
        errors.append("high < low の異常行あり")
    if not (df["open"] >= df["low"]).all():
        errors.append("open < low の異常行あり")
    if not (df["open"] <= df["high"]).all():
        errors.append("open > high の異常行あり")
    if not (df["close"] >= df["low"]).all():
        errors.append("close < low の異常行あり")
    if not (df["close"] <= df["high"]).all():
        errors.append("close > high の異常行あり")

    # ③ ゼロ値・欠損値
    if not (df["close"] > 0).all():
        errors.append("close=0 の異常値あり")
    nulls = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if nulls > 0:
        errors.append(f"欠損値 {nulls} 個あり")

    # ④ カラム数
    if len(df.columns) > 7:
        errors.append(f"カラム数異常: {len(df.columns)}個")
    if isinstance(df.columns, pd.MultiIndex):
        errors.append("MultiIndexカラムが存在する")

    if errors:
        for e in errors:
            print(f"  [ERROR] {label}: {e}")
        return False
    return True


# ── メイン処理 ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("全通貨ペア IS/OOS データ取得スクリプト")
    print(f"対象: {len(PAIRS)} ペア × {len(PERIODS)} 期間 × {len(GRANULARITIES)} 時間軸")
    print(f"合計: 最大 {len(PAIRS) * len(PERIODS) * len(GRANULARITIES)} ファイル")
    print(f"保存先: {DATA_DIR}")
    print("=" * 60)

    summary = []
    skipped = 0
    fetched = 0
    failed = 0

    for pair_name, instrument in PAIRS.items():
        for period_name, (start, end) in PERIODS.items():
            for tf_name, granularity in GRANULARITIES.items():
                filename = f"{pair_name}_{period_name}_{tf_name}.csv"
                out_path = os.path.join(DATA_DIR, filename)

                # 既存ファイルがあればスキップ
                if os.path.exists(out_path):
                    existing_lines = sum(1 for _ in open(out_path)) - 1  # ヘッダー除く
                    print(f"[SKIP] {filename} ({existing_lines:,} 行)")
                    summary.append(f"[SKIP] {filename}: {existing_lines:,} 行")
                    skipped += 1
                    continue

                print(f"\n[FETCH] {filename}")
                print(f"  {instrument} | {granularity} | {start[:10]} → {end[:10]}")

                rows = fetch_candles(instrument, granularity, start, end)
                print()  # 改行

                if not rows:
                    print(f"  [WARN] データ取得失敗またはデータなし")
                    summary.append(f"[FAIL] {filename}: データなし")
                    failed += 1
                    continue

                # DataFrame作成
                df = pd.DataFrame(rows)

                # ① 重複除去・ソート
                before = len(df)
                df = df.drop_duplicates(subset=["timestamp"], keep="first")
                df = df.sort_values("timestamp").reset_index(drop=True)
                after = len(df)
                if before != after:
                    print(f"  重複 {before - after} 行を除去")

                # 品質チェック
                if not quality_check(df, filename):
                    print(f"  [ERROR] 品質チェック失敗、保存をスキップ")
                    summary.append(f"[FAIL] {filename}: 品質チェック失敗")
                    failed += 1
                    continue

                # 保存
                df.to_csv(out_path, index=False)
                ts_first = df["timestamp"].iloc[0]
                ts_last = df["timestamp"].iloc[-1]
                print(f"  [OK] {filename}: {len(df):,} 行, {ts_first} 〜 {ts_last}, カラム数={len(df.columns)}")
                summary.append(f"[OK] {filename}: {len(df):,} 行, {ts_first} 〜 {ts_last}")
                fetched += 1

                time.sleep(0.5)  # ペア間のクールダウン

    # サマリー出力
    print("\n" + "=" * 60)
    print("取得結果サマリー")
    print("=" * 60)
    for s in summary:
        print(s)
    print(f"\n取得: {fetched} | スキップ: {skipped} | 失敗: {failed} | 合計: {fetched + skipped + failed}")
    print("完了")


if __name__ == "__main__":
    main()
