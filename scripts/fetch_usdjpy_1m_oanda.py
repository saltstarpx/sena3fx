"""
OANDA APIからUSDJPY 1分足OHLCデータを取得してCSVに保存する。

対象:
  IS  (In-Sample):  2024-07-01 〜 2025-02-28 UTC → usdjpy_is_1m.csv
  OOS (Out-of-Sample): 2025-03-03 〜 2026-02-27 UTC → usdjpy_oos_1m.csv

品質チェック:
  - タイムスタンプ重複除去
  - OHLC整合性チェック
  - ゼロ値・欠損値チェック
  - カラム数チェック（MultiIndexバグ防止）
  - 未確定足の除外
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
INSTRUMENT  = "USD_JPY"
GRANULARITY = "M1"
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

PERIODS = [
    {
        "label": "IS",
        "start": datetime(2024, 7, 1, tzinfo=timezone.utc),
        "end":   datetime(2025, 3, 1, tzinfo=timezone.utc),   # 2025-02-28 末まで
        "out":   os.path.join(DATA_DIR, "usdjpy_is_1m.csv"),
    },
    {
        "label": "OOS",
        "start": datetime(2025, 3, 3, tzinfo=timezone.utc),
        "end":   datetime(2026, 2, 28, tzinfo=timezone.utc),
        "out":   os.path.join(DATA_DIR, "usdjpy_oos_1m.csv"),
    },
]

CHUNK = 5000          # OANDA API 1リクエスト上限
SLEEP = 0.3           # レート制限対策（秒）


# ── データ取得 ─────────────────────────────────────────
def fetch_chunk(from_dt: datetime, to_dt: datetime) -> list:
    """1リクエスト分（最大5000本）のローソク足を取得して返す"""
    url = f"{BASE_URL}/v3/instruments/{INSTRUMENT}/candles"
    params = {
        "granularity": GRANULARITY,
        "price": "M",
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": CHUNK,
    }
    # from と to を同時指定するとcountは無視されるため、
    # 1チャンク分の to を計算して渡す
    params_send = {
        "granularity": GRANULARITY,
        "price": "M",
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": CHUNK,
    }
    resp = requests.get(url, headers=HEADERS, params=params_send, timeout=30)
    if resp.status_code != 200:
        print(f"  [ERROR] HTTP {resp.status_code}: {resp.text[:200]}")
        return []
    return resp.json().get("candles", [])


def fetch_period(start: datetime, end: datetime, label: str) -> pd.DataFrame:
    """期間全体を分割取得して結合する"""
    all_rows = []
    cursor = start
    total = 0

    print(f"\n[{label}] 取得開始: {start.date()} 〜 {end.date()}")

    while cursor < end:
        candles = fetch_chunk(cursor, end)
        if not candles:
            print(f"  [{label}] データなし or エラー（cursor={cursor}）、スキップ")
            cursor += timedelta(minutes=CHUNK)
            continue

        for c in candles:
            # 未確定足は除外
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            ts_str = c["time"][:19]  # "2024-07-01T00:00:00"
            all_rows.append({
                "timestamp": ts_str,
                "open":   float(mid.get("o", 0)),
                "high":   float(mid.get("h", 0)),
                "low":    float(mid.get("l", 0)),
                "close":  float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        # 次のカーソルは最後に取得した足の1分後
        last_ts = candles[-1]["time"][:19]
        last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        cursor = last_dt + timedelta(minutes=1)

        total += len(candles)
        print(f"  [{label}] 累計: {total:,}本 | 最新: {last_ts}", flush=True)

        # 取得本数が CHUNK 未満 → 期間終端に到達
        if len(candles) < CHUNK:
            break

        time.sleep(SLEEP)

    if not all_rows:
        print(f"  [{label}] データ取得失敗")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


# ── 品質チェック ───────────────────────────────────────
def quality_check(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """品質チェックと修正を行い、クリーンなDataFrameを返す"""
    print(f"\n[{label}] 品質チェック開始 ({len(df):,}行)")

    # ① タイムスタンプをUTC付きに変換
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ② タイムスタンプ重複除去（月またぎ取得で月末に重複が発生する既知バグ）
    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        print(f"  [修正] タイムスタンプ重複 {dup}件 → 除去")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # ③ 時系列ソート
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ④ OHLC整合性チェック
    err_hl = (df["high"] < df["low"]).sum()
    err_oh = (df["open"] > df["high"]).sum()
    err_ol = (df["open"] < df["low"]).sum()
    err_ch = (df["close"] > df["high"]).sum()
    err_cl = (df["close"] < df["low"]).sum()
    ohlc_errors = err_hl + err_oh + err_ol + err_ch + err_cl
    if ohlc_errors > 0:
        print(f"  [ERROR] OHLC整合性エラー: high<low={err_hl}, open>high={err_oh}, "
              f"open<low={err_ol}, close>high={err_ch}, close<low={err_cl}")
        sys.exit(1)
    else:
        print("  [OK] OHLC整合性: 問題なし")

    # ⑤ ゼロ値・欠損値チェック
    zero_close = (df["close"] == 0).sum()
    null_count = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if zero_close > 0 or null_count > 0:
        print(f"  [ERROR] ゼロ値={zero_close}件, 欠損値={null_count}件")
        sys.exit(1)
    else:
        print("  [OK] ゼロ値・欠損値: 問題なし")

    # ⑥ カラム数チェック（MultiIndexバグ防止）
    if isinstance(df.columns, pd.MultiIndex):
        print("  [ERROR] MultiIndexカラムが存在する（バグ）")
        sys.exit(1)
    if len(df.columns) > 7:
        print(f"  [ERROR] カラム数異常: {len(df.columns)}個 → {df.columns.tolist()}")
        sys.exit(1)
    print(f"  [OK] カラム数: {len(df.columns)}個 → {df.columns.tolist()}")

    print(f"  [OK] チェック完了: {len(df):,}行, "
          f"{df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    return df


# ── メイン ─────────────────────────────────────────────
def main():
    if not API_KEY:
        print("[ERROR] OANDA_API_KEY が設定されていません")
        sys.exit(1)

    print("=" * 60)
    print("USDJPY 1分足OHLCデータ取得スクリプト")
    print(f"環境: {ENVIRONMENT.upper()}")
    print("=" * 60)

    for period in PERIODS:
        label = period["label"]
        start = period["start"]
        end   = period["end"]
        out   = period["out"]

        # 取得
        df = fetch_period(start, end, label)
        if df.empty:
            print(f"[{label}] スキップ（データなし）")
            continue

        # 品質チェック
        df = quality_check(df, label)

        # 保存（index=False で timestamp を列として保存）
        df.to_csv(out, index=False)
        print(f"\n[{label}] 保存完了: {out}")
        print(f"  行数: {len(df):,}行")
        print(f"  期間: {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
        print(f"  カラム: {df.columns.tolist()}")

    print("\n" + "=" * 60)
    print("全期間の取得・保存完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
