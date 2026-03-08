"""
ohlcフォルダのデータをIS/OOS形式に変換してdata/直下に統合するスクリプト

対象:
- FX銘柄（1H足あり）: USDJPY, EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD
  → 1H足からリサンプリングで15分足・4時間足を生成
  → 1分足は既存のis/oosファイルがあれば使用、なければスキップ
- 指数（4H足あり）: NAS100
  → 4H足のみ（1分足・15分足なし）
- XAUUSD: 既存の2025_15m, 2025_4hを使用

IS期間: 2024-07-01 〜 2025-02-28
OOS期間: 2025-03-03 〜 2026-02-27
"""

import pandas as pd
import numpy as np
import os
import sys

IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

DATA_DIR = "/home/ubuntu/sena3fx/data"
OHLC_DIR = "/home/ubuntu/sena3fx/data/ohlc"

def load_ohlc(path):
    """ohlcフォルダのCSVを読み込む（datetime列をインデックスに）"""
    df = pd.read_csv(path)
    # datetime列を特定
    dt_col = None
    for c in ["datetime", "timestamp", "time", "date", "Date", "Time"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.rename(columns={dt_col: "timestamp"})
    df = df.set_index("timestamp")
    # カラム名を小文字に
    df.columns = [c.lower() for c in df.columns]
    # 必要カラムのみ
    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            if c == "volume":
                df["volume"] = 0
    df = df[[c for c in needed if c in df.columns]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def resample_to(df, rule):
    """DataFrameを指定時間足にリサンプリング"""
    result = pd.DataFrame({
        "open":   df["open"].resample(rule).first(),
        "high":   df["high"].resample(rule).max(),
        "low":    df["low"].resample(rule).min(),
        "close":  df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum() if "volume" in df.columns else 0,
    }).dropna(subset=["open", "close"])
    return result

def save_is_oos(df, symbol, tf):
    """IS/OOSに分割してdata/直下に保存"""
    is_path  = os.path.join(DATA_DIR, f"{symbol}_is_{tf}.csv")
    oos_path = os.path.join(DATA_DIR, f"{symbol}_oos_{tf}.csv")

    # 既存ファイルがある場合はスキップ
    if os.path.exists(is_path) and os.path.exists(oos_path):
        is_rows  = len(pd.read_csv(is_path))
        oos_rows = len(pd.read_csv(oos_path))
        print(f"  [SKIP] {symbol}_{tf}: 既存ファイルあり (IS={is_rows}行, OOS={oos_rows}行)")
        return False

    df_is  = df[(df.index >= IS_START)  & (df.index <= IS_END)]
    df_oos = df[(df.index >= OOS_START) & (df.index <= OOS_END)]

    if len(df_is) == 0 and len(df_oos) == 0:
        print(f"  [SKIP] {symbol}_{tf}: 対象期間のデータなし")
        return False

    if len(df_is) > 0:
        df_is.reset_index().to_csv(is_path, index=False)
        print(f"  [OK]   {symbol}_is_{tf}.csv  → {len(df_is)}行 ({df_is.index[0].date()} 〜 {df_is.index[-1].date()})")
    else:
        print(f"  [WARN] {symbol}_is_{tf}: IS期間データなし")

    if len(df_oos) > 0:
        df_oos.reset_index().to_csv(oos_path, index=False)
        print(f"  [OK]   {symbol}_oos_{tf}.csv → {len(df_oos)}行 ({df_oos.index[0].date()} 〜 {df_oos.index[-1].date()})")
    else:
        print(f"  [WARN] {symbol}_oos_{tf}: OOS期間データなし")

    return True

# ============================================================
# FX銘柄: 1H足 → 15分足(×4リサンプル不可のためスキップ)・4H足を生成
# 注: 1H→15分は逆方向（ダウンサンプル）なので不可。
#     1H足はそのまま1H足として保存。4H足はリサンプリングで生成。
# ============================================================

FX_SYMBOLS = {
    "usdjpy": "USDJPY_1h.csv",
    "eurusd": "EURUSD_1h.csv",
    "gbpusd": "GBPUSD_1h.csv",
    "audusd": "AUDUSD_1h.csv",
    "usdcad": "USDCAD_1h.csv",
    "usdchf": "USDCHF_1h.csv",
    "nzdusd": "NZDUSD_1h.csv",
    "eurjpy": "EURJPY_1h.csv",
    "gbpjpy": "GBPJPY_1h.csv",
    "eurgbp": "EURGBP_1h.csv",
}

print("=" * 60)
print("ohlcフォルダ → data/ 統合処理")
print(f"IS: {IS_START} 〜 {IS_END}")
print(f"OOS: {OOS_START} 〜 {OOS_END}")
print("=" * 60)

print("\n【FX銘柄: 1H足 + 4H足（1Hからリサンプリング）】")
for symbol, fname in FX_SYMBOLS.items():
    fpath = os.path.join(OHLC_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [MISS] {fname} が見つかりません")
        continue
    print(f"\n--- {symbol.upper()} ---")
    df_1h = load_ohlc(fpath)

    # 1H足を保存
    save_is_oos(df_1h, symbol, "1h")

    # 4H足: 1Hからリサンプリング
    df_4h = resample_to(df_1h, "4h")
    save_is_oos(df_4h, symbol, "4h")

# ============================================================
# NAS100: 4H足のみ
# ============================================================
print("\n【NAS100: 4H足】")
nas_path = os.path.join(OHLC_DIR, "NAS100_4h.csv")
if os.path.exists(nas_path):
    df_nas = load_ohlc(nas_path)
    save_is_oos(df_nas, "nas100", "4h")
else:
    print("  [MISS] NAS100_4h.csv が見つかりません")

# ============================================================
# XAUUSD: 2025_15m, 2025_4h を使用
# ============================================================
print("\n【XAUUSD: 15分足・4H足（2025年データ）】")
xau_15m_path = os.path.join(OHLC_DIR, "XAUUSD_2025_15m.csv")
xau_4h_path  = os.path.join(OHLC_DIR, "XAUUSD_2025_4h.csv")
xau_1h_path  = os.path.join(OHLC_DIR, "XAUUSD_1h.csv")

if os.path.exists(xau_15m_path):
    df_xau_15m = load_ohlc(xau_15m_path)
    save_is_oos(df_xau_15m, "xauusd", "15m")

if os.path.exists(xau_4h_path):
    df_xau_4h = load_ohlc(xau_4h_path)
    save_is_oos(df_xau_4h, "xauusd", "4h")

if os.path.exists(xau_1h_path):
    df_xau_1h = load_ohlc(xau_1h_path)
    save_is_oos(df_xau_1h, "xauusd", "1h")

# ============================================================
# XAGUSD: 2025_15m, 2025_4h を使用
# ============================================================
print("\n【XAGUSD: 15分足・4H足（2025年データ）】")
xag_15m_path = os.path.join(OHLC_DIR, "XAGUSD_2025_15m.csv")
xag_4h_path  = os.path.join(OHLC_DIR, "XAGUSD_2025_4h.csv")
xag_1h_path  = os.path.join(OHLC_DIR, "XAGUSD_1h.csv")

if os.path.exists(xag_15m_path):
    df_xag_15m = load_ohlc(xag_15m_path)
    save_is_oos(df_xag_15m, "xagusd", "15m")

if os.path.exists(xag_4h_path):
    df_xag_4h = load_ohlc(xag_4h_path)
    save_is_oos(df_xag_4h, "xagusd", "4h")

if os.path.exists(xag_1h_path):
    df_xag_1h = load_ohlc(xag_1h_path)
    save_is_oos(df_xag_1h, "xagusd", "1h")

# ============================================================
# 最終: 全データ状況サマリー
# ============================================================
print("\n" + "=" * 60)
print("【全銘柄データ状況サマリー】")
print("=" * 60)

all_symbols = [
    "usdjpy", "eurusd", "gbpusd", "audusd", "usdcad", "usdchf", "nzdusd",
    "eurjpy", "gbpjpy", "eurgbp",
    "nas100", "spx500", "us30",
    "xauusd", "xagusd",
]
timeframes = ["1m", "15m", "1h", "4h"]

print(f"{'銘柄':<10} {'1m_IS':>8} {'1m_OOS':>8} {'15m_IS':>8} {'15m_OOS':>8} {'1h_IS':>8} {'1h_OOS':>8} {'4h_IS':>8} {'4h_OOS':>8}")
print("-" * 90)

for sym in all_symbols:
    row = f"{sym:<10}"
    for tf in timeframes:
        for period in ["is", "oos"]:
            fpath = os.path.join(DATA_DIR, f"{sym}_{period}_{tf}.csv")
            if os.path.exists(fpath):
                rows = len(pd.read_csv(fpath))
                row += f" {rows:>8,}"
            else:
                row += f" {'---':>8}"
    print(row)

print("\n完了。")
