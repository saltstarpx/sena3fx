#!/usr/bin/env python3.11
"""
scripts/generate_htf_from_1m.py
================================
1分足CSVから15分足・4時間足を自動生成する
データ品質チェック機能付き

【入力】 data/ohlc/{SYMBOL}_1m.csv
【出力】 data/ohlc/{SYMBOL}_15m.csv
         data/ohlc/{SYMBOL}_4h.csv
         results/data_quality_report.txt

【4時間足区切り】
  UTC 22:00/02:00/06:00/10:00/14:00/18:00 （既存ファイルと一致）
  pandas: resample('4h', offset='2h')

使い方:
    python3.11 scripts/generate_htf_from_1m.py [SYMBOL ...]
    python3.11 scripts/generate_htf_from_1m.py           # 全銘柄処理
    python3.11 scripts/generate_htf_from_1m.py USDJPY EURUSD
"""

from __future__ import annotations
import sys
import os
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── パス設定 ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ohlc"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_PATH = RESULTS_DIR / "data_quality_report.txt"

# ── 対象銘柄（data/ohlcに_1m.csvが存在するものを自動検出）─────
ALL_SYMBOLS = sorted(
    p.stem.replace("_1m", "")
    for p in DATA_DIR.glob("*_1m.csv")
)

# ── 異常値検出閾値 ────────────────────────────────────────────
PRICE_DEVIATION_THRESHOLD = 0.50   # 前後足と比べて50%以上乖離
MAX_GAP_MINUTES_FX = 120           # FX: 120分超の連続ギャップを警告
MAX_GAP_MINUTES_IDX = 120          # 指数も同様


# ─────────────────────────────────────────────────────────────
#  品質チェック
# ─────────────────────────────────────────────────────────────
def check_quality(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """
    データ品質チェック。問題レコードを除外した clean DataFrame と
    ログメッセージリストを返す。
    """
    logs: list[str] = []
    original_len = len(df)

    # ── 0. カラム確認 ──────────────────────────────────────────
    required = {"open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logs.append(f"[ERROR] 必要カラムが存在しない: {missing_cols}")
        return df, logs

    # ── 1. タイムスタンプチェック ──────────────────────────────
    # 重複タイムスタンプ
    dup_mask = df.index.duplicated(keep="first")
    n_dup = dup_mask.sum()
    if n_dup > 0:
        logs.append(f"[WARN]  重複タイムスタンプ: {n_dup}件 → 最初の行を採用")
        df = df[~dup_mask]

    # 逆順タイムスタンプ
    if not df.index.is_monotonic_increasing:
        logs.append("[WARN]  逆順タイムスタンプを検出 → ソートして修正")
        df = df.sort_index()

    # 間隔チェック（1分足: 想定60秒）
    diffs = df.index.to_series().diff().dropna()
    non_1m = diffs[diffs != pd.Timedelta("1min")]
    gaps = non_1m[non_1m > pd.Timedelta("1min")]
    big_gaps = gaps[gaps > pd.Timedelta(minutes=MAX_GAP_MINUTES_FX)]
    if len(big_gaps) > 0:
        logs.append(f"[INFO]  大きなギャップ({MAX_GAP_MINUTES_FX}分超): {len(big_gaps)}件")
        for ts, td in big_gaps.head(5).items():
            logs.append(f"          {ts}  gap={td}")

    # ── 2. 価格データチェック ──────────────────────────────────
    # NaN行
    nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logs.append(f"[WARN]  NaNを含む行: {n_nan}件 → 除外")
        df = df[~nan_mask]

    # ゼロ/マイナス価格
    zero_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_zero = zero_mask.sum()
    if n_zero > 0:
        logs.append(f"[WARN]  ゼロ/マイナス価格: {n_zero}件 → 除外")
        df = df[~zero_mask]

    # high < low
    hl_mask = df["high"] < df["low"]
    n_hl = hl_mask.sum()
    if n_hl > 0:
        logs.append(f"[WARN]  high < low: {n_hl}件 → 除外")
        df = df[~hl_mask]

    # open/close が [low, high] 範囲外
    oc_out = (
        (df["open"] < df["low"]) | (df["open"] > df["high"]) |
        (df["close"] < df["low"]) | (df["close"] > df["high"])
    )
    n_oc = oc_out.sum()
    if n_oc > 0:
        logs.append(f"[WARN]  open/closeがhigh-low範囲外: {n_oc}件")
        # 除外はせずログのみ（スプレッドバグ等で稀に発生）

    # 異常値（前後と比べて50%以上乖離）
    mid = (df["high"] + df["low"]) / 2
    mid_pct_change = mid.pct_change().abs()
    anomaly_mask = mid_pct_change > PRICE_DEVIATION_THRESHOLD
    n_anomaly = anomaly_mask.sum()
    if n_anomaly > 0:
        logs.append(f"[WARN]  異常価格（±50%超乖離）: {n_anomaly}件")
        for ts in df[anomaly_mask].head(3).index:
            logs.append(f"          {ts}  pct_change={mid_pct_change[ts]:.2%}")

    # ── 3. 重複行チェック ──────────────────────────────────────
    n_full_dup = df.duplicated().sum()
    if n_full_dup > 0:
        logs.append(f"[INFO]  完全重複行（タイムスタンプ除外後）: {n_full_dup}件")

    # ── 4. 欠損率 ──────────────────────────────────────────────
    # 実営業時間（月〜金 22:00〜22:00）のギャップ推定
    total_missing_pct = (original_len - len(df)) / original_len * 100 if original_len > 0 else 0
    logs.append(f"[INFO]  除外合計: {original_len - len(df)}行 / {original_len}行 "
                f"({total_missing_pct:.2f}%)")

    return df, logs


# ─────────────────────────────────────────────────────────────
#  resample
# ─────────────────────────────────────────────────────────────
def resample_to_htf(df: pd.DataFrame, rule: str, offset: str | None = None) -> pd.DataFrame:
    """1m DataFrame を rule 足にリサンプル。空足は除外。"""
    kwargs: dict = {}
    if offset:
        kwargs["offset"] = offset

    agg_dict = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    r = df.resample(rule, **kwargs).agg(agg_dict)
    r = r.dropna(subset=["open", "close"])   # ボリューム0の空足を除去
    r = r[r["volume"] > 0]
    return r


# ─────────────────────────────────────────────────────────────
#  1銘柄処理
# ─────────────────────────────────────────────────────────────
def process_symbol(symbol: str) -> dict:
    """
    1銘柄の1m→15m/4h変換を実行。
    結果サマリーを dict で返す。
    """
    result = {
        "symbol": symbol,
        "status": "OK",
        "logs": [],
        "rows_1m_raw": 0,
        "rows_1m_clean": 0,
        "rows_15m": 0,
        "rows_4h": 0,
    }

    input_path = DATA_DIR / f"{symbol}_1m.csv"
    if not input_path.exists():
        result["status"] = "SKIP"
        result["logs"].append(f"[SKIP] {input_path} が存在しない")
        return result

    # ── 読み込み ──────────────────────────────────────────────
    logs = result["logs"]
    logs.append(f"=== {symbol} ===")
    logs.append(f"入力: {input_path}")

    df = pd.read_csv(input_path, parse_dates=["timestamp"], index_col="timestamp")
    result["rows_1m_raw"] = len(df)

    # ── フォーマット確認 ──────────────────────────────────────
    logs.append(f"カラム: {list(df.columns)}")
    logs.append(f"タイムゾーン: {'UTC表記なし（naive）' if df.index.tz is None else str(df.index.tz)}")
    logs.append(f"最初の行: {df.index[0]}  {df.iloc[0].to_dict()}")
    logs.append(f"最後の行: {df.index[-1]}  {df.iloc[-1].to_dict()}")
    logs.append(f"行数: {len(df):,}")

    # ── 4時間足区切り確認 ─────────────────────────────────────
    sample_00 = df.resample("4h", origin="start_day").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().head(3)
    sample_02 = df.resample("4h", offset="2h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().head(3)
    logs.append("4時間足区切り候補:")
    logs.append("  [UTC 00:00基準] 00/04/08/12/16/20 →")
    for ts in sample_00.index:
        logs.append(f"    {ts}")
    logs.append("  [UTC 02:00基準] 22/02/06/10/14/18 →")
    for ts in sample_02.index:
        logs.append(f"    {ts}")
    logs.append("  採用: UTC 22:00基準 (offset='2h') ← 既存ファイルと一致")

    # ── 品質チェック ──────────────────────────────────────────
    df_clean, qc_logs = check_quality(df, symbol)
    logs.extend(qc_logs)
    result["rows_1m_clean"] = len(df_clean)

    if len(df_clean) == 0:
        result["status"] = "ERROR"
        logs.append("[ERROR] クリーン後にデータが0件")
        return result

    # ── resample 実行 ─────────────────────────────────────────
    # 15分足: origin='start_day'（00:00/15/30/45）
    df_15m = resample_to_htf(df_clean, "15min", offset=None)
    result["rows_15m"] = len(df_15m)

    # 4時間足: offset='2h'（22:00/02:00/06:00/10:00/14:00/18:00）
    df_4h = resample_to_htf(df_clean, "4h", offset="2h")
    result["rows_4h"] = len(df_4h)

    # ── 保存 ─────────────────────────────────────────────────
    out_15m = DATA_DIR / f"{symbol}_15m.csv"
    out_4h = DATA_DIR / f"{symbol}_4h.csv"
    df_15m.to_csv(out_15m)
    df_4h.to_csv(out_4h)
    logs.append(f"出力: {out_15m.name}  {len(df_15m):,}行")
    logs.append(f"出力: {out_4h.name}   {len(df_4h):,}行")

    return result


# ─────────────────────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────────────────────
def main():
    # コマンドライン引数で銘柄指定可能
    if len(sys.argv) > 1:
        symbols = [s.upper() for s in sys.argv[1:]]
    else:
        symbols = ALL_SYMBOLS

    if not symbols:
        print("data/ohlcに*_1m.csvが見つかりません")
        print(f"対象ディレクトリ: {DATA_DIR}")
        sys.exit(1)

    print(f"対象銘柄: {symbols}")
    print(f"出力レポート: {REPORT_PATH}")
    print()

    all_results = []
    report_lines: list[str] = [
        "=" * 70,
        f"データ品質チェック + 高時間足生成レポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    for sym in symbols:
        print(f"処理中: {sym} ... ", end="", flush=True)
        result = process_symbol(sym)
        all_results.append(result)
        print(f"{result['status']}  "
              f"1m={result['rows_1m_clean']:,}行, "
              f"15m={result['rows_15m']:,}行, "
              f"4h={result['rows_4h']:,}行")
        report_lines.extend(result["logs"])
        report_lines.append("")

    # ── サマリーテーブル ──────────────────────────────────────
    report_lines.append("=" * 70)
    report_lines.append("サマリー")
    report_lines.append("=" * 70)
    report_lines.append(
        f"{'銘柄':<10} {'状態':<6} {'1m(raw)':>10} {'1m(clean)':>10} "
        f"{'15m':>8} {'4h':>6}"
    )
    report_lines.append("-" * 60)
    for r in all_results:
        report_lines.append(
            f"{r['symbol']:<10} {r['status']:<6} {r['rows_1m_raw']:>10,} "
            f"{r['rows_1m_clean']:>10,} {r['rows_15m']:>8,} {r['rows_4h']:>6,}"
        )

    # ── レポート書き出し ──────────────────────────────────────
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print()
    print(f"レポート保存: {REPORT_PATH}")

    # コンソールにもサマリー表示
    print()
    print("\n".join(report_lines[-len(all_results)-5:]))


if __name__ == "__main__":
    main()
