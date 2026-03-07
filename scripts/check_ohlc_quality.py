#!/usr/bin/env python3
"""
ローソク足データ品質チェックスクリプト
- カラム重複チェック
- タイムスタンプ重複チェック
- 期間の重複チェック（ファイル間）
- OHLC整合性チェック（high >= open/close >= low）
- 欠損値チェック
- 異常値チェック（スパイク）
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# チェック対象ファイル（正規版のみ）
TARGET_FILES = {
    "USDJPY": {
        "1m":  ["usdjpy_1m.csv", "USDJPY_1m.csv", "usdjpy_is_1m.csv", "usdjpy_oos_1m.csv"],
        "15m": ["usdjpy_15m.csv", "usdjpy_15m_fixed.csv", "usdjpy_15m_old.csv",
                "USDJPY_15m.csv", "usdjpy_is_15m.csv", "usdjpy_oos_15m.csv"],
        "1h":  ["usdjpy_1h.csv", "usdjpy_is_1h.csv", "usdjpy_oos_1h.csv"],
        "4h":  ["usdjpy_4h.csv", "USDJPY_4h.csv", "usdjpy_is_4h.csv", "usdjpy_oos_4h.csv"],
    },
    "EURJPY": {
        "1m":  ["eurjpy_1m.csv"],
        "15m": ["eurjpy_15m.csv"],
        "4h":  ["eurjpy_4h.csv"],
    },
    "GBPJPY": {
        "1m":  ["gbpjpy_1m.csv"],
        "15m": ["gbpjpy_15m.csv"],
        "4h":  ["gbpjpy_4h.csv"],
    },
}

ISSUES = []

def check_file(filepath, label):
    """単一ファイルの品質チェック"""
    issues = []
    
    if not filepath.exists():
        return issues
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        issues.append(f"[ERROR] {label}: 読み込み失敗 - {e}")
        return issues
    
    cols = df.columns.tolist()
    
    # 1. カラム数チェック（正常は6〜7: timestamp,open,high,low,close,volume[,tick_count]）
    if len(cols) > 8:
        issues.append(f"[CRITICAL] {label}: カラム数異常 ({len(cols)}個) - カラム: {cols[:10]}...")
        return issues  # 以降のチェックは意味がないのでスキップ
    
    # 2. 必須カラム確認
    required = ['open', 'high', 'low', 'close']
    ts_col = cols[0]
    missing_cols = [c for c in required if c not in cols]
    if missing_cols:
        issues.append(f"[ERROR] {label}: 必須カラム欠損 {missing_cols}")
        return issues
    
    # 3. タイムスタンプをパース
    try:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    except Exception as e:
        issues.append(f"[ERROR] {label}: タイムスタンプパース失敗 - {e}")
        return issues
    
    n = len(df)
    period_start = df[ts_col].min()
    period_end = df[ts_col].max()
    
    # 4. タイムスタンプ重複チェック
    dup_ts = df[ts_col].duplicated().sum()
    if dup_ts > 0:
        dup_examples = df[df[ts_col].duplicated(keep=False)][ts_col].head(3).tolist()
        issues.append(f"[ERROR] {label}: タイムスタンプ重複 {dup_ts}件 - 例: {dup_examples}")
    
    # 5. OHLC整合性チェック
    invalid_hl = (df['high'] < df['low']).sum()
    if invalid_hl > 0:
        issues.append(f"[ERROR] {label}: high < low の異常ローソク足 {invalid_hl}件")
    
    invalid_oh = (df['open'] > df['high']).sum()
    if invalid_oh > 0:
        issues.append(f"[ERROR] {label}: open > high の異常ローソク足 {invalid_oh}件")
    
    invalid_ol = (df['open'] < df['low']).sum()
    if invalid_ol > 0:
        issues.append(f"[ERROR] {label}: open < low の異常ローソク足 {invalid_ol}件")
    
    invalid_ch = (df['close'] > df['high']).sum()
    if invalid_ch > 0:
        issues.append(f"[ERROR] {label}: close > high の異常ローソク足 {invalid_ch}件")
    
    invalid_cl = (df['close'] < df['low']).sum()
    if invalid_cl > 0:
        issues.append(f"[ERROR] {label}: close < low の異常ローソク足 {invalid_cl}件")
    
    # 6. 欠損値チェック
    null_counts = df[required].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        issues.append(f"[WARN] {label}: 欠損値 {total_nulls}件 - {null_counts.to_dict()}")
    
    # 7. 異常スパイクチェック（前後比で10%以上の変動）
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    close_pct = df_sorted['close'].pct_change().abs()
    spikes = (close_pct > 0.10).sum()
    if spikes > 0:
        spike_rows = df_sorted[close_pct > 0.10][[ts_col, 'close']].head(3)
        issues.append(f"[WARN] {label}: 10%超スパイク {spikes}件 - 例: {spike_rows.values.tolist()}")
    
    # 8. ゼロ値チェック
    zero_close = (df['close'] == 0).sum()
    if zero_close > 0:
        issues.append(f"[ERROR] {label}: close=0 の異常値 {zero_close}件")
    
    # サマリー出力（問題なし）
    if not issues:
        print(f"  [OK] {label}: {n}行, {period_start.date()} 〜 {period_end.date()}, カラム={len(cols)}")
    else:
        print(f"  [NG] {label}: {n}行, {period_start.date()} 〜 {period_end.date()}, カラム={len(cols)}")
        for iss in issues:
            print(f"       {iss}")
    
    return issues


def check_period_overlap(files_info):
    """ファイル間の期間重複チェック"""
    overlap_issues = []
    periods = []
    
    for fname, start, end in files_info:
        for fname2, start2, end2 in periods:
            # 重複判定
            if start <= end2 and end >= start2:
                overlap_issues.append(
                    f"[WARN] 期間重複: {fname} ({start.date()}〜{end.date()}) と "
                    f"{fname2} ({start2.date()}〜{end2.date()})"
                )
        periods.append((fname, start, end))
    
    return overlap_issues


def main():
    print("=" * 70)
    print("ローソク足データ品質チェック")
    print("=" * 70)
    
    all_issues = []
    
    for pair, timeframes in TARGET_FILES.items():
        print(f"\n{'='*50}")
        print(f"通貨ペア: {pair}")
        print(f"{'='*50}")
        
        for tf, filenames in timeframes.items():
            print(f"\n  [{tf}]")
            files_info = []
            
            for fname in filenames:
                fpath = DATA_DIR / fname
                if not fpath.exists():
                    continue
                
                issues = check_file(fpath, fname)
                all_issues.extend(issues)
                
                # 期間情報を収集（正常ファイルのみ）
                if not any("CRITICAL" in i or "ERROR" in i for i in issues):
                    try:
                        df = pd.read_csv(fpath, usecols=[0])
                        ts_col = df.columns[0]
                        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                        files_info.append((fname, df[ts_col].min(), df[ts_col].max()))
                    except:
                        pass
            
            # 期間重複チェック
            if len(files_info) > 1:
                overlap_issues = check_period_overlap(files_info)
                if overlap_issues:
                    for oi in overlap_issues:
                        print(f"  {oi}")
                    all_issues.extend(overlap_issues)
    
    print("\n" + "=" * 70)
    print("チェック結果サマリー")
    print("=" * 70)
    
    critical = [i for i in all_issues if "CRITICAL" in i]
    errors = [i for i in all_issues if "ERROR" in i]
    warns = [i for i in all_issues if "WARN" in i]
    
    print(f"CRITICAL: {len(critical)}件")
    print(f"ERROR:    {len(errors)}件")
    print(f"WARN:     {len(warns)}件")
    
    if critical:
        print("\n--- CRITICAL ---")
        for i in critical:
            print(f"  {i}")
    
    if errors:
        print("\n--- ERROR ---")
        for i in errors:
            print(f"  {i}")
    
    if warns:
        print("\n--- WARN ---")
        for i in warns:
            print(f"  {i}")
    
    return len(critical) + len(errors)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(0)  # エラーがあっても0で終了（報告のみ）
