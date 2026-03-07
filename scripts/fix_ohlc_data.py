#!/usr/bin/env python3
"""
ローソク足データ修正スクリプト
1. 異常ファイル（27カラム）を削除
2. タイムスタンプ重複を修正（月末重複行を削除）
3. 旧版・重複ファイルをarchiveへ移動
"""
import pandas as pd
import shutil
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
ARCHIVE_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("ローソク足データ修正")
print("=" * 60)

# =====================================================
# 1. 異常ファイル（27カラム）を削除
# =====================================================
print("\n[1] 異常ファイル（27カラム）を削除")

bad_files = ["USDJPY_15m.csv", "USDJPY_4h.csv"]
for fname in bad_files:
    fpath = DATA_DIR / fname
    if fpath.exists():
        # archiveへ移動（念のため保管）
        dest = ARCHIVE_DIR / fname
        shutil.move(str(fpath), str(dest))
        print(f"  移動: {fname} → data/archive/{fname}")

# =====================================================
# 2. タイムスタンプ重複を修正
# =====================================================
print("\n[2] タイムスタンプ重複を修正")

def fix_duplicate_timestamps(fname):
    """重複タイムスタンプを修正（月末重複は後者を削除）"""
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"  スキップ（ファイルなし）: {fname}")
        return
    
    df = pd.read_csv(fpath)
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    
    before = len(df)
    
    # 重複タイムスタンプを確認
    dup_mask = df[ts_col].duplicated(keep=False)
    dup_rows = df[dup_mask]
    
    if len(dup_rows) == 0:
        print(f"  OK（重複なし）: {fname}")
        return
    
    print(f"  修正前: {fname} - {before}行, 重複={df[ts_col].duplicated().sum()}件")
    print(f"  重複タイムスタンプ例:")
    for ts in df[ts_col][df[ts_col].duplicated()].head(5):
        rows = df[df[ts_col] == ts]
        print(f"    {ts}: {len(rows)}行")
        print(f"      {rows[['open','high','low','close']].to_string(index=False)}")
    
    # 重複を削除（最初の行を保持）
    # 月末の重複は、OANDAのAPIが月末最終足と翌月最初足を二重に返すバグ
    # → keep='first' で最初の行を保持
    df_fixed = df.drop_duplicates(subset=[ts_col], keep='first').reset_index(drop=True)
    after = len(df_fixed)
    
    # タイムスタンプをソート
    df_fixed = df_fixed.sort_values(ts_col).reset_index(drop=True)
    
    # 保存
    df_fixed.to_csv(fpath, index=False)
    print(f"  修正後: {fname} - {after}行 (削除: {before - after}行)")

fix_files = [
    "usdjpy_is_4h.csv",
    "usdjpy_oos_4h.csv",
    "eurjpy_4h.csv",
    "gbpjpy_4h.csv",
]

for fname in fix_files:
    fix_duplicate_timestamps(fname)

# =====================================================
# 3. 旧版・重複ファイルをarchiveへ移動
# =====================================================
print("\n[3] 旧版・重複ファイルをarchiveへ移動")

# 移動対象: is/oos分割版が正規版なので、旧版の統合ファイルをarchiveへ
# ただし、usdjpy_1m.csv, usdjpy_15m.csv, usdjpy_4h.csv, usdjpy_1h.csv は
# IS期間（2024/7〜2025/2）のデータとして有効なので保持
# 完全に重複しているものだけarchiveへ

archive_targets = [
    # 完全重複（usdjpy_15m.csvと同一内容）
    "usdjpy_15m_fixed.csv",   # usdjpy_15m.csvと同一
    "usdjpy_15m_old.csv",     # 旧形式（tick_count付き）
    # USDJPY_1m.csvはusdjpy_1m.csvと同一
    "USDJPY_1m.csv",
]

for fname in archive_targets:
    fpath = DATA_DIR / fname
    if fpath.exists():
        dest = ARCHIVE_DIR / fname
        shutil.move(str(fpath), str(dest))
        print(f"  移動: {fname} → data/archive/{fname}")

print("\n[完了] データ修正が完了しました")
print(f"\narchiveディレクトリ: {ARCHIVE_DIR}")
print("archiveの内容:")
for f in sorted(ARCHIVE_DIR.iterdir()):
    print(f"  {f.name}")
