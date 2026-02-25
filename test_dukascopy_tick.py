"""
Dukascopyティックデータ取得テスト
XAUUSDとXAGUSDのティックデータが取得できるか検証する
"""
import sys
import os
from datetime import datetime, timedelta

# まず利用可能なインストゥルメントを確認
print("=== Dukascopy Python ティックデータ取得テスト ===")
print()

import dukascopy_python
from dukascopy_python import instruments

# 利用可能なインストゥルメント一覧からXAU/XAGを探す
print("--- 利用可能なコモディティ/金属関連インストゥルメント ---")
for name in dir(instruments):
    if 'XAU' in name or 'XAG' in name or 'GOLD' in name or 'SILVER' in name:
        print(f"  {name} = {getattr(instruments, name)}")

print()

# 全インストゥルメントカテゴリを確認
print("--- インストゥルメントカテゴリ ---")
categories = set()
for name in dir(instruments):
    if name.startswith('INSTRUMENT_'):
        parts = name.split('_')
        if len(parts) >= 2:
            categories.add(parts[1])
for cat in sorted(categories):
    print(f"  {cat}")

print()

# XAUUSDのティックデータを短期間で取得テスト
print("--- XAUUSDティックデータ取得テスト（直近1日分） ---")
try:
    # 直近のデータを少量取得
    end = datetime(2025, 1, 2)
    start = datetime(2025, 1, 1)
    
    # インストゥルメント名を探す
    xau_instrument = None
    for name in dir(instruments):
        if 'XAU' in name and 'USD' in name:
            xau_instrument = getattr(instruments, name)
            print(f"  使用インストゥルメント: {name} = {xau_instrument}")
            break
    
    if xau_instrument is None:
        print("  XAUUSDインストゥルメントが見つかりません")
        # 全インストゥルメントを表示
        print("  全インストゥルメント:")
        for name in sorted(dir(instruments)):
            if name.startswith('INSTRUMENT_'):
                print(f"    {name}")
        sys.exit(1)
    
    print(f"  期間: {start} ~ {end}")
    print(f"  取得中...")
    
    df = dukascopy_python.fetch(
        xau_instrument,
        dukascopy_python.INTERVAL_TICK,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
        debug=False,
    )
    
    print(f"  取得完了!")
    print(f"  行数: {len(df)}")
    print(f"  列: {list(df.columns)}")
    print(f"  先頭5行:")
    print(df.head())
    print(f"  末尾5行:")
    print(df.tail())
    print(f"  データ型:")
    print(df.dtypes)
    
    # 保存テスト
    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'XAUUSD_tick_sample.csv')
    df.to_csv(outpath)
    print(f"  保存先: {outpath}")
    print(f"  ファイルサイズ: {os.path.getsize(outpath)/1024:.1f} KB")
    
except Exception as e:
    print(f"  エラー: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== テスト完了 ===")
