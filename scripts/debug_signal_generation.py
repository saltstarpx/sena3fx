#!/usr/bin/env python3
"""
RUN-012 シグナル生成のデバッグスクリプト
各ステップでの判定結果を詳細にログ出力し、問題箇所を特定します。
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def main():
    # データ読み込み
    data_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m_2026_Jan.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print("=" * 80)
    print("RUN-012 シグナル生成デバッグ")
    print("=" * 80)
    print(f"データ期間: {df.index[0]} 〜 {df.index[-1]}")
    print(f"1分足バー数: {len(df)}\n")
    
    # 15分足・4時間足へのリサンプリング
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    print(f"15分足バー数: {len(df_15m)}")
    print(f"4時間足バー数: {len(df_4h)}\n")
    
    # === ステップ1: 4時間足の環境認識 ===
    print("=" * 80)
    print("【ステップ1】4時間足の環境認識")
    print("=" * 80)
    
    # ATR計算
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(14).mean()
    
    # トレンド判定
    df_4h['trend'] = 0
    df_4h.loc[df_4h['close'] > df_4h['close'].shift(3), 'trend'] = 1
    df_4h.loc[df_4h['close'] < df_4h['close'].shift(3), 'trend'] = -1
    
    print(f"トレンド分布:")
    print(df_4h['trend'].value_counts())
    print(f"\n直近5本の4時間足:")
    print(df_4h[['close', 'trend', 'atr']].tail(5))
    
    # === ステップ2: 15分足のパターン認識 ===
    print("\n" + "=" * 80)
    print("【ステップ2】15分足のパターン認識")
    print("=" * 80)
    
    # 下髭の長さを計算
    df_15m['lower_wick'] = df_15m[['open', 'close']].min(axis=1) - df_15m['low']
    df_15m['upper_wick'] = df_15m['high'] - df_15m[['open', 'close']].max(axis=1)
    df_15m['body_size'] = abs(df_15m['close'] - df_15m['open'])
    df_15m['bar_range'] = df_15m['high'] - df_15m['low']
    
    # 下髭が長い陽線を探す
    df_15m['long_lower_wick_bull'] = (df_15m['lower_wick'] > 0.02) & (df_15m['close'] > df_15m['open'])
    df_15m['long_upper_wick_bear'] = (df_15m['upper_wick'] > 0.02) & (df_15m['close'] < df_15m['open'])
    
    print(f"下髭が長い陽線: {df_15m['long_lower_wick_bull'].sum()} 件")
    print(f"上髭が長い陰線: {df_15m['long_upper_wick_bear'].sum()} 件")
    
    print(f"\n下髭の長さ統計:")
    print(df_15m['lower_wick'].describe())
    print(f"\n上髭の長さ統計:")
    print(df_15m['upper_wick'].describe())
    
    # === ステップ3: 1分足の「横軸」検出 ===
    print("\n" + "=" * 80)
    print("【ステップ3】1分足の「横軸」検出")
    print("=" * 80)
    
    # 直近10本の1分足でのボラティリティを計算
    df['volatility_10'] = df['high'].rolling(10).max() - df['low'].rolling(10).min()
    
    print(f"1分足ボラティリティ（直近10本）の統計:")
    print(df['volatility_10'].describe())
    print(f"\nボラティリティが5pips未満の期間: {(df['volatility_10'] < 0.05).sum()} 本 ({(df['volatility_10'] < 0.05).sum()/len(df)*100:.1f}%)")
    
    # === 総合判定 ===
    print("\n" + "=" * 80)
    print("【総合判定】")
    print("=" * 80)
    
    # トレンドが存在する期間
    h4_with_trend = df_4h[df_4h['trend'] != 0]
    print(f"トレンドが存在する4時間足: {len(h4_with_trend)} 本 ({len(h4_with_trend)/len(df_4h)*100:.1f}%)")
    
    # パターンが存在する期間
    m15_with_pattern = df_15m[df_15m['long_lower_wick_bull'] | df_15m['long_upper_wick_bear']]
    print(f"反転パターンが存在する15分足: {len(m15_with_pattern)} 本 ({len(m15_with_pattern)/len(df_15m)*100:.1f}%)")
    
    # 結論
    print("\n" + "=" * 80)
    print("【結論】")
    print("=" * 80)
    
    if len(h4_with_trend) == 0:
        print("⚠️ 4時間足でトレンドが検出されていません。トレンド判定ロジックを見直してください。")
    
    if len(m15_with_pattern) == 0:
        print("⚠️ 15分足で反転パターンが検出されていません。パターン認識ロジックを見直してください。")
    
    if len(h4_with_trend) > 0 and len(m15_with_pattern) > 0:
        print("✓ トレンドとパターンの両方が検出されています。1分足の「横軸」検出ロジックを確認してください。")

if __name__ == '__main__':
    main()
