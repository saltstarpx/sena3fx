#!/usr/bin/env python3
"""
通貨強弱エンジン + XAGクロスペア動的選択ロジック
==============================================
1. XAGUSDとFXペアからXAGクロスペア(XAGJPY, XAGEUR等)を合成
2. 主要FXペアから通貨強弱を計算
3. 「XAGロング時は最弱通貨を相手に、ショート時は最強通貨を相手に」する動的選択
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== STEP 1: XAGクロスペア合成 ==========

SYNTH_RULES = {
    'XAGJPY': ('USDJPY', 'multiply'),
    'XAGEUR': ('EURUSD', 'divide'),
    'XAGGBP': ('GBPUSD', 'divide'),
    'XAGAUD': ('AUDUSD', 'divide'),
    'XAGNZD': ('NZDUSD', 'divide'),
    'XAGCAD': ('USDCAD', 'multiply'),
    'XAGCHF': ('USDCHF', 'multiply'),
}

def synthesize_xag_pairs(timeframe='1d'):
    """XAGUSDとFXペアからXAGクロスペアを合成"""
    xag_file = os.path.join(DATA_DIR, f'XAGUSD_{timeframe}.csv')
    xag = pd.read_csv(xag_file, index_col='datetime', parse_dates=True)
    
    pairs = {'XAGUSD': xag}
    
    for pair_name, (fx_pair, op) in SYNTH_RULES.items():
        fx_file = os.path.join(DATA_DIR, f'{fx_pair}_{timeframe}.csv')
        if not os.path.exists(fx_file):
            continue
        fx = pd.read_csv(fx_file, index_col='datetime', parse_dates=True)
        
        # 共通日付で結合
        common_idx = xag.index.intersection(fx.index)
        x = xag.loc[common_idx]
        f = fx.loc[common_idx]
        
        if op == 'multiply':
            synth = pd.DataFrame({
                'Open': x['Open'] * f['Open'],
                'High': x['High'] * f['High'],
                'Low': x['Low'] * f['Low'],
                'Close': x['Close'] * f['Close'],
                'Volume': x['Volume'],
            }, index=common_idx)
        else:
            synth = pd.DataFrame({
                'Open': x['Open'] / f['Open'],
                'High': x['High'] / f['High'],
                'Low': x['Low'] / f['Low'],
                'Close': x['Close'] / f['Close'],
                'Volume': x['Volume'],
            }, index=common_idx)
        
        synth.index.name = 'datetime'
        out_file = os.path.join(DATA_DIR, f'{pair_name}_{timeframe}.csv')
        synth.to_csv(out_file)
        pairs[pair_name] = synth
        print(f"  Synthesized {pair_name}_{timeframe}: {len(synth)} bars")
    
    return pairs


# ========== STEP 2: 通貨強弱計算 ==========

# 各通貨が含まれるペアの定義（通貨が「前」ならそのまま、「後」なら反転）
CURRENCY_PAIRS = {
    'USD': [('USDJPY', 1), ('EURUSD', -1), ('GBPUSD', -1), ('AUDUSD', -1), 
            ('NZDUSD', -1), ('USDCAD', 1), ('USDCHF', 1)],
    'JPY': [('USDJPY', -1), ('EURJPY', -1), ('GBPJPY', -1)],
    'EUR': [('EURUSD', 1), ('EURJPY', 1), ('EURGBP', 1)],
    'GBP': [('GBPUSD', 1), ('GBPJPY', 1), ('EURGBP', -1)],
    'AUD': [('AUDUSD', 1)],
    'NZD': [('NZDUSD', 1)],
    'CAD': [('USDCAD', -1)],
    'CHF': [('USDCHF', -1)],
}

def calc_currency_strength(timeframe='1d', lookback=20):
    """
    各通貨の強弱を計算する。
    lookback期間の変化率を各通貨が含まれるペアで平均化。
    """
    # 全FXペアの終値変化率を計算
    fx_returns = {}
    needed_pairs = set()
    for currency, pairs in CURRENCY_PAIRS.items():
        for pair_name, _ in pairs:
            needed_pairs.add(pair_name)
    
    for pair_name in needed_pairs:
        f = os.path.join(DATA_DIR, f'{pair_name}_{timeframe}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, index_col='datetime', parse_dates=True)
        fx_returns[pair_name] = df['Close'].pct_change(lookback)
    
    # 共通インデックス
    all_idx = None
    for name, ret in fx_returns.items():
        if all_idx is None:
            all_idx = ret.dropna().index
        else:
            all_idx = all_idx.intersection(ret.dropna().index)
    
    # 各通貨の強弱を計算
    strength = pd.DataFrame(index=all_idx)
    for currency, pairs in CURRENCY_PAIRS.items():
        vals = []
        for pair_name, direction in pairs:
            if pair_name in fx_returns:
                vals.append(fx_returns[pair_name].loc[all_idx] * direction)
        if vals:
            strength[currency] = pd.concat(vals, axis=1).mean(axis=1) * 100
    
    return strength


# ========== STEP 3: 動的ペア選択ロジック ==========

CURRENCY_TO_PAIR = {
    'USD': 'XAGUSD',
    'JPY': 'XAGJPY',
    'EUR': 'XAGEUR',
    'GBP': 'XAGGBP',
    'AUD': 'XAGAUD',
    'NZD': 'XAGNZD',
    'CAD': 'XAGCAD',
    'CHF': 'XAGCHF',
}

def select_best_pair(strength_row, direction='long'):
    """
    通貨強弱に基づいて最適なXAGペアを選択する。
    
    direction='long':  XAGを買う → 最弱通貨を相手にする
    direction='short': XAGを売る → 最強通貨を相手にする
    """
    if direction == 'long':
        # 最弱通貨を見つける（XAGロングに最適）
        weakest = strength_row.idxmin()
        return CURRENCY_TO_PAIR.get(weakest, 'XAGUSD'), weakest, strength_row[weakest]
    else:
        # 最強通貨を見つける（XAGショートに最適）
        strongest = strength_row.idxmax()
        return CURRENCY_TO_PAIR.get(strongest, 'XAGUSD'), strongest, strength_row[strongest]


def dynamic_pair_selection_backtest(timeframe='1d', lookback=20):
    """
    通貨強弱ベースの動的ペア選択バックテスト。
    各時点で最適なペアを選択し、そのペアでのリターンを計算。
    """
    print(f"\n{'='*60}")
    print(f"動的ペア選択バックテスト (TF={timeframe}, lookback={lookback})")
    print(f"{'='*60}")
    
    # XAGクロスペアを合成
    xag_pairs = synthesize_xag_pairs(timeframe)
    
    # 通貨強弱を計算
    strength = calc_currency_strength(timeframe, lookback)
    print(f"\n通貨強弱データ: {len(strength)} bars")
    print(f"期間: {strength.index[0]} ~ {strength.index[-1]}")
    
    # 各時点での最適ペア選択と結果を記録
    records = []
    
    for i in range(len(strength) - 1):
        date = strength.index[i]
        next_date = strength.index[i + 1]
        row = strength.iloc[i]
        
        # ロング方向: 最弱通貨を相手に
        long_pair, weak_ccy, weak_str = select_best_pair(row, 'long')
        
        # ショート方向: 最強通貨を相手に
        short_pair, strong_ccy, strong_str = select_best_pair(row, 'short')
        
        # 選択されたペアの翌日リターンを計算
        long_ret = 0
        short_ret = 0
        
        if long_pair in xag_pairs:
            pair_df = xag_pairs[long_pair]
            if date in pair_df.index and next_date in pair_df.index:
                long_ret = (pair_df.loc[next_date, 'Close'] / pair_df.loc[date, 'Close'] - 1) * 100
        
        if short_pair in xag_pairs:
            pair_df = xag_pairs[short_pair]
            if date in pair_df.index and next_date in pair_df.index:
                short_ret = -(pair_df.loc[next_date, 'Close'] / pair_df.loc[date, 'Close'] - 1) * 100
        
        # XAGUSD固定の場合のリターン（ベンチマーク）
        xagusd_ret = 0
        if date in xag_pairs['XAGUSD'].index and next_date in xag_pairs['XAGUSD'].index:
            xagusd_ret = (xag_pairs['XAGUSD'].loc[next_date, 'Close'] / xag_pairs['XAGUSD'].loc[date, 'Close'] - 1) * 100
        
        records.append({
            'date': date,
            'long_pair': long_pair,
            'weak_currency': weak_ccy,
            'weak_strength': round(weak_str, 4),
            'long_return_pct': round(long_ret, 4),
            'short_pair': short_pair,
            'strong_currency': strong_ccy,
            'strong_strength': round(strong_str, 4),
            'short_return_pct': round(short_ret, 4),
            'xagusd_return_pct': round(xagusd_ret, 4),
        })
    
    results_df = pd.DataFrame(records)
    
    # === 結果分析 ===
    print(f"\n--- 結果サマリー ---")
    
    # 動的ロングの累積リターン vs XAGUSD固定
    dynamic_long_cum = results_df['long_return_pct'].cumsum().iloc[-1]
    xagusd_cum = results_df['xagusd_return_pct'].cumsum().iloc[-1]
    dynamic_short_cum = results_df['short_return_pct'].cumsum().iloc[-1]
    
    print(f"\n[ロング方向]")
    print(f"  動的ペア選択 累積リターン: {dynamic_long_cum:.2f}%")
    print(f"  XAGUSD固定   累積リターン: {xagusd_cum:.2f}%")
    print(f"  改善幅: {dynamic_long_cum - xagusd_cum:+.2f}%")
    
    print(f"\n[ショート方向]")
    print(f"  動的ペア選択 累積リターン: {dynamic_short_cum:.2f}%")
    
    # ペア選択頻度
    print(f"\n[ロング方向 ペア選択頻度]")
    freq = results_df['long_pair'].value_counts()
    for pair, count in freq.items():
        avg_ret = results_df[results_df['long_pair'] == pair]['long_return_pct'].mean()
        print(f"  {pair}: {count}回 ({count/len(results_df)*100:.1f}%), 平均リターン: {avg_ret:.4f}%")
    
    print(f"\n[ショート方向 ペア選択頻度]")
    freq_s = results_df['short_pair'].value_counts()
    for pair, count in freq_s.items():
        avg_ret = results_df[results_df['short_pair'] == pair]['short_return_pct'].mean()
        print(f"  {pair}: {count}回 ({count/len(results_df)*100:.1f}%), 平均リターン: {avg_ret:.4f}%")
    
    # 最新の通貨強弱と推奨ペア
    latest = strength.iloc[-1]
    print(f"\n[最新の通貨強弱 ({strength.index[-1].strftime('%Y-%m-%d')})]")
    sorted_str = latest.sort_values(ascending=False)
    for ccy, val in sorted_str.items():
        bar = '█' * int(abs(val) * 5)
        sign = '+' if val > 0 else ''
        print(f"  {ccy}: {sign}{val:.2f}  {bar}")
    
    best_long, weak, _ = select_best_pair(latest, 'long')
    best_short, strong, _ = select_best_pair(latest, 'short')
    print(f"\n  → ロング推奨: {best_long} (相手通貨{weak}が最弱)")
    print(f"  → ショート推奨: {best_short} (相手通貨{strong}が最強)")
    
    # CSVに保存
    out_csv = os.path.join(RESULTS_DIR, f'dynamic_pair_selection_{timeframe}.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")
    
    return results_df, strength


if __name__ == '__main__':
    # 日足で検証
    results_1d, strength_1d = dynamic_pair_selection_backtest('1d', lookback=20)
    
    # 1時間足でも検証
    results_1h, strength_1h = dynamic_pair_selection_backtest('1h', lookback=20)
    
    # 最終サマリーをJSON保存
    summary = {
        'generated_at': datetime.now().isoformat(),
        'daily': {
            'dynamic_long_cumulative': round(results_1d['long_return_pct'].cumsum().iloc[-1], 2),
            'xagusd_fixed_cumulative': round(results_1d['xagusd_return_pct'].cumsum().iloc[-1], 2),
            'dynamic_short_cumulative': round(results_1d['short_return_pct'].cumsum().iloc[-1], 2),
            'total_bars': len(results_1d),
        },
        'hourly': {
            'dynamic_long_cumulative': round(results_1h['long_return_pct'].cumsum().iloc[-1], 2),
            'xagusd_fixed_cumulative': round(results_1h['xagusd_return_pct'].cumsum().iloc[-1], 2),
            'dynamic_short_cumulative': round(results_1h['short_return_pct'].cumsum().iloc[-1], 2),
            'total_bars': len(results_1h),
        },
        'latest_recommendation': {
            'long': select_best_pair(strength_1d.iloc[-1], 'long')[0],
            'short': select_best_pair(strength_1d.iloc[-1], 'short')[0],
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'currency_strength_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("全処理完了")
    print(f"{'='*60}")
