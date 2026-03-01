#!/usr/bin/env python3
"""
通貨強弱エンジン v2 - 改良版
==============================
v1の問題: 単純な最弱通貨選択ではXAGUSD固定に勝てない
v2の改良:
  1. 複数lookback期間のブレンド（短期5日+中期20日+長期60日）
  2. 強弱差分の閾値フィルター（差が小さい場合はXAGUSD固定）
  3. EMA200トレンドフィルター（上昇トレンドでのみロング）
  4. 季節フィルター（7月・9月回避）
  5. モメンタム加速度（強弱の変化速度）
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== 合成ルール ==========
SYNTH_RULES = {
    'XAGJPY': ('USDJPY', 'multiply'),
    'XAGEUR': ('EURUSD', 'divide'),
    'XAGGBP': ('GBPUSD', 'divide'),
    'XAGAUD': ('AUDUSD', 'divide'),
    'XAGNZD': ('NZDUSD', 'divide'),
    'XAGCAD': ('USDCAD', 'multiply'),
    'XAGCHF': ('USDCHF', 'multiply'),
}

CURRENCY_PAIRS = {
    'USD': [('USDJPY', 1), ('EURUSD', -1), ('GBPUSD', -1), ('AUDUSD', -1), 
            ('NZDUSD', -1), ('USDCAD', 1), ('USDCHF', 1)],
    'JPY': [('USDJPY', -1)],
    'EUR': [('EURUSD', 1)],
    'GBP': [('GBPUSD', 1)],
    'AUD': [('AUDUSD', 1)],
    'NZD': [('NZDUSD', 1)],
    'CAD': [('USDCAD', -1)],
    'CHF': [('USDCHF', -1)],
}

CURRENCY_TO_PAIR = {
    'USD': 'XAGUSD', 'JPY': 'XAGJPY', 'EUR': 'XAGEUR', 'GBP': 'XAGGBP',
    'AUD': 'XAGAUD', 'NZD': 'XAGNZD', 'CAD': 'XAGCAD', 'CHF': 'XAGCHF',
}


def load_xag_pairs(timeframe='1d'):
    """XAGクロスペアを合成して読み込む"""
    xag_file = os.path.join(DATA_DIR, f'XAGUSD_{timeframe}.csv')
    xag = pd.read_csv(xag_file, index_col='datetime', parse_dates=True)
    pairs = {'XAGUSD': xag}
    
    for pair_name, (fx_pair, op) in SYNTH_RULES.items():
        fx_file = os.path.join(DATA_DIR, f'{fx_pair}_{timeframe}.csv')
        if not os.path.exists(fx_file):
            continue
        fx = pd.read_csv(fx_file, index_col='datetime', parse_dates=True)
        common_idx = xag.index.intersection(fx.index)
        x, f = xag.loc[common_idx], fx.loc[common_idx]
        
        if op == 'multiply':
            synth = pd.DataFrame({
                'Open': x['Open']*f['Open'], 'High': x['High']*f['High'],
                'Low': x['Low']*f['Low'], 'Close': x['Close']*f['Close'],
                'Volume': x['Volume']}, index=common_idx)
        else:
            synth = pd.DataFrame({
                'Open': x['Open']/f['Open'], 'High': x['High']/f['High'],
                'Low': x['Low']/f['Low'], 'Close': x['Close']/f['Close'],
                'Volume': x['Volume']}, index=common_idx)
        synth.index.name = 'datetime'
        pairs[pair_name] = synth
    return pairs


def calc_blended_strength(timeframe='1d'):
    """
    複数lookback期間をブレンドした通貨強弱
    短期(5) x 0.5 + 中期(20) x 0.3 + 長期(60) x 0.2
    """
    weights = {5: 0.5, 20: 0.3, 60: 0.2}
    
    fx_data = {}
    needed = set()
    for ccy, pairs in CURRENCY_PAIRS.items():
        for pn, _ in pairs:
            needed.add(pn)
    
    for pn in needed:
        f = os.path.join(DATA_DIR, f'{pn}_{timeframe}.csv')
        if os.path.exists(f):
            fx_data[pn] = pd.read_csv(f, index_col='datetime', parse_dates=True)['Close']
    
    blended = None
    for lookback, weight in weights.items():
        fx_returns = {}
        for pn, close in fx_data.items():
            fx_returns[pn] = close.pct_change(lookback)
        
        all_idx = None
        for pn, ret in fx_returns.items():
            idx = ret.dropna().index
            all_idx = idx if all_idx is None else all_idx.intersection(idx)
        
        strength = pd.DataFrame(index=all_idx)
        for ccy, pairs in CURRENCY_PAIRS.items():
            vals = []
            for pn, direction in pairs:
                if pn in fx_returns:
                    vals.append(fx_returns[pn].loc[all_idx] * direction)
            if vals:
                strength[ccy] = pd.concat(vals, axis=1).mean(axis=1) * 100
        
        if blended is None:
            blended = strength * weight
        else:
            common = blended.index.intersection(strength.index)
            blended = blended.loc[common] * 1 + strength.loc[common] * weight  # 累積加算
            # 修正: 初回以降は累積なので再計算
    
    # 正しいブレンド計算
    all_strengths = {}
    for lookback, weight in weights.items():
        fx_returns = {}
        for pn, close in fx_data.items():
            fx_returns[pn] = close.pct_change(lookback)
        
        all_idx = None
        for pn, ret in fx_returns.items():
            idx = ret.dropna().index
            all_idx = idx if all_idx is None else all_idx.intersection(idx)
        
        strength = pd.DataFrame(index=all_idx)
        for ccy, pairs in CURRENCY_PAIRS.items():
            vals = []
            for pn, direction in pairs:
                if pn in fx_returns:
                    vals.append(fx_returns[pn].loc[all_idx] * direction)
            if vals:
                strength[ccy] = pd.concat(vals, axis=1).mean(axis=1) * 100
        all_strengths[lookback] = strength
    
    # 共通インデックスでブレンド
    common_idx = all_strengths[5].index
    for lb in [20, 60]:
        common_idx = common_idx.intersection(all_strengths[lb].index)
    
    blended = pd.DataFrame(0, index=common_idx, columns=all_strengths[5].columns)
    for lb, w in weights.items():
        blended += all_strengths[lb].loc[common_idx] * w
    
    return blended


def calc_momentum_acceleration(strength, window=5):
    """通貨強弱の変化速度（加速度）を計算"""
    return strength.diff(window)


def run_enhanced_backtest(timeframe='1d', threshold=0.3, seasonal_filter=True):
    """
    改良版バックテスト
    
    Parameters:
    - threshold: 通貨強弱差がこの値以上の場合のみペア切替
    - seasonal_filter: 7月・9月を除外
    """
    print(f"\n{'='*70}")
    print(f"改良版通貨強弱バックテスト v2")
    print(f"TF={timeframe}, threshold={threshold}, seasonal={seasonal_filter}")
    print(f"{'='*70}")
    
    xag_pairs = load_xag_pairs(timeframe)
    strength = calc_blended_strength(timeframe)
    accel = calc_momentum_acceleration(strength)
    
    # EMA200をXAGUSDで計算（トレンドフィルター）
    xagusd = xag_pairs['XAGUSD']
    ema200 = xagusd['Close'].ewm(span=200, adjust=False).mean()
    
    print(f"データ期間: {strength.index[0]} ~ {strength.index[-1]}")
    print(f"通貨強弱バー数: {len(strength)}")
    
    records = []
    
    for i in range(1, len(strength) - 1):
        date = strength.index[i]
        next_date = strength.index[i + 1]
        row = strength.iloc[i]
        accel_row = accel.iloc[i] if i < len(accel) else None
        
        # 季節フィルター
        if seasonal_filter and date.month in [7, 9]:
            records.append({
                'date': date, 'action': 'SKIP_SEASONAL',
                'selected_pair': 'NONE', 'direction': 'none',
                'return_pct': 0, 'xagusd_return_pct': 0,
                'reason': f'Month {date.month} filtered'
            })
            continue
        
        # XAGUSDのトレンド判定
        if date not in ema200.index:
            continue
        xag_trend = 'up' if xagusd.loc[date, 'Close'] > ema200.loc[date] else 'down'
        
        # 最弱・最強通貨の特定
        weakest_ccy = row.idxmin()
        strongest_ccy = row.idxmax()
        weakest_val = row[weakest_ccy]
        strongest_val = row[strongest_ccy]
        usd_val = row.get('USD', 0)
        
        # XAGUSD固定のリターン
        xagusd_ret = 0
        if date in xagusd.index and next_date in xagusd.index:
            xagusd_ret = (xagusd.loc[next_date, 'Close'] / xagusd.loc[date, 'Close'] - 1) * 100
        
        # === 動的選択ロジック ===
        selected_pair = 'XAGUSD'
        direction = 'long' if xag_trend == 'up' else 'short'
        reason = 'default'
        
        if direction == 'long':
            # ロング: 最弱通貨を相手にする
            # 条件: USD強弱 - 最弱通貨強弱 > threshold
            diff = usd_val - weakest_val
            if diff > threshold and weakest_ccy != 'USD':
                # 加速度チェック: 最弱通貨がさらに弱くなっている場合のみ
                if accel_row is not None and accel_row[weakest_ccy] < 0:
                    selected_pair = CURRENCY_TO_PAIR.get(weakest_ccy, 'XAGUSD')
                    reason = f'{weakest_ccy} weakening (diff={diff:.2f}, accel={accel_row[weakest_ccy]:.3f})'
                else:
                    reason = f'accel not confirmed for {weakest_ccy}'
            else:
                reason = f'diff too small ({diff:.2f} < {threshold})'
        else:
            # ショート: 最強通貨を相手にする
            diff = strongest_val - usd_val
            if diff > threshold and strongest_ccy != 'USD':
                if accel_row is not None and accel_row[strongest_ccy] > 0:
                    selected_pair = CURRENCY_TO_PAIR.get(strongest_ccy, 'XAGUSD')
                    reason = f'{strongest_ccy} strengthening (diff={diff:.2f}, accel={accel_row[strongest_ccy]:.3f})'
                else:
                    reason = f'accel not confirmed for {strongest_ccy}'
            else:
                reason = f'diff too small ({diff:.2f} < {threshold})'
        
        # 選択ペアのリターン計算
        ret = 0
        if selected_pair in xag_pairs:
            pair_df = xag_pairs[selected_pair]
            if date in pair_df.index and next_date in pair_df.index:
                raw_ret = (pair_df.loc[next_date, 'Close'] / pair_df.loc[date, 'Close'] - 1) * 100
                ret = raw_ret if direction == 'long' else -raw_ret
        
        records.append({
            'date': date,
            'action': 'TRADE',
            'selected_pair': selected_pair,
            'direction': direction,
            'return_pct': round(ret, 4),
            'xagusd_return_pct': round(xagusd_ret if direction == 'long' else -xagusd_ret, 4),
            'reason': reason,
        })
    
    df = pd.DataFrame(records)
    trades = df[df['action'] == 'TRADE']
    
    # === 結果分析 ===
    print(f"\n--- 結果サマリー ---")
    print(f"総バー数: {len(df)}, トレード: {len(trades)}, スキップ: {len(df)-len(trades)}")
    
    dynamic_cum = trades['return_pct'].cumsum().iloc[-1] if len(trades) > 0 else 0
    xagusd_cum = trades['xagusd_return_pct'].cumsum().iloc[-1] if len(trades) > 0 else 0
    
    print(f"\n動的ペア選択 累積: {dynamic_cum:.2f}%")
    print(f"XAGUSD固定   累積: {xagusd_cum:.2f}%")
    print(f"改善幅: {dynamic_cum - xagusd_cum:+.2f}%")
    
    # ペア切替が発生した場合のみの分析
    switched = trades[trades['selected_pair'] != 'XAGUSD']
    if len(switched) > 0:
        print(f"\nペア切替発生: {len(switched)}回 ({len(switched)/len(trades)*100:.1f}%)")
        switch_ret = switched['return_pct'].mean()
        default_ret = trades[trades['selected_pair'] == 'XAGUSD']['return_pct'].mean()
        print(f"  切替時平均リターン: {switch_ret:.4f}%")
        print(f"  XAGUSD時平均リターン: {default_ret:.4f}%")
        
        print(f"\n  切替先ペア別:")
        for pair in switched['selected_pair'].unique():
            subset = switched[switched['selected_pair'] == pair]
            print(f"    {pair}: {len(subset)}回, 平均: {subset['return_pct'].mean():.4f}%")
    
    # 方向別分析
    for d in ['long', 'short']:
        subset = trades[trades['direction'] == d]
        if len(subset) > 0:
            print(f"\n[{d.upper()}方向] {len(subset)}回, 累積: {subset['return_pct'].cumsum().iloc[-1]:.2f}%")
    
    # 年別分析
    trades_copy = trades.copy()
    trades_copy['year'] = pd.to_datetime(trades_copy['date']).dt.year
    print(f"\n[年別パフォーマンス]")
    for year, grp in trades_copy.groupby('year'):
        dyn = grp['return_pct'].sum()
        base = grp['xagusd_return_pct'].sum()
        print(f"  {year}: 動的={dyn:+.2f}%, XAGUSD={base:+.2f}%, 差={dyn-base:+.2f}%")
    
    # 最新推奨
    latest = strength.iloc[-1]
    latest_accel = accel.iloc[-1]
    print(f"\n[最新の通貨強弱ブレンド ({strength.index[-1]})]")
    for ccy in latest.sort_values(ascending=False).index:
        a = latest_accel[ccy] if ccy in latest_accel.index else 0
        arrow = '↑' if a > 0 else '↓' if a < 0 else '→'
        print(f"  {ccy}: {latest[ccy]:+.3f} {arrow} (加速度: {a:+.3f})")
    
    # 保存
    out_csv = os.path.join(RESULTS_DIR, f'dynamic_pair_v2_{timeframe}.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")
    
    return df, strength


def parameter_sweep():
    """閾値パラメータの最適化"""
    print(f"\n{'='*70}")
    print("パラメータスイープ: 閾値の最適化")
    print(f"{'='*70}")
    
    results = []
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        for seasonal in [True, False]:
            df, _ = run_enhanced_backtest('1d', threshold=threshold, seasonal_filter=seasonal)
            trades = df[df['action'] == 'TRADE']
            if len(trades) == 0:
                continue
            
            dynamic_cum = trades['return_pct'].cumsum().iloc[-1]
            xagusd_cum = trades['xagusd_return_pct'].cumsum().iloc[-1]
            switched = len(trades[trades['selected_pair'] != 'XAGUSD'])
            
            results.append({
                'threshold': threshold,
                'seasonal': seasonal,
                'dynamic_cum': round(dynamic_cum, 2),
                'xagusd_cum': round(xagusd_cum, 2),
                'improvement': round(dynamic_cum - xagusd_cum, 2),
                'switch_count': switched,
                'total_trades': len(trades),
            })
    
    sweep_df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("パラメータスイープ結果")
    print(f"{'='*70}")
    print(sweep_df.to_string(index=False))
    
    best = sweep_df.loc[sweep_df['improvement'].idxmax()]
    print(f"\n最適パラメータ: threshold={best['threshold']}, seasonal={best['seasonal']}")
    print(f"  改善幅: {best['improvement']:+.2f}%")
    
    sweep_df.to_csv(os.path.join(RESULTS_DIR, 'parameter_sweep.csv'), index=False)
    return sweep_df


if __name__ == '__main__':
    # パラメータスイープで最適閾値を探索
    sweep = parameter_sweep()
    
    print(f"\n{'='*70}")
    print("全処理完了")
    print(f"{'='*70}")
