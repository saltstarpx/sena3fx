#!/usr/bin/env python3
"""
通貨強弱ポートフォリオエンジン - 最終版
==========================================
結論: 通貨強弱による「常時ペア切替」はXAGUSD固定に勝てない。
代わりに、通貨強弱を「追加エントリーフィルター」として活用する。

戦略:
1. メイン: XAGUSD（常時監視、通常のテクニカル戦略）
2. サブ1: XAGCHF（CHFが弱い時のみロング追加、WR=68.9%の実績）
3. サブ2: XAGJPY（JPYが弱い時のみロング追加、大きなリターン実績）

通貨強弱の役割:
- メインのXAGUSD戦略のフィルターとして使用
- USDが最強の時はXAGUSDロングを避ける
- サブペアの追加エントリー判断に使用

出力: Claude Codeが統合可能なJSON形式のシグナル
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SYNTH_RULES = {
    'XAGJPY': ('USDJPY', 'multiply'),
    'XAGCHF': ('USDCHF', 'multiply'),
}

def load_pair(pair_name, timeframe='1d'):
    """ペアデータを読み込む（合成含む）"""
    direct_file = os.path.join(DATA_DIR, f'{pair_name}_{timeframe}.csv')
    if os.path.exists(direct_file):
        return pd.read_csv(direct_file, index_col='datetime', parse_dates=True)
    
    if pair_name in SYNTH_RULES:
        xag = pd.read_csv(os.path.join(DATA_DIR, f'XAGUSD_{timeframe}.csv'), 
                          index_col='datetime', parse_dates=True)
        fx_pair, op = SYNTH_RULES[pair_name]
        fx = pd.read_csv(os.path.join(DATA_DIR, f'{fx_pair}_{timeframe}.csv'),
                        index_col='datetime', parse_dates=True)
        common = xag.index.intersection(fx.index)
        x, f = xag.loc[common], fx.loc[common]
        if op == 'multiply':
            return pd.DataFrame({
                'Open': x['Open']*f['Open'], 'High': x['High']*f['High'],
                'Low': x['Low']*f['Low'], 'Close': x['Close']*f['Close'],
                'Volume': x['Volume']}, index=common)
        else:
            return pd.DataFrame({
                'Open': x['Open']/f['Open'], 'High': x['High']/f['High'],
                'Low': x['Low']/f['Low'], 'Close': x['Close']/f['Close'],
                'Volume': x['Volume']}, index=common)
    return None


def calc_usd_strength(timeframe='1d', lookback=20):
    """USD強弱のみを高精度で計算"""
    pairs = [('USDJPY', 1), ('EURUSD', -1), ('GBPUSD', -1), ('AUDUSD', -1), 
             ('NZDUSD', -1), ('USDCAD', 1), ('USDCHF', 1)]
    
    returns = []
    for pair_name, direction in pairs:
        f = os.path.join(DATA_DIR, f'{pair_name}_{timeframe}.csv')
        if os.path.exists(f):
            close = pd.read_csv(f, index_col='datetime', parse_dates=True)['Close']
            ret = close.pct_change(lookback) * direction * 100
            returns.append(ret)
    
    if not returns:
        return pd.Series(dtype=float)
    
    combined = pd.concat(returns, axis=1)
    return combined.mean(axis=1).dropna()


def calc_chf_strength(timeframe='1d', lookback=20):
    """CHF強弱を計算"""
    f = os.path.join(DATA_DIR, f'USDCHF_{timeframe}.csv')
    if os.path.exists(f):
        close = pd.read_csv(f, index_col='datetime', parse_dates=True)['Close']
        return -close.pct_change(lookback) * 100  # CHF強 = USDCHF下落
    return pd.Series(dtype=float)


def calc_jpy_strength(timeframe='1d', lookback=20):
    """JPY強弱を計算"""
    f = os.path.join(DATA_DIR, f'USDJPY_{timeframe}.csv')
    if os.path.exists(f):
        close = pd.read_csv(f, index_col='datetime', parse_dates=True)['Close']
        return -close.pct_change(lookback) * 100  # JPY強 = USDJPY下落
    return pd.Series(dtype=float)


def portfolio_backtest(timeframe='1d'):
    """
    ポートフォリオバックテスト
    
    ルール:
    - XAGUSD: EMA200上でロング（USD強弱フィルター付き）
    - XAGCHF: CHFが弱い（下位25%）時のみロング追加
    - XAGJPY: JPYが弱い（下位25%）時のみロング追加
    - 季節フィルター: 7月・9月はポジション縮小
    - 各ペアのロットは均等配分（1/3ずつ）
    """
    print(f"\n{'='*70}")
    print(f"ポートフォリオバックテスト (TF={timeframe})")
    print(f"{'='*70}")
    
    # データ読み込み
    xagusd = load_pair('XAGUSD', timeframe)
    xagchf = load_pair('XAGCHF', timeframe)
    xagjpy = load_pair('XAGJPY', timeframe)
    
    usd_str = calc_usd_strength(timeframe, 20)
    chf_str = calc_chf_strength(timeframe, 20)
    jpy_str = calc_jpy_strength(timeframe, 20)
    
    # EMA200
    ema200_xagusd = xagusd['Close'].ewm(span=200, adjust=False).mean()
    ema200_xagchf = xagchf['Close'].ewm(span=200, adjust=False).mean()
    ema200_xagjpy = xagjpy['Close'].ewm(span=200, adjust=False).mean()
    
    # 共通インデックス
    common = xagusd.index.intersection(xagchf.index).intersection(xagjpy.index)
    common = common.intersection(usd_str.dropna().index)
    common = common.intersection(chf_str.dropna().index)
    common = common.intersection(jpy_str.dropna().index)
    common = common[common.isin(ema200_xagusd.dropna().index)]
    common = sorted(common)
    
    # CHF/JPY強弱の25%タイル（弱い判定の閾値）
    chf_weak_threshold = chf_str.quantile(0.25)
    jpy_weak_threshold = jpy_str.quantile(0.25)
    usd_strong_threshold = usd_str.quantile(0.75)
    
    print(f"データ期間: {common[0]} ~ {common[-1]}")
    print(f"バー数: {len(common)}")
    print(f"CHF弱閾値: {chf_weak_threshold:.3f}")
    print(f"JPY弱閾値: {jpy_weak_threshold:.3f}")
    print(f"USD強閾値: {usd_strong_threshold:.3f}")
    
    records = []
    
    for i in range(200, len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        
        month = date.month
        seasonal_penalty = 0.5 if month in [7, 9] else 1.0
        
        # === XAGUSD ===
        xagusd_long = xagusd.loc[date, 'Close'] > ema200_xagusd.loc[date]
        usd_too_strong = usd_str.loc[date] > usd_strong_threshold
        
        xagusd_signal = 1.0 if (xagusd_long and not usd_too_strong) else 0.0
        xagusd_signal *= seasonal_penalty
        
        xagusd_ret = (xagusd.loc[next_date, 'Close'] / xagusd.loc[date, 'Close'] - 1) * 100
        
        # === XAGCHF ===
        xagchf_long = xagchf.loc[date, 'Close'] > ema200_xagchf.loc[date]
        chf_weak = chf_str.loc[date] < chf_weak_threshold
        
        xagchf_signal = 1.0 if (xagchf_long and chf_weak) else 0.0
        xagchf_signal *= seasonal_penalty
        
        xagchf_ret = (xagchf.loc[next_date, 'Close'] / xagchf.loc[date, 'Close'] - 1) * 100
        
        # === XAGJPY ===
        xagjpy_long = xagjpy.loc[date, 'Close'] > ema200_xagjpy.loc[date]
        jpy_weak = jpy_str.loc[date] < jpy_weak_threshold
        
        xagjpy_signal = 1.0 if (xagjpy_long and jpy_weak) else 0.0
        xagjpy_signal *= seasonal_penalty
        
        xagjpy_ret = (xagjpy.loc[next_date, 'Close'] / xagjpy.loc[date, 'Close'] - 1) * 100
        
        # ポートフォリオリターン（均等配分）
        active_count = sum([1 for s in [xagusd_signal, xagchf_signal, xagjpy_signal] if s > 0])
        
        if active_count > 0:
            weight = 1.0 / 3  # 常に1/3ずつ配分
            portfolio_ret = (xagusd_signal * xagusd_ret * weight +
                           xagchf_signal * xagchf_ret * weight +
                           xagjpy_signal * xagjpy_ret * weight)
        else:
            portfolio_ret = 0
        
        # XAGUSD単独（ベンチマーク）
        benchmark_ret = xagusd_ret if xagusd_long else 0
        benchmark_ret *= seasonal_penalty
        
        records.append({
            'date': date,
            'xagusd_signal': xagusd_signal,
            'xagchf_signal': xagchf_signal,
            'xagjpy_signal': xagjpy_signal,
            'active_pairs': active_count,
            'xagusd_ret': round(xagusd_ret, 4),
            'xagchf_ret': round(xagchf_ret, 4),
            'xagjpy_ret': round(xagjpy_ret, 4),
            'portfolio_ret': round(portfolio_ret, 4),
            'benchmark_ret': round(benchmark_ret, 4),
        })
    
    df = pd.DataFrame(records)
    
    # === 結果分析 ===
    active = df[df['active_pairs'] > 0]
    
    port_cum = df['portfolio_ret'].cumsum().iloc[-1]
    bench_cum = df['benchmark_ret'].cumsum().iloc[-1]
    
    print(f"\n{'='*50}")
    print(f"結果サマリー")
    print(f"{'='*50}")
    print(f"ポートフォリオ累積: {port_cum:.2f}%")
    print(f"XAGUSD単独累積:    {bench_cum:.2f}%")
    print(f"改善幅: {port_cum - bench_cum:+.2f}%")
    
    # リスク指標
    port_std = df['portfolio_ret'].std()
    bench_std = df['benchmark_ret'].std()
    port_sharpe = df['portfolio_ret'].mean() / port_std * np.sqrt(252) if port_std > 0 else 0
    bench_sharpe = df['benchmark_ret'].mean() / bench_std * np.sqrt(252) if bench_std > 0 else 0
    
    # 最大ドローダウン
    port_equity = (1 + df['portfolio_ret'] / 100).cumprod()
    port_peak = port_equity.cummax()
    port_dd = ((port_equity - port_peak) / port_peak * 100).min()
    
    bench_equity = (1 + df['benchmark_ret'] / 100).cumprod()
    bench_peak = bench_equity.cummax()
    bench_dd = ((bench_equity - bench_peak) / bench_peak * 100).min()
    
    print(f"\nボラティリティ: ポート={port_std:.3f}% vs ベンチ={bench_std:.3f}%")
    print(f"シャープ比:     ポート={port_sharpe:.2f} vs ベンチ={bench_sharpe:.2f}")
    print(f"最大DD:         ポート={port_dd:.2f}% vs ベンチ={bench_dd:.2f}%")
    
    # アクティブペア数の分布
    print(f"\nアクティブペア数分布:")
    for n in range(4):
        count = (df['active_pairs'] == n).sum()
        print(f"  {n}ペア: {count}回 ({count/len(df)*100:.1f}%)")
    
    # 各ペアの貢献度
    print(f"\n各ペアの貢献度:")
    for pair, sig_col, ret_col in [
        ('XAGUSD', 'xagusd_signal', 'xagusd_ret'),
        ('XAGCHF', 'xagchf_signal', 'xagchf_ret'),
        ('XAGJPY', 'xagjpy_signal', 'xagjpy_ret'),
    ]:
        active_mask = df[sig_col] > 0
        if active_mask.sum() > 0:
            avg_ret = df.loc[active_mask, ret_col].mean()
            cum_ret = df.loc[active_mask, ret_col].sum()
            wr = (df.loc[active_mask, ret_col] > 0).sum() / active_mask.sum() * 100
            print(f"  {pair}: {active_mask.sum()}回アクティブ, "
                  f"累積={cum_ret:.2f}%, 平均={avg_ret:.4f}%, WR={wr:.1f}%")
    
    # 年別
    df_copy = df.copy()
    df_copy['year'] = pd.to_datetime(df_copy['date']).dt.year
    print(f"\n年別パフォーマンス:")
    for year, grp in df_copy.groupby('year'):
        p = grp['portfolio_ret'].sum()
        b = grp['benchmark_ret'].sum()
        print(f"  {year}: ポート={p:+.2f}%, ベンチ={b:+.2f}%, 差={p-b:+.2f}%")
    
    # 保存
    df.to_csv(os.path.join(RESULTS_DIR, f'portfolio_backtest_{timeframe}.csv'), index=False)
    
    # サマリーJSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'strategy': 'Currency Strength Portfolio (XAGUSD + XAGCHF + XAGJPY)',
        'performance': {
            'portfolio_cumulative': round(port_cum, 2),
            'benchmark_cumulative': round(bench_cum, 2),
            'improvement': round(port_cum - bench_cum, 2),
            'portfolio_sharpe': round(port_sharpe, 2),
            'benchmark_sharpe': round(bench_sharpe, 2),
            'portfolio_max_dd': round(port_dd, 2),
            'benchmark_max_dd': round(bench_dd, 2),
            'portfolio_volatility': round(port_std, 4),
            'benchmark_volatility': round(bench_std, 4),
        },
        'pair_contributions': {},
    }
    
    for pair, sig_col, ret_col in [
        ('XAGUSD', 'xagusd_signal', 'xagusd_ret'),
        ('XAGCHF', 'xagchf_signal', 'xagchf_ret'),
        ('XAGJPY', 'xagjpy_signal', 'xagjpy_ret'),
    ]:
        active_mask = df[sig_col] > 0
        if active_mask.sum() > 0:
            summary['pair_contributions'][pair] = {
                'active_count': int(active_mask.sum()),
                'cumulative_return': round(df.loc[active_mask, ret_col].sum(), 2),
                'win_rate': round((df.loc[active_mask, ret_col] > 0).sum() / active_mask.sum() * 100, 1),
            }
    
    with open(os.path.join(RESULTS_DIR, 'portfolio_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果保存完了")
    return df


if __name__ == '__main__':
    df = portfolio_backtest('1d')
    print(f"\n{'='*70}")
    print("ポートフォリオバックテスト完了")
    print(f"{'='*70}")
