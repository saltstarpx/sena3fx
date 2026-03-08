#!/usr/bin/env python3
"""
通貨強弱エンジン v3 - テクニカル統合型
========================================
v2の問題: 通貨強弱だけでは不十分。ペア切替のリターンがXAGUSD固定に勝てない。
v3の根本的再設計:
  通貨強弱は「フィルター」として使い、最終判断は各XAGペアの
  テクニカル指標スコアで行う。

  スコア = トレンド整合性 + モメンタム + 通貨強弱ボーナス

  1. 各XAGペアのEMA20/50/200の位置関係（パーフェクトオーダー）
  2. RSI(14)の方向性
  3. ATR正規化モメンタム
  4. 通貨強弱による加点/減点
  5. 季節フィルター（7月・9月回避）
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
    'XAGEUR': ('EURUSD', 'divide'),
    'XAGGBP': ('GBPUSD', 'divide'),
    'XAGAUD': ('AUDUSD', 'divide'),
    'XAGNZD': ('NZDUSD', 'divide'),
    'XAGCAD': ('USDCAD', 'multiply'),
    'XAGCHF': ('USDCHF', 'multiply'),
}

PAIR_TO_CURRENCY = {
    'XAGUSD': 'USD', 'XAGJPY': 'JPY', 'XAGEUR': 'EUR', 'XAGGBP': 'GBP',
    'XAGAUD': 'AUD', 'XAGNZD': 'NZD', 'XAGCAD': 'CAD', 'XAGCHF': 'CHF',
}

CURRENCY_PAIRS_FOR_STRENGTH = {
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


def calc_currency_strength_blended(timeframe='1d'):
    """ブレンド通貨強弱"""
    weights = {5: 0.5, 20: 0.3, 60: 0.2}
    fx_data = {}
    needed = set()
    for ccy, pairs in CURRENCY_PAIRS_FOR_STRENGTH.items():
        for pn, _ in pairs:
            needed.add(pn)
    for pn in needed:
        f = os.path.join(DATA_DIR, f'{pn}_{timeframe}.csv')
        if os.path.exists(f):
            fx_data[pn] = pd.read_csv(f, index_col='datetime', parse_dates=True)['Close']
    
    all_strengths = {}
    for lookback, weight in weights.items():
        fx_returns = {pn: close.pct_change(lookback) for pn, close in fx_data.items()}
        all_idx = None
        for pn, ret in fx_returns.items():
            idx = ret.dropna().index
            all_idx = idx if all_idx is None else all_idx.intersection(idx)
        strength = pd.DataFrame(index=all_idx)
        for ccy, pairs in CURRENCY_PAIRS_FOR_STRENGTH.items():
            vals = [fx_returns[pn].loc[all_idx] * d for pn, d in pairs if pn in fx_returns]
            if vals:
                strength[ccy] = pd.concat(vals, axis=1).mean(axis=1) * 100
        all_strengths[lookback] = strength
    
    common_idx = all_strengths[5].index
    for lb in [20, 60]:
        common_idx = common_idx.intersection(all_strengths[lb].index)
    blended = pd.DataFrame(0, index=common_idx, columns=all_strengths[5].columns)
    for lb, w in weights.items():
        blended += all_strengths[lb].loc[common_idx] * w
    return blended


def calc_technical_score(close_series, direction='long'):
    """
    テクニカルスコアを計算する（各バーに対して）
    
    スコア構成:
    - EMAパーフェクトオーダー: +3 (完全一致), +1 (部分一致), 0 (不一致)
    - RSI方向性: +2 (方向一致), 0 (中立), -1 (逆方向)
    - モメンタム: +2 (強い), +1 (中程度), 0 (弱い)
    - 価格位置: +1 (EMA200上/下), 0 (逆)
    
    最大スコア: 8
    """
    close = close_series.copy()
    
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14)
    # 簡易版: 終値の変化率の絶対値の移動平均
    atr = close.pct_change().abs().rolling(14).mean() * 100
    
    # モメンタム(10日変化率)
    momentum = close.pct_change(10) * 100
    
    scores = pd.Series(0.0, index=close.index)
    
    if direction == 'long':
        # EMAパーフェクトオーダー (Close > EMA20 > EMA50 > EMA200)
        perfect = (close > ema20) & (ema20 > ema50) & (ema50 > ema200)
        partial = (close > ema200) & (ema20 > ema200)
        scores += perfect.astype(float) * 3 + (~perfect & partial).astype(float) * 1
        
        # RSI方向性
        rsi_bull = rsi > 55
        rsi_bear = rsi < 40
        scores += rsi_bull.astype(float) * 2 - rsi_bear.astype(float) * 1
        
        # モメンタム
        scores += (momentum > atr).astype(float) * 2 + ((momentum > 0) & (momentum <= atr)).astype(float) * 1
        
        # 価格位置
        scores += (close > ema200).astype(float) * 1
    else:
        # ショート方向
        perfect = (close < ema20) & (ema20 < ema50) & (ema50 < ema200)
        partial = (close < ema200) & (ema20 < ema200)
        scores += perfect.astype(float) * 3 + (~perfect & partial).astype(float) * 1
        
        rsi_bear = rsi < 45
        rsi_bull = rsi > 60
        scores += rsi_bear.astype(float) * 2 - rsi_bull.astype(float) * 1
        
        scores += (momentum < -atr).astype(float) * 2 + ((momentum < 0) & (momentum >= -atr)).astype(float) * 1
        scores += (close < ema200).astype(float) * 1
    
    return scores


def run_v3_backtest(timeframe='1d', strength_weight=1.0, seasonal_filter=True, 
                    min_score_diff=1.0, holding_period=1):
    """
    v3バックテスト: テクニカルスコア + 通貨強弱
    
    Parameters:
    - strength_weight: 通貨強弱のスコアへの寄与度
    - min_score_diff: XAGUSD対比でこのスコア差以上ある場合のみ切替
    - holding_period: ポジション保有期間（バー数）
    """
    xag_pairs = load_xag_pairs(timeframe)
    strength = calc_currency_strength_blended(timeframe)
    
    # XAGUSDのトレンド方向判定
    xagusd = xag_pairs['XAGUSD']
    ema200_xagusd = xagusd['Close'].ewm(span=200, adjust=False).mean()
    
    # 各ペアのテクニカルスコアを事前計算
    tech_scores_long = {}
    tech_scores_short = {}
    for pair_name, pair_df in xag_pairs.items():
        tech_scores_long[pair_name] = calc_technical_score(pair_df['Close'], 'long')
        tech_scores_short[pair_name] = calc_technical_score(pair_df['Close'], 'short')
    
    # 共通インデックス
    common_idx = strength.index
    for pair_name in xag_pairs:
        common_idx = common_idx.intersection(xag_pairs[pair_name].index)
        common_idx = common_idx.intersection(tech_scores_long[pair_name].dropna().index)
    
    records = []
    
    for i in range(200, len(common_idx) - holding_period):
        date = common_idx[i]
        future_date = common_idx[i + holding_period]
        
        # 季節フィルター
        if seasonal_filter and date.month in [7, 9]:
            records.append({
                'date': date, 'action': 'SKIP_SEASONAL',
                'selected_pair': 'NONE', 'direction': 'none',
                'total_score': 0, 'xagusd_score': 0,
                'return_pct': 0, 'xagusd_return_pct': 0,
            })
            continue
        
        # XAGのトレンド方向
        if date not in ema200_xagusd.index:
            continue
        direction = 'long' if xagusd.loc[date, 'Close'] > ema200_xagusd.loc[date] else 'short'
        
        # 各ペアの総合スコアを計算
        pair_scores = {}
        for pair_name in xag_pairs:
            if date not in tech_scores_long[pair_name].index:
                continue
            
            # テクニカルスコア
            if direction == 'long':
                tech = tech_scores_long[pair_name].loc[date]
            else:
                tech = tech_scores_short[pair_name].loc[date]
            
            # 通貨強弱ボーナス
            ccy = PAIR_TO_CURRENCY[pair_name]
            if date in strength.index and ccy in strength.columns:
                ccy_str = strength.loc[date, ccy]
                if direction == 'long':
                    # ロング: 相手通貨が弱いほどボーナス
                    bonus = -ccy_str * strength_weight
                else:
                    # ショート: 相手通貨が強いほどボーナス
                    bonus = ccy_str * strength_weight
            else:
                bonus = 0
            
            pair_scores[pair_name] = tech + bonus
        
        if not pair_scores:
            continue
        
        # 最高スコアのペアを選択
        best_pair = max(pair_scores, key=pair_scores.get)
        best_score = pair_scores[best_pair]
        xagusd_score = pair_scores.get('XAGUSD', 0)
        
        # スコア差が閾値未満ならXAGUSD固定
        if best_score - xagusd_score < min_score_diff:
            best_pair = 'XAGUSD'
            best_score = xagusd_score
        
        # リターン計算
        ret = 0
        if best_pair in xag_pairs:
            pair_df = xag_pairs[best_pair]
            if date in pair_df.index and future_date in pair_df.index:
                raw_ret = (pair_df.loc[future_date, 'Close'] / pair_df.loc[date, 'Close'] - 1) * 100
                ret = raw_ret if direction == 'long' else -raw_ret
        
        xagusd_ret = 0
        if date in xagusd.index and future_date in xagusd.index:
            raw = (xagusd.loc[future_date, 'Close'] / xagusd.loc[date, 'Close'] - 1) * 100
            xagusd_ret = raw if direction == 'long' else -raw
        
        records.append({
            'date': date, 'action': 'TRADE',
            'selected_pair': best_pair, 'direction': direction,
            'total_score': round(best_score, 2), 'xagusd_score': round(xagusd_score, 2),
            'return_pct': round(ret, 4), 'xagusd_return_pct': round(xagusd_ret, 4),
        })
    
    return pd.DataFrame(records)


def analyze_results(df, label=''):
    """結果分析"""
    trades = df[df['action'] == 'TRADE']
    if len(trades) == 0:
        return {'improvement': 0}
    
    dynamic_cum = trades['return_pct'].cumsum().iloc[-1]
    xagusd_cum = trades['xagusd_return_pct'].cumsum().iloc[-1]
    improvement = dynamic_cum - xagusd_cum
    
    switched = trades[trades['selected_pair'] != 'XAGUSD']
    switch_pct = len(switched) / len(trades) * 100 if len(trades) > 0 else 0
    
    # 勝率
    wins = (trades['return_pct'] > 0).sum()
    wr = wins / len(trades) * 100
    
    # シャープレシオ（簡易版）
    mean_ret = trades['return_pct'].mean()
    std_ret = trades['return_pct'].std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
    
    result = {
        'label': label,
        'dynamic_cum': round(dynamic_cum, 2),
        'xagusd_cum': round(xagusd_cum, 2),
        'improvement': round(improvement, 2),
        'total_trades': len(trades),
        'switch_pct': round(switch_pct, 1),
        'win_rate': round(wr, 1),
        'sharpe': round(sharpe, 2),
    }
    return result


def full_parameter_sweep():
    """全パラメータのグリッドサーチ"""
    print(f"\n{'='*70}")
    print("v3 パラメータグリッドサーチ")
    print(f"{'='*70}")
    
    results = []
    
    for sw in [0.0, 0.5, 1.0, 2.0]:
        for msd in [0.0, 0.5, 1.0, 2.0]:
            for hp in [1, 3, 5]:
                for sf in [True, False]:
                    label = f"sw={sw}_msd={msd}_hp={hp}_sf={sf}"
                    try:
                        df = run_v3_backtest('1d', strength_weight=sw, 
                                           seasonal_filter=sf, min_score_diff=msd,
                                           holding_period=hp)
                        r = analyze_results(df, label)
                        r['strength_weight'] = sw
                        r['min_score_diff'] = msd
                        r['holding_period'] = hp
                        r['seasonal'] = sf
                        results.append(r)
                    except Exception as e:
                        print(f"  ERROR {label}: {e}")
    
    sweep_df = pd.DataFrame(results)
    sweep_df = sweep_df.sort_values('improvement', ascending=False)
    
    print(f"\n--- TOP 20 パラメータ組み合わせ ---")
    cols = ['strength_weight', 'min_score_diff', 'holding_period', 'seasonal',
            'dynamic_cum', 'xagusd_cum', 'improvement', 'switch_pct', 'win_rate', 'sharpe']
    print(sweep_df[cols].head(20).to_string(index=False))
    
    # 保存
    sweep_df.to_csv(os.path.join(RESULTS_DIR, 'v3_parameter_sweep.csv'), index=False)
    
    # 最適パラメータで詳細分析
    best = sweep_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"最適パラメータ:")
    print(f"  strength_weight = {best['strength_weight']}")
    print(f"  min_score_diff  = {best['min_score_diff']}")
    print(f"  holding_period  = {best['holding_period']}")
    print(f"  seasonal_filter = {best['seasonal']}")
    print(f"  改善幅: {best['improvement']:+.2f}%")
    print(f"  動的累積: {best['dynamic_cum']:.2f}%")
    print(f"  XAGUSD: {best['xagusd_cum']:.2f}%")
    print(f"  勝率: {best['win_rate']:.1f}%")
    print(f"  シャープ: {best['sharpe']:.2f}")
    
    # 最適パラメータで詳細実行
    best_df = run_v3_backtest('1d', 
                              strength_weight=best['strength_weight'],
                              seasonal_filter=best['seasonal'],
                              min_score_diff=best['min_score_diff'],
                              holding_period=int(best['holding_period']))
    
    trades = best_df[best_df['action'] == 'TRADE']
    
    # ペア別分析
    print(f"\n--- 最適パラメータでのペア別分析 ---")
    for pair in sorted(trades['selected_pair'].unique()):
        subset = trades[trades['selected_pair'] == pair]
        cum = subset['return_pct'].cumsum().iloc[-1] if len(subset) > 0 else 0
        avg = subset['return_pct'].mean()
        wr = (subset['return_pct'] > 0).sum() / len(subset) * 100 if len(subset) > 0 else 0
        print(f"  {pair}: {len(subset)}回 ({len(subset)/len(trades)*100:.1f}%), "
              f"累積={cum:.2f}%, 平均={avg:.4f}%, WR={wr:.1f}%")
    
    # 年別分析
    trades_copy = trades.copy()
    trades_copy['year'] = pd.to_datetime(trades_copy['date']).dt.year
    print(f"\n--- 年別パフォーマンス ---")
    for year, grp in trades_copy.groupby('year'):
        dyn = grp['return_pct'].sum()
        base = grp['xagusd_return_pct'].sum()
        wr = (grp['return_pct'] > 0).sum() / len(grp) * 100
        print(f"  {year}: 動的={dyn:+.2f}%, XAGUSD={base:+.2f}%, 差={dyn-base:+.2f}%, WR={wr:.1f}%")
    
    # 方向別
    for d in ['long', 'short']:
        subset = trades[trades['direction'] == d]
        if len(subset) > 0:
            cum = subset['return_pct'].cumsum().iloc[-1]
            wr = (subset['return_pct'] > 0).sum() / len(subset) * 100
            print(f"\n  [{d.upper()}] {len(subset)}回, 累積={cum:.2f}%, WR={wr:.1f}%")
    
    # 詳細結果を保存
    best_df.to_csv(os.path.join(RESULTS_DIR, 'v3_best_result.csv'), index=False)
    
    # サマリーJSON
    summary = {
        'generated_at': datetime.now().isoformat(),
        'best_params': {
            'strength_weight': float(best['strength_weight']),
            'min_score_diff': float(best['min_score_diff']),
            'holding_period': int(best['holding_period']),
            'seasonal_filter': bool(best['seasonal']),
        },
        'performance': {
            'dynamic_cumulative': float(best['dynamic_cum']),
            'xagusd_cumulative': float(best['xagusd_cum']),
            'improvement': float(best['improvement']),
            'win_rate': float(best['win_rate']),
            'sharpe_ratio': float(best['sharpe']),
        }
    }
    with open(os.path.join(RESULTS_DIR, 'v3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return sweep_df, best_df


if __name__ == '__main__':
    sweep_df, best_df = full_parameter_sweep()
    print(f"\n{'='*70}")
    print("v3 全処理完了")
    print(f"{'='*70}")
