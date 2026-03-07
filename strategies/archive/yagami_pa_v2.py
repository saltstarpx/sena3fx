"""
やがみプライスアクション戦略 v2 - 画像解析に基づく改善版
======================================================
Noteの画像解析から得られた「実体の揃い」「インサイドバー」「横軸の形成」を重視したロジック。

改善点:
1. 実体の揃い（Body Alignment）: 終値の分散をチェックし、水平なサポート/レジスタンスを特定。
2. インサイドバー（Inside Bar）: ボラティリティ収束と逆三尊右肩の予兆を検知。
3. 横軸の形成（Time Axis）: 単発の反転ではなく、数本の停滞を経てからのエントリー。
"""
import numpy as np
import pandas as pd

def _calc_atr(bars, period=14):
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()

def _is_inside_bar(bars, idx):
    """現在の足が1つ前の足のレンジ内に収まっているか判定"""
    if idx < 1: return False
    cur = bars.iloc[idx]
    prev = bars.iloc[idx-1]
    return (cur['high'] <= prev['high']) and (cur['low'] >= prev['low'])

def _body_alignment_score(bars, idx, lookback=3):
    """
    実体が揃っているかスコアリング。終値の標準偏差で判定。
    値が小さいほど「実体が揃っている」。
    """
    if idx < lookback: return np.nan
    closes = bars['close'].iloc[idx-lookback+1 : idx+1]
    return np.std(closes)

def signal_pa_v2_improved(bars, atr_period=14, lookback=20, alignment_lookback=3, alignment_threshold=0.15):
    """
    改善版やがみ式ロジック:
    - 条件1: 安値圏/高値圏であること（直近lookback本のレンジ端）
    - 条件2: 実体が揃っていること（終値の標準偏差がATRの一定以下）
    - 条件3: インサイドバーが出現していること（収束の確認）
    - 条件4: 15分足で逆の色が確定していること
    """
    atr = _calc_atr(bars, atr_period)
    signals = pd.Series(index=bars.index, dtype=object)
    
    for i in range(max(lookback, alignment_lookback), len(bars)):
        bar = bars.iloc[i]
        prev = bars.iloc[i-1]
        a = atr.iloc[i]
        if np.isnan(a) or a == 0: continue
        
        # 直近レンジの取得
        recent_bars = bars.iloc[i-lookback : i]
        r_max = recent_bars['high'].max()
        r_min = recent_bars['low'].min()
        
        # 実体の揃いスコア（ATR比で正規化）
        align_score = _body_alignment_score(bars, i, alignment_lookback) / a
        
        # インサイドバー判定
        inside = _is_inside_bar(bars, i)
        
        # --- ロングエントリー条件 ---
        # 1. 安値圏（直近安値から1.5*ATR以内）
        in_low_zone = bar['low'] <= r_min + 1.5 * a
        # 2. 実体が揃っている
        bodies_aligned = align_score < alignment_threshold
        # 3. 直近で陽線が出ている（逆の色）
        has_bull_confirmation = (bar['close'] > bar['open']) or (prev['close'] > prev['open'])
        # 4. インサイドバーまたはボラ収束
        converged = inside or (abs(bar['high'] - bar['low']) < a * 0.8)
        
        if in_low_zone and bodies_aligned and has_bull_confirmation and converged:
            signals.iloc[i] = 'long'
            continue

        # --- ショートエントリー条件 ---
        # 1. 高値圏（直近高値から1.5*ATR以内）
        in_high_zone = bar['high'] >= r_max - 1.5 * a
        # 2. 実体が揃っている
        bodies_aligned = align_score < alignment_threshold
        # 3. 直近で陰線が出ている
        has_bear_confirmation = (bar['close'] < bar['open']) or (prev['close'] < prev['open'])
        # 4. ボラ収束
        converged = inside or (abs(bar['high'] - bar['low']) < a * 0.8)
        
        if in_high_zone and bodies_aligned and has_bear_confirmation and converged:
            signals.iloc[i] = 'short'

    return signals
