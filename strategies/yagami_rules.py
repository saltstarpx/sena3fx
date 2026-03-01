"""
Teammate A: Yagami — 高勝率シグナル追求
========================================
既存のやがみメソッド（5条件評価）をさらに深化させ、
高勝率・高プロフィットファクターのシグナルを追求する。

評価基準: 勝率(WR) > 60%, プロフィットファクター(PF) > 1.8

担当タスク:
  - 薄いゾーン戦略のフォワードテスト
  - overheat_monitor.py の監視
  - やがみシグナルの改善

本ファイルは lib/yagami.py のシグナル関数を re-export し、
USD強弱フィルターやロット調整ロジックを組み合わせるハブ。
"""
from lib.yagami import (
    sig_yagami_A,
    sig_yagami_B,
    sig_yagami_reversal_only,
    sig_yagami_double_bottom,
    sig_yagami_pattern_break,
    sig_yagami_london_ny,
    sig_yagami_vol_regime,
    sig_yagami_trend_regime,
    sig_yagami_prime_time,
    sig_yagami_full_filter,
    sig_yagami_A_full_filter,
)
from strategies.market_filters import make_usd_filtered_signal


# ── USD強弱フィルター付きバリアント ──
sig_yagami_A_usd   = make_usd_filtered_signal(sig_yagami_A, threshold=75)
sig_yagami_B_usd   = make_usd_filtered_signal(sig_yagami_B, threshold=75)
sig_yagami_full_usd = make_usd_filtered_signal(sig_yagami_full_filter, threshold=75)
sig_yagami_lonny_usd = make_usd_filtered_signal(sig_yagami_london_ny, threshold=75)


# ── 戦略バリアントリスト (バックテスト用) ──
def yagami_variants(freq='4h'):
    """Teammate A の全戦略バリアントを返す"""
    return [
        ('YagamiA',          sig_yagami_A(freq=freq)),
        ('YagamiB',          sig_yagami_B(freq=freq)),
        ('YagamiReversal',   sig_yagami_reversal_only(freq=freq)),
        ('YagamiLonNY',      sig_yagami_london_ny(freq=freq)),
        ('YagamiFull',       sig_yagami_full_filter(freq=freq)),
        ('YagamiA+USD',      sig_yagami_A_usd(freq=freq)),
        ('YagamiB+USD',      sig_yagami_B_usd(freq=freq)),
        ('YagamiFull+USD',   sig_yagami_full_usd(freq=freq)),
        ('YagamiLonNY+USD',  sig_yagami_lonny_usd(freq=freq)),
    ]
