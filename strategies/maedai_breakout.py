"""
Teammate B: Maedai — 高RRトレンドフォロー
==========================================
低勝率・高リスクリワード（RR）のトレンドフォロー戦略。
「背を近くして何度も挑戦し、大きな値動きを取る」思想を追求する。

評価基準: Sharpe Ratio > 1.5, Calmar Ratio > 3.0, 年間トレード数 > 50

担当タスク:
  - Donchianブレイクアウト戦略のパラメータ最適化
  - ATRベースのトレーリングストップの改良

本ファイルは lib/yagami.py の Maedai 系シグナル関数を re-export し、
パラメータ探索やUSDフィルターとの組み合わせを管理する。
"""
from lib.yagami import (
    sig_maedai_breakout,
    sig_maedai_breakout_v2,
    sig_maedai_best,
    sig_maedai_htf_breakout,
    sig_maedai_htf_pullback,
    sig_maedai_dc_ema_tf,
    sig_maedai_yagami_union,
)
from strategies.market_filters import make_usd_filtered_signal


# ── USD強弱フィルター付きバリアント ──
sig_maedai_dc_usd = make_usd_filtered_signal(sig_maedai_dc_ema_tf, threshold=75)
sig_maedai_union_usd = make_usd_filtered_signal(sig_maedai_yagami_union, threshold=75)
sig_maedai_best_usd = make_usd_filtered_signal(sig_maedai_best, threshold=75)


# ── Donchian パラメータ探索グリッド (Task 3) ──
DC_PARAM_GRID = [
    {'lookback_days': 10, 'ema_days': 200},
    {'lookback_days': 15, 'ema_days': 200},
    {'lookback_days': 20, 'ema_days': 200},
    {'lookback_days': 30, 'ema_days': 200},
    {'lookback_days': 40, 'ema_days': 200},
    {'lookback_days': 15, 'ema_days': 100},
    {'lookback_days': 20, 'ema_days': 100},
    {'lookback_days': 30, 'ema_days': 100},
]


def maedai_dc_variants(freq='4h'):
    """Donchianチャネル期間のパラメータバリアントを返す"""
    variants = []
    for p in DC_PARAM_GRID:
        name = f"DC{p['lookback_days']}d_EMA{p['ema_days']}"
        sig = sig_maedai_dc_ema_tf(
            freq=freq,
            lookback_days=p['lookback_days'],
            ema_days=p['ema_days'],
        )
        variants.append((name, sig))
        # USD フィルター付き
        sig_usd = sig_maedai_dc_usd(
            freq=freq,
            lookback_days=p['lookback_days'],
            ema_days=p['ema_days'],
        )
        variants.append((f"{name}+USD", sig_usd))
    return variants


def maedai_full_variants(freq='4h'):
    """Teammate B の全戦略バリアントを返す"""
    variants = maedai_dc_variants(freq)
    variants.extend([
        ('MaedaiBest',       sig_maedai_best(freq=freq)),
        ('MaedaiBest+USD',   sig_maedai_best_usd(freq=freq)),
    ])
    return variants
