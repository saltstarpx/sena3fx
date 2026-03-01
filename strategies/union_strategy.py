"""
Union戦略 — 単独実行可能スクリプト
====================================
Maedai(DCブレイク) OR Yagami(RSI押し目) の複合シグナル。

実行方法:
  python strategies/union_strategy.py

評価基準: Sharpe Ratio > 1.5, Calmar Ratio > 3.0

v8実績: MaedaiUnion_4H で Sharpe 1.859 を記録。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from strategies.market_filters import make_usd_filtered_signal, SEASON_ALL

# ──────────────────────────────────────────────────────
#  シグナル定義
# ──────────────────────────────────────────────────────

# 素のUnion (v8実績: Sharpe 1.859)
sig_union = sig_maedai_yagami_union(
    freq='4h',
    lookback_days=15,
    ema_days=200,
    confirm_bars=2,
    rsi_oversold=45,
)

# USD強弱フィルター付き
sig_union_usd = make_usd_filtered_signal(
    sig_maedai_yagami_union, threshold=75
)(freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45)


# ──────────────────────────────────────────────────────
#  データ読み込み
# ──────────────────────────────────────────────────────

def load_ohlc(path):
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols].astype(float)


def fmt(v, prec=3):
    if v is None or (isinstance(v, float) and v != v):
        return 'N/A'
    return f'{v:.{prec}f}'


# ──────────────────────────────────────────────────────
#  バックテスト実行
# ──────────────────────────────────────────────────────

def run_backtest():
    ohlc_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    if not os.path.exists(ohlc_4h):
        ohlc_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_4h.csv')

    print(f'Loading: {ohlc_4h}')
    df = load_ohlc(ohlc_4h)
    print(f'Bars: {len(df)}, {df.index[0].date()} ~ {df.index[-1].date()}')

    engine = BacktestEngine(
        init_cash=5_000_000,
        risk_pct=0.05,
        default_sl_atr=2.0,
        default_tp_atr=4.0,
        pyramid_entries=0,
        target_max_dd=0.30,
        target_min_wr=0.35,
        target_rr_threshold=2.0,
        target_min_trades=20,
    )

    variants = [
        ('Union_4H (素)',    sig_union,     SEASON_ALL),
        ('Union_4H+USD',    sig_union_usd, SEASON_ALL),
    ]

    results = []
    for name, sig, months in variants:
        r = engine.run(
            data=df,
            signal_func=sig,
            freq='4h',
            name=name,
            allowed_months=months,
        )
        results.append((name, r))

    # ── 結果表示 ──
    print('\n' + '=' * 70)
    print('  Union戦略 バックテスト結果')
    print('  v8実績 Sharpe 1.859 の再現確認')
    print('=' * 70)
    header = f"{'戦略':<25} {'PF':>6} {'WR%':>6} {'MDD%':>7} {'Sharpe':>8} {'Calmar':>8} {'Trades':>7}"
    print(header)
    print('-' * 70)

    for name, r in results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') is not None else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') is not None else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<25} {pf:>6} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7}')

    print('=' * 70)

    # Sharpe 確認
    union_sharpe = None
    for name, r in results:
        if '素' in name or 'USD' not in name:
            union_sharpe = r.get('sharpe_ratio')
            break

    if union_sharpe is not None:
        target = 1.5
        status = 'PASS' if union_sharpe >= target else 'FAIL'
        print(f'\nSharpe確認: {union_sharpe:.3f} (目標 > {target}) → {status}')
        if union_sharpe >= 1.5:
            print(f'  → v8実績 (1.859) 対比: {union_sharpe:.3f}')

    return results


if __name__ == '__main__':
    run_backtest()
