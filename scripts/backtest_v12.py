"""
v12 統合バックテスト — 新エンジン比較
=======================================
Task 1: VolatilityAdjustedSizer
Task 2: KellyCriterionSizer
Task 3: MetaStrategy (HMM)

実行:
  python scripts/backtest_v12.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import VolatilityAdjustedSizer, KellyCriterionSizer
from lib.regime import HiddenMarkovRegimeDetector
from strategies.meta_strategy import make_meta_signal


# ──────────────────────────────────────────────────────
def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    return df.set_index('datetime').sort_index()[
        [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    ].astype(float)


def fmt(v, p=3):
    return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'


# ──────────────────────────────────────────────────────
def main():
    path_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    path_1d = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')

    df_4h = load_ohlc(path_4h)
    df_1d = load_ohlc(path_1d)
    print(f'データ: {df_4h.index[0].date()} ~ {df_4h.index[-1].date()}  ({len(df_4h)} bars 4H)')

    # ベースシグナル
    sig_union = sig_maedai_yagami_union(
        freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45,
    )

    # ── Task 1: VolatilityAdjustedSizer ──
    vol_sizer = VolatilityAdjustedSizer(atr_lookback=100, clip_min=0.25, clip_max=2.0)
    print(f'\nTask 1: {vol_sizer}')

    # ── Task 2: KellyCriterionSizer ──
    kelly_sizer = KellyCriterionSizer(kelly_fraction=0.25, strategy_name='Union_4H')
    print(f'Task 2: {kelly_sizer}')
    print(f'        → final risk_pct = {kelly_sizer.kelly_f:.4f} '
          f'(multiplier = {kelly_sizer.get_multiplier():.3f}x)')

    # ── Task 3: HMM レジーム ──
    print('\nTask 3: HMM学習中...')
    detector = HiddenMarkovRegimeDetector()
    detector.fit(df_1d['close'])
    regimes = detector.predict(df_1d['close'])
    stats = detector.regime_stats()
    for k, v in stats.items():
        print(f'  {k}: label={v["label"]}, vol={v["volatility"]:.5f}')
    trend_pct = (regimes == 1).mean() * 100
    print(f'  トレンド比率: {trend_pct:.1f}%')

    meta_sig = make_meta_signal(df_1d['close'])

    # ── エンジン共通設定 ──
    engine = BacktestEngine(
        init_cash=5_000_000,
        risk_pct=0.05,
        default_sl_atr=2.0,
        default_tp_atr=4.0,
        pyramid_entries=0,
        target_max_dd=0.30,
        target_min_wr=0.35,
        target_rr_threshold=2.0,
        target_min_trades=10,
    )

    variants = [
        ('Union_4H_Base',     sig_union, None),
        ('Union_4H+VolSizer', sig_union, vol_sizer),
        ('Union_4H+Kelly',    sig_union, kelly_sizer),
        ('MetaStrategy_4H',   meta_sig,  None),
    ]

    all_results = []
    for name, sig, sizer in variants:
        print(f'\n  実行: {name}...', end='', flush=True)
        r = engine.run(data=df_4h, signal_func=sig, freq='4h', name=name, sizer=sizer)
        all_results.append((name, r))
        pf     = fmt(r.get('profit_factor'))
        sh     = fmt(r.get('sharpe_ratio'))
        cal    = fmt(r.get('calmar_ratio'))
        mdd    = fmt(r.get('max_drawdown_pct'), 1)
        wr     = fmt(r.get('win_rate_pct'), 1)
        trades = r.get('total_trades', 0)
        print(f' done → Sharpe={sh}, PF={pf}, MDD={mdd}%, WR={wr}%, Trades={trades}, Calmar={cal}')

    # ── 結果表示 ──
    print('\n' + '=' * 80)
    print('  v12 新エンジン比較 バックテスト結果')
    print('  ベースライン: Union_4H_Base (v8実績 Sharpe 2.817)')
    print('=' * 80)
    header = (f"{'戦略':<28} {'PF':>7} {'WR%':>6} {'MDD%':>7}"
              f" {'Sharpe':>8} {'Calmar':>8} {'Trades':>7} {'判定':>8}")
    print(header)
    print('-' * 80)

    base_sharpe = None
    for name, r in all_results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') is not None else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') is not None else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        sh_val = r.get('sharpe_ratio')
        mdd_val = r.get('max_drawdown_pct', 100)
        pf_val  = r.get('profit_factor', 0)

        if 'Base' in name:
            base_sharpe = sh_val
            verdict = 'BASE'
        elif sh_val and mdd_val and pf_val:
            if sh_val >= 1.5 and mdd_val <= 30 and pf_val >= 1.5:
                verdict = 'PASS'
            else:
                verdict = 'CHECK'
        else:
            verdict = 'N/A'

        print(f'{name:<28} {pf:>7} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7} {verdict:>8}')

    print('=' * 80)

    if base_sharpe:
        print(f'\nベースライン Sharpe: {base_sharpe:.3f}')
        for name, r in all_results:
            if 'Base' not in name:
                sh = r.get('sharpe_ratio')
                if sh:
                    diff = sh - base_sharpe
                    sign = '+' if diff >= 0 else ''
                    print(f'  {name:<28} Sharpe {sh:.3f} ({sign}{diff:.3f})')

    # ── equity curves (トレードごと) を CSV 保存 ──
    curves_path = os.path.join(ROOT, 'results', 'v12_equity_curves.csv')
    os.makedirs(os.path.dirname(curves_path), exist_ok=True)

    curve_rows = []
    for name, r in all_results:
        trades_list = r.get('trades', [])
        cash = 5_000_000
        for t in trades_list:
            cash += t.get('pnl', 0)
            exit_t = t.get('exit_time', '')
            curve_rows.append({
                'strategy': name,
                'exit_time': str(exit_t)[:10],
                'equity': round(cash, 0),
                'pnl': round(t.get('pnl', 0), 0),
            })
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
        print(f'\n資産曲線データ保存: {curves_path}')

    return all_results


if __name__ == '__main__':
    main()
