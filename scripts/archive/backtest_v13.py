"""
v13 統合バックテスト
====================
Task 1: HybridKellySizer (kelly_fraction=0.5)
Task 2: MetaStrategy v2.0 (3状態HMM + グリッドサーチ)
Task 3: VolSizer + disable_atr_sl

実行:
  python scripts/backtest_v13.py
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
from lib.risk_manager import (
    VolatilityAdjustedSizer, KellyCriterionSizer, HybridKellySizer
)
from lib.regime import HiddenMarkovRegimeDetector
from strategies.meta_strategy import make_meta_signal_v2, grid_search_meta


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


def main():
    path_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    path_1d = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')
    df_4h = load_ohlc(path_4h)
    df_1d = load_ohlc(path_1d)
    print(f'データ: {df_4h.index[0].date()} ~ {df_4h.index[-1].date()} ({len(df_4h)} 4H bars)')

    sig_union = sig_maedai_yagami_union(
        freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45,
    )

    engine = BacktestEngine(
        init_cash=5_000_000,
        risk_pct=0.05,
        default_sl_atr=2.0,
        default_tp_atr=4.0,
        pyramid_entries=0,
        target_max_dd=0.30,
        target_min_wr=0.30,
        target_min_trades=5,
    )

    all_results = []

    # ── ベースライン ──
    print('\n[BASE] Union_4H...', end='', flush=True)
    r = engine.run(data=df_4h, signal_func=sig_union, freq='4h', name='Union_4H_Base')
    all_results.append(('Union_4H_Base', r))
    print(f' Sharpe={fmt(r.get("sharpe_ratio"))}, MDD={fmt(r.get("max_drawdown_pct"),1)}%')

    # ── Task 1a: Kelly f=0.25 (v12再掲) ──
    kelly_025 = KellyCriterionSizer(kelly_fraction=0.25, strategy_name='Union_4H_Base')
    print(f'\n[T1a] Union+Kelly(f=0.25): {kelly_025}')
    r = engine.run(data=df_4h, signal_func=sig_union, freq='4h',
                   name='Union+Kelly_0.25', sizer=kelly_025)
    all_results.append(('Union+Kelly_0.25', r))
    print(f'      Sharpe={fmt(r.get("sharpe_ratio"))}, MDD={fmt(r.get("max_drawdown_pct"),1)}%, '
          f'Calmar={fmt(r.get("calmar_ratio"))}')

    # ── Task 1b: HybridKelly f=0.5 (v13新設) ──
    hybrid = HybridKellySizer(kelly_fraction=0.5, strategy_name='Union_4H_Base')
    print(f'\n[T1b] Union+HybridKelly(f=0.5): kelly_f={hybrid._kelly.kelly_f:.4f}')
    r = engine.run(data=df_4h, signal_func=sig_union, freq='4h',
                   name='Union+HybridKelly_0.5', sizer=hybrid)
    all_results.append(('Union+HybridKelly_0.5', r))
    print(f'      Sharpe={fmt(r.get("sharpe_ratio"))}, MDD={fmt(r.get("max_drawdown_pct"),1)}%, '
          f'Calmar={fmt(r.get("calmar_ratio"))}')

    # ── Task 3: VolSizer + disable_atr_sl ──
    vol_sizer = VolatilityAdjustedSizer(atr_lookback=100, clip_min=0.25, clip_max=2.0)
    print(f'\n[T3] Union+VolSizer(disable_atr_sl=True)...')
    r = engine.run(data=df_4h, signal_func=sig_union, freq='4h',
                   name='Union+VolSizer_NoATRSL', sizer=vol_sizer, disable_atr_sl=True)
    all_results.append(('Union+VolSizer_NoATRSL', r))
    print(f'     Sharpe={fmt(r.get("sharpe_ratio"))}, MDD={fmt(r.get("max_drawdown_pct"),1)}%')

    # ── Task 2: MetaStrategy v2.0 グリッドサーチ ──
    print('\n[T2] MetaStrategy v2.0 グリッドサーチ...')

    # 3状態HMM レジーム分布確認
    det = HiddenMarkovRegimeDetector(n_states=3)
    det.fit(df_1d['close'])
    dist = det.regime_distribution(df_1d['close'])
    print(f'     HMM分布: range={dist["range"]:.1%}, '
          f'low_trend={dist["low_trend"]:.1%}, '
          f'high_trend={dist["high_trend"]:.1%}')

    best_params, best_r, gs_all = grid_search_meta(df_4h, df_1d, engine, verbose=True)
    all_results.append(('MetaStrategy_v2_Best', best_r))
    print(f'\n     ベスト Sharpe={fmt(best_r.get("sharpe_ratio"))}, '
          f'MDD={fmt(best_r.get("max_drawdown_pct"),1)}%, '
          f'Trades={best_r.get("total_trades",0)}')
    if best_params:
        yp, mp, up = best_params
        print(f'     ベストParams: yagami={yp}, maedai_lb={mp.get("lookback_days")}, '
              f'union_rsi={up.get("rsi_oversold")}')

    # ── 結果テーブル ──
    print('\n' + '=' * 84)
    print('  v13 新エンジン比較 バックテスト結果 (初期資金 500万円)')
    print('=' * 84)
    header = (f"{'戦略':<30} {'PF':>7} {'WR%':>6} {'MDD%':>7}"
              f" {'Sharpe':>8} {'Calmar':>8} {'Trades':>7}")
    print(header)
    print('-' * 84)

    for name, r in all_results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<30} {pf:>7} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7}')

    print('=' * 84)

    # ── 資産曲線保存 ──
    curve_rows = []
    for name, r in all_results:
        cash = 5_000_000
        for t in r.get('trades', []):
            cash += t.get('pnl', 0)
            curve_rows.append({
                'strategy': name,
                'exit_time': str(t.get('exit_time', ''))[:10],
                'equity': round(cash, 0),
                'pnl': round(t.get('pnl', 0), 0),
            })
    if curve_rows:
        out_path = os.path.join(ROOT, 'results', 'v13_equity_curves.csv')
        pd.DataFrame(curve_rows).to_csv(out_path, index=False)
        print(f'\n資産曲線保存: {out_path}')

    # ── HMM レジーム分布保存 ──
    reg_series = det.predict(df_1d['close'])
    reg_df = pd.DataFrame({
        'date': df_1d.index,
        'regime': reg_series.values,
        'close': df_1d['close'].values,
    })
    reg_path = os.path.join(ROOT, 'results', 'v13_hmm_regimes.csv')
    reg_df.to_csv(reg_path, index=False)
    print(f'HMMレジームデータ保存: {reg_path}')

    return all_results, dist, best_params


if __name__ == '__main__':
    main()
