"""
v14 統合バックテスト
====================
Task 1: MetaStrategy v3.0 (5D HMM: log_return, abs_return, ATR, ADX, RSI)
         MetaStrategy v2.0 との比較グリッドサーチ

実行:
  python scripts/backtest_v14.py
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
from lib.regime import HiddenMarkovRegimeDetector
from strategies.meta_strategy import (
    make_meta_signal_v2, grid_search_meta,
    make_meta_signal_v3, grid_search_meta_v3,
)


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
    print(f'データ: {df_4h.index[0].date()} ~ {df_4h.index[-1].date()} ({len(df_4h)} 4H bars, {len(df_1d)} 1D bars)')

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
    sig_union = sig_maedai_yagami_union(
        freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45,
    )
    r = engine.run(data=df_4h, signal_func=sig_union, freq='4h', name='Union_4H_Base')
    all_results.append(('Union_4H_Base', r))
    print(f' Sharpe={fmt(r.get("sharpe_ratio"))}, MDD={fmt(r.get("max_drawdown_pct"),1)}%')

    # ── v2.0 MetaStrategy (2D HMM) ──
    print('\n[T1a] MetaStrategy v2.0 (2D HMM) グリッドサーチ...')
    det2 = HiddenMarkovRegimeDetector(n_states=3)
    det2.fit(df_1d['close'])
    dist2 = det2.regime_distribution(df_1d['close'])
    print(f'      2D HMM分布: range={dist2["range"]:.1%}, '
          f'low_trend={dist2["low_trend"]:.1%}, '
          f'high_trend={dist2["high_trend"]:.1%}')

    best_p2, best_r2, gs2 = grid_search_meta(df_4h, df_1d, engine, verbose=True)
    all_results.append(('MetaStrategy_v2_Best', best_r2))
    print(f'\n      v2.0 Best: Sharpe={fmt(best_r2.get("sharpe_ratio"))}, '
          f'MDD={fmt(best_r2.get("max_drawdown_pct"),1)}%, '
          f'Trades={best_r2.get("total_trades",0)}')
    if best_p2:
        yp, mp, up = best_p2
        print(f'      Params: maedai_lb={mp.get("lookback_days")}, union_rsi={up.get("rsi_oversold")}')

    # ── v3.0 MetaStrategy (5D HMM) ──
    print('\n[T1b] MetaStrategy v3.0 (5D HMM) グリッドサーチ...')
    det3 = HiddenMarkovRegimeDetector(n_states=3)
    det3.fit(df_1d['close'], ohlc_df=df_1d)
    dist3 = det3.regime_distribution(df_1d['close'], ohlc_df=df_1d)
    print(f'      5D HMM分布: range={dist3["range"]:.1%}, '
          f'low_trend={dist3["low_trend"]:.1%}, '
          f'high_trend={dist3["high_trend"]:.1%}')

    fstats = det3.feature_stats_by_regime(df_1d['close'], ohlc_df=df_1d)
    for regime_name, fst in fstats.items():
        if fst.get('count', 0) > 0:
            print(f'      [{regime_name}] count={fst["count"]}, '
                  f'avg_ATR={fst["avg_atr"]:.4f}, '
                  f'avg_ADX={fst["avg_adx"]:.1f}, '
                  f'avg_RSI={fst["avg_rsi"]:.1f}, '
                  f'mean_ret={fst["mean_ret"]:.4f}%')

    best_p3, best_r3, gs3 = grid_search_meta_v3(df_4h, df_1d, engine, verbose=True)
    all_results.append(('MetaStrategy_v3_Best', best_r3))
    print(f'\n      v3.0 Best: Sharpe={fmt(best_r3.get("sharpe_ratio"))}, '
          f'MDD={fmt(best_r3.get("max_drawdown_pct"),1)}%, '
          f'Trades={best_r3.get("total_trades",0)}')
    if best_p3:
        yp, mp, up = best_p3
        print(f'      Params: maedai_lb={mp.get("lookback_days")}, union_rsi={up.get("rsi_oversold")}')

    # ── 結果テーブル ──
    print('\n' + '=' * 90)
    print('  v14 MetaStrategy v2.0 vs v3.0 (5D HMM) バックテスト結果 (初期資金 500万円)')
    print('=' * 90)
    header = (f"{'戦略':<32} {'PF':>7} {'WR%':>6} {'MDD%':>7}"
              f" {'Sharpe':>8} {'Calmar':>8} {'Trades':>7}")
    print(header)
    print('-' * 90)

    for name, r in all_results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<32} {pf:>7} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7}')

    print('=' * 90)

    # ── performance_log.csv 追記 ──
    import csv, datetime
    log_path = os.path.join(ROOT, 'results', 'performance_log.csv')
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'strategy_name', 'parameters',
                             'timeframe', 'sharpe_ratio', 'profit_factor',
                             'max_drawdown', 'win_rate', 'trades'])
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for name, r in all_results:
            writer.writerow([
                ts, name, 'v14',
                '4H',
                round(r.get('sharpe_ratio') or 0, 4),
                round(r.get('profit_factor') or 0, 4),
                round(r.get('max_drawdown_pct') or 0, 2),
                round(r.get('win_rate_pct') or 0, 2),
                r.get('total_trades', 0),
            ])
    print(f'\nperformance_log.csv 更新: {log_path}')

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
        out_path = os.path.join(ROOT, 'results', 'v14_equity_curves.csv')
        pd.DataFrame(curve_rows).to_csv(out_path, index=False)
        print(f'資産曲線保存: {out_path}')

    # ── 5D HMM レジームデータ保存 ──
    reg_series = det3.predict(df_1d['close'], ohlc_df=df_1d)
    h, l, c = df_1d['high'], df_1d['low'], df_1d['close']
    from lib.regime import _compute_atr, _compute_adx, _compute_rsi
    atr_s = _compute_atr(h, l, c, p=14)
    adx_s = _compute_adx(h, l, c, p=14)
    rsi_s = _compute_rsi(c, p=14)
    reg_df = pd.DataFrame({
        'date':   df_1d.index,
        'regime': reg_series.values,
        'close':  c.values,
        'atr':    atr_s.values,
        'adx':    adx_s.values,
        'rsi':    rsi_s.values,
    })
    reg_path = os.path.join(ROOT, 'results', 'v14_hmm_regimes.csv')
    reg_df.to_csv(reg_path, index=False)
    print(f'5D HMMレジームデータ保存: {reg_path}')

    return all_results, dist2, dist3, fstats, best_p2, best_p3


if __name__ == '__main__':
    main()
