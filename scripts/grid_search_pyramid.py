"""
LivermorePyramidingSizer パラメータ・グリッドサーチ (v17 Task 1)
================================================================
XAUUSD + Union_4H_Base + KellyCriterionSizer(f=0.25) をベースに、
LivermorePyramidingSizer の step_pct / max_pyramids を探索する。

探索グリッド:
  step_pct   : [0.02, 0.03, 0.04]  (2%, 3%, 4%)
  max_pyramids: [1, 2]
  pyramid_ratios: [0.5, 0.3, 0.2] 固定

出力:
  results/pyramid_grid_search.csv — 全6パターンの結果
  results/performance_log.csv     — 自動追記 (BacktestEngine)

実行:
  python scripts/grid_search_pyramid.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glob
import csv
import datetime
import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import KellyCriterionSizer, LivermorePyramidingSizer

SYMBOL = 'XAUUSD'
KELLY_FRACTION = 0.25

ENGINE_CFG = dict(
    init_cash=5_000_000,
    risk_pct=0.05,
    default_sl_atr=2.0,
    default_tp_atr=4.0,
    pyramid_entries=0,
    target_max_dd=0.30,
    target_min_wr=0.30,
    target_rr_threshold=2.0,
    target_min_trades=5,
)

SIGNAL_PARAMS = dict(
    freq='4h', lookback_days=15, ema_days=200,
    confirm_bars=2, rsi_oversold=45,
)

# グリッド
STEP_PCTS    = [0.02, 0.03, 0.04]
MAX_PYRAMIDS = [1, 2]
PYRAMID_RATIOS = [0.5, 0.3, 0.2]

OUTPUT_CSV = os.path.join(ROOT, 'results', 'pyramid_grid_search.csv')


def find_ohlc_4h(symbol: str) -> str | None:
    pat = os.path.join(ROOT, 'data', 'ohlc', f'{symbol}*4h.csv')
    matches = sorted(glob.glob(pat))
    return matches[-1] if matches else None


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df.set_index('datetime').sort_index()[cols].astype(float)


def fmt(v, p=3):
    return '' if v is None or (isinstance(v, float) and v != v) else round(v, p)


def main():
    path = find_ohlc_4h(SYMBOL)
    if path is None:
        print(f'データなし: {SYMBOL}')
        sys.exit(1)

    df = load_ohlc(path)
    print(f'[{SYMBOL}] {len(df)} バー, {df.index[0].date()} ~ {df.index[-1].date()}')

    engine = BacktestEngine(**ENGINE_CFG)
    sig_func = sig_maedai_yagami_union(**SIGNAL_PARAMS)

    # ── ベースライン (Kelly のみ) ──
    print('\n[BASE] Union+Kelly(f=0.25)...', end='', flush=True)
    kelly_base = KellyCriterionSizer(
        kelly_fraction=KELLY_FRACTION,
        strategy_name='Union_XAUUSD_Base',
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    r_base = engine.run(data=df, signal_func=sig_func, freq='4h',
                        name='XAUUSD+Kelly(f=0.25)', sizer=kelly_base)
    base_wr  = r_base.get('win_rate_pct', 50) / 100
    base_pf  = r_base.get('profit_factor', 2.0)
    base_end = r_base.get('end_value', ENGINE_CFG['init_cash'])
    print(f' Sharpe={fmt(r_base.get("sharpe_ratio"),3)}, '
          f'MDD={fmt(r_base.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_base.get("total_trades",0)}, '
          f'¥{base_end:,.0f}')

    # ── グリッドサーチ ──
    combos = [(sp, mp) for sp in STEP_PCTS for mp in MAX_PYRAMIDS]
    print(f'\nグリッドサーチ: {len(combos)}組み合わせ (step_pct × max_pyramids)\n')

    header = (f"{'step_pct':>9} {'max_pyr':>7} | "
              f"{'Sharpe':>7} {'Calmar':>7} {'MDD%':>6} "
              f"{'PF':>6} {'WR%':>5} {'Trades':>6} | "
              f"{'最終資産':>12} {'pyramid%':>9}")
    print(header)
    print('-' * 90)

    grid_results = []

    for step_pct, max_pyr in combos:
        name = (f'XAUUSD+Kelly+LV(step={step_pct:.0%},pyr={max_pyr})')

        kelly = KellyCriterionSizer(
            win_rate=base_wr,
            profit_factor=base_pf,
            kelly_fraction=KELLY_FRACTION,
            base_risk_pct=ENGINE_CFG['risk_pct'],
        )
        livermore = LivermorePyramidingSizer(
            base_sizer=kelly,
            step_pct=step_pct,
            pyramid_ratios=PYRAMID_RATIOS,
            max_pyramids=max_pyr,
        )

        r = engine.run(data=df, signal_func=sig_func, freq='4h',
                       name=name, sizer=livermore)

        sh   = fmt(r.get('sharpe_ratio'), 3)
        ca   = fmt(r.get('calmar_ratio'), 3)
        mdd  = fmt(r.get('max_drawdown_pct'), 1)
        pf_  = fmt(r.get('profit_factor'), 3)
        wr   = fmt(r.get('win_rate_pct'), 1)
        tr   = r.get('total_trades', 0)
        end_v = r.get('end_value', 0) or 0

        trades = r.get('trades', [])
        pyr_trades = [t for t in trades if t.get('pyramid_layers', 1) > 1]
        pyr_pct = len(pyr_trades) / max(len(trades), 1) * 100

        print(f'{step_pct:>8.0%} {max_pyr:>7} | '
              f'{sh:>7} {ca:>7} {mdd:>6} '
              f'{pf_:>6} {wr:>5} {tr:>6} | '
              f'¥{end_v:>11,.0f} {pyr_pct:>8.1f}%')

        grid_results.append({
            'step_pct':     step_pct,
            'max_pyramids': max_pyr,
            'pyramid_ratios': str(PYRAMID_RATIOS),
            'sharpe_ratio': fmt(r.get('sharpe_ratio'), 4),
            'calmar_ratio': fmt(r.get('calmar_ratio'), 4),
            'max_drawdown_pct': fmt(r.get('max_drawdown_pct'), 2),
            'profit_factor': fmt(r.get('profit_factor'), 4),
            'win_rate_pct':  fmt(r.get('win_rate_pct'), 2),
            'total_trades':  tr,
            'end_value':     round(end_v),
            'pyramid_trade_pct': round(pyr_pct, 1),
            'strategy_name': name,
        })

    print('-' * 90)
    print(f'\n[ベースライン比較]')
    print(f'  Union+Kelly(f=0.25): Sharpe={fmt(r_base.get("sharpe_ratio"),3)}, '
          f'MDD={fmt(r_base.get("max_drawdown_pct"),1)}%, '
          f'¥{base_end:,.0f}')

    # ── 最優秀パラメータ ──
    valid = [r for r in grid_results if r['sharpe_ratio'] != '' and float(str(r['sharpe_ratio'])) > 0]
    if valid:
        best = max(valid, key=lambda r: float(str(r['sharpe_ratio'])))
        print(f'\n  ベストパラメータ:')
        print(f'    step_pct={best["step_pct"]:.0%}, max_pyramids={best["max_pyramids"]}')
        print(f'    Sharpe={best["sharpe_ratio"]}, Calmar={best["calmar_ratio"]}, '
              f'MDD={best["max_drawdown_pct"]}%')

    # ── CSV 保存 ──
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if grid_results:
        pd.DataFrame(grid_results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f'\n結果保存: {OUTPUT_CSV}')

    return grid_results, r_base


if __name__ == '__main__':
    main()
