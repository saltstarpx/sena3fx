"""
XAUUSD + KellyCriterionSizer(f=0.25) バックテスト (v16 Task 1)
===============================================================
XAUUSDにKellyサイジングを適用した場合のパフォーマンスを確認する。

比較:
  - Union_XAUUSD_Base   : フィルターなし（ベースライン）
  - XAUUSD+Kelly(f=0.25): KellyCriterionSizer(f=0.25) 適用

結果は performance_log.csv に記録される。

実行:
  python scripts/backtest_xau_kelly.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glob
import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import KellyCriterionSizer

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
    return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'


def main():
    path = find_ohlc_4h(SYMBOL)
    if path is None:
        print(f'データなし: {SYMBOL}')
        sys.exit(1)

    df = load_ohlc(path)
    print(f'[{SYMBOL}] {len(df)} バー, {df.index[0].date()} ~ {df.index[-1].date()}')

    engine = BacktestEngine(**ENGINE_CFG)
    sig_func = sig_maedai_yagami_union(**SIGNAL_PARAMS)

    # ── ベースライン (Kelly なし) ──
    print('\nバックテスト実行: Union_XAUUSD_Base...', end='', flush=True)
    r_base = engine.run(data=df, signal_func=sig_func, freq='4h',
                        name='Union_XAUUSD_Base')
    print(f' Sharpe={fmt(r_base.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_base.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_base.get("total_trades",0)}')

    # ── Kelly(f=0.25) ──
    kelly = KellyCriterionSizer(
        win_rate=r_base.get('win_rate_pct', 50) / 100,
        profit_factor=r_base.get('profit_factor', 2.0),
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    print(f'\nKellySizer: WR={kelly._win_rate:.1%}, PF={kelly._profit_factor:.3f}, '
          f'f*={kelly._kelly_f:.4f}, 乗数={kelly.get_multiplier(0):.2f}x')

    print(f'バックテスト実行: XAUUSD+Kelly(f={KELLY_FRACTION})...', end='', flush=True)
    r_kelly = engine.run(data=df, signal_func=sig_func, freq='4h',
                         name=f'XAUUSD+Kelly(f={KELLY_FRACTION})',
                         sizer=kelly)
    print(f' Sharpe={fmt(r_kelly.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_kelly.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_kelly.get("total_trades",0)}')

    # ── 比較テーブル ──
    print('\n' + '=' * 80)
    print(f'  {SYMBOL} — Union vs Union+Kelly(f={KELLY_FRACTION}) 比較')
    print('=' * 80)
    header = (f"{'戦略':<36} {'Sharpe':>8} {'Calmar':>8} {'MDD%':>7}"
              f" {'PF':>7} {'WR%':>6} {'Trades':>7} {'最終資産':>10}")
    print(header)
    print('-' * 80)

    for name, r in [
        ('Union_XAUUSD_Base', r_base),
        (f'XAUUSD+Kelly(f={KELLY_FRACTION})', r_kelly),
    ]:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        end_v  = f"¥{r.get('end_value',0):,.0f}" if r.get('end_value') else 'N/A'
        print(f'{name:<36} {sharpe:>8} {calmar:>8} {mdd:>7} {pf:>7} {wr:>6} {trades:>7} {end_v:>10}')

    print('=' * 80)
    print(f'\nperformance_log.csv に記録済み (BacktestEngine 自動追記)')

    return r_base, r_kelly


if __name__ == '__main__':
    main()
