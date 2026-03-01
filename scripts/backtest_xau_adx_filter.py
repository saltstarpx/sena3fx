"""
XAUUSD + Kelly(f=0.25) ADXフィルター有効性検証 (v17 Task 2)
============================================================
XAGUSDでは非推奨だったADX(>25)フィルターが
XAUUSDで有効か否かを最終判断する。

比較:
  A: XAUUSD+Kelly(f=0.25)          — フィルターなし (v16 Task1と同じ)
  B: XAUUSD+Kelly(f=0.25)+ADX(>25) — ADX(14) > 25 のバーのみエントリー

実行:
  python scripts/backtest_xau_adx_filter.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glob
import numpy as np
import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import KellyCriterionSizer

SYMBOL = 'XAUUSD'
KELLY_FRACTION = 0.25
ADX_PERIOD = 14
ADX_THRESH = 25

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


def compute_adx(h: pd.Series, l: pd.Series, c: pd.Series,
                p: int = ADX_PERIOD) -> pd.Series:
    up   = h.diff()
    down = -l.diff()
    dm_plus  = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))],
                   axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    di_plus  = 100 * pd.Series(dm_plus,  index=h.index).ewm(
        alpha=1 / p, min_periods=p, adjust=False).mean() / atr_s
    di_minus = 100 * pd.Series(dm_minus, index=h.index).ewm(
        alpha=1 / p, min_periods=p, adjust=False).mean() / atr_s
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()


def make_kelly_adx_signal(base_sig_func, adx_thresh: float = ADX_THRESH):
    """Union シグナルに ADX(14) > adx_thresh フィルターを重ねる。"""
    def _f(bars: pd.DataFrame) -> pd.Series:
        base = base_sig_func(bars)
        adx  = compute_adx(bars['high'], bars['low'], bars['close'], p=ADX_PERIOD)
        result = base.copy()
        result[adx < adx_thresh] = 'flat'
        return result
    return _f


def fmt(v, p=3):
    return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'


def main():
    path = find_ohlc_4h(SYMBOL)
    if path is None:
        print(f'データなし: {SYMBOL}')
        sys.exit(1)

    df = load_ohlc(path)
    print(f'[{SYMBOL}] {len(df)} バー, {df.index[0].date()} ~ {df.index[-1].date()}')

    adx_vals = compute_adx(df['high'], df['low'], df['close'], p=ADX_PERIOD)
    above_25 = (adx_vals > ADX_THRESH).sum()
    total_valid = adx_vals.notna().sum()
    print(f'ADX(14) 統計: mean={adx_vals.mean():.1f}, median={adx_vals.median():.1f}, '
          f'> {ADX_THRESH}: {above_25}/{total_valid} ({above_25/total_valid*100:.1f}%)')

    engine = BacktestEngine(**ENGINE_CFG)
    base_sig = sig_maedai_yagami_union(**SIGNAL_PARAMS)
    adx_sig  = make_kelly_adx_signal(base_sig, adx_thresh=ADX_THRESH)

    # ── ケリー乗数 (ベースラインWR/PFを仮用) ──
    # 先にベースランを走らせてWR/PFを取得する
    print('\n[A] Union_XAUUSD_Base (Kelly なし)...', end='', flush=True)
    r_raw = engine.run(data=df, signal_func=base_sig, freq='4h',
                       name='Union_XAUUSD_Base_ref')
    print(f' WR={fmt(r_raw.get("win_rate_pct"),1)}%, PF={fmt(r_raw.get("profit_factor"))}')

    kelly_a = KellyCriterionSizer(
        win_rate=r_raw.get('win_rate_pct', 50) / 100,
        profit_factor=r_raw.get('profit_factor', 2.0),
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    kelly_b = KellyCriterionSizer(
        win_rate=r_raw.get('win_rate_pct', 50) / 100,
        profit_factor=r_raw.get('profit_factor', 2.0),
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )

    print(f'Kelly乗数: {kelly_a.get_multiplier(0):.2f}x')

    # ── パターン A: Kelly のみ ──
    name_a = f'XAUUSD+Kelly(f={KELLY_FRACTION})'
    print(f'\n[A] {name_a}...', end='', flush=True)
    r_a = engine.run(data=df, signal_func=base_sig, freq='4h',
                     name=name_a, sizer=kelly_a)
    print(f' Sharpe={fmt(r_a.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_a.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_a.get("total_trades",0)}')

    # ── パターン B: Kelly + ADXフィルター ──
    name_b = f'XAUUSD+Kelly(f={KELLY_FRACTION})+ADX{ADX_THRESH}'
    print(f'[B] {name_b}...', end='', flush=True)
    r_b = engine.run(data=df, signal_func=adx_sig, freq='4h',
                     name=name_b, sizer=kelly_b)
    print(f' Sharpe={fmt(r_b.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_b.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_b.get("total_trades",0)}')

    # ── 比較テーブル ──
    print('\n' + '=' * 90)
    print(f'  {SYMBOL} — Kelly vs Kelly+ADX({ADX_THRESH}) 最終比較')
    print('=' * 90)
    header = (f"{'戦略':<42} {'Sharpe':>7} {'Calmar':>7} {'MDD%':>6}"
              f" {'PF':>6} {'WR%':>5} {'Trades':>6} {'最終資産':>12}")
    print(header)
    print('-' * 90)

    for name, r in [(name_a, r_a), (name_b, r_b)]:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1)
        mdd    = fmt(r.get('max_drawdown_pct'), 1)
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        end_v  = f"¥{r.get('end_value',0):,.0f}" if r.get('end_value') else 'N/A'
        print(f'{name:<42} {sharpe:>7} {calmar:>7} {mdd:>6} '
              f'{pf:>6} {wr:>5} {trades:>6} {end_v:>12}')

    print('=' * 90)

    sh_a  = r_a.get('sharpe_ratio') or 0
    sh_b  = r_b.get('sharpe_ratio') or 0
    mdd_a = r_a.get('max_drawdown_pct') or 0
    mdd_b = r_b.get('max_drawdown_pct') or 0
    tr_a  = r_a.get('total_trades', 0)
    tr_b  = r_b.get('total_trades', 0)

    print(f'\n[ADXフィルター効果 in XAUUSD]')
    print(f'  Sharpe   : {sh_a:.3f} → {sh_b:.3f} ({sh_b-sh_a:+.3f})')
    print(f'  MDD%     : {mdd_a:.1f}% → {mdd_b:.1f}% ({mdd_b-mdd_a:+.1f}pt)')
    print(f'  Trades   : {tr_a} → {tr_b} ({tr_b-tr_a:+d}件)')

    verdict = ('ADXフィルター有効 (Sharpe改善)' if sh_b > sh_a
               else 'ADXフィルター中立' if abs(sh_b - sh_a) < 0.1
               else 'ADXフィルター非推奨 (Sharpe低下)')
    print(f'  判定: {verdict}')
    print(f'\nperformance_log.csv に記録済み (BacktestEngine 自動追記)')

    return r_a, r_b


if __name__ == '__main__':
    main()
