"""
ADXフィルター有効性検証
========================
Union_4H_Base 戦略に ADX(14) > 25 フィルターを追加し、
フィルターなしと比較する。

対象商品: XAGUSD (データが2商品のうち、Unionと別の商品での検証)
比較:
  - Union_4H_XAGUSD       : フィルターなし (ベースライン)
  - Union_4H_XAGUSD+ADX25 : ADX(14) > 25 のバーのみでエントリー

ADXフィルターの意味:
  ADX > 25 → トレンド相場 → Union (トレンドフォロー) が有効
  ADX < 25 → レンジ相場  → エントリー見送り

実行:
  python scripts/backtest_adx_filter.py

出力:
  - 標準出力: 比較テーブル
  - results/performance_log.csv: 追記 (strategy_name='Union+ADXFilter')
"""
import os
import sys
import csv
import datetime
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union

ADX_PERIOD   = 14
ADX_THRESH   = 25
SYMBOL       = 'XAGUSD'
DATA_PATH    = os.path.join(ROOT, 'data', 'ohlc', 'XAGUSD_2025_4h.csv')

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


# ── ADX計算 ──────────────────────────────────────

def compute_adx(h: pd.Series, l: pd.Series, c: pd.Series,
                p: int = ADX_PERIOD) -> pd.Series:
    """ADX (Average Directional Index) を返す。"""
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


# ── ADXフィルター付きシグナルファクトリー ──────────

def make_union_adx_filtered(base_sig_func, adx_thresh: float = ADX_THRESH):
    """
    Union シグナルに ADX(14) > adx_thresh フィルターを重ねる。
    ADX条件を満たさないバーでは 'flat' に変換する。
    """
    def _f(bars: pd.DataFrame) -> pd.Series:
        base = base_sig_func(bars)
        adx  = compute_adx(bars['high'], bars['low'], bars['close'], p=ADX_PERIOD)

        # ADX < thresh のバーでシグナルをフラットに
        result = base.copy()
        mask_no_trend = adx < adx_thresh
        result[mask_no_trend] = 'flat'
        return result

    return _f


# ── データ読み込み ────────────────────────────────

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


# ── メイン ───────────────────────────────────────

def main():
    if not os.path.exists(DATA_PATH):
        print(f'データなし: {DATA_PATH}')
        sys.exit(1)

    df = load_ohlc(DATA_PATH)
    print(f'[{SYMBOL}] {len(df)} バー, {df.index[0].date()} ~ {df.index[-1].date()}')

    # ADX統計を事前確認
    adx_vals = compute_adx(df['high'], df['low'], df['close'], p=ADX_PERIOD)
    above_25 = (adx_vals > ADX_THRESH).sum()
    pct_above = above_25 / len(adx_vals[adx_vals.notna()]) * 100
    print(f'\nADX(14) > {ADX_THRESH}: {above_25}バー / {len(adx_vals[adx_vals.notna()])}バー ({pct_above:.1f}%)')
    print(f'ADX(14) 統計: mean={adx_vals.mean():.1f}, median={adx_vals.median():.1f}, '
          f'min={adx_vals.min():.1f}, max={adx_vals.max():.1f}')

    engine = BacktestEngine(**ENGINE_CFG)
    base_sig   = sig_maedai_yagami_union(**SIGNAL_PARAMS)
    adx_sig    = make_union_adx_filtered(base_sig, adx_thresh=ADX_THRESH)

    print(f'\nバックテスト実行中...')

    # ── ベースライン ──
    r_base = engine.run(data=df, signal_func=base_sig,
                        freq='4h', name=f'Union_{SYMBOL}_Base')

    # ── ADXフィルター付き ──
    r_adx  = engine.run(data=df, signal_func=adx_sig,
                        freq='4h', name=f'Union_{SYMBOL}+ADX{ADX_THRESH}')

    # ── 比較テーブル ──
    print('\n' + '=' * 80)
    print(f'  {SYMBOL} — Union vs Union+ADX({ADX_THRESH}) 比較')
    print('=' * 80)
    header = (f"{'戦略':<34} {'Sharpe':>8} {'Calmar':>8} {'MDD%':>7}"
              f" {'PF':>7} {'WR%':>6} {'Trades':>7}")
    print(header)
    print('-' * 80)

    for name, r in [
        (f'Union_{SYMBOL}_Base', r_base),
        (f'Union_{SYMBOL}+ADX{ADX_THRESH}', r_adx),
    ]:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<34} {sharpe:>8} {calmar:>8} {mdd:>7} {pf:>7} {wr:>6} {trades:>7}')

    print('=' * 80)

    # ADXフィルターの効果を分析
    base_tr  = r_base.get('total_trades', 0)
    adx_tr   = r_adx.get('total_trades', 0)
    base_sh  = r_base.get('sharpe_ratio', 0) or 0
    adx_sh   = r_adx.get('sharpe_ratio', 0)  or 0
    base_mdd = r_base.get('max_drawdown_pct', 0) or 0
    adx_mdd  = r_adx.get('max_drawdown_pct', 0)  or 0

    print(f'\n[ADXフィルター効果]')
    print(f'  トレード数: {base_tr} → {adx_tr} ({adx_tr-base_tr:+d}件)')
    print(f'  Sharpe   : {base_sh:.3f} → {adx_sh:.3f} ({adx_sh-base_sh:+.3f})')
    print(f'  MDD%     : {base_mdd:.1f}% → {adx_mdd:.1f}% ({adx_mdd-base_mdd:+.1f}pt)')

    verdict = 'ADXフィルター有効 (Sharpe改善)' if adx_sh > base_sh else \
              'ADXフィルター中立 (有意差なし)' if abs(adx_sh - base_sh) < 0.1 else \
              'ADXフィルター非推奨 (Sharpe低下)'
    print(f'  判定: {verdict}')

    # ── performance_log.csv 追記 ──
    log_path = os.path.join(ROOT, 'results', 'performance_log.csv')
    write_header = not os.path.exists(log_path)
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'strategy_name', 'parameters',
                             'timeframe', 'sharpe_ratio', 'profit_factor',
                             'max_drawdown', 'win_rate', 'trades'])
        for name, r in [
            (f'Union_{SYMBOL}_Base',    r_base),
            ('Union+ADXFilter',         r_adx),
        ]:
            writer.writerow([
                ts, name, f'v15_adx_filter_thresh{ADX_THRESH}',
                '4H',
                round(r.get('sharpe_ratio') or 0, 4),
                round(r.get('profit_factor') or 0, 4),
                round(r.get('max_drawdown_pct') or 0, 2),
                round(r.get('win_rate_pct') or 0, 2),
                r.get('total_trades', 0),
            ])
    print(f'\nperformance_log.csv 追記: {log_path}')

    return r_base, r_adx


if __name__ == '__main__':
    main()
