"""
メイン戦略: XAUUSD + Kelly(f=0.25) + ADX(14)>25 (v18 確定版)
=============================================================
v17 検証で最高性能が確認された戦略の最終定義。

バックテスト結果 (2023-10-06 ~ 2026-02-27, XAUUSD 4H, 初期資金¥5,000,000):
  Sharpe = 2.250
  Calmar = 11.534
  MDD%   = 16.1%
  WR%    = 61.0%
  Trades = 41
  最終資産 = ¥27,175,560 (+443.5%)

実行方法:
  python strategies/main_strategy.py                 # バックテスト実行
  python strategies/main_strategy.py --signal-only   # 最新シグナルのみ出力

モジュールとして使用:
  from strategies.main_strategy import make_signal_func, KELLY_FRACTION, ENGINE_CFG
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import KellyCriterionSizer

# ── 戦略パラメータ (確定値) ──────────────────────────────
SYMBOL         = 'XAUUSD'
GRANULARITY    = '4h'
KELLY_FRACTION = 0.25
ADX_PERIOD     = 14
ADX_THRESH     = 25.0

SIGNAL_PARAMS = dict(
    freq='4h', lookback_days=15, ema_days=200,
    confirm_bars=2, rsi_oversold=45,
)

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

# 確認済みバックテスト結果 (参照用)
CONFIRMED_RESULTS = dict(
    sharpe_ratio=2.250,
    calmar_ratio=11.534,
    max_drawdown_pct=16.1,
    win_rate_pct=61.0,
    total_trades=41,
    end_value=27_175_560,
    period='2023-10-06 ~ 2026-02-27',
    bars=3714,
)


# ── ADX 計算 ────────────────────────────────────────────

def compute_adx(h: pd.Series, l: pd.Series, c: pd.Series,
                p: int = ADX_PERIOD) -> pd.Series:
    """ADX(p) を計算して返す。"""
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


# ── シグナル関数 ─────────────────────────────────────────

def make_signal_func(adx_thresh: float = ADX_THRESH):
    """
    Union_4H_Base に ADX(14) > adx_thresh フィルターを重ねた
    メインシグナル関数を返す。

    Returns:
        callable: bars (pd.DataFrame) -> pd.Series[str]
    """
    _base = sig_maedai_yagami_union(**SIGNAL_PARAMS)

    def _signal(bars: pd.DataFrame) -> pd.Series:
        base = _base(bars)
        adx  = compute_adx(bars['high'], bars['low'], bars['close'], p=ADX_PERIOD)
        result = base.copy()
        result[adx < adx_thresh] = 'flat'
        return result

    _signal.__name__ = f'MainStrategy_XAUUSD_ADX{int(adx_thresh)}'
    return _signal


# ── データロード ─────────────────────────────────────────

def _find_ohlc_4h() -> str | None:
    pat = os.path.join(ROOT, 'data', 'ohlc', f'{SYMBOL}*4h.csv')
    matches = sorted(glob.glob(pat))
    return matches[-1] if matches else None


def _load_ohlc(path: str) -> pd.DataFrame:
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


def load_ohlc() -> pd.DataFrame:
    """XAUUSD 4H OHLCデータを返す。"""
    path = _find_ohlc_4h()
    if path is None:
        raise FileNotFoundError(
            f'XAUUSD 4H データが見つかりません: data/ohlc/{SYMBOL}*4h.csv\n'
            f'先に scripts/fetch_data.py を実行してください。'
        )
    return _load_ohlc(path)


# ── バックテスト実行 ─────────────────────────────────────

def run_backtest(verbose: bool = True) -> dict:
    """
    メイン戦略のバックテストを実行する。

    Returns:
        dict: BacktestEngine の結果辞書
    """
    from lib.backtest import BacktestEngine

    df = load_ohlc()
    if verbose:
        print(f'[{SYMBOL}] {len(df)} バー, '
              f'{df.index[0].date()} ~ {df.index[-1].date()}')

    adx_vals = compute_adx(df['high'], df['low'], df['close'])
    above    = (adx_vals > ADX_THRESH).sum()
    total_v  = adx_vals.notna().sum()
    if verbose:
        print(f'ADX({ADX_PERIOD}) 統計: mean={adx_vals.mean():.1f}, '
              f'median={adx_vals.median():.1f}, '
              f'>={ADX_THRESH}: {above}/{total_v} ({above/total_v*100:.1f}%)')

    engine = BacktestEngine(**ENGINE_CFG)

    # ── ベースラインで Kelly パラメータを推定 ──
    base_sig = sig_maedai_yagami_union(**SIGNAL_PARAMS)
    r_ref    = engine.run(data=df, signal_func=base_sig, freq='4h',
                          name='MainStrategy_ref_base')
    base_wr = r_ref.get('win_rate_pct', 50) / 100
    base_pf = r_ref.get('profit_factor', 2.0)

    kelly = KellyCriterionSizer(
        win_rate=base_wr,
        profit_factor=base_pf,
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    if verbose:
        print(f'Kelly乗数: {kelly.get_multiplier(0):.2f}x '
              f'(WR={base_wr*100:.1f}%, PF={base_pf:.3f})')

    # ── メイン戦略 ──
    sig_func = make_signal_func(adx_thresh=ADX_THRESH)
    strategy_name = f'XAUUSD+Kelly(f={KELLY_FRACTION})+ADX{int(ADX_THRESH)}'

    if verbose:
        print(f'\n実行中: {strategy_name}...', end='', flush=True)

    result = engine.run(data=df, signal_func=sig_func, freq='4h',
                        name=strategy_name, sizer=kelly)

    if verbose:
        print(' 完了')

    return result


# ── フォーマット出力 ─────────────────────────────────────

def _fmt(v, p=3):
    return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'


def print_results(result: dict) -> None:
    """バックテスト結果を整形して標準出力する。"""
    strategy_name = f'XAUUSD+Kelly(f={KELLY_FRACTION})+ADX{int(ADX_THRESH)}'
    print('\n' + '=' * 70)
    print(f'  メイン戦略バックテスト結果')
    print(f'  {strategy_name}')
    print('=' * 70)
    print(f'  Sharpe Ratio     : {_fmt(result.get("sharpe_ratio"))}')
    print(f'  Calmar Ratio     : {_fmt(result.get("calmar_ratio"))}')
    print(f'  Max Drawdown     : {_fmt(result.get("max_drawdown_pct"), 1)}%')
    print(f'  Profit Factor    : {_fmt(result.get("profit_factor"))}')
    print(f'  Win Rate         : {_fmt(result.get("win_rate_pct"), 1)}%')
    print(f'  Total Trades     : {result.get("total_trades", 0)}')
    end_v = result.get('end_value', 0) or 0
    init  = ENGINE_CFG['init_cash']
    print(f'  Final Equity     : ¥{end_v:,.0f} ({(end_v/init - 1)*100:+.1f}%)')
    print('=' * 70)

    # 承認条件チェック
    sh  = result.get('sharpe_ratio') or 0
    ca  = result.get('calmar_ratio') or 0
    mdd = result.get('max_drawdown_pct') or 99
    ok  = sh >= 1.5 and ca >= 5.0 and mdd <= 30
    status = '✅ 承認 (Sharpe≥1.5, Calmar≥5.0, MDD≤30%)' if ok else '❌ 要再検討'
    print(f'  承認ステータス   : {status}')
    print('=' * 70)


# ── シグナル確認 ─────────────────────────────────────────

def get_latest_signal(df: pd.DataFrame | None = None) -> dict:
    """
    最新バーのシグナルを返す。

    Args:
        df: OHLCデータ (None の場合はローカルCSVを使用)

    Returns:
        dict: {signal, price, bar_time, adx}
    """
    if df is None:
        df = load_ohlc()

    sig_func = make_signal_func()
    signals  = sig_func(df)
    adx_vals = compute_adx(df['high'], df['low'], df['close'])

    bar_time  = signals.index[-1]
    signal    = str(signals.iloc[-1]) if signals.iloc[-1] is not None else 'flat'
    price     = float(df['close'].iloc[-1])
    adx_val   = float(adx_vals.iloc[-1]) if adx_vals.iloc[-1] == adx_vals.iloc[-1] else None

    return {
        'signal':   signal,
        'price':    price,
        'bar_time': bar_time,
        'adx':      adx_val,
    }


# ── エントリーポイント ────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='メイン戦略バックテスト: XAUUSD+Kelly+ADX(>25)'
    )
    parser.add_argument('--signal-only', action='store_true',
                        help='バックテストを行わず最新シグナルのみ出力')
    parser.add_argument('--adx-thresh', type=float, default=ADX_THRESH,
                        help=f'ADX閾値 (デフォルト: {ADX_THRESH})')
    args = parser.parse_args()

    if args.signal_only:
        try:
            df  = load_ohlc()
            sig = get_latest_signal(df)
            adx_str = f'{sig["adx"]:.1f}' if sig['adx'] is not None else 'N/A'
            print(f'[最新シグナル] {sig["bar_time"]} | {SYMBOL} | '
                  f'{sig["signal"].upper()} | Price={sig["price"]:.2f} | '
                  f'ADX={adx_str}')
        except FileNotFoundError as e:
            print(f'エラー: {e}', file=sys.stderr)
            sys.exit(1)
        return

    try:
        result = run_backtest(verbose=True)
        print_results(result)
    except FileNotFoundError as e:
        print(f'エラー: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
