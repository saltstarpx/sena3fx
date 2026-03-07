"""
MetaStrategy v3.0 — 5D HMMレジーム連動型戦略セレクター
=========================================================
v1.0: 2状態 (range→YagamiA / trend→Union)
v2.0: 3状態 + レジーム別パラメータ最適化 — v13新設
v3.0: 5D HMM (log_return, abs_return, ATR, ADX, RSI) — v14新設

レジーム → 戦略マッピング:
  0: range      → YagamiA_4H  (RSI逆張り、閾値を厳しく)
  1: low_trend  → Maedai_DC   (DC期間短め=感度高め)
  2: high_trend → Union_4H    (強トレンドフォロー)

実行方法:
  python strategies/meta_strategy.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from lib.backtest import BacktestEngine
from lib.regime import HiddenMarkovRegimeDetector
from lib.yagami import sig_yagami_A, sig_maedai_yagami_union
from strategies.maedai_breakout import sig_maedai_dc_ema_tf


# ──────────────────────────────────────────────────────
#  レジーム別パラメータグリッド (簡易グリッドサーチ用)
# ──────────────────────────────────────────────────────

# YagamiA (range) — RSI閾値を厳しめに
YAGAMI_GRID = [
    {'freq': '4h'},          # デフォルト
]

# Maedai DC (low_trend) — lookback短め→感度高め
MAEDAI_GRID = [
    {'freq': '4h', 'lookback_days': 20, 'ema_days': 200, 'confirm_bars': 1},
    {'freq': '4h', 'lookback_days': 10, 'ema_days': 200, 'confirm_bars': 1},
    {'freq': '4h', 'lookback_days': 15, 'ema_days': 200, 'confirm_bars': 1},
]

# Union (high_trend) — v12実績パラメータ
UNION_GRID = [
    {'freq': '4h', 'lookback_days': 15, 'ema_days': 200, 'confirm_bars': 2, 'rsi_oversold': 45},
]


# ──────────────────────────────────────────────────────
#  シグナル生成ヘルパー
# ──────────────────────────────────────────────────────

def _build_yagami(params: dict):
    return sig_yagami_A(freq=params.get('freq', '4h'))


def _build_maedai(params: dict):
    return sig_maedai_dc_ema_tf(
        freq=params.get('freq', '4h'),
        lookback_days=params.get('lookback_days', 15),
        ema_days=params.get('ema_days', 200),
        confirm_bars=params.get('confirm_bars', 1),
    )


def _build_union(params: dict):
    return sig_maedai_yagami_union(
        freq=params.get('freq', '4h'),
        lookback_days=params.get('lookback_days', 15),
        ema_days=params.get('ema_days', 200),
        confirm_bars=params.get('confirm_bars', 2),
        rsi_oversold=params.get('rsi_oversold', 45),
    )


# ──────────────────────────────────────────────────────
#  MetaStrategy v2.0 シグナル生成
# ──────────────────────────────────────────────────────

def make_meta_signal_v2(daily_close: pd.Series,
                        yagami_params: dict | None = None,
                        maedai_params: dict | None = None,
                        union_params:  dict | None = None) -> callable:
    """
    3状態HMMレジームに基づくシグナル生成関数を返す。

    Args:
        daily_close  : 日足終値 (HMM学習用)
        yagami_params: レンジ時の YagamiA パラメータ
        maedai_params: 低ボラトレンド時の Maedai パラメータ
        union_params : 高ボラトレンド時の Union パラメータ
    """
    yp = yagami_params or YAGAMI_GRID[0]
    mp = maedai_params or MAEDAI_GRID[0]
    up = union_params  or UNION_GRID[0]

    detector = HiddenMarkovRegimeDetector(n_states=3)

    def _signal(bars: pd.DataFrame) -> pd.Series:
        try:
            detector.fit(daily_close)
            regimes = detector.predict(daily_close)
        except Exception:
            regimes = pd.Series(2, index=daily_close.index)  # フォールバック: high_trend

        # 日付インデックスに正規化
        regime_daily = regimes.copy()
        regime_daily.index = pd.DatetimeIndex(regime_daily.index).normalize()

        sig_range  = _build_yagami(yp)(bars)
        sig_low    = _build_maedai(mp)(bars)
        sig_high   = _build_union(up)(bars)

        result = pd.Series('flat', index=bars.index, dtype=object)
        for idx in bars.index:
            day = pd.Timestamp(idx).normalize()
            try:
                regime = int(regime_daily.get(day, regime_daily.iloc[-1]))
            except Exception:
                regime = 2

            if regime == 0:
                v = sig_range.get(idx, 'flat')
            elif regime == 1:
                v = sig_low.get(idx, 'flat')
            else:
                v = sig_high.get(idx, 'flat')

            result[idx] = v if v is not None else 'flat'

        return result

    return _signal


# ──────────────────────────────────────────────────────
#  グリッドサーチ
# ──────────────────────────────────────────────────────

def grid_search_meta(df_4h: pd.DataFrame, df_1d: pd.DataFrame,
                     engine: BacktestEngine, verbose: bool = True) -> tuple:
    """
    各レジームのパラメータグリッドを探索して最良の組み合わせを返す。

    Returns:
        (best_params, best_result, all_results)
    """
    best_sharpe = -np.inf
    best_params = None
    best_result = None
    all_results = []

    combos = [
        (yp, mp, up)
        for yp in YAGAMI_GRID
        for mp in MAEDAI_GRID
        for up in UNION_GRID
    ]

    if verbose:
        print(f'グリッドサーチ: {len(combos)}組み合わせ')

    for i, (yp, mp, up) in enumerate(combos):
        name = (f"Meta_y{yp.get('freq','4h')}"
                f"_m{mp.get('lookback_days',15)}"
                f"_u{up.get('rsi_oversold',45)}")
        sig = make_meta_signal_v2(df_1d['close'], yp, mp, up)
        r = engine.run(data=df_4h, signal_func=sig, freq='4h', name=name)
        sh = r.get('sharpe_ratio') or -9999
        all_results.append((name, r, yp, mp, up))

        if verbose:
            pf  = f"{r.get('profit_factor', 0):.3f}" if r.get('profit_factor') else 'N/A'
            mdd = f"{r.get('max_drawdown_pct', 0):.1f}" if r.get('max_drawdown_pct') else 'N/A'
            print(f"  [{i+1}/{len(combos)}] {name:<40} "
                  f"Sharpe={sh:.3f}, PF={pf}, MDD={mdd}%, "
                  f"Trades={r.get('total_trades',0)}")

        if sh > best_sharpe:
            best_sharpe = sh
            best_params = (yp, mp, up)
            best_result = r

    return best_params, best_result, all_results


# ──────────────────────────────────────────────────────
#  MetaStrategy v3.0 — 5D HMM シグナル生成
# ──────────────────────────────────────────────────────

def make_meta_signal_v3(daily_close: pd.Series,
                        daily_ohlc: pd.DataFrame,
                        yagami_params: dict | None = None,
                        maedai_params: dict | None = None,
                        union_params:  dict | None = None) -> callable:
    """
    5D HMM (log_return, abs_return, ATR, ADX, RSI) レジームに基づく
    シグナル生成関数を返す。

    Args:
        daily_close : 日足終値 (HMM学習用)
        daily_ohlc  : 日足OHLC DataFrame (ATR/ADX/RSI計算用)
        yagami_params: レンジ時の YagamiA パラメータ
        maedai_params: 低ボラトレンド時の Maedai パラメータ
        union_params : 高ボラトレンド時の Union パラメータ
    """
    yp = yagami_params or YAGAMI_GRID[0]
    mp = maedai_params or MAEDAI_GRID[0]
    up = union_params  or UNION_GRID[0]

    detector = HiddenMarkovRegimeDetector(n_states=3)

    def _signal(bars: pd.DataFrame) -> pd.Series:
        try:
            detector.fit(daily_close, ohlc_df=daily_ohlc)
            regimes = detector.predict(daily_close, ohlc_df=daily_ohlc)
        except Exception:
            regimes = pd.Series(2, index=daily_close.index)

        regime_daily = regimes.copy()
        regime_daily.index = pd.DatetimeIndex(regime_daily.index).normalize()

        sig_range = _build_yagami(yp)(bars)
        sig_low   = _build_maedai(mp)(bars)
        sig_high  = _build_union(up)(bars)

        result = pd.Series('flat', index=bars.index, dtype=object)
        for idx in bars.index:
            day = pd.Timestamp(idx).normalize()
            try:
                regime = int(regime_daily.get(day, regime_daily.iloc[-1]))
            except Exception:
                regime = 2

            if regime == 0:
                v = sig_range.get(idx, 'flat')
            elif regime == 1:
                v = sig_low.get(idx, 'flat')
            else:
                v = sig_high.get(idx, 'flat')

            result[idx] = v if v is not None else 'flat'

        return result

    return _signal


def grid_search_meta_v3(df_4h: pd.DataFrame, df_1d: pd.DataFrame,
                        engine: 'BacktestEngine', verbose: bool = True) -> tuple:
    """
    5D HMM MetaStrategy v3.0 のグリッドサーチ。

    Returns:
        (best_params, best_result, all_results)
    """
    best_sharpe = -np.inf
    best_params = None
    best_result = None
    all_results = []

    combos = [
        (yp, mp, up)
        for yp in YAGAMI_GRID
        for mp in MAEDAI_GRID
        for up in UNION_GRID
    ]

    if verbose:
        print(f'[v3] グリッドサーチ: {len(combos)}組み合わせ (5D HMM)')

    for i, (yp, mp, up) in enumerate(combos):
        name = (f"Meta3_y{yp.get('freq','4h')}"
                f"_m{mp.get('lookback_days',15)}"
                f"_u{up.get('rsi_oversold',45)}")
        sig = make_meta_signal_v3(df_1d['close'], df_1d, yp, mp, up)
        r = engine.run(data=df_4h, signal_func=sig, freq='4h', name=name)
        sh = r.get('sharpe_ratio') or -9999
        all_results.append((name, r, yp, mp, up))

        if verbose:
            pf  = f"{r.get('profit_factor', 0):.3f}" if r.get('profit_factor') else 'N/A'
            mdd = f"{r.get('max_drawdown_pct', 0):.1f}" if r.get('max_drawdown_pct') else 'N/A'
            print(f"  [{i+1}/{len(combos)}] {name:<40} "
                  f"Sharpe={sh:.3f}, PF={pf}, MDD={mdd}%, "
                  f"Trades={r.get('total_trades',0)}")

        if sh > best_sharpe:
            best_sharpe = sh
            best_params = (yp, mp, up)
            best_result = r

    return best_params, best_result, all_results


# ──────────────────────────────────────────────────────
#  データ読み込み
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
    df = df.set_index('datetime').sort_index()
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols].astype(float)


# ──────────────────────────────────────────────────────
#  バックテスト実行
# ──────────────────────────────────────────────────────

def run_meta_v2_backtest():
    path_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    path_1d = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')

    df_4h = load_ohlc(path_4h)
    df_1d = load_ohlc(path_1d)

    print(f'4H: {len(df_4h)} bars | 1D: {len(df_1d)} bars')
    print(f'期間: {df_4h.index[0].date()} ~ {df_4h.index[-1].date()}')

    # HMM 学習 & レジーム分布確認
    print('\n=== 3状態HMMレジーム分析 ===')
    detector = HiddenMarkovRegimeDetector(n_states=3)
    detector.fit(df_1d['close'])
    dist = detector.regime_distribution(df_1d['close'])
    stats = detector.regime_stats()

    for k, v in stats.items():
        print(f"  {k}: mean={v['mean_return']:.5f}, vol={v['volatility']:.5f}")
    print(f'  レジーム分布: range={dist["range"]:.1%}, '
          f'low_trend={dist["low_trend"]:.1%}, '
          f'high_trend={dist["high_trend"]:.1%}')

    engine = BacktestEngine(
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

    # グリッドサーチ
    print('\n=== グリッドサーチ ===')
    best_params, best_result, all_gs = grid_search_meta(df_4h, df_1d, engine)

    # ベースライン比較
    sig_union = sig_maedai_yagami_union(
        freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45,
    )
    r_base = engine.run(data=df_4h, signal_func=sig_union, freq='4h', name='Union_4H_Base')

    def fmt(v, p=3):
        return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'

    print('\n' + '=' * 72)
    print('  MetaStrategy v2.0 最終比較')
    print('=' * 72)
    header = f"{'戦略':<32} {'PF':>6} {'WR%':>6} {'MDD%':>7} {'Sharpe':>8} {'Calmar':>8} {'Trades':>7}"
    print(header)
    print('-' * 72)

    for name, r in [('Union_4H_Base', r_base), ('MetaStrategy_v2_Best', best_result)]:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<32} {pf:>6} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7}')

    print('=' * 72)
    if best_params:
        yp, mp, up = best_params
        print(f'\nベストパラメータ:')
        print(f'  range (YagamiA): {yp}')
        print(f'  low_trend (Maedai): {mp}')
        print(f'  high_trend (Union): {up}')

    return best_params, best_result, r_base


if __name__ == '__main__':
    run_meta_v2_backtest()
