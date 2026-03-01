"""
MetaStrategy — HMMレジーム連動型戦略セレクター v1.0
====================================================
現在のレジームに応じてYagamiA(レンジ) / Union(トレンド)を自動切替。

実行方法:
  python strategies/meta_strategy.py

評価基準:
  Union_4H の Sharpe 2.817 を基準として改善を確認
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


# ──────────────────────────────────────────────────────
#  ベースシグナル定義
# ──────────────────────────────────────────────────────

_sig_yagami = sig_yagami_A(freq='4h')

_sig_union = sig_maedai_yagami_union(
    freq='4h', lookback_days=15, ema_days=200,
    confirm_bars=2, rsi_oversold=45,
)


# ──────────────────────────────────────────────────────
#  MetaStrategy シグナル生成
# ──────────────────────────────────────────────────────

def make_meta_signal(daily_close: pd.Series,
                     hmm_fit_bars: int = 252,
                     retrain_interval: int = 20) -> callable:
    """
    HMMレジームに基づいてシグナルを切り替える関数を生成する。

    Args:
        daily_close      : 日足終値（HMM学習用）
        hmm_fit_bars     : 初期学習に使用するバー数（デフォルト252 = 1年）
        retrain_interval : 再学習間隔（バー数）

    Returns:
        signal_func(bars) -> pd.Series
    """
    detector = HiddenMarkovRegimeDetector()

    def _meta_signal(bars: pd.DataFrame) -> pd.Series:
        # HMM を日足データで学習
        fit_data = daily_close.iloc[-hmm_fit_bars:] if len(daily_close) > hmm_fit_bars \
            else daily_close
        try:
            detector.fit(fit_data)
            regimes = detector.predict(fit_data)
        except Exception:
            regimes = pd.Series(1, index=fit_data.index)

        # 4H バーの各インデックスに対してレジームを割り当て
        # daily_close.index を DatetimeIndex として使い、4H バーを最寄り日付にマップ
        regime_daily = regimes.copy()
        regime_daily.index = pd.DatetimeIndex(regime_daily.index).normalize()

        sig_yagami = _sig_yagami(bars)
        sig_union  = _sig_union(bars)

        result = pd.Series('flat', index=bars.index, dtype=object)

        for idx in bars.index:
            day = pd.Timestamp(idx).normalize()
            # レジーム取得（最寄り日で検索）
            try:
                regime = int(regime_daily.get(day, regime_daily.iloc[-1]))
            except Exception:
                regime = 1  # デフォルト: トレンド

            if regime == 0:
                # レンジ → YagamiA
                v = sig_yagami.get(idx, 'flat')
            else:
                # トレンド → Union
                v = sig_union.get(idx, 'flat')

            result[idx] = v if v is not None else 'flat'

        return result

    return _meta_signal


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

def run_meta_backtest():
    path_4h = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    path_1d = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')

    df_4h = load_ohlc(path_4h)
    df_1d = load_ohlc(path_1d)

    print(f'4H bars: {len(df_4h)}, 1D bars: {len(df_1d)}')
    print(f'期間: {df_4h.index[0].date()} ~ {df_4h.index[-1].date()}')

    # HMM 学習
    print('\nHMMレジーム検出 学習中...')
    detector = HiddenMarkovRegimeDetector()
    detector.fit(df_1d['close'])
    regimes = detector.predict(df_1d['close'])
    stats = detector.regime_stats()

    print('レジーム統計:')
    for k, v in stats.items():
        print(f"  {k}: {v['label']}, "
              f"mean_return={v['mean_return']:.5f}, "
              f"vol={v['volatility']:.5f}")

    trend_days = int((regimes == 1).sum())
    range_days = int((regimes == 0).sum())
    print(f'  トレンド日: {trend_days}日 / レンジ日: {range_days}日')

    # MetaStrategy シグナル
    meta_sig = make_meta_signal(df_1d['close'])

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

    # ベースライン Union_4H との比較
    sig_union_base = sig_maedai_yagami_union(
        freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45,
    )

    variants = [
        ('Union_4H (ベース)',  sig_union_base),
        ('MetaStrategy_4H',   meta_sig),
    ]

    results = []
    for name, sig in variants:
        r = engine.run(data=df_4h, signal_func=sig, freq='4h', name=name)
        results.append((name, r))

    # 結果表示
    def fmt(v, p=3):
        return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'

    print('\n' + '=' * 72)
    print('  MetaStrategy バックテスト結果')
    print('=' * 72)
    header = f"{'戦略':<28} {'PF':>6} {'WR%':>6} {'MDD%':>7} {'Sharpe':>8} {'Calmar':>8} {'Trades':>7}"
    print(header)
    print('-' * 72)
    for name, r in results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        print(f'{name:<28} {pf:>6} {wr:>6} {mdd:>7} {sharpe:>8} {calmar:>8} {trades:>7}')
    print('=' * 72)

    return results


if __name__ == '__main__':
    run_meta_backtest()
