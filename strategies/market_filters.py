"""
Teammate C: Risk Manager — 市場環境フィルター
=============================================
ポートフォリオ全体のリスクを調整する市場環境フィルターの開発・評価。

評価基準:
  - ポートフォリオ全体の最大ドローダウン(MDD)低減率
  - ボラティリティ低減率
  - Sharpe Ratio 改善率

フィルター一覧:
  1. USD強弱フィルター — USD上位25%の強さではロング回避
  2. 季節フィルター     — 特定月の除外
"""
import numpy as np
import pandas as pd


# ================================================================== #
#  1. USD 強弱フィルター                                               #
# ================================================================== #
def calc_usd_strength(bars: pd.DataFrame,
                      lookback: int = 20,
                      rank_window: int = 100) -> pd.Series:
    """
    XAUUSD の逆モメンタムから USD 強弱プロキシを算出する。

    ロジック:
      - XAUUSD リターン(lookback本) を算出
      - ローリングパーセンタイルランク (rank_window本) に変換
      - USD強弱 = 100 - gold_momentum_percentile
        (金が下落 = USD強、金が上昇 = USD弱)

    Args:
        bars:        OHLC DataFrame (XAUUSD)
        lookback:    モメンタム計算の振り返り本数
        rank_window: パーセンタイルランクの窓幅

    Returns:
        pd.Series: 0-100 のUSD強弱スコア (100=最強)
    """
    close = bars['close']
    # 金のモメンタム (lookback本リターン)
    gold_ret = close.pct_change(lookback)

    # ローリングパーセンタイルランク
    def percentile_rank(s, window):
        """各時点の値が過去window本中の何パーセンタイルにいるか"""
        out = pd.Series(np.nan, index=s.index)
        vals = s.values
        for i in range(window, len(vals)):
            w = vals[i - window:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) < 5:
                continue
            rank = np.sum(valid < vals[i]) / len(valid) * 100
            out.iloc[i] = rank
        return out

    gold_pctrank = percentile_rank(gold_ret, rank_window)
    # USD強弱 = 金が弱い(下落中)ほどUSD強い
    usd_strength = 100.0 - gold_pctrank
    return usd_strength.fillna(50.0)


def usd_strength_filter(bars: pd.DataFrame,
                        threshold: float = 75.0,
                        lookback: int = 20,
                        rank_window: int = 100) -> pd.Series:
    """
    USD強弱フィルター。

    USD が上位 threshold% の強さのとき True (= ロング回避すべき)。

    Args:
        bars:      OHLC DataFrame
        threshold: USD強弱のフィルター閾値 (デフォルト75 = 上位25%)

    Returns:
        pd.Series[bool]: True = USD強 → ロング回避
    """
    usd = calc_usd_strength(bars, lookback, rank_window)
    return usd >= threshold


def wrap_signal_with_usd_filter(sig_func, bars: pd.DataFrame,
                                 threshold: float = 75.0,
                                 lookback: int = 20,
                                 rank_window: int = 100):
    """
    既存シグナル関数にUSD強弱フィルターを適用するラッパー。

    USD が強い期間のロングシグナルを除去する。
    ショートシグナルは影響を受けない。

    Returns:
        sig_func と同じ形式の pd.Series
    """
    raw_signals = sig_func(bars)
    usd_strong = usd_strength_filter(bars, threshold, lookback, rank_window)

    filtered = raw_signals.copy()
    # USD強 + ロングシグナル → 除去
    mask = usd_strong & (filtered == 'long')
    filtered[mask] = None
    return filtered


def make_usd_filtered_signal(sig_factory, threshold=75.0,
                              lookback=20, rank_window=100):
    """
    シグナルファクトリにUSD強弱フィルターを組み込む。

    Usage:
        from lib.yagami import sig_yagami_A
        filtered_sig = make_usd_filtered_signal(sig_yagami_A, threshold=75)
        result = engine.run(data=df, signal_func=filtered_sig(freq='4h'))
    """
    def _factory(*args, **kwargs):
        inner_fn = sig_factory(*args, **kwargs)

        def _filtered(bars):
            return wrap_signal_with_usd_filter(
                inner_fn, bars, threshold, lookback, rank_window
            )
        return _filtered
    return _factory


# ================================================================== #
#  2. 季節フィルター                                                   #
# ================================================================== #
# BacktestEngine.run() の allowed_months パラメータで制御
# ここではユーティリティ関数のみ提供

SEASON_ALL      = None                      # 全月
SEASON_SKIP_JUL = [1,2,3,4,5,6,8,9,10,11,12]   # 7月除外
SEASON_SKIP_JUL_SEP = [1,2,3,4,5,6,8,10,11,12]  # 7月+9月除外
SEASON_ACTIVE   = [1,2,3,10,11,12]          # 冬季(1-3)+秋季(10-12)
SEASON_SKIP_SUMMER = [1,2,3,4,5,10,11,12]   # 6-9月除外


def seasonal_effectiveness(trades: list, skip_months: tuple = (7, 9)) -> dict:
    """
    季節フィルターの有効性を評価する。

    Args:
        trades: バックテストのトレードリスト
        skip_months: 除外する月

    Returns:
        dict: 月別PnL・フィルタ有無の比較
    """
    monthly = {}
    for t in trades:
        m = t['entry_time'].month if hasattr(t['entry_time'], 'month') else 0
        if m not in monthly:
            monthly[m] = {'count': 0, 'pnl': 0, 'wins': 0}
        monthly[m]['count'] += 1
        monthly[m]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            monthly[m]['wins'] += 1

    included_pnl = sum(v['pnl'] for k, v in monthly.items() if k not in skip_months)
    excluded_pnl = sum(v['pnl'] for k, v in monthly.items() if k in skip_months)
    included_cnt = sum(v['count'] for k, v in monthly.items() if k not in skip_months)
    excluded_cnt = sum(v['count'] for k, v in monthly.items() if k in skip_months)

    return {
        'monthly_breakdown': monthly,
        'included_months_pnl': included_pnl,
        'excluded_months_pnl': excluded_pnl,
        'included_months_count': included_cnt,
        'excluded_months_count': excluded_cnt,
        'skip_months': skip_months,
    }
