"""
リアルタイムシグナルエンジン
============================
バックテストで実証済みの2戦略を実装:

  SEAS_DC2d       ... Gold (XAU_USD) 推奨
    - 2日間ドンチャンブレイク (4H足)
    - 6月・7月スキップ (シーズナルフィルター)
    - SL=1.5×ATR, TP=4.5×ATR

  UNION3d50_SEAS  ... Silver (XAG_USD) 推奨
    - RSI(14)が50を上抜け & EMA21上 OR 3日ドンチャンブレイク
    - 6月・7月スキップ
    - SL=2.0×ATR, TP=6.0×ATR

バックテスト結果 (2025-09 〜 2026-02, リスク5%):
  Gold:   +195.7% ROI, MaxDD 4.6%, Calmar 42.5
  Silver: +214.7% ROI, MaxDD 5.1%, Calmar 42.1
"""

import os
import sys
import logging
import pandas as pd

# プロジェクトルートを sys.path に追加して lib/ をインポート可能にする
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.yagami import sig_dc_fast, sig_aggressive_union

log = logging.getLogger('sena3fx')


# ------------------------------------------------------------------ #
#  シーズナルフィルター                                               #
# ------------------------------------------------------------------ #

def seasonal_filter(sig_fn, skip_months: tuple = (6, 7)):
    """
    指定月をスキップするシーズナルフィルター。
    デフォルトは6月・7月 (金・銀ともにレンジ相場になりやすい)。
    """
    def wrapped(bars: pd.DataFrame) -> pd.Series:
        sigs = sig_fn(bars)
        mask = pd.Series(bars.index.month, index=bars.index).isin(skip_months)
        sigs[mask] = 'flat'
        return sigs
    return wrapped


# ------------------------------------------------------------------ #
#  戦略定義                                                           #
# ------------------------------------------------------------------ #

_STRATEGIES = {
    # Gold 推奨: 2日ドンチャン + シーズナル
    'SEAS_DC2d': seasonal_filter(
        sig_dc_fast(freq='4h', lookback_days=2),
        skip_months=(6, 7)
    ),
    # Silver 推奨: RSI50クロス OR 3日DC + シーズナル
    'UNION3d50_SEAS': seasonal_filter(
        sig_aggressive_union(freq='4h', rsi_thresh=50, lookback_days_dc=3),
        skip_months=(6, 7)
    ),
    # フィルターなし版 (テスト・比較用)
    'DC2d':      sig_dc_fast(freq='4h', lookback_days=2),
    'UNION3d50': sig_aggressive_union(freq='4h', rsi_thresh=50, lookback_days_dc=3),
}


# ------------------------------------------------------------------ #
#  ATR計算                                                            #
# ------------------------------------------------------------------ #

def calc_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(14) を計算して Series を返す"""
    h = bars['high']
    l = bars['low']
    c = bars['close']
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ------------------------------------------------------------------ #
#  シグナル取得                                                       #
# ------------------------------------------------------------------ #

def get_signal(bars: pd.DataFrame, strategy: str) -> dict:
    """
    最新の完成バーのシグナルを返す。

    Args:
        bars:     4H OHLCデータ (完成バーのみ)
                  columns: open, high, low, close, volume
                  index: datetime (UTC, TZなし)
        strategy: 'SEAS_DC2d' | 'UNION3d50_SEAS' | 'DC2d' | 'UNION3d50'

    Returns:
        dict: {
            'signal':   'long' | 'flat',
            'atr':      float,    # ATR(14)値
            'close':    float,    # 最新終値
            'bar_time': datetime  # 最新バーの時刻
        }
    """
    _empty = {'signal': 'flat', 'atr': None, 'close': None, 'bar_time': None}

    if bars is None or len(bars) < 30:
        log.debug(f"データ不足 ({len(bars) if bars is not None else 0} bars < 30)")
        return _empty

    if strategy not in _STRATEGIES:
        raise ValueError(
            f"未知の戦略: '{strategy}'. "
            f"利用可能: {list(_STRATEGIES.keys())}"
        )

    sig_fn = _STRATEGIES[strategy]

    try:
        sigs = sig_fn(bars)
        atr  = calc_atr(bars, period=14)
    except Exception as e:
        log.error(f"シグナル計算エラー: {e}")
        return _empty

    last_sig   = sigs.iloc[-1]
    last_atr   = atr.iloc[-1]
    last_close = bars['close'].iloc[-1]
    last_time  = bars.index[-1]

    # 'long' / 'short' 以外はすべて 'flat' として扱う
    signal = last_sig if last_sig in ('long', 'short') else 'flat'

    return {
        'signal':   signal,
        'atr':      float(last_atr) if pd.notna(last_atr) else None,
        'close':    float(last_close),
        'bar_time': last_time,
    }
