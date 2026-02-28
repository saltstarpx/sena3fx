"""
USD強弱プロキシフィルター
===========================
FXペアデータがない環境でXAUUSDの逆相関を利用してUSD強弱を推定する。

Gold ↔ USD の逆相関:
  - Goldが上昇 → USD が弱い → XAGUSD ロングに追い風
  - Goldが下落 → USD が強い → XAGUSD ロングに逆風

計算方法:
  1. XAUUSD 1日足の N日ローリングリターンを計算
  2. M日窓でZスコア化 → 正規化済みモメンタム
  3. 符号反転 → USD強弱プロキシ (正=USD強い, 負=USD弱い)

実行: python scripts/usd_strength_filter.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
#  USD強弱プロキシ計算                                               #
# ------------------------------------------------------------------ #
def calc_usd_proxy(xauusd_1d: pd.DataFrame,
                   ret_window: int = 20,
                   zscore_window: int = 60) -> pd.Series:
    """
    XAUUSD 1日足データからUSD強弱プロキシを計算する。

    Args:
        xauusd_1d:     XAUUSD 1日足 OHLC (index=datetime)
        ret_window:    ローリングリターン期間 (デフォルト20日)
        zscore_window: Zスコア標準化の窓 (デフォルト60日)

    Returns:
        pd.Series: USD強弱スコア
            - 正の値 = USD強い (Goldが下落傾向)
            - 負の値 = USD弱い (Goldが上昇傾向)
            - index = xauusd_1d と同じ日付
    """
    close = xauusd_1d['close']

    # N日リターン (% change over ret_window days)
    ret_n = close.pct_change(ret_window) * 100

    # Zスコア化 (zscore_window日の移動平均・移動標準偏差)
    roll_mean = ret_n.rolling(zscore_window).mean()
    roll_std  = ret_n.rolling(zscore_window).std()
    zscore    = (ret_n - roll_mean) / roll_std.replace(0, np.nan)

    # 符号反転: Gold下落 = USD強い (+)
    usd_proxy = -zscore
    usd_proxy.name = 'usd_strength_proxy'

    return usd_proxy


def calc_usd_strength(timeframe: str = '1d', lookback: int = 20,
                      data_root: str = None) -> pd.Series:
    """
    互換インターフェース: currency_strength_portfolio.calc_usd_strength と同名。

    Args:
        timeframe: '1d' のみ対応 (将来拡張用)
        lookback:  ローリングリターン期間
        data_root: データルートパス (省略時は自動検出)

    Returns:
        pd.Series: USD強弱プロキシ (日次, float)
    """
    if data_root is None:
        data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # XAUUSD 1日足を読み込む
    path_1d = os.path.join(data_root, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')
    if not os.path.exists(path_1d):
        raise FileNotFoundError(f"XAUUSD 1日足データが見つかりません: {path_1d}")

    df = pd.read_csv(path_1d)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if hasattr(dt, 'dt') and dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    return calc_usd_proxy(df, ret_window=lookback)


# ------------------------------------------------------------------ #
#  フィルターマスク生成                                              #
# ------------------------------------------------------------------ #
def get_usd_filter_mask(usd_strength: pd.Series,
                        target_index: pd.DatetimeIndex,
                        threshold: float = 1.0) -> pd.Series:
    """
    USD強弱がthresholdを超える期間にFalseを返すブールマスク。

    Args:
        usd_strength:  calc_usd_proxy() の出力 (日次)
        target_index:  15m足などのターゲットインデックス
        threshold:     このスコアを超えるとロングをブロック

    Returns:
        pd.Series[bool]: True = エントリー許可, False = ブロック
    """
    # 日次 → 15m に前向き補完
    combined = usd_strength.reindex(
        usd_strength.index.union(target_index)
    ).ffill()
    mask = combined.reindex(target_index).fillna(0) < threshold
    mask.name = 'usd_allow'
    return mask


def get_seasonal_filter_mask(target_index: pd.DatetimeIndex,
                              skip_months: tuple = (7, 9)) -> pd.Series:
    """
    指定した月をスキップするブールマスク。

    Args:
        target_index:  15m足などのターゲットインデックス
        skip_months:   スキップする月番号のタプル (1=1月, ..., 12=12月)
                       デフォルト: (7, 9) = 7月・9月を回避

    Returns:
        pd.Series[bool]: True = エントリー許可, False = ブロック
    """
    months = pd.Series(target_index.month, index=target_index)
    mask   = ~months.isin(skip_months)
    mask.name = 'seasonal_allow'
    return mask


def get_usd_scale_series(usd_strength: pd.Series,
                          target_index: pd.DatetimeIndex,
                          strong_threshold: float = 0.5,
                          scale_factor: float = 0.7) -> pd.Series:
    """
    USDが適度に強い時にポジションサイズを縮小するスケール係数を返す。

    Args:
        strong_threshold: このスコアを超えると scale_factor を適用
        scale_factor:     縮小率 (デフォルト: 0.7 = 30%縮小)

    Returns:
        pd.Series[float]: 各足のリスク倍率 (1.0 or scale_factor)
    """
    combined = usd_strength.reindex(
        usd_strength.index.union(target_index)
    ).ffill()
    score = combined.reindex(target_index).fillna(0)
    scale = pd.Series(1.0, index=target_index)
    scale[score >= strong_threshold] = scale_factor
    scale.name = 'risk_scale'
    return scale


# ------------------------------------------------------------------ #
#  スタンドアロン動作確認                                            #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'ohlc', 'XAUUSD_2025_1d.csv')

    import sys
    print("USD強弱プロキシ計算テスト")
    print(f"データ: {path}")

    usd = calc_usd_strength(lookback=20)
    print(f"\n計算完了: {len(usd)} 日分")
    print(f"  期間: {usd.index[0].date()} 〜 {usd.index[-1].date()}")
    print(f"  平均: {usd.mean():.4f}")
    print(f"  標準偏差: {usd.std():.4f}")
    print(f"  最大: {usd.max():.4f} (USD最強)")
    print(f"  最小: {usd.min():.4f} (USD最弱)")
    print(f"\n最新20日:")
    print(usd.dropna().tail(20).to_string())

    threshold = 1.0
    blocked = (usd.dropna() >= threshold).sum()
    pct = blocked / len(usd.dropna()) * 100
    print(f"\nthreshold={threshold}: {blocked}日/{len(usd.dropna())}日 ({pct:.1f}%) がロングブロック対象")
