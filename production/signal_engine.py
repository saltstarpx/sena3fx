"""
signal_engine.py - Gold Logic リアルタイムシグナル検出エンジン
=============================================================
バックテスト（backtest_all_1m_adaptive.py）の generate_signals() を
MetaTrader5 リアルタイム環境向けに適合させたモジュール。

【シグナル検出フロー】
  1. 直近2本の1H足でダブルボトム/トップパターンを検出
  2. 直前4H足でトレンド・KMID・KLOW・EMA距離チェック
  3. 日足EMA20方向一致（Goldロジック）
  4. 条件通過 → シグナルdict を返す（E2エントリーはbot本体で実施）

【シグナルdict】
  {
    "dir":      1 or -1,
    "raw_ep":   スプレッド前エントリー価格（=パターン足終値近辺）,
    "sl":       SL価格,
    "tp":       TP価格（RR=2.5）,
    "risk":     リスク幅（raw_ep - sl の絶対値）,
    "h1_time":  起点1H足の開始時刻,
    "expire_at":E2エントリーウィンドウ終了時刻（h1_time + 3分）,
  }
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# ── パラメータ（バックテストと完全一致） ─────────────────────────
KLOW_THR        = 0.0015   # 4H足下ヒゲ比率上限
EMA_DIST_MIN    = 1.0      # EMA距離フィルター（ATR倍）
PATTERN_TOL     = 0.30     # ダブルボトム/トップ許容幅（1H ATR倍）
E2_SPIKE_ATR    = 2.0      # スパイク判定（1m足レンジ > ATR×この値でスキップ）
E2_WINDOW_MIN   = 3        # E2エントリーウィンドウ（分）
RR_RATIO        = 2.5      # リスクリワード比
ATR_PERIOD      = 14       # ATR計算期間


# ────────────────────────────────────────────────────────────────
# インジケーター計算
# ────────────────────────────────────────────────────────────────
def _calc_atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()


def _calc_ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_indicators(df4h: pd.DataFrame,
                     df1d: pd.DataFrame | None = None,
                     df15m: pd.DataFrame | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    4H足・日足にATR/EMA20/トレンドを付与。
    1H足はdf15mから生成（渡す場合）。

    Returns
    -------
    (df4h_ind, df1d_ind, df1h_ind or None)
    """
    d4 = df4h.copy()
    d4["atr"]   = _calc_atr(d4)
    d4["ema20"] = _calc_ema(d4["close"])
    d4["trend"] = np.where(d4["close"] > d4["ema20"], 1, -1)

    # 日足
    if df1d is not None and len(df1d) >= 20:
        d1 = df1d.copy()
    else:
        # 4H足からリサンプル
        d1 = d4.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open", "close"])
    d1["ema20"]   = _calc_ema(d1["close"])
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)

    # 1H足（15m足から生成）
    d1h = None
    if df15m is not None and len(df15m) > 0:
        d1h = df15m.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open", "close"])
        d1h["atr"] = _calc_atr(d1h)

    return d4, d1, d1h


# ────────────────────────────────────────────────────────────────
# フィルター関数（4H直前足に対して適用）
# ────────────────────────────────────────────────────────────────
def _check_kmid(bar: pd.Series, direction: int) -> bool:
    """直前4H足の実体方向がエントリー方向と一致"""
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])


def _check_klow(bar: pd.Series) -> bool:
    """直前4H足の下ヒゲ比率 < 0.15%"""
    o = bar["open"]
    if o <= 0:
        return False
    lower_wick = min(bar["open"], bar["close"]) - bar["low"]
    return (lower_wick / o) < KLOW_THR


def _check_ema_dist(bar: pd.Series) -> bool:
    """4H終値とEMA20の距離 ≥ ATR × 1.0"""
    dist = abs(bar["close"] - bar["ema20"])
    atr  = bar.get("atr", np.nan)
    return not pd.isna(atr) and atr > 0 and dist >= atr * EMA_DIST_MIN


# ────────────────────────────────────────────────────────────────
# シグナル検出（リアルタイム版）
# ────────────────────────────────────────────────────────────────
def check_signal(
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    df1d: pd.DataFrame,
    spread: float,
    now: datetime | None = None,
) -> dict | None:
    """
    直近1H足2本のパターン + 4H/1Dフィルターでシグナルを検出。

    Parameters
    ----------
    df1h : DataFrame（1H足、ATR付き。直近3本以上必要）
    df4h : DataFrame（4H足、ATR/EMA20/trend付き。直近2本以上必要）
    df1d : DataFrame（日足、trend1d付き。直近1本以上必要）
    spread : float（価格単位でのスプレッド）
    now   : 現在時刻（UTC）。Noneなら datetime.now(UTC)を使用。

    Returns
    -------
    シグナルdict または None
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # 直近のクローズ済み1H足を取得（nowより前のもの）
    h1_closed = df1h[df1h.index < now].copy()
    if len(h1_closed) < 3:
        return None

    h1_cur  = h1_closed.iloc[-1]   # 最新クローズ足（シグナル起点）
    h1_prev = h1_closed.iloc[-2]   # 1本前
    # h1_cur の時刻 = 今のH足開始時刻に相当

    # E2エントリーウィンドウ確認
    # シグナルは h1_cur がクローズした直後の次の1H足開始時に有効
    # = h1_cur.name + 1H が「今の1H足開始」
    next_h1_open = h1_cur.name + pd.Timedelta(hours=1)
    expire_at    = next_h1_open + pd.Timedelta(minutes=E2_WINDOW_MIN)

    if now < next_h1_open or now > expire_at:
        return None   # まだ前の足がクローズしていない or ウィンドウ外

    atr_1h = h1_cur["atr"]
    if pd.isna(atr_1h) or atr_1h <= 0:
        return None

    # 直前4H足（h1_cur より前で最後のもの）
    h4_before = df4h[df4h.index < next_h1_open]
    if len(h4_before) < 2:
        return None
    h4_lat = h4_before.iloc[-1]
    if pd.isna(h4_lat.get("atr", np.nan)) or h4_lat["atr"] <= 0:
        return None

    trend = int(h4_lat["trend"])

    # 日足EMA20方向一致（Goldロジック）
    d1_before = df1d[df1d.index.normalize() < next_h1_open.normalize()]
    if len(d1_before) == 0:
        return None
    if int(d1_before.iloc[-1]["trend1d"]) != trend:
        return None

    # KMID / KLOW / EMA距離
    if not _check_kmid(h4_lat, trend):
        return None
    if not _check_klow(h4_lat):
        return None
    if not _check_ema_dist(h4_lat):
        return None

    # ダブルボトム/トップパターン（h1_prev と h1_cur）
    tol = atr_1h * PATTERN_TOL
    if trend == 1:
        v1, v2 = h1_prev["low"],  h1_cur["low"]
    else:
        v1, v2 = h1_prev["high"], h1_cur["high"]

    if abs(v1 - v2) > tol:
        return None

    # SL / TP 計算
    if trend == 1:
        sl   = min(v1, v2) - atr_1h * 0.15
        raw_ep = h1_cur["close"]
        risk   = raw_ep - sl
    else:
        sl   = max(v1, v2) + atr_1h * 0.15
        raw_ep = h1_cur["close"]
        risk   = sl - raw_ep

    # リスク幅チェック（SL距離 ≤ 4H ATR × 2）
    if risk <= 0 or risk > h4_lat["atr"] * 2:
        return None

    tp = raw_ep + trend * risk * RR_RATIO

    return {
        "dir":      trend,
        "raw_ep":   raw_ep,
        "sl":       sl,
        "tp":       tp,
        "risk":     risk,
        "h1_time":  h1_cur.name,
        "expire_at": expire_at,
    }


# ────────────────────────────────────────────────────────────────
# 1m足スパイクチェック（E2エントリー前に呼ぶ）
# ────────────────────────────────────────────────────────────────
def is_spike_bar(bar_high: float, bar_low: float, atr_1m: float) -> bool:
    """1m足のレンジがATR×E2_SPIKE_ATRを超えるならスパイク（スキップ）"""
    return (bar_high - bar_low) > atr_1m * E2_SPIKE_ATR


def calc_entry_price(raw_ep: float, direction: int, spread: float) -> float:
    """スプレッドを加味したエントリー価格"""
    return raw_ep + (spread if direction == 1 else -spread)
