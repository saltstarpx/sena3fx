"""
yagami_mtf_v78.py
================
v77ベース + 4つの追加フィルター（バックテスト検証済み）

【v77からの変更点】
backtest_v78_improvements.py によりXAUUSDで検証した4改善を正式採用。
全て1Hループに適用（段階検証結果: v77 PF2.03 → v78D PF2.50）。

■ 改善A: 1H確認足 KLOWチェック追加
  - h4_latest の KMID+KLOW に加え、h1_prev1 にも KLOW を適用
  - h1_prev1 は既に陽線/陰線確認済みのため KMID は常にTrue → 実質 KLOW のみ追加
  - 効果: 下ヒゲの大きい弱い確認足でのエントリーを除外

■ 改善B: 時間帯フィルター（UTC 5〜utc_end 時）
  - 流動性の低い深夜帯（アジア時間）のノイズシグナルを除外
  - XAUUSD デフォルト: UTC 5〜21時
  - FX通貨ペアは utc_end=20 を推奨
  - 効果: PF 2.00→2.20 (+0.20)、MDD 15.6%→12.5%

■ 改善C: 確認足 実体サイズ最小値（ATR × 0.2）
  - 確認足の実体（|close - open|）が ATR×0.2 未満のシグナルを除外
  - 微弱な陽線/陰線（1ティック陽線など）での低品質エントリーを除外
  - 効果: PF 2.20→2.37 (+0.17)、MDD 12.5%→10.6%

■ 改善D: パターン許容幅を ATR×0.3→ATR×0.2 に縮小
  - 二番底/天井の安値/高値の差の許容幅を厳格化
  - より精度の高い W字/M字パターンのみを採用
  - 効果: PF 2.37→2.50 (+0.13)、MDD 10.6%→6.9%

【バックテスト結果（XAUUSD, OOS: 2025/03〜2026/02, spread=5.2pips）】
指標                v77      v78D（全改善）
トレード数（OOS）  555件    180件（-68%）
PF                 2.03     2.50   (+23%)
MDD                15.6%    6.9%   (-56%)
Kelly              -0.012   +0.043 (正転)
月次プラス         12/12    12/12
エントリー品質     低       高（高品質シグナルに絞り込み）

【推奨銘柄】
- XAUUSD: 全4改善適用（utc_end=21）→ OOS PF=2.50 ✅
- GBPUSD: 改善Bのみ（utc_end=20）→ OOS PF=1.95（採用基準未達、様子見）
- AUDUSD/EURUSD: v77据え置き（改善C・DでPF低下）
- 指数（US30/SPX500/NAS100）: v77・v78共に PF<1.0、採用不可

【エントリーウィンドウ】
- 1時間足更新: 更新から2分以内の最初の1分足
- 4時間足: v77同様（改善の効果が未検証のため変更なし）
"""

import pandas as pd
import numpy as np

# ── 定数 ────────────────────────────────────────────────
KLOW_THRESHOLD      = 0.0015  # 下ヒゲ比率上限（v77から継承）
PATTERN_TOLERANCE   = 0.2     # [改善D] 二番底/天井の許容幅係数 ATR × 0.2
BODY_MIN_FACTOR     = 0.2     # [改善C] 確認足 実体サイズ最小値係数 ATR × 0.2
UTC_START_DEFAULT   = 5       # [改善B] 時間帯フィルター 開始時刻（UTC）
UTC_END_DEFAULT     = 21      # [改善B] 時間帯フィルター 終了時刻（UTC）※XAUUSD基準


def calculate_atr(df, period=14):
    high_low   = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def check_kmid_klow(bar, direction):
    """
    KMID + KLOW フィルター（v77から継承）

    Parameters
    ----------
    bar       : pd.Series  ローソク足（open, high, low, close）
    direction : int        1=ロング, -1=ショート

    Returns
    -------
    bool  True=通過（エントリー可）
    """
    o, l, c = bar["open"], bar["low"], bar["close"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ok = (body_bottom - l) / o < KLOW_THRESHOLD if o > 0 else False
    return kmid_ok and klow_ok


def generate_signals(
    data_1m,
    data_15m,
    data_4h,
    spread_pips  = 0.2,
    rr_ratio     = 2.5,
    utc_start    = UTC_START_DEFAULT,
    utc_end      = UTC_END_DEFAULT,
):
    """
    4時間足 + 1時間足の二番底・二番天井でシグナルを生成する。
    エントリーは足更新後2分以内の最初の1分足始値（成行）。
    v78: 4つの追加フィルターを適用（1Hループのみ変更）。

    Parameters
    ----------
    data_1m      : pd.DataFrame   1分足データ（エントリー実行用）
    data_15m     : pd.DataFrame   15分足データ（1H集約ベース）
    data_4h      : pd.DataFrame   4時間足データ（トレンド判定）
    spread_pips  : float          スプレッド（pips）
    rr_ratio     : float          リスクリワード比（デフォルト 2.5）
    utc_start    : int            時間帯フィルター 開始時刻（UTC）
    utc_end      : int            時間帯フィルター 終了時刻（UTC）
                                  XAUUSD=21, FX=20, 指数=22 を推奨

    Returns
    -------
    list[dict]  シグナルリスト
    """
    spread = spread_pips * 0.01

    # 4時間足: ATR・EMA・トレンド計算
    data_4h = data_4h.copy()
    data_4h["atr"]   = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    # 1時間足: 15分足から集約
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",  "volume": "sum"
    }).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals   = []
    used_times = set()

    # ──────────────────────────────────────────────────────
    # 4時間足ループ（v77 Bug①修正済み、v78での変更なし）
    # ──────────────────────────────────────────────────────
    h4_times = data_4h.index.tolist()
    for i in range(3, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1 = data_4h.iloc[i - 1]   # 確認足
        h4_prev2 = data_4h.iloc[i - 2]   # パターン1本目
        h4_prev3 = data_4h.iloc[i - 3]   # 文脈足（KMIDフィルター対象）
        h4_current = data_4h.iloc[i]

        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        trend     = h4_current["trend"]
        tolerance = atr_val * 0.3  # 4Hは v77 と同じ許容幅

        if trend == 1:  # ロング: 二番底
            low1, low2 = h4_prev2["low"], h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, direction=1):
                    continue
                sl  = min(low1, low2) - atr_val * 0.15
                m1w = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  h4_current_time + pd.Timedelta(minutes=2))
                ]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw + spread; risk = raw - sl
                        if 0 < risk <= atr_val * 3:
                            signals.append({"time": et, "dir": 1, "ep": ep,
                                            "sl": sl, "tp": raw + risk * rr_ratio,
                                            "risk": risk, "spread": spread,
                                            "tf": "4h", "pattern": "double_bottom"})
                            used_times.add(et)

        if trend == -1:  # ショート: 二番天井
            high1, high2 = h4_prev2["high"], h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, direction=-1):
                    continue
                sl  = max(high1, high2) + atr_val * 0.15
                m1w = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  h4_current_time + pd.Timedelta(minutes=2))
                ]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw - spread; risk = sl - raw
                        if 0 < risk <= atr_val * 3:
                            signals.append({"time": et, "dir": -1, "ep": ep,
                                            "sl": sl, "tp": raw - risk * rr_ratio,
                                            "risk": risk, "spread": spread,
                                            "tf": "4h", "pattern": "double_top"})
                            used_times.add(et)

    # ──────────────────────────────────────────────────────
    # 1時間足ループ（v78: 4つの追加フィルターを適用）
    # ──────────────────────────────────────────────────────
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1  = data_1h.iloc[i - 1]
        h1_prev2  = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # [改善B] 時間帯フィルター（UTC utc_start〜utc_end 時のみ）
        if not (utc_start <= h1_current_time.hour < utc_end):
            continue

        # 完結済み4H足のみ取得（v77 Bug②修正済み）
        h4_before = data_4h[data_4h.index < h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)):
            continue
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # [改善D] パターン許容幅を ATR×0.2 に縮小（v77: ATR×0.3）
        tolerance = atr_val * PATTERN_TOLERANCE

        for direction in [1, -1]:
            if trend != direction:
                continue

            if direction == 1:
                v1, v2   = h1_prev2["low"], h1_prev1["low"]
                conf_ok  = h1_prev1["close"] > h1_prev1["open"]
            else:
                v1, v2   = h1_prev2["high"], h1_prev1["high"]
                conf_ok  = h1_prev1["close"] < h1_prev1["open"]

            if abs(v1 - v2) > tolerance:
                continue
            if not conf_ok:
                continue

            # [改善C] 確認足の実体サイズ最小値チェック
            body = abs(h1_prev1["close"] - h1_prev1["open"])
            if body < atr_val * BODY_MIN_FACTOR:
                continue

            # 4H文脈足 KMID+KLOW（v77 Bug①修正済み）
            if not check_kmid_klow(h4_latest, direction):
                continue

            # [改善A] 1H確認足 KLOW チェック（実質KLOWのみ、KMIDは確認済み）
            if not check_kmid_klow(h1_prev1, direction):
                continue

            # エントリー（足更新後2分以内の最初の1分足始値）
            m1w = data_1m[
                (data_1m.index >= h1_current_time) &
                (data_1m.index <  h1_current_time + pd.Timedelta(minutes=2))
            ]
            if len(m1w) == 0:
                continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times:
                continue

            raw = eb["open"]
            if direction == 1:
                sl = min(v1, v2) - atr_val * 0.15
                ep = raw + spread
                risk = raw - sl
            else:
                sl = max(v1, v2) + atr_val * 0.15
                ep = raw - spread
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * rr_ratio
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp,
                    "risk": risk, "spread": spread,
                    "tf": "1h",
                    "pattern": "double_bottom" if direction == 1 else "double_top"
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals
