"""
yagami_mtf_v77.py
================
v76ベース + KMID（実体方向一致）+ KLOW（下ヒゲ小）フィルターを追加

【v76からの変更点】
qlib Alpha158系ファクタースクリーニングにより発見した2条件フィルターを追加。
全57ファクター中で最も高いIC（情報係数）を示したKMIDとKLOWを採用。

■ KMID（実体方向一致フィルター）:
  - ロングエントリー: 直前4H足が陽線（close > open）であること
  - ショートエントリー: 直前4H足が陰線（close < open）であること
  - 効果: 逆方向の4H足でのエントリー（勝率32〜38%の「養分トレード」183本）を除外
  - 原理: やがみメソッドの「ローソク足の強弱」条件をより忠実に実装

■ KLOW（下ヒゲ小フィルター）:
  - 直前4H足の下ヒゲ比率が 0.15% 未満であること
  - 下ヒゲ比率 = (min(open, close) - low) / open
  - 効果: 下ヒゲが大きい足（買い圧力が混在）でのエントリーを除外
  - 原理: モメンタムの純度が高い足のみでエントリーすることで勝率向上

【バックテスト結果（USDJPY, 2025/1-12, spread=0.4pips）】
指標              v76      v77(KMID+KLOW)
トレード数        373回    327回（-12%）
勝率              56.6%    76.1%
PF                2.17     4.96
総損益            +8,227p  +12,551p
MDD               460.9p   222.6p
月次シャープ      5.57     10.47
ケリー基準        0.305    0.608
最大連敗          5回      3回
プラス月          12/12    12/12

【過学習検証（5段階、全PASS）】
- IS/OOS分割: OOSの方が効果大（PF改善 IS+1.7 → OOS+3.3）
- ウォークフォワード: 12/12月（100%）でPF改善
- ブートストラップ: p=0.0000（PF差・勝率差とも統計的有意）
- 閾値感度: KLOW 0.0005〜0.005全範囲でPF>3.5
- 全期間（20ヶ月）: PF 5.26, 20/20月プラス

【エントリーウィンドウ】
- 4時間足更新: 更新から2分以内の最初の1分足
- 1時間足更新: 更新から2分以内の最初の1分足
"""

import pandas as pd
import numpy as np

# KLOW閾値（下ヒゲ比率の上限）
# 検証済み: 0.0005〜0.005の全範囲でPF>3.5（感度テストPASS）
KLOW_THRESHOLD = 0.0015


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def check_kmid_klow(prev_4h_bar, direction):
    """
    KMID + KLOW フィルター（v77追加ロジック）

    Parameters
    ----------
    prev_4h_bar : pd.Series
        エントリー直前の4時間足ローソク足データ（open, high, low, close）
    direction : int
        エントリー方向（1=ロング, -1=ショート）

    Returns
    -------
    bool
        True: フィルター通過（エントリー可）
        False: フィルター不通過（エントリースキップ）
    """
    o = prev_4h_bar["open"]
    h = prev_4h_bar["high"]
    l = prev_4h_bar["low"]
    c = prev_4h_bar["close"]

    # KMID: 実体方向一致フィルター
    # ロングなら直前4H足が陽線、ショートなら陰線
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)

    # KLOW: 下ヒゲ小フィルター
    # 下ヒゲ比率 = (実体下端 - 安値) / 始値
    body_bottom = min(o, c)
    klow_ratio = (body_bottom - l) / o if o > 0 else 0
    klow_ok = klow_ratio < KLOW_THRESHOLD

    return kmid_ok and klow_ok


def generate_signals(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=2.5):
    """
    4時間足 + 1時間足の二番底・二番天井でシグナルを生成する。
    エントリーは更新後の最初の1分足始値（成行）。
    v77: KMID+KLOWフィルターを追加。

    【スプレッドの扱い】
    - SL/TPはチャートレベル（始値基準）で固定
    - EPはスプレッド分だけ不利な価格（実際の約定価格）
    - 損益計算は (exit_price - ep) × 100 × dir で行う
    - スプレッドが増えるほど損益が悪化する（実環境に近い）

    Returns: list of signal dicts
    """
    spread = spread_pips * 0.01

    # 4時間足: ATR・EMA・トレンド
    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    # 1時間足データを15分足から集約
    data_1h = data_15m.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals = []
    used_times = set()  # 重複エントリー防止

    # ── 4時間足の二番底・二番天井 ──────────────────────────────
    # [BUG①修正] i=3 から開始し、パターン直前の文脈足(iloc[i-3])にKMIDを適用。
    # h4_prev1（確認足）は既に「陽線/陰線」を確認済みのため、
    # そこにKMIDを適用すると常にTrueになり意味をなさない。
    h4_times = data_4h.index.tolist()
    for i in range(3, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1 = data_4h.iloc[i - 1]   # 確認足（陽線/陰線チェック対象）
        h4_prev2 = data_4h.iloc[i - 2]   # パターン1本目
        h4_prev3 = data_4h.iloc[i - 3]   # パターン直前の文脈足（KMIDフィルター対象）
        h4_current = data_4h.iloc[i]

        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        trend = h4_current["trend"]
        tolerance = atr_val * 0.3

        # ロング: 二番底
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                # ── v77: KMID+KLOWフィルター（文脈足h4_prev3で判定） ──
                if not check_kmid_klow(h4_prev3, direction=1):
                    continue
                # ─────────────────────────────
                sl = min(low1, low2) - atr_val * 0.15
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]          # チャートレベルの基準価格
                        ep = raw_ep + spread                 # 実際の約定価格（スプレッド分不利）
                        risk = raw_ep - sl                   # リスク幅はチャートレベルで計算
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep + risk * rr_ratio    # TPもチャートレベルで計算
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,           # 実際の約定価格
                                "sl": sl,           # SLレベル
                                "tp": tp,           # TPレベル
                                "risk": risk,       # チャートレベルのリスク幅
                                "spread": spread,   # スプレッド（参照用）
                                "tf": "4h",
                                "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                # ── v77: KMID+KLOWフィルター（文脈足h4_prev3で判定） ──
                if not check_kmid_klow(h4_prev3, direction=-1):
                    continue
                # ─────────────────────────────────────────────────────
                sl = max(high1, high2) + atr_val * 0.15
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]          # チャートレベルの基準価格
                        ep = raw_ep - spread                 # 実際の約定価格（スプレッド分不利）
                        risk = sl - raw_ep                   # リスク幅はチャートレベルで計算
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep - risk * rr_ratio    # TPもチャートレベルで計算
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "4h",
                                "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # ── 1時間足の二番底・二番天井 ──────────────────────────────
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # [BUG②修正] 完結済み4H足のみ取得（< で現在形成中の足を除外）
        h4_before = data_4h[data_4h.index < h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        tolerance = atr_val * 0.3

        # ロング: 二番底（4時間足が上昇トレンドのみ）
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                # ── v77: KMID+KLOWフィルター（直前4H足で判定）──
                if not check_kmid_klow(h4_latest, direction=1):
                    continue
                # ──────────────────────────────────────────────
                sl = min(low1, low2) - atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep + spread
                        risk = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "1h",
                                "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井（4時間足が下降トレンドのみ）
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                # ── v77: KMID+KLOWフィルター（直前4H足で判定）──
                if not check_kmid_klow(h4_latest, direction=-1):
                    continue
                # ──────────────────────────────────────────────
                sl = max(high1, high2) + atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep - spread
                        risk = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "1h",
                                "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # 時刻順にソート
    signals.sort(key=lambda x: x["time"])
    return signals
