"""
yagami_candle_patterns.py
===========================
やがみメソッド（ローソク足の本1・本2）に基づくローソク足パターン検出関数群。

【設計方針】
- check_kmid_klow（v77）と同形式で実装
- パラメータはすべてATR比率で表現（絶対値禁止）
- 閾値は経済的・構造的根拠で固定（OOSデータ非参照）

【PDFで確認した主要定義】
  大陽線     : 実体が足全体を支配（body/range≥0.7相当）。逆張り禁止
  下ヒゲ陽線 : 実体上部・長い下ヒゲ。底打ちシグナル
  リバーサルロー/ハイ: 最強の反転パターン。前落幅を包む大陽線/大陰線
  インサイドバー継続: 高値安値を更新せず複数本継続。チャートパターン形成の基礎
  エンゴルフィング: 前回足の実体を包む足。転換点で鉄板
  上下ヒゲ中ピンバー: 実体が中央・両ヒゲ大。レンジ相場のサイン（エントリー不可）
  実体が揃う: 複数足の実体が同一価格帯に集中。反発強度の指標

【バックテスト知見との対応】
  F2_pin_bar（谷足逆ヒゲ≥body×2）がFX/USDJPYで最強 → リバーサルロー/ハイ に包含
  F3_no_shadow（確認足逆ヒゲ<body×0.5）がXAUUSDで有効 → candle_strength_score に反映
"""

import numpy as np
import pandas as pd


# ── ATR計算ユーティリティ ──────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range の移動平均（ATR）"""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def _body(bar: pd.Series) -> float:
    """実体サイズ"""
    return abs(bar["close"] - bar["open"])


def _range(bar: pd.Series) -> float:
    """High - Low"""
    return bar["high"] - bar["low"]


def _upper_wick(bar: pd.Series) -> float:
    """上ヒゲ = high - max(open, close)"""
    return bar["high"] - max(bar["open"], bar["close"])


def _lower_wick(bar: pd.Series) -> float:
    """下ヒゲ = min(open, close) - low"""
    return min(bar["open"], bar["close"]) - bar["low"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ローソク足強弱スコア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def candle_strength_score(df: pd.DataFrame, atr_period: int = 14) -> float:
    """
    直近2本のローソク足からローソク足強弱スコアを返す。

    PDFのやがみメソッド7分類を数値化し、加重合計で -1.0 〜 +1.0 を返す。
    ロング方向がプラス、ショート方向がマイナス。

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame（最低3本以上必要。直近2本を使用）
    atr_period : int
        ATR計算期間

    Returns
    -------
    float
        -1.0 〜 +1.0  （0 = ニュートラル / エンゴルフィング等で±1.0超えを±1.0にclamp）

    分類とスコア（PDFのやがみメソッド7種 + バックテスト知見を反映）
    ---
    分類                   スコア    根拠
    大陽線                 +1.0     body/range≥0.7、逆張り厳禁（PDF書1）
    大陰線                 -1.0     同上
    エンゴルフィング陽線   +0.85    前回足実体を包む転換→鉄板ロング（PDF書1）
    エンゴルフィング陰線   -0.85    同上逆
    リバーサルロー         +0.90    最強底打ちパターン、F2_pin_bar知見（PDF書1+BT）
    リバーサルハイ         -0.90    同上逆
    下ヒゲ陽線             +0.65    底打ちシグナル・ヒゲ長いほど強い（PDF書1）
    上ヒゲ陰線             -0.65    同上逆
    上下ヒゲ中ピンバー     0.0      何もできない足（PDF書1: 逆×、順×）
    インサイドバー         ±0.25   前足の方向を継続（PDF書2）
    中陽線 / 中陰線        ±0.20   モブ（PDF書1）
    """
    if len(df) < 3:
        return 0.0

    atr_series = _atr(df, atr_period)
    atr_val = atr_series.iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return 0.0

    cur  = df.iloc[-1]   # 最新足
    prev = df.iloc[-2]   # 直前足

    body_cur  = _body(cur)
    rng_cur   = _range(cur)
    uw_cur    = _upper_wick(cur)
    lw_cur    = _lower_wick(cur)

    body_prev = _body(prev)

    if rng_cur <= 0:
        return 0.0

    body_ratio = body_cur / rng_cur
    is_bullish = cur["close"] > cur["open"]
    is_bearish = cur["close"] < cur["open"]

    score = 0.0

    # ─ 1. 上下ヒゲ中ピンバー（最優先: 何もできない）─────────────
    # 条件: 上下両ヒゲともにbody以上（body が中央に存在）
    # PDF: 「上下髭中ピンバーが出たらポジション不可」
    if body_cur > 0 and uw_cur >= body_cur * 0.8 and lw_cur >= body_cur * 0.8:
        return 0.0  # ニュートラル確定

    # ─ 2. 大陽線 / 大陰線（body/range ≥ 0.7）───────────────────
    # PDF: 「大陽線の中で逆張りは一切しない」
    # 閾値0.7 = ヒゲ30%以下、実体が支配的。業界標準水準
    if body_ratio >= 0.7:
        score = 1.0 if is_bullish else -1.0
        return float(np.clip(score, -1.0, 1.0))

    # ─ 3. リバーサルロー / リバーサルハイ ──────────────────────
    # PDF: 「最強のパターン。ほぼ安値確定」
    # 定義: 前足が陰線 かつ 現足が陽線で前足の実体を包む（body基準）
    # + 下ヒゲがbody×2以上（F2_pin_barのバックテスト知見を反映）
    reversal_low  = (
        is_bullish and
        prev["close"] < prev["open"] and       # 前足が陰線
        cur["close"] >= prev["open"] and        # 現足closeが前足openを超え（包む）
        lw_cur >= body_cur * 1.5               # 下ヒゲ強い（PDF: 長ければ長いほど強い）
    )
    reversal_high = (
        is_bearish and
        prev["close"] > prev["open"] and
        cur["close"] <= prev["open"] and
        uw_cur >= body_cur * 1.5
    )
    if reversal_low:
        return 0.90
    if reversal_high:
        return -0.90

    # ─ 4. エンゴルフィング（前回足の実体を包む足）──────────────
    # PDF: 「前回陰線を包む陽線。ロング鉄板」
    # 定義: 現足の実体がprev実体を完全に包む
    if is_bullish and body_prev > 0:
        if (cur["close"] >= max(prev["open"], prev["close"]) and
                cur["open"]  <= min(prev["open"], prev["close"])):
            return 0.85
    if is_bearish and body_prev > 0:
        if (cur["close"] <= min(prev["open"], prev["close"]) and
                cur["open"]  >= max(prev["open"], prev["close"])):
            return -0.85

    # ─ 5. 下ヒゲ陽線 / 上ヒゲ陰線 ─────────────────────────────
    # PDF: 「底打ちシグナル。下ヒゲ長いほど強い」
    # 条件: ヒゲが実体の1.5倍以上（F2ピンバーの教科書的2倍より緩い）
    # + 逆方向ヒゲがbody×0.5未満（F3_no_shadow知見: 逆抵抗が小さい）
    if is_bullish and body_cur > 0:
        if lw_cur >= body_cur * 1.5 and uw_cur < body_cur * 0.5:
            return 0.65
    if is_bearish and body_cur > 0:
        if uw_cur >= body_cur * 1.5 and lw_cur < body_cur * 0.5:
            return -0.65

    # ─ 6. インサイドバー（前足の値幅内に収まる）────────────────
    # PDF: 「インサイドバーを継続する仮定でポジション構築」
    # 方向は前足のclose方向を継承
    inside = (cur["high"] <= prev["high"] and cur["low"] >= prev["low"])
    if inside:
        prev_dir = 1 if prev["close"] > prev["open"] else -1
        return float(0.25 * prev_dir)

    # ─ 7. 中陽線 / 中陰線（モブ）───────────────────────────────
    # PDF: 「エントリーの指標にならないモブ」
    if is_bullish:
        return 0.20
    if is_bearish:
        return -0.20

    return 0.0  # 十字線


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. インサイドバー継続フィルター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_inside_bar_cluster(df: pd.DataFrame,
                              min_consecutive: int = 2,
                              body_atr_ratio: float = 0.3,
                              atr_period: int = 14) -> bool:
    """
    インサイドバー継続フィルター。

    PDFより:「インサイドバーが複数本連続している＋実体の幅に厚みがある」
    単独インサイドバーではエントリー厳禁。2本以上の連続が必要。

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame（最低 min_consecutive+1 本以上）
    min_consecutive : int
        最低連続インサイドバー本数（PDF: 複数本 → 2以上）
    body_atr_ratio : float
        実体の最小幅（ATR比率）。薄すぎる実体を除外（PDF: 実体に厚みが必要）
        固定値: ATR×0.3（v77の許容幅基準と同じスケール）
    atr_period : int
        ATR計算期間

    Returns
    -------
    bool
        True: インサイドバーが min_consecutive 本以上連続 かつ 実体が十分厚い
    """
    need = min_consecutive + 1   # 親足 + 連続N本
    if len(df) < need:
        return False

    atr_val = _atr(df, atr_period).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return False

    # 直近 min_consecutive 本がすべてインサイドバーか確認
    consecutive = 0
    for i in range(1, min_consecutive + 1):
        cur  = df.iloc[-i]
        prev = df.iloc[-i - 1]
        is_inside = (cur["high"] <= prev["high"]) and (cur["low"] >= prev["low"])
        body_ok   = _body(cur) >= atr_val * body_atr_ratio
        if is_inside and body_ok:
            consecutive += 1
        else:
            break

    return consecutive >= min_consecutive


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 実体ゾーン揃い判定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_body_zone_aligned(df: pd.DataFrame,
                             lookback: int = 5,
                             zone_atr_ratio: float = 1.5,
                             min_body_atr_ratio: float = 0.2,
                             atr_period: int = 14) -> bool:
    """
    実体ゾーン揃い判定。

    PDFより:「複数足の実体（始値・終値）が同一価格帯に集中 + 実体に厚みあり」
    反発強度の指標。実体が揃っている = レジサポの信頼度が高い。

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame
    lookback : int
        確認本数（直近N本）
    zone_atr_ratio : float
        実体が集中すべきATR幅の倍率（PDF: 銘柄次第で定義不可 → ATR×1.5を基準）
        固定値: 1.5（v77のパターン許容幅ATR×0.3の5倍 = 広めだが収束判定として妥当）
    min_body_atr_ratio : float
        各足の実体最小幅（ATR比率）。薄い実体はゾーン計算から除外
        固定値: 0.2（v78Cと同基準）
    atr_period : int
        ATR計算期間

    Returns
    -------
    bool
        True: lookback本の実体がATR×zone_atr_ratio の帯域内に収まっている
    """
    if len(df) < lookback + atr_period:
        return False

    atr_val = _atr(df, atr_period).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return False

    recent = df.iloc[-lookback:]
    zone_width = atr_val * zone_atr_ratio
    min_body   = atr_val * min_body_atr_ratio

    # 各足の open/close を収集（実体が thin な足は除外）
    prices = []
    for _, bar in recent.iterrows():
        if _body(bar) >= min_body:
            prices.append(bar["open"])
            prices.append(bar["close"])

    if len(prices) < 4:   # 最低2本分（4点）必要
        return False

    # 全open/closeがzone_width以内に収まるか
    price_range = max(prices) - min(prices)
    return price_range <= zone_width


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 足確定タイミングフィルター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_candle_close_timing(ts: pd.Timestamp,
                               timeframe: str = "15m",
                               window_minutes: int = 2) -> bool:
    """
    足確定タイミングフィルター。

    PDFより:「足更新でポジションを取る。2分以内が超初動の狙い目」
    「5分刻みでポジションを取る。中途半端な時間は消耗の原因」

    Parameters
    ----------
    ts : pd.Timestamp
        エントリー時刻（UTC）
    timeframe : str
        時間足（'1m', '5m', '15m', '1h', '4h', '1d'）
    window_minutes : int
        足確定前の許容ウィンドウ（分）。PDF: 2分以内が理想
        固定値: 2（PDF直接記載）

    Returns
    -------
    bool
        True: 足確定の window_minutes 分前以内
    """
    tf_minutes = {
        "1m":  1,
        "5m":  5,
        "15m": 15,
        "30m": 30,
        "1h":  60,
        "2h":  120,
        "4h":  240,
        "1d":  1440,
    }
    if timeframe not in tf_minutes:
        return False

    period = tf_minutes[timeframe]
    minute_in_period = (ts.hour * 60 + ts.minute) % period
    # 足確定 = minute_in_period が 0 の瞬間
    # window: (period - window_minutes) 〜 period-1 の間
    threshold = period - window_minutes
    return minute_in_period >= threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 上位足方向確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_higher_tf_direction(df_1h: pd.DataFrame,
                               df_4h: pd.DataFrame) -> int:
    """
    上位足方向確認（MTF整合性チェック）。

    PDFより:「上位足の色に順張り。1Hと4Hの色が一致する場合のみ有効」
    「4H更新時に多くの時間軸のトレーダーが同方向にポジション構築する」

    Parameters
    ----------
    df_1h : pd.DataFrame
        1時間足 OHLC DataFrame
    df_4h : pd.DataFrame
        4時間足 OHLC DataFrame

    Returns
    -------
    int
        +1 : ロング優勢（1H・4Hとも陽線）
        -1 : ショート優勢（1H・4Hとも陰線）
         0 : 中立（不一致 / データ不足）
    """
    if len(df_1h) < 2 or len(df_4h) < 2:
        return 0

    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]

    dir_1h = 1 if last_1h["close"] > last_1h["open"] else -1
    dir_4h = 1 if last_4h["close"] > last_4h["open"] else -1

    # PDF: 「1Hと4Hの色が一致する場合のみ有効」
    if dir_1h == dir_4h:
        return dir_4h
    return 0
