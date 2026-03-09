"""
backtest_candle_filters.py
===========================
5つのローソク足フィルターを v77 ベースラインに追加し、
IS/OOS 両期間で効果を検証する。

【設計方針（過学習防止）】
  - 5つのフィルターは閾値を経済的根拠で固定（OOSデータを一切参照しない）
  - 各フィルターは単独適用（組み合わせ最適化しない）
  - OOS採用基準: OOS PF ≥ IS PF × 0.7 かつ OOS PF > v77 OOS PF

【5つのフィルター】
  F1: 確認足実体比率 ≥ 0.5  （body/range、方向確信の最低水準）
  F2: 谷足ピンバー           （逆ヒゲ > body×2、反転の証拠、教科書的定義）
  F3: 確認足逆ヒゲ抑制       （逆方向ヒゲ < body×0.5、ラリー途中でない確認）
  F4: 確認足エングルフィング  （confirm_body > prev_body、勢いの増大）
  F5: 谷足下値拒絶           （谷足low < 実体中央値、実際に押された形）

【銘柄】
  FX     : EURUSD / GBPUSD / AUDUSD
  METALS : XAUUSD
  INDICES: US30 / SPX500 / NAS100
  JPY    : USDJPY（1m なし、15m で代用）

IS : 2024-07-01 〜 2025-02-28
OOS: 2025-03-03 〜 2026-02-27
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 定数 ──────────────────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
KLOW_THR  = 0.0015    # v77継承・固定
IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# OOS採用基準
OOS_PASS_RATIO = 0.7  # OOS PF ≥ IS PF × 0.7

# ── 銘柄定義 ──────────────────────────────────────────────────────
SYMBOLS = [
    # (display, sym_lower, category, use_15m_entry)
    ("EURUSD", "eurusd", "FX",      False),
    ("GBPUSD", "gbpusd", "FX",      False),
    ("AUDUSD", "audusd", "FX",      False),
    ("XAUUSD", "xauusd", "METALS",  False),
    ("US30",   "us30",   "INDICES", False),
    ("SPX500", "spx500", "INDICES", False),
    ("NAS100", "nas100", "INDICES", False),
    ("USDJPY", "usdjpy", "JPY",     True),   # 1m なし → 15m 代用
]

# ── フィルター定義 ────────────────────────────────────────────────
# 各フィルターは独立して適用（組み合わせなし）
# フラグ名が generate_signals_1h の引数名と対応する
VARIANTS = [
    # (name, filter_kwargs)
    ("v77",    dict(f1=False, f2=False, f3=False, f4=False, f5=False)),
    ("F1_body_ratio",  dict(f1=True,  f2=False, f3=False, f4=False, f5=False)),
    ("F2_pin_bar",     dict(f1=False, f2=True,  f3=False, f4=False, f5=False)),
    ("F3_no_shadow",   dict(f1=False, f2=False, f3=True,  f4=False, f5=False)),
    ("F4_engulf",      dict(f1=False, f2=False, f3=False, f4=True,  f5=False)),
    ("F5_low_reject",  dict(f1=False, f2=False, f3=False, f4=False, f5=True)),
]

# ── データロード ────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def slice_period(df, start, end):
    if df is None:
        return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def load_data(sym_lower, pair_name, is_period):
    """
    優先順位:
      1. data/{sym}_{is|oos}_{tf}.csv
      2. data/ohlc/{PAIR}_{tf}.csv （全期間 → slice）
    1m が見つからない場合は None を返す（JPY系は呼ばない）
    """
    tag   = "is" if is_period else "oos"
    start = IS_START if is_period else OOS_START
    end   = IS_END   if is_period else OOS_END

    def try_load(tf):
        # 試し1: is/oos 分割ファイル
        p = os.path.join(DATA_DIR, f"{sym_lower}_{tag}_{tf}.csv")
        df = load_csv(p)
        if df is not None and len(df) > 0:
            return slice_period(df, start, end)
        # 試し2: ohlc/ 全期間ファイル（大文字）
        p2 = os.path.join(DATA_DIR, "ohlc", f"{pair_name}_{tf}.csv")
        df2 = load_csv(p2)
        if df2 is not None and len(df2) > 0:
            return slice_period(df2, start, end)
        return None

    d1m  = try_load("1m")
    d15m = try_load("15m")
    d4h  = try_load("4h")
    return d1m, d15m, d4h

# ── インジケーター ──────────────────────────────────────────────
def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def add_4h_indicators(df):
    df = df.copy()
    df["atr"]   = calculate_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ok = (body_bottom - l) / o < KLOW_THR if o > 0 else False
    return kmid_ok and klow_ok

# ── 5フィルター関数 ────────────────────────────────────────────
def filter_f1_body_ratio(confirm_bar, direction):
    """F1: 確認足 実体/レンジ ≥ 0.5（固定閾値: 自然な中間値）"""
    body  = abs(confirm_bar["close"] - confirm_bar["open"])
    rng   = confirm_bar["high"] - confirm_bar["low"]
    if rng <= 0:
        return False
    return (body / rng) >= 0.5

def filter_f2_pin_bar(pattern_bar, direction):
    """
    F2: 谷足（二番底）または天頂足（二番天井）にピンバー
    ロング: 下ヒゲ > body × 2（押し目での強い反転）
    ショート: 上ヒゲ > body × 2
    固定閾値: 2.0（ピンバーの教科書的定義）
    """
    body = abs(pattern_bar["close"] - pattern_bar["open"])
    if body == 0:
        body = 1e-10  # ゼロ除算防止（十字線もピンバー候補）
    if direction == 1:
        wick = min(pattern_bar["open"], pattern_bar["close"]) - pattern_bar["low"]
    else:
        wick = pattern_bar["high"] - max(pattern_bar["open"], pattern_bar["close"])
    return wick >= body * 2.0

def filter_f3_no_shadow(confirm_bar, direction):
    """
    F3: 確認足の逆方向ヒゲ < body × 0.5
    ロング: 上ヒゲ（利食い圧力）を抑制
    ショート: 下ヒゲ（買い戻し圧力）を抑制
    固定閾値: 0.5（実体の半分以内 = 抵抗は小さい）
    """
    body = abs(confirm_bar["close"] - confirm_bar["open"])
    if body == 0:
        return False  # 十字線は方向不明なので除外
    if direction == 1:
        # ロング: 上ヒゲ = high - close
        counter_wick = confirm_bar["high"] - confirm_bar["close"]
    else:
        # ショート: 下ヒゲ = open - low (close < open なので)
        counter_wick = confirm_bar["open"] - confirm_bar["low"]
    return counter_wick < body * 0.5

def filter_f4_engulf(confirm_bar, prev_bar):
    """
    F4: 確認足の実体 > 直前足の実体（エングルフィング的な勢い増大）
    パラメータなし、純粋な大小比較
    """
    confirm_body = abs(confirm_bar["close"] - confirm_bar["open"])
    prev_body    = abs(prev_bar["close"]    - prev_bar["open"])
    return confirm_body > prev_body

def filter_f5_low_reject(pattern_bar, direction):
    """
    F5: 谷足/天頂足で価格が実体中央を越えた下値/上値を試した（ヒゲによる拒絶）
    ロング: pattern_bar.low < 実体中央値（下値を試してから反発）
    ショート: pattern_bar.high > 実体中央値（上値を試してから反落）
    パラメータなし、実体中央との比較（相対的な構造的条件）
    """
    body_mid = (pattern_bar["open"] + pattern_bar["close"]) / 2
    if direction == 1:
        return pattern_bar["low"] < body_mid
    else:
        return pattern_bar["high"] > body_mid

# ── シグナル生成 ────────────────────────────────────────────────
def generate_signals(data_entry, data_15m, data_4h,
                     spread_pips, pip_size,
                     f1=False, f2=False, f3=False, f4=False, f5=False,
                     entry_is_15m=False):
    """
    data_entry: 1m足（通常）または 15m足（USDJPY）
    entry_is_15m: Trueの場合、1m足の代わりに15m足でエントリー
    """
    spread = spread_pips * pip_size

    data_4h = add_4h_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    signals   = []
    used_times = set()
    h1_times  = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]  # 直前足（確認足）
        h1_prev2 = data_1h.iloc[i - 2]  # 2本前足（パターン足）
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # 完結済み4H足
        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)):
            continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tol    = atr_val * 0.3

        for direction in [1, -1]:
            if trend != direction:
                continue

            if direction == 1:
                v1, v2  = h1_prev2["low"],  h1_prev1["low"]
                conf_ok = h1_prev1["close"] > h1_prev1["open"]
                pattern_bar = h1_prev1  # 二番底は prev1 が谷
            else:
                v1, v2  = h1_prev2["high"], h1_prev1["high"]
                conf_ok = h1_prev1["close"] < h1_prev1["open"]
                pattern_bar = h1_prev1  # 二番天井は prev1 が天頂

            if abs(v1 - v2) > tol:
                continue
            if not conf_ok:
                continue

            # v77 KMID + KLOW（4H文脈足）
            if not check_kmid_klow(h4_latest, direction):
                continue

            # ── 5つのフィルター（各々独立、固定閾値） ────────────
            # F1: 確認足実体比率 ≥ 0.5
            if f1 and not filter_f1_body_ratio(h1_prev1, direction):
                continue

            # F2: 谷足（pattern_bar = h1_prev1）ピンバー
            if f2 and not filter_f2_pin_bar(pattern_bar, direction):
                continue

            # F3: 確認足の逆ヒゲ < body × 0.5
            if f3 and not filter_f3_no_shadow(h1_prev1, direction):
                continue

            # F4: 確認足実体 > 2本前足実体（エングルフィング）
            if f4 and not filter_f4_engulf(h1_prev1, h1_prev2):
                continue

            # F5: 谷足low < 実体中央（下値拒絶の証拠）
            if f5 and not filter_f5_low_reject(pattern_bar, direction):
                continue

            # エントリー足を取得（1m または 15m）
            window_end = h1_ct + pd.Timedelta(minutes=2 if not entry_is_15m else 15)
            entry_bars = data_entry[
                (data_entry.index >= h1_ct) &
                (data_entry.index <  window_end)
            ]
            if len(entry_bars) == 0:
                continue
            eb = entry_bars.iloc[0]
            et = eb.name
            if et in used_times:
                continue

            raw = eb["open"]
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                ep   = raw + spread
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                ep   = raw - spread
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ────────────────────────────────────────────
def simulate(signals, data_entry, symbol):
    if not signals:
        return [], [INIT_CASH]
    rm     = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH
    trades = []
    eq_curve = [INIT_CASH]

    for sig in signals:
        direction = sig["dir"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot    = rm.calc_lot(equity, risk, ep, usdjpy_rate=150.0)
        future = data_entry[data_entry.index > sig["time"]]
        if len(future) == 0:
            continue

        half_done = False
        be_sl     = None
        result    = None
        exit_price = None
        exit_time  = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                cur_sl = be_sl if half_done else sl
                if bar["low"] <= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(1, ep, exit_price, lot * rem, 150.0, ep)
                    equity += pnl
                    result = "win" if pnl > 0 else "loss"
                    break
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * HALF_R:
                        equity += rm.calc_pnl_jpy(1, ep, ep + risk * HALF_R, lot * 0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(1, ep, tp, lot * rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["high"] >= ep + risk * HALF_R:
                    equity += rm.calc_pnl_jpy(1, ep, ep + risk * HALF_R, lot * 0.5, 150.0, ep)
                    half_done = True; be_sl = ep
            else:
                cur_sl = be_sl if half_done else sl
                if bar["high"] >= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(-1, ep, exit_price, lot * rem, 150.0, ep)
                    equity += pnl
                    result = "win" if pnl > 0 else "loss"
                    break
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * HALF_R:
                        equity += rm.calc_pnl_jpy(-1, ep, ep - risk * HALF_R, lot * 0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(-1, ep, tp, lot * rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["low"] <= ep - risk * HALF_R:
                    equity += rm.calc_pnl_jpy(-1, ep, ep - risk * HALF_R, lot * 0.5, 150.0, ep)
                    half_done = True; be_sl = ep

        if result is None:
            continue
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve

# ── 統計計算 ────────────────────────────────────────────────────
def calc_stats(trades, eq_curve, pair, variant, period):
    if not trades:
        return {
            "pair": pair, "variant": variant, "period": period,
            "n": 0, "wr": 0.0, "pf": 0.0, "mdd": 0.0,
            "kelly": -1.0, "monthly_plus": "0/0"
        }
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["result"] == "win"]
    wr   = len(wins) / n

    eq     = np.array(eq_curve)
    deltas = np.diff(eq)
    gw     = deltas[deltas > 0].sum()
    gl     = abs(deltas[deltas < 0].sum())
    pf     = gw / gl if gl > 0 else float("inf")

    peak  = np.maximum.accumulate(eq)
    mdd   = abs(((eq - peak) / peak).min()) * 100
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev_eq = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > prev_eq).sum()}/{len(monthly)}"

    return {
        "pair":         pair,
        "variant":      variant,
        "period":       period,
        "n":            n,
        "wr":           round(wr * 100, 1),
        "pf":           round(pf, 2),
        "mdd":          round(mdd, 1),
        "kelly":        round(kelly, 3),
        "monthly_plus": mp,
    }

# ── OOS採用チェック ────────────────────────────────────────────
def oos_survival_check(is_pf, oos_pf, baseline_oos_pf):
    """
    採用条件（AND）:
      1. OOS PF ≥ IS PF × 0.7  （過学習チェック）
      2. OOS PF > baseline（v77）のOOS PF  （ベースライン超え）
    """
    if is_pf == 0:
        return False, "IS=0"
    ratio = oos_pf / is_pf
    cond1 = ratio >= OOS_PASS_RATIO
    cond2 = oos_pf > baseline_oos_pf
    if cond1 and cond2:
        return True, f"✅ OOS/IS={ratio:.2f} PF↑{oos_pf - baseline_oos_pf:+.2f}"
    elif not cond1:
        return False, f"❌ OOS/IS={ratio:.2f}<{OOS_PASS_RATIO}"
    else:
        return False, f"❌ OOS PF {oos_pf:.2f}≤v77({baseline_oos_pf:.2f})"

# ── メイン ──────────────────────────────────────────────────────
print("=" * 110)
print("ローソク足フィルター 5種類 IS/OOS バックテスト（過学習対策版）")
print(f"IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
print(f"採用基準: OOS PF ≥ IS PF × {OOS_PASS_RATIO}  かつ  OOS PF > v77 OOS PF")
print()
print("【フィルター定義（全て固定閾値・OOSデータ非参照）】")
print("  F1_body_ratio : 確認足実体/レンジ ≥ 0.5（自然な中間値）")
print("  F2_pin_bar    : 谷足/天頂足の逆ヒゲ ≥ body×2（教科書的ピンバー）")
print("  F3_no_shadow  : 確認足の逆ヒゲ < body×0.5（利食い/買い戻し圧力小）")
print("  F4_engulf     : 確認足実体 > 直前足実体（勢いの増大、比率比較のみ）")
print("  F5_low_reject : 谷足/天頂足のlow/high が実体中央を超えた（拒絶の証拠）")
print("=" * 110)

all_results = []

for pair_name, sym_lower, category, use_15m_entry in SYMBOLS:
    try:
        rm     = RiskManager(pair_name, risk_pct=RISK_PCT)
        spread = rm.spread_pips
        pip    = rm.pip_size
    except Exception as e:
        print(f"\n{pair_name}: [SKIP] RiskManager エラー: {e}")
        continue

    print(f"\n{'─'*110}")
    print(f"  {pair_name}  [{category}]  spread:{spread}pips"
          f"{'  ※15m足でエントリー代用' if use_15m_entry else ''}")

    # データロード（IS / OOS）
    d1m_is,  d15m_is,  d4h_is  = load_data(sym_lower, pair_name, is_period=True)
    d1m_oos, d15m_oos, d4h_oos = load_data(sym_lower, pair_name, is_period=False)

    if use_15m_entry:
        # USDJPY: 15m を entry 足として使用（1m は不要）
        entry_is  = d15m_is
        entry_oos = d15m_oos
    else:
        entry_is  = d1m_is
        entry_oos = d1m_oos

    missing = [(n, d) for n, d in [
        ("entry_IS",  entry_is),  ("15m_IS",  d15m_is),  ("4h_IS",  d4h_is),
        ("entry_OOS", entry_oos), ("15m_OOS", d15m_oos), ("4h_OOS", d4h_oos),
    ] if d is None or len(d) == 0]

    if missing:
        print(f"  [SKIP] データ不足: {[n for n, _ in missing]}")
        continue

    print(f"  {'バリアント':<18} {'期間':<4} {'件数':>5} {'勝率':>7} "
          f"{'PF':>6} {'MDD':>7} {'Kelly':>7} {'月次+':<8}  OOS採用判定")
    print(f"  {'─'*100}")

    pair_results = {}  # variant -> {IS: stats, OOS: stats}

    for vname, vflags in VARIANTS:
        pair_results[vname] = {}
        for period_label, entry_df, d15m, d4h in [
            ("IS",  entry_is,  d15m_is,  d4h_is),
            ("OOS", entry_oos, d15m_oos, d4h_oos),
        ]:
            sigs = generate_signals(
                entry_df, d15m, d4h,
                spread, pip,
                entry_is_15m=use_15m_entry,
                **vflags
            )
            trades, eq = simulate(sigs, entry_df, pair_name)
            st = calc_stats(trades, eq, pair_name, vname, period_label)
            pair_results[vname][period_label] = st
            all_results.append(st)

            # v77 は採用判定なし
            if vname == "v77":
                print(f"  {vname:<18} {period_label:<4} {st['n']:>5} "
                      f"{st['wr']:>6.1f}% {st['pf']:>6.2f} "
                      f"{st['mdd']:>6.1f}% {st['kelly']:>7.3f} "
                      f"{st['monthly_plus']:<8}")

        # フィルター行: IS + OOS を並べて採用判定
        if vname != "v77":
            is_st  = pair_results[vname]["IS"]
            oos_st = pair_results[vname]["OOS"]
            v77_oos_pf = pair_results["v77"]["OOS"]["pf"]
            adopted, reason = oos_survival_check(is_st["pf"], oos_st["pf"], v77_oos_pf)
            print(f"  {vname:<18} IS   {is_st['n']:>5} "
                  f"{is_st['wr']:>6.1f}% {is_st['pf']:>6.2f} "
                  f"{is_st['mdd']:>6.1f}% {is_st['kelly']:>7.3f} "
                  f"{is_st['monthly_plus']:<8}")
            print(f"  {'':18} OOS  {oos_st['n']:>5} "
                  f"{oos_st['wr']:>6.1f}% {oos_st['pf']:>6.2f} "
                  f"{oos_st['mdd']:>6.1f}% {oos_st['kelly']:>7.3f} "
                  f"{oos_st['monthly_plus']:<8}  {reason}")

# ── フィルター横断サマリー ──────────────────────────────────────
print("\n" + "=" * 110)
print("フィルター横断サマリー（OOS PF）")
print("=" * 110)

df_all = pd.DataFrame(all_results)

variants_list = [v for v, _ in VARIANTS]
pairs_list    = [p for p, _, _, _ in SYMBOLS if
                 len(df_all[(df_all["pair"] == p) & (df_all["period"] == "OOS")]) > 0]

# ヘッダ
header = f"  {'銘柄':<8} {'Cat':<8}"
for vname in variants_list:
    header += f"  {vname:<18}"
print(header)
print(f"  {'─'*100}")

# 各銘柄行
adoption_map = {}  # (pair, variant) -> adopted bool

for pair_name, _, category, _ in SYMBOLS:
    row_v77 = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == "v77") & (df_all["period"] == "OOS")]
    if len(row_v77) == 0:
        continue
    v77_oos_pf = row_v77.iloc[0]["pf"]

    line = f"  {pair_name:<8} {category:<8}"
    for vname in variants_list:
        row_is  = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == vname) & (df_all["period"] == "IS")]
        row_oos = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == vname) & (df_all["period"] == "OOS")]
        if len(row_is) == 0 or len(row_oos) == 0:
            line += f"  {'N/A':<18}"
            continue
        is_pf  = row_is.iloc[0]["pf"]
        oos_pf = row_oos.iloc[0]["pf"]

        if vname == "v77":
            line += f"  {oos_pf:5.2f}{'':>13}"
        else:
            adopted, _ = oos_survival_check(is_pf, oos_pf, v77_oos_pf)
            adoption_map[(pair_name, vname)] = adopted
            mark = "✅" if adopted else "  "
            delta = oos_pf - v77_oos_pf
            line += f"  {mark}{oos_pf:5.2f}({delta:+.2f}){'':>5}"
    print(line)

# フィルター別採用銘柄数まとめ
print(f"\n  {'─'*100}")
print(f"  {'合計採用':<8} {'':8}", end="")
for vname in variants_list:
    if vname == "v77":
        print(f"  {'baseline':<18}", end="")
    else:
        count = sum(1 for (p, v), ok in adoption_map.items() if v == vname and ok)
        total = sum(1 for (p, v) in adoption_map if v == vname)
        print(f"  {count}/{total}銘柄採用{'':>9}", end="")
print()

# ── カテゴリ別サマリー ──────────────────────────────────────────
print("\n" + "=" * 110)
print("カテゴリ別 採用状況")
print("=" * 110)

categories = {}
for pair_name, _, cat, _ in SYMBOLS:
    categories.setdefault(cat, []).append(pair_name)

for cat, pairs in categories.items():
    print(f"\n  [{cat}]  銘柄: {', '.join(pairs)}")
    for vname in variants_list:
        if vname == "v77":
            continue
        cat_adopted = [(p, adoption_map.get((p, vname), False)) for p in pairs
                       if (p, vname) in adoption_map]
        if not cat_adopted:
            continue
        n_ok = sum(1 for _, ok in cat_adopted if ok)
        marks = "  ".join(f"{'✅' if ok else '❌'}{p}" for p, ok in cat_adopted)
        print(f"    {vname:<18}: {n_ok}/{len(cat_adopted)}  {marks}")

# ── CSV保存 ──────────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, "candle_filters.csv")
df_all.to_csv(csv_out, index=False)
print(f"\n結果CSV: {csv_out}")

# ── 可視化 ──────────────────────────────────────────────────────
cat_colors = {
    "FX":      "#3b82f6",
    "METALS":  "#f59e0b",
    "INDICES": "#10b981",
    "JPY":     "#8b5cf6",
}
filter_names = [v for v, _ in VARIANTS if v != "v77"]

n_pairs = len(pairs_list)
fig, axes = plt.subplots(2, max(4, (n_pairs + 1) // 2), figsize=(22, 10))
axes = axes.flatten()

for idx, (pair_name, _, category, _) in enumerate(
    [(p, s, c, u) for p, s, c, u in SYMBOLS if p in pairs_list]
):
    ax = axes[idx]
    color = cat_colors.get(category, "#6b7280")

    # v77 baseline
    row_v77 = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == "v77") & (df_all["period"] == "OOS")]
    v77_pf  = row_v77.iloc[0]["pf"] if len(row_v77) > 0 else 0

    is_pfs  = []
    oos_pfs = []
    labels  = []
    adopted = []

    for vname in filter_names:
        row_is  = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == vname) & (df_all["period"] == "IS")]
        row_oos = df_all[(df_all["pair"] == pair_name) & (df_all["variant"] == vname) & (df_all["period"] == "OOS")]
        if len(row_is) == 0 or len(row_oos) == 0:
            continue
        is_pfs.append(row_is.iloc[0]["pf"])
        oos_pfs.append(row_oos.iloc[0]["pf"])
        labels.append(vname.replace("_", "\n"))
        adopted.append(adoption_map.get((pair_name, vname), False))

    x = range(len(labels))
    bar_colors = [("#22c55e" if ok else "#ef4444") for ok in adopted]
    bars = ax.bar(x, oos_pfs, color=bar_colors, alpha=0.7, label="OOS PF")
    ax.plot(x, is_pfs, "ko--", markersize=4, linewidth=1, label="IS PF", alpha=0.6)
    ax.axhline(v77_pf, color="blue", linestyle="--", linewidth=1.5, alpha=0.8, label=f"v77 OOS PF={v77_pf:.2f}")
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_title(f"{pair_name} [{category}]", fontsize=9, fontweight="bold", color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("PF", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # 採用マーク
    for xi, (pf, ok) in enumerate(zip(oos_pfs, adopted)):
        if ok:
            ax.text(xi, pf + 0.05, "✓", ha="center", fontsize=10, color="green")

# 未使用軸を非表示
for ax in axes[len(pairs_list):]:
    ax.set_visible(False)

fig.suptitle(
    "ローソク足フィルター OOS効果検証\n"
    "緑=採用（OOS/IS≥0.7 かつ OOS>v77）  赤=不採用\n"
    "黒点線=IS PF  青点線=v77 OOS PF  灰点=PF2.0",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
chart_out = os.path.join(OUT_DIR, "candle_filters.png")
plt.savefig(chart_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート: {chart_out}")
print("\n全処理完了。")
