"""
backtest_subtraction_aesthetics.py
====================================
「引き算の美学」— フィルター除去による感度検証

【目的】
現行 v79BC（FX）/ v79A（XAUUSD）+ E1/E2 エントリーの各フィルターを
1つずつ除去して、どの条件が「本当に必要か」を定量検証する。

フィルターが正当化されるには：
  「除いた時に OOS PF が下がる」こと
  除いても PF が維持 or 改善するなら → そのフィルターは不要（引き算対象）

【検証対象フィルター】（F1〜F3は戦略の根幹、固定）
  F4: KMID  — 4H文脈足の実体がエントリー方向
  F5: KLOW  — 4H文脈足の下ヒゲ < 0.15%（= 0.0015）
  F6: ADX ≥ 20           （FXカテゴリのみ、v79B）
  F7: Streak ≥ 4         （FXカテゴリのみ、v79C）
  F8: 日足EMA20 方向一致  （XAUUSDのみ、v79A）
  F9: セッション UTC7-22  （FX/XAUUSD共通）

【バリアント設計】
  baseline      : 全フィルター ON（現行 v79 + E1/E2 エントリー）
  -KMID         : F4 除去
  -KLOW         : F5 除去
  -KMID-KLOW    : F4+F5 除去（v77のキモを全部外す）
  -ADX          : F6 除去（FXのみ）
  -Streak       : F7 除去（FXのみ）
  -ADX-Streak   : F6+F7 除去 = 純v77相当（FXのみ）
  -1D_EMA       : F8 除去（XAUUSDのみ）
  -Session      : F9 除去（全銘柄）
  -conf_candle  : F3 除去（確認足方向チェックを外す）

【エントリー方式】
  FX:     E1（1m方向確認待ち 最大5分）
  XAUUSD: E2（スパイクフィルター）
  ← 前回 backtest_1m_entry_accuracy.py の推奨結果を採用

【期間】
  OOS: 2025-03-03 〜 2026-02-27

【過学習チェック基準】
  カテゴリPASS: フィルター除去が2/3銘柄（FX）or 1/1（XAUUSD）でPF改善 → 引き算推奨
  カテゴリFAIL: 除去でPFが低下 → そのフィルターは有効（存続）
"""
import os
import sys
import warnings
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
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
TOL_FACTOR  = 0.3
USDJPY_RATE = 150.0

OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# E1 パラメータ
E1_MAX_WAIT_MIN   = 5
E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3

# ── バリアント定義 ─────────────────────────────────────────────────
# 各バリアントは「除去するフィルター」をフラグで制御
# use_kmid, use_klow, use_adx, use_streak, use_1d_ema, use_session, use_conf_candle
VARIANTS = [
    # label,             use_kmid, use_klow, use_adx, use_streak, use_1d_ema, use_session, use_conf
    ("baseline",         True,     True,     True,    True,       True,       True,        True),
    ("-KMID",            False,    True,     True,    True,       True,       True,        True),
    ("-KLOW",            True,     False,    True,    True,       True,       True,        True),
    ("-KMID-KLOW",       False,    False,    True,    True,       True,       True,        True),
    ("-ADX",             True,     True,     False,   True,       True,       True,        True),
    ("-Streak",          True,     True,     True,    False,      True,       True,        True),
    ("-ADX-Streak",      True,     True,     False,   False,      True,       True,        True),
    ("-1D_EMA",          True,     True,     True,    True,       False,      True,        True),
    ("-Session",         True,     True,     True,    True,       True,       False,       True),
    ("-conf_candle",     True,     True,     True,    True,       True,       True,        False),
]

# ── 銘柄設定 ──────────────────────────────────────────────────────
SYMBOLS = [
    {
        "name":        "EURUSD",
        "lower":       "eurusd",
        "category":    "FX",
        "entry_method": "E1",     # 前回推奨
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "GBPUSD",
        "lower":       "gbpusd",
        "category":    "FX",
        "entry_method": "E1",
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "AUDUSD",
        "lower":       "audusd",
        "category":    "FX",
        "entry_method": "E1",
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "XAUUSD",
        "lower":       "xauusd",
        "category":    "METALS",
        "entry_method": "E2",    # 前回推奨
        "adx_min":     0,
        "streak_min":  0,
        "use_1d":      True,
        "utc_start":   0,
        "utc_end":     24,
    },
]


# ── データロード ──────────────────────────────────────────────────
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
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def load_data(sym_lower, sym_upper):
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")

    def _load(tf):
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p):
            return load_csv(p)
        p2 = os.path.join(DATA_DIR, f"{sym_lower}_oos_{tf}.csv")
        return load_csv(p2)

    return _load("1m"), _load("15m"), _load("4h")


# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def calc_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high - high.shift(1)
    down = low.shift(1) - low
    pdm  = np.where((up > down) & (up > 0),   up,   0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    a    = 1.0 / period
    atr_ = pd.Series(tr.values,  index=df.index).ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    mdi  = 100 * pd.Series(mdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.ewm(alpha=a, adjust=False).mean()


def build_indicators(df4h, need_adx, need_1d):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if need_adx:
        df["adx"] = calc_adx(df, 14)
    if need_1d:
        d1 = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None


# ── フィルター関数 ────────────────────────────────────────────────
def check_kmid(bar, direction):
    o, c = bar["open"], bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)


def check_klow(bar):
    o, c, l = bar["open"], bar["close"], bar["low"]
    body_bot = min(o, c)
    return (body_bot - l) / o < KLOW_THR if o > 0 else False


# ── エントリー価格決定（searchsorted で高速化）──────────────────
def pick_entry_1m(data_1m, signal_time, direction, spread, method,
                  atr_1m=None,
                  m1_idx=None, m1_opens=None, m1_closes=None,
                  m1_highs=None, m1_lows=None):
    """
    m1_idx/m1_opens/m1_closes/m1_highs/m1_lows: 事前キャッシュ済みnumpy配列
    """
    if m1_idx is None:
        m1_idx    = data_1m.index
        m1_opens  = data_1m["open"].values
        m1_closes = data_1m["close"].values
        m1_highs  = data_1m["high"].values
        m1_lows   = data_1m["low"].values

    start = m1_idx.searchsorted(signal_time, side="left")

    if method == "E1":
        end_time = signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            o = m1_opens[i]; c = m1_closes[i]
            if direction == 1 and c > o:
                pass
            elif direction == -1 and c < o:
                pass
            else:
                continue
            ni = i + 1
            if ni >= len(m1_idx):
                return None, None
            return m1_idx[ni], m1_opens[ni] + (spread if direction == 1 else -spread)
        return None, None

    else:  # E2: スパイクフィルター
        win_min = max(2, E2_ALT_WINDOW_MIN)
        end_time = signal_time + pd.Timedelta(minutes=win_min)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            bar_time  = m1_idx[i]
            bar_range = m1_highs[i] - m1_lows[i]
            if atr_1m is not None:
                atr_val = atr_1m.get(bar_time, np.nan)
                if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                    continue
            return bar_time, m1_opens[i] + (spread if direction == 1 else -spread)
        return None, None


# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(data_1m, data_15m, data_4h,
                     spread_pips, pip_size,
                     sym_cfg, variant_flags,
                     atr_1m=None,
                     m1_cache=None):
    """
    variant_flags: (use_kmid, use_klow, use_adx, use_streak, use_1d_ema, use_session, use_conf)
    """
    use_kmid, use_klow, use_adx, use_streak, use_1d_ema, use_session, use_conf = variant_flags
    spread = spread_pips * pip_size

    adx_min    = sym_cfg["adx_min"]    if use_adx    else 0
    streak_min = sym_cfg["streak_min"] if use_streak  else 0
    need_1d    = sym_cfg["use_1d"]     and use_1d_ema
    utc_start  = sym_cfg["utc_start"]  if use_session else 0
    utc_end    = sym_cfg["utc_end"]    if use_session else 24

    need_adx = adx_min > 0
    data_4h, data_1d = build_indicators(data_4h, need_adx, need_1d)

    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h["atr"] = calc_atr(data_1h, 14)

    # 1m配列キャッシュ（searchsortedで高速アクセス）
    if m1_cache is None:
        m1_cache = {
            "idx":    data_1m.index,
            "opens":  data_1m["open"].values,
            "closes": data_1m["close"].values,
            "highs":  data_1m["high"].values,
            "lows":   data_1m["low"].values,
        }

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()
    min_idx    = max(2, streak_min if streak_min > 0 else 0)

    for i in range(min_idx, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if not (utc_start <= h1_ct.hour < utc_end):
            continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak_min if streak_min > 0 else 2, 2):
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)):
            continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # F6: ADX フィルター
        if adx_min > 0:
            adx_val = h4_latest.get("adx", 0)
            if pd.isna(adx_val) or adx_val < adx_min:
                continue

        # F7: Streak フィルター
        if streak_min > 0:
            recent = h4_before["trend"].iloc[-streak_min:].values
            if not all(t == trend for t in recent):
                continue

        # F8: 日足EMA20 フィルター
        if need_1d and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0:
                continue
            if d1_before.iloc[-1]["trend1d"] != trend:
                continue

        tol = atr_val * TOL_FACTOR

        for direction in [1, -1]:
            if trend != direction:
                continue

            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
                conf_ok = h1_prev1["close"] > h1_prev1["open"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
                conf_ok = h1_prev1["close"] < h1_prev1["open"]

            if abs(v1 - v2) > tol:
                continue

            # F3: 確認足方向チェック
            if use_conf and not conf_ok:
                continue

            # F4: KMID
            if use_kmid and not check_kmid(h4_latest, direction):
                continue

            # F5: KLOW
            if use_klow and not check_klow(h4_latest):
                continue

            # エントリー
            et, ep = pick_entry_1m(
                data_1m, h1_ct, direction, spread,
                sym_cfg["entry_method"], atr_1m,
                m1_idx=m1_cache["idx"],
                m1_opens=m1_cache["opens"],
                m1_closes=m1_cache["closes"],
                m1_highs=m1_cache["highs"],
                m1_lows=m1_cache["lows"],
            )
            if et is None or et in used_times:
                continue

            raw = ep - spread if direction == 1 else ep + spread
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# 最大探索上限: 20,000本（1m足で約14日分、これで未決済はスキップ）
MAX_LOOKAHEAD = 20_000

# ── シミュレーション（ベクトル演算版）────────────────────────────
def _find_exit(highs, lows, times, ep, sl, tp, risk, direction):
    half_price = ep + direction * risk * HALF_R
    limit = min(len(highs), MAX_LOOKAHEAD)
    for i in range(limit):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            sl_hit   = lo <= sl
            tp_hit   = h  >= tp
            half_hit = h  >= half_price
        else:
            sl_hit   = h  >= sl
            tp_hit   = lo <= tp
            half_hit = lo <= half_price
        if sl_hit:
            return i, sl, "loss", False, -1
        if tp_hit:
            return i, tp, "win", False, -1
        if half_hit:
            be_sl = ep
            for j in range(i + 1, limit):
                h2 = highs[j]; lo2 = lows[j]
                if direction == 1:
                    if lo2 <= be_sl: return j, be_sl, "win", True, i
                    if h2  >= tp:    return j, tp,    "win", True, i
                else:
                    if h2  >= be_sl: return j, be_sl, "win", True, i
                    if lo2 <= tp:    return j, tp,    "win", True, i
            return -1, None, None, True, i
    return -1, None, None, False, -1


def simulate(signals, data_1m, symbol):
    if not signals:
        return [], [INIT_CASH]
    rm       = RiskManager(symbol, risk_pct=RISK_PCT)
    equity   = INIT_CASH
    trades   = []; eq_curve = [INIT_CASH]
    m1_times = data_1m.index
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=USDJPY_RATE)

        start_pos = m1_times.searchsorted(sig["time"], side="right")
        if start_pos >= len(m1_times):
            continue

        exit_i, exit_price, result, half_done, half_i = _find_exit(
            m1_highs[start_pos:], m1_lows[start_pos:],
            m1_times[start_pos:], ep, sl, tp, risk, direction
        )
        if result is None:
            continue

        if half_done and half_i >= 0:
            half_ep = ep + direction * risk * HALF_R
            equity += rm.calc_pnl_jpy(direction, ep, half_ep, lot * 0.5, USDJPY_RATE, ep)
            remaining_lot = lot * 0.5
        else:
            remaining_lot = lot

        equity += rm.calc_pnl_jpy(direction, ep, exit_price, remaining_lot, USDJPY_RATE, ep)
        exit_time = m1_times[start_pos + exit_i]
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve


# ── 統計計算 ──────────────────────────────────────────────────────
def calc_stats(trades, eq_curve, symbol, variant_label):
    if not trades:
        return {
            "symbol": symbol, "variant": variant_label,
            "n": 0, "wr": 0.0, "pf": 0.0, "mdd_pct": 0.0,
            "kelly": -1.0, "monthly_plus": "0/0"
        }
    df  = pd.DataFrame(trades)
    n   = len(df)
    wr  = (df["result"] == "win").mean()

    eq  = np.array(eq_curve)
    dlt = np.diff(eq)
    gw  = dlt[dlt > 0].sum()
    gl  = abs(dlt[dlt < 0].sum())
    pf  = gw / gl if gl > 0 else float("inf")

    peak  = np.maximum.accumulate(eq)
    mdd   = abs(((eq - peak) / peak).min()) * 100
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev_eq = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > prev_eq).sum()}/{len(monthly)}"

    return {
        "symbol":       symbol,
        "variant":      variant_label,
        "n":            n,
        "wr":           round(wr * 100, 1),
        "pf":           round(pf, 2),
        "mdd_pct":      round(mdd, 1),
        "kelly":        round(kelly, 3),
        "monthly_plus": mp,
    }


# ── メイン ───────────────────────────────────────────────────────
def main():
    all_results = []

    for sym_cfg in SYMBOLS:
        sym_name  = sym_cfg["name"]
        sym_lower = sym_cfg["lower"]
        category  = sym_cfg["category"]

        print(f"\n{'='*65}")
        print(f"  {sym_name}  [{category}]  entry={sym_cfg['entry_method']}")
        print(f"{'='*65}")

        d1m, d15m, d4h = load_data(sym_lower, sym_name)

        if d1m is None or d15m is None or d4h is None:
            print(f"  [SKIP] データ不足")
            continue

        d1m_oos  = slice_period(d1m,  OOS_START, OOS_END)
        d15m_oos = slice_period(d15m, OOS_START, OOS_END)
        d4h_oos  = slice_period(d4h,  OOS_START, OOS_END)

        if d1m_oos is None or len(d1m_oos) == 0:
            print(f"  [SKIP] OOS 1m データなし")
            continue

        print(f"  OOS 1m: {len(d1m_oos):,} bars | 4h: {len(d4h_oos):,} bars")

        # スパイクフィルター用 1m ATR（dict化でO(1)ルックアップ）
        atr_1m_series = calc_atr(d1m_oos, period=10)
        atr_1m = atr_1m_series.to_dict()

        # 1m配列キャッシュ（シグナル生成を高速化）
        m1_cache = {
            "idx":    d1m_oos.index,
            "opens":  d1m_oos["open"].values,
            "closes": d1m_oos["close"].values,
            "highs":  d1m_oos["high"].values,
            "lows":   d1m_oos["low"].values,
        }

        cfg         = SYMBOL_CONFIG.get(sym_name, {})
        spread_pips = cfg.get("spread", 0.0)
        pip_size    = cfg.get("pip", 0.0001)

        baseline_pf = None

        for (label, use_kmid, use_klow, use_adx, use_streak,
             use_1d_ema, use_session, use_conf) in VARIANTS:

            # FX に無関係なフィルターはスキップ
            if category == "FX" and label in ("-1D_EMA",):
                continue
            # XAUUSD に無関係なフィルターはスキップ
            if category == "METALS" and label in ("-ADX", "-Streak", "-ADX-Streak"):
                continue

            variant_flags = (use_kmid, use_klow, use_adx, use_streak,
                             use_1d_ema, use_session, use_conf)

            sigs = generate_signals(
                d1m_oos, d15m_oos, d4h_oos,
                spread_pips=spread_pips,
                pip_size=pip_size,
                sym_cfg=sym_cfg,
                variant_flags=variant_flags,
                atr_1m=atr_1m,
                m1_cache=m1_cache,
            )
            trades, eq_curve = simulate(sigs, d1m_oos, sym_name)
            stats = calc_stats(trades, eq_curve, sym_name, label)
            all_results.append(stats)

            if label == "baseline":
                baseline_pf = stats["pf"]
                diff_str = ""
            else:
                diff = stats["pf"] - baseline_pf if baseline_pf else 0
                sign = "+" if diff >= 0 else ""
                diff_str = f"  ({sign}{diff:.2f} vs baseline)"

            # 引き算判定
            judgement = ""
            if label != "baseline" and baseline_pf is not None:
                diff = stats["pf"] - baseline_pf
                judgement = "  ← ✅ 引き算候補" if diff >= 0 else "  ← ❌ 必要"

            print(f"  [{label:15s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                  f"PF={stats['pf']:.2f}{diff_str}{judgement}")

    # ── サマリー ────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  引き算の美学 — PF変化サマリー（vs baseline）")
    print("="*70)

    df_res = pd.DataFrame(all_results)

    # ベースラインの PF を取得
    baseline_df = df_res[df_res["variant"] == "baseline"].set_index("symbol")["pf"]

    # PF変化テーブル
    pf_table = df_res.pivot_table(
        index="symbol", columns="variant", values="pf", aggfunc="first"
    )
    variant_order = [v[0] for v in VARIANTS]
    available_variants = [v for v in variant_order if v in pf_table.columns]
    pf_table = pf_table[available_variants]
    print("\n■ プロフィットファクター")
    print(pf_table.to_string())

    # PF変化（baseline比）
    print("\n■ PF変化（baseline比、正=引き算候補）")
    diff_table = pf_table.copy()
    for sym in diff_table.index:
        bl = baseline_df.get(sym, np.nan)
        diff_table.loc[sym] = diff_table.loc[sym] - bl
    print(diff_table.drop(columns=["baseline"], errors="ignore").to_string())

    # ── カテゴリ別引き算判定 ────────────────────────────────────────
    print("\n\n■ フィルター別 引き算判定")

    # FXカテゴリ
    fx_syms = [c["name"] for c in SYMBOLS if c["category"] == "FX"]
    fx_df   = df_res[df_res["symbol"].isin(fx_syms)]
    fx_bl   = baseline_df[baseline_df.index.isin(fx_syms)]

    fx_variants = [v[0] for v in VARIANTS
                   if v[0] not in ("baseline", "-1D_EMA")]
    print("\nFXカテゴリ（EURUSD / GBPUSD / AUDUSD）:")
    for vname in fx_variants:
        vdf = fx_df[fx_df["variant"] == vname]
        improved = 0
        for _, row in vdf.iterrows():
            bl = fx_bl.get(row["symbol"], np.nan)
            if not np.isnan(bl) and row["pf"] >= bl:
                improved += 1
        total    = len(vdf)
        verdict  = "✅ 引き算推奨" if improved >= 2 else "❌ 存続"
        avg_diff = (vdf.set_index("symbol")["pf"] - fx_bl).mean()
        sign = "+" if avg_diff >= 0 else ""
        print(f"  {vname:15s}: {improved}/{total}銘柄改善  avg_diff={sign}{avg_diff:.2f}  → {verdict}")

    # XAUUSDカテゴリ
    xau_df = df_res[df_res["symbol"] == "XAUUSD"]
    xau_bl = baseline_df.get("XAUUSD", np.nan)
    xau_variants = [v[0] for v in VARIANTS
                    if v[0] not in ("baseline", "-ADX", "-Streak", "-ADX-Streak")]
    print("\nMETALSカテゴリ（XAUUSD）:")
    for vname in xau_variants:
        vdf    = xau_df[xau_df["variant"] == vname]
        if len(vdf) == 0:
            continue
        pf_val = vdf.iloc[0]["pf"]
        diff   = pf_val - xau_bl
        sign   = "+" if diff >= 0 else ""
        verdict = "✅ 引き算推奨" if diff >= 0 else "❌ 存続"
        print(f"  {vname:15s}: PF={pf_val:.2f} ({sign}{diff:.2f})  → {verdict}")

    # ── CSV 保存 ─────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "backtest_subtraction_aesthetics.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")

    # ── グラフ ────────────────────────────────────────────────────
    symbols_with_data = [s["name"] for s in SYMBOLS
                         if s["name"] in df_res["symbol"].unique()]
    n_sym = len(symbols_with_data)
    if n_sym == 0:
        return

    fig, axes = plt.subplots(1, n_sym, figsize=(6 * n_sym, 5), squeeze=False)

    for ax_idx, sym in enumerate(symbols_with_data):
        ax      = axes[0][ax_idx]
        sym_df  = df_res[df_res["symbol"] == sym].copy()
        bl_pf   = sym_df[sym_df["variant"] == "baseline"]["pf"].values[0]
        sym_df  = sym_df[sym_df["variant"] != "baseline"]

        colors = ["green" if pf >= bl_pf else "tomato"
                  for pf in sym_df["pf"]]

        bars = ax.bar(sym_df["variant"], sym_df["pf"],
                      color=colors, alpha=0.85)
        ax.axhline(bl_pf, color="navy", linestyle="--", linewidth=1.2,
                   label=f"baseline={bl_pf:.2f}")
        ax.axhline(2.0,   color="red",  linestyle=":",  linewidth=0.8,
                   label="PF=2.0")
        ax.set_title(sym, fontsize=12)
        ax.set_ylabel("PF (OOS)")
        ax.set_ylim(0, max(sym_df["pf"].max(), bl_pf) * 1.25)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)

        for bar_obj, pf_val in zip(bars, sym_df["pf"]):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 0.02,
                    f"{pf_val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("引き算の美学 — フィルター除去 OOS PF比較\n"
                 "緑=baseline以上（引き算候補）/ 赤=baseline以下（存続）",
                 fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "backtest_subtraction_aesthetics.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"グラフ保存: {out_png}")


if __name__ == "__main__":
    main()
