"""
backtest_addition_filters.py
==============================
「引き算の美学」後の「足し算」— 追加フィルター効果検証

【前提: 引き算後の確定ベース（lean baseline）】
  ✅ KMID（4H文脈足 実体方向一致）
  ✅ KLOW（4H文脈足 下ヒゲ < 0.15%）
  ✅ Streak ≥ 4（4H連続同方向、FXのみ）
  ✅ 日足EMA20方向一致（XAUUSDのみ）
  ❌ ADX → 除去確定
  ❌ セッションフィルター → 完全除去
  ❌ 確認足方向チェック → 除去（微改善）
  ✅ E1エントリー（FX）/ E2エントリー（XAUUSD）

【追加フィルター候補（1つずつ lean baseline に追加）】

  A1: +EMA距離（v81C）
      4H close が 4H EMA20 から ATR×1.0 以上離れている場合のみエントリー
      根拠: OOS定量分析「EMA遠2-3ATR時に勝率+5.7%」(p=0.078)
      パラメータ: 固定値1.0（「ATR1本分のエッジ」は古典TA業界標準）

  A2: +1H EMA傾き
      直前3本の1H EMA20が上昇中（ロング）/ 下落中（ショート）場合のみ
      根拠: 「乗り込む波が加速しているか」— モメンタム古典TA手法
      パラメータ: 3本（「短すぎず長すぎない」標準的ルックバック）

  A3: +パターン精度強化（v81A 厳格化）
      二番底・二番天井の許容幅を ATR×0.3 → ATR×0.15 に縮小
      根拠: より正確なパターンのみに絞る（データ非依存の固定値）
      パラメータ: 0.15（0.3の半分、銘柄共通で過学習なし）

  A4: +ATR拡張（ボラティリティ拡大局面）
      現在の1H ATR が直近5本の1H ATR 平均より大きい場合のみ
      根拠: 収縮→拡大のブレイクアウト局面でトレンドが継続しやすい（古典TA）
      パラメータ: period=5（短期ボラ変化の業界標準的観察窓）

  A5: +A1+A2 複合（EMA距離 + 1H傾き）
      独立した2条件の組み合わせ（過学習チェック用）

【過学習対策】
  全パラメータは固定値（OOSデータを見て調整しない）
  追加フィルターのPASS基準: カテゴリ内過半数銘柄でOOS PF改善
  IS/OOS乖離が大きい場合は採用しない

【期間】
  OOS: 2025-03-03 〜 2026-02-27
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
USDJPY_RATE = 150.0

OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# エントリー方式パラメータ
E1_MAX_WAIT_MIN   = 5
E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3
MAX_LOOKAHEAD     = 20_000

# 追加フィルターパラメータ（固定値、データ非依存）
A1_EMA_DIST_MIN   = 1.0   # 4H ATR×1.0 以上 EMA から離れていること
A2_EMA_SLOPE_BARS = 3     # 1H EMA20 が直近N本で上昇/下降中
A3_TIGHT_TOL      = 0.15  # パターン許容幅（ATR×0.15）
A3_DEFAULT_TOL    = 0.30  # 現行デフォルト
A4_ATR_PERIOD     = 5     # ATR拡張判定ウィンドウ（本数）

# ── バリアント定義 ─────────────────────────────────────────────────
# (label, ema_dist, slope, tight_tol, atr_expand, both_a1a2)
VARIANTS = [
    ("lean_baseline", False, False, False, False),
    ("+EMA_dist",     True,  False, False, False),
    ("+1H_slope",     False, True,  False, False),
    ("+tight_tol",    False, False, True,  False),
    ("+ATR_expand",   False, False, False, True),
    ("+A1+A2",        True,  True,  False, False),
]

# ── 銘柄設定（lean baseline 準拠）────────────────────────────────
SYMBOLS = [
    {
        "name":        "EURUSD",
        "lower":       "eurusd",
        "category":    "FX",
        "entry_method": "E1",
        "streak_min":  4,
        "use_1d":      False,
    },
    {
        "name":        "GBPUSD",
        "lower":       "gbpusd",
        "category":    "FX",
        "entry_method": "E1",
        "streak_min":  4,
        "use_1d":      False,
    },
    {
        "name":        "AUDUSD",
        "lower":       "audusd",
        "category":    "FX",
        "entry_method": "E1",
        "streak_min":  4,
        "use_1d":      False,
    },
    {
        "name":        "XAUUSD",
        "lower":       "xauusd",
        "category":    "METALS",
        "entry_method": "E2",
        "streak_min":  0,
        "use_1d":      True,
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
        return load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_{tf}.csv"))
    return _load("1m"), _load("15m"), _load("4h")


# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def build_4h(df4h, need_1d):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if need_1d:
        d1 = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None


def build_1h(data_15m):
    df = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    # ATR拡張用ローリング平均
    df["atr_avg"] = df["atr"].rolling(A4_ATR_PERIOD).mean()
    return df


# ── フィルター関数 ────────────────────────────────────────────────
def check_kmid(bar, direction):
    o, c = bar["open"], bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)


def check_klow(bar):
    o, c, l = bar["open"], bar["close"], bar["low"]
    body_bot = min(o, c)
    return (body_bot - l) / o < KLOW_THR if o > 0 else False


# ── A1: EMA距離フィルター ─────────────────────────────────────────
def check_ema_dist(h4_bar, direction, dist_min=A1_EMA_DIST_MIN):
    dist = abs(h4_bar["close"] - h4_bar["ema20"])
    atr  = h4_bar["atr"]
    if pd.isna(atr) or atr <= 0:
        return False
    return dist >= atr * dist_min


# ── A2: 1H EMA傾きフィルター ─────────────────────────────────────
def check_1h_ema_slope(data_1h, signal_time, direction, n=A2_EMA_SLOPE_BARS):
    """1H EMA20 が直近n本で方向一致の傾きを持つか"""
    h1_before = data_1h[data_1h.index < signal_time]
    if len(h1_before) < n + 1:
        return False
    ema_vals = h1_before["ema20"].iloc[-(n + 1):]
    slope    = ema_vals.iloc[-1] - ema_vals.iloc[0]
    if direction == 1:
        return slope > 0
    else:
        return slope < 0


# ── A4: ATR拡張フィルター ─────────────────────────────────────────
def check_atr_expand(data_1h, signal_time):
    """現在の1H ATR が直近A4_ATR_PERIOD本の平均より大きいか"""
    h1_before = data_1h[data_1h.index < signal_time]
    if len(h1_before) < 2:
        return False
    latest = h1_before.iloc[-1]
    atr_now = latest["atr"]
    atr_avg = latest["atr_avg"]
    if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg <= 0:
        return False
    return atr_now > atr_avg


# ── エントリー価格決定 ────────────────────────────────────────────
def pick_entry_1m(signal_time, direction, spread, method,
                  atr_1m, m1_cache):
    m1_idx    = m1_cache["idx"]
    m1_opens  = m1_cache["opens"]
    m1_closes = m1_cache["closes"]
    m1_highs  = m1_cache["highs"]
    m1_lows   = m1_cache["lows"]

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

    else:  # E2
        win_min  = max(2, E2_ALT_WINDOW_MIN)
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
                     atr_1m=None, m1_cache=None):
    """
    variant_flags: (use_ema_dist, use_slope, use_tight_tol, use_atr_expand)
    """
    use_ema_dist, use_slope, use_tight_tol, use_atr_expand, _ = variant_flags
    spread = spread_pips * pip_size

    streak_min = sym_cfg["streak_min"]
    need_1d    = sym_cfg["use_1d"]

    data_4h, data_1d = build_4h(data_4h, need_1d)
    data_1h = build_1h(data_15m)

    if m1_cache is None:
        m1_cache = {
            "idx":    data_1m.index,
            "opens":  data_1m["open"].values,
            "closes": data_1m["close"].values,
            "highs":  data_1m["high"].values,
            "lows":   data_1m["low"].values,
        }

    tol_factor = A3_TIGHT_TOL if use_tight_tol else A3_DEFAULT_TOL

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

        # セッションフィルター: 完全除去（lean baseline）

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak_min if streak_min > 0 else 2, 2):
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)):
            continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # Streak フィルター（lean baseline で存続）
        if streak_min > 0:
            recent = h4_before["trend"].iloc[-streak_min:].values
            if not all(t == trend for t in recent):
                continue

        # 日足EMA20（XAUUSD のみ）
        if need_1d and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0:
                continue
            if d1_before.iloc[-1]["trend1d"] != trend:
                continue

        # KMID（lean baseline で存続）
        if not check_kmid(h4_latest, trend):
            continue

        # KLOW（lean baseline で存続）
        if not check_klow(h4_latest):
            continue

        # A1: EMA距離フィルター
        if use_ema_dist and not check_ema_dist(h4_latest, trend):
            continue

        # A2: 1H EMA傾き
        if use_slope and not check_1h_ema_slope(data_1h, h1_ct, trend):
            continue

        # A4: ATR拡張
        if use_atr_expand and not check_atr_expand(data_1h, h1_ct):
            continue

        tol = atr_val * tol_factor

        for direction in [1, -1]:
            if trend != direction:
                continue

            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
                # 確認足: lean baseline では除去（-conf_candle）
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]

            if abs(v1 - v2) > tol:
                continue

            et, ep = pick_entry_1m(
                h1_ct, direction, spread,
                sym_cfg["entry_method"], atr_1m, m1_cache
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


# ── シミュレーション（ベクトル演算 + 最大探索上限）────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half_price = ep + direction * risk * HALF_R
    limit = min(len(highs), MAX_LOOKAHEAD)
    for i in range(limit):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl:              return i, sl, "loss", False, -1
            if h  >= tp:              return i, tp, "win",  False, -1
            if h  >= half_price:
                be_sl = ep
                for j in range(i + 1, limit):
                    if lows[j]  <= be_sl: return j, be_sl, "win", True, i
                    if highs[j] >= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
        else:
            if h  >= sl:              return i, sl, "loss", False, -1
            if lo <= tp:              return i, tp, "win",  False, -1
            if lo <= half_price:
                be_sl = ep
                for j in range(i + 1, limit):
                    if highs[j] >= be_sl: return j, be_sl, "win", True, i
                    if lows[j]  <= tp:    return j, tp,    "win", True, i
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
            ep, sl, tp, risk, direction
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
        return {"symbol": symbol, "variant": variant_label,
                "n": 0, "wr": 0.0, "pf": 0.0, "mdd_pct": 0.0,
                "kelly": -1.0, "monthly_plus": "0/0"}
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
    return {"symbol": symbol, "variant": variant_label,
            "n": n, "wr": round(wr * 100, 1), "pf": round(pf, 2),
            "mdd_pct": round(mdd, 1), "kelly": round(kelly, 3),
            "monthly_plus": mp}


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
            print("  [SKIP] データ不足"); continue

        d1m_oos  = slice_period(d1m,  OOS_START, OOS_END)
        d15m_oos = slice_period(d15m, OOS_START, OOS_END)
        d4h_oos  = slice_period(d4h,  OOS_START, OOS_END)

        if d1m_oos is None or len(d1m_oos) == 0:
            print("  [SKIP] OOS 1m なし"); continue

        print(f"  OOS 1m: {len(d1m_oos):,} bars | 4h: {len(d4h_oos):,} bars")

        atr_1m_series = calc_atr(d1m_oos, period=10)
        atr_1m = atr_1m_series.to_dict()

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

        for (label, use_ema_dist, use_slope,
             use_tight_tol, use_atr_expand) in VARIANTS:
            # +A1+A2 の場合
            is_both = (label == "+A1+A2")
            vflags  = (use_ema_dist, use_slope, use_tight_tol,
                       use_atr_expand, is_both)

            sigs = generate_signals(
                d1m_oos, d15m_oos, d4h_oos,
                spread_pips=spread_pips,
                pip_size=pip_size,
                sym_cfg=sym_cfg,
                variant_flags=vflags,
                atr_1m=atr_1m,
                m1_cache=m1_cache,
            )
            trades, eq_curve = simulate(sigs, d1m_oos, sym_name)
            stats = calc_stats(trades, eq_curve, sym_name, label)
            all_results.append(stats)

            if label == "lean_baseline":
                baseline_pf = stats["pf"]
                print(f"  [{label:16s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                      f"PF={stats['pf']:.2f}  MDD={stats['mdd_pct']:.1f}%  "
                      f"月次+={stats['monthly_plus']}")
            else:
                diff    = stats["pf"] - baseline_pf if baseline_pf else 0
                sign    = "+" if diff >= 0 else ""
                verdict = "  ← ✅ 足し算候補" if diff >= 0 else "  ← ❌ 逆効果"
                print(f"  [{label:16s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                      f"PF={stats['pf']:.2f}  ({sign}{diff:.2f}){verdict}")

    # ── サマリー ────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  足し算フィルター — PF変化サマリー（vs lean_baseline）")
    print("="*70)

    df_res = pd.DataFrame(all_results)
    bl_df  = df_res[df_res["variant"] == "lean_baseline"].set_index("symbol")["pf"]

    pf_table = df_res.pivot_table(
        index="symbol", columns="variant", values="pf", aggfunc="first"
    )
    variant_order = [v[0] for v in VARIANTS]
    avail = [v for v in variant_order if v in pf_table.columns]
    pf_table = pf_table[avail]
    print("\n■ プロフィットファクター")
    print(pf_table.to_string())

    print("\n■ PF変化（lean_baseline比、正=足し算候補）")
    diff_table = pf_table.copy()
    for sym in diff_table.index:
        diff_table.loc[sym] = diff_table.loc[sym] - bl_df.get(sym, 0)
    print(diff_table.drop(columns=["lean_baseline"], errors="ignore").to_string())

    # フィルター別判定
    print("\n\n■ フィルター別 足し算判定")
    add_variants = [v[0] for v in VARIANTS if v[0] != "lean_baseline"]

    fx_syms  = [c["name"] for c in SYMBOLS if c["category"] == "FX"]
    xau_syms = [c["name"] for c in SYMBOLS if c["category"] == "METALS"]

    print("\nFXカテゴリ（EURUSD / GBPUSD / AUDUSD）:")
    for vname in add_variants:
        vdf = df_res[(df_res["symbol"].isin(fx_syms)) & (df_res["variant"] == vname)]
        improved = sum(
            row["pf"] >= bl_df.get(row["symbol"], 0)
            for _, row in vdf.iterrows()
        )
        total    = len(vdf)
        avg_diff = (vdf.set_index("symbol")["pf"] - bl_df[bl_df.index.isin(fx_syms)]).mean()
        sign     = "+" if avg_diff >= 0 else ""
        verdict  = "✅ 採用候補" if improved >= 2 else "❌ 逆効果"
        print(f"  {vname:16s}: {improved}/{total}銘柄改善  avg_diff={sign}{avg_diff:.2f}  → {verdict}")

    print("\nMETALSカテゴリ（XAUUSD）:")
    for vname in add_variants:
        vdf    = df_res[(df_res["symbol"].isin(xau_syms)) & (df_res["variant"] == vname)]
        if len(vdf) == 0: continue
        pf_val = vdf.iloc[0]["pf"]
        diff   = pf_val - bl_df.get("XAUUSD", 0)
        sign   = "+" if diff >= 0 else ""
        verdict = "✅ 採用候補" if diff >= 0 else "❌ 逆効果"
        print(f"  {vname:16s}: PF={pf_val:.2f} ({sign}{diff:.2f})  → {verdict}")

    # ── CSV / PNG 保存 ────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "backtest_addition_filters.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")

    symbols_data = [s["name"] for s in SYMBOLS if s["name"] in df_res["symbol"].unique()]
    n_sym = len(symbols_data)
    if n_sym == 0: return

    fig, axes = plt.subplots(1, n_sym, figsize=(6 * n_sym, 5), squeeze=False)
    for ax_idx, sym in enumerate(symbols_data):
        ax     = axes[0][ax_idx]
        sym_df = df_res[df_res["symbol"] == sym].copy()
        bl_pf  = sym_df[sym_df["variant"] == "lean_baseline"]["pf"].values[0]
        plot_df = sym_df[sym_df["variant"] != "lean_baseline"]
        colors  = ["green" if p >= bl_pf else "tomato" for p in plot_df["pf"]]
        bars    = ax.bar(plot_df["variant"], plot_df["pf"], color=colors, alpha=0.85)
        ax.axhline(bl_pf, color="navy", linestyle="--", linewidth=1.2,
                   label=f"lean_baseline={bl_pf:.2f}")
        ax.axhline(2.0, color="red", linestyle=":", linewidth=0.8, label="PF=2.0")
        ax.set_title(sym, fontsize=12)
        ax.set_ylabel("PF (OOS)")
        ax.set_ylim(0, max(plot_df["pf"].max(), bl_pf) * 1.25)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
        for b, pf_v in zip(bars, plot_df["pf"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{pf_v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("足し算フィルター — lean baseline への追加効果（OOS PF）\n"
                 "緑=baseline以上（採用候補）/ 赤=baseline以下（逆効果）", fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "backtest_addition_filters.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"グラフ保存: {out_png}")


if __name__ == "__main__":
    main()
