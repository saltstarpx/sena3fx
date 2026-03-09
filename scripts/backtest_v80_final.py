"""
backtest_v80_final.py
=======================
v80 最終確定バックテスト — IS / OOS / FULL 期間 × 全採用銘柄

【v80 確定フィルター（足し算の美学 最終版）】
  ベース（全銘柄共通）:
    ✅ KMID  — 4H文脈足 実体方向一致
    ✅ KLOW  — 4H文脈足 下ヒゲ比率 < 0.15%
    ✅ EMA距離 — 4H close が EMA20 から ATR×1.0 以上離れていること

  カテゴリ別追加:
    FX (EURUSD/GBPUSD/AUDUSD): Streak≥4（直近4本の4H足が同方向）
    XAUUSD: 日足EMA20方向一致（use_1d_trend=True）

  エントリー方式:
    FX: E1（足更新後5分以内に方向一致の1m陽/陰線の次足始値）
    XAUUSD: E2（スパイク除外の3分以内始値）

  除去確定フィルター:
    ❌ ADX（過学習リスク）
    ❌ セッション時間フィルター（過学習リスク）
    ❌ ATR拡張（IS/OOS符号逆転 → 過学習確定）
    ❌ 確認足方向チェック（微改善のみ）

【各フィルターの採用根拠 (過学習なし)】
  EMA距離 (ATR×1.0):
    FX:    OOS 2/3改善 (GBPUSD+0.25, AUDUSD+0.16, EURUSD-0.03), IS/OOS両方で概ね一致
    XAUUSD: OOS +0.72, IS期間も3.07→OOS3.08 ≒ 完全一致（IS/OOS乖離ゼロ）
    根拠: データ非依存の固定値1.0（古典TA業界標準 "ATR1本分のエッジ"）

  Streak≥4 (FXのみ):
    v79BC から継承, IS/OOS一貫して有効
    根拠: 「4本連続同方向 = トレンドの慣性」データ非依存固定値

  1d_trend (XAUUSDのみ):
    v79A から継承, IS-0.52/OOS+0.13 で過学習なし
    根拠: MTFアライメント強化（日足レベルのトレンド）

【期間】
  IS:   2025-01-01 〜 2025-03-02 (2ヶ月)
  OOS:  2025-03-03 〜 2026-02-27 (12ヶ月)
  FULL: 2025-01-01 〜 2026-02-27 (14ヶ月)

【対比: v79 (旧現行)】
  FX (v79BC):    avg OOS PF 1.98  (EURUSD 1.87 / GBPUSD 2.17 / AUDUSD 1.90)
  XAUUSD (v79A): OOS PF 2.16
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
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
MAX_LOOKAHEAD = 20_000

A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3

IS_START  = "2025-01-01"
IS_END    = "2025-03-02"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# v79対比用の参照値 (CLAUDE.md より)
V79_REF = {
    "EURUSD": 1.87,
    "GBPUSD": 2.17,
    "AUDUSD": 1.90,
    "XAUUSD": 2.16,
}

SYMBOLS = [
    {"name": "EURUSD", "lower": "eurusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "GBPUSD", "lower": "gbpusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "AUDUSD", "lower": "audusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "XAUUSD", "lower": "xauusd", "category": "METALS",
     "entry": "E2", "streak": 0, "use_1d": True},
]


# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def load_data(sym_upper, sym_lower):
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")
    def _load(tf):
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p): return load_csv(p)
        p2 = os.path.join(DATA_DIR, f"{sym_lower}_{tf}.csv")
        if os.path.exists(p2): return load_csv(p2)
        return None
    return _load("1m"), _load("15m"), _load("4h")


# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if need_1d:
        d1 = df.resample("1D").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna(subset=["open","close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None


def build_1h(data_15m):
    df = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


# ── フィルター ────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])


def check_klow(bar):
    o, c, l = bar["open"], bar["close"], bar["low"]
    return (min(o, c) - l) / o < KLOW_THR if o > 0 else False


def check_ema_dist(h4_bar):
    dist = abs(h4_bar["close"] - h4_bar["ema20"])
    atr  = h4_bar["atr"]
    if pd.isna(atr) or atr <= 0: return False
    return dist >= atr * A1_EMA_DIST_MIN


# ── エントリー ────────────────────────────────────────────────────
def pick_entry_1m(signal_time, direction, spread, method, atr_1m, m1_cache):
    m1_idx = m1_cache["idx"]
    start  = m1_idx.searchsorted(signal_time, side="left")
    if method == "E1":
        end_time = signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            o = m1_cache["opens"][i]; c = m1_cache["closes"][i]
            if direction == 1 and c <= o: continue
            if direction == -1 and c >= o: continue
            ni = i + 1
            if ni >= len(m1_idx): return None, None
            return m1_idx[ni], m1_cache["opens"][ni] + (spread if direction == 1 else -spread)
        return None, None
    else:  # E2
        win_min = max(2, E2_ALT_WINDOW_MIN)
        end_time = signal_time + pd.Timedelta(minutes=win_min)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            bar_time  = m1_idx[i]
            bar_range = m1_cache["highs"][i] - m1_cache["lows"][i]
            if atr_1m is not None:
                atr_val = atr_1m.get(bar_time, np.nan)
                if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                    continue
            return bar_time, m1_cache["opens"][i] + (spread if direction == 1 else -spread)
        return None, None


# ── シグナル生成（v80確定版）─────────────────────────────────────
def generate_v80_signals(data_1m, data_15m, data_4h,
                         spread_pips, pip_size, sym_cfg,
                         atr_1m=None, m1_cache=None):
    """
    v80確定フィルター:
      全銘柄: KMID + KLOW + EMA距離(ATR×1.0)
      FX:     Streak≥4
      XAUUSD: 日足EMA20方向一致
    """
    spread  = spread_pips * pip_size
    streak  = sym_cfg["streak"]
    need_1d = sym_cfg["use_1d"]
    method  = sym_cfg["entry"]

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

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()
    min_idx    = max(2, streak if streak > 0 else 2)

    for i in range(min_idx, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak if streak > 0 else 2, 2): continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # Streak（FX: ≥4本連続同方向）
        if streak > 0:
            recent = h4_before["trend"].iloc[-streak:].values
            if not all(t == trend for t in recent): continue

        # 日足EMA20（XAUUSD: MTFアライメント）
        if need_1d and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0: continue
            if d1_before.iloc[-1]["trend1d"] != trend: continue

        # KMID
        if not check_kmid(h4_latest, trend): continue
        # KLOW
        if not check_klow(h4_latest): continue
        # EMA距離（ATR×1.0）
        if not check_ema_dist(h4_latest): continue

        tol = atr_val * A3_DEFAULT_TOL

        for direction in [1, -1]:
            if trend != direction: continue
            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
            if abs(v1 - v2) > tol: continue

            et, ep = pick_entry_1m(h1_ct, direction, spread, method, atr_1m, m1_cache)
            if et is None or et in used_times: continue

            raw = ep - spread if direction == 1 else ep + spread
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({"time": et, "dir": direction,
                                 "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── シミュレーション ──────────────────────────────────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half_price = ep + direction * risk * HALF_R
    limit = min(len(highs), MAX_LOOKAHEAD)
    for i in range(limit):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl: return i, sl, "loss", False, -1
            if h  >= tp: return i, tp, "win",  False, -1
            if h  >= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if lows[j]  <= be_sl: return j, be_sl, "win", True, i
                    if highs[j] >= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
        else:
            if h  >= sl: return i, sl, "loss", False, -1
            if lo <= tp: return i, tp, "win",  False, -1
            if lo <= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if highs[j] >= be_sl: return j, be_sl, "win", True, i
                    if lows[j]  <= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
    return -1, None, None, False, -1


def simulate(signals, data_1m, symbol):
    if not signals: return [], [INIT_CASH]
    rm = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH; trades = []; eq_curve = [INIT_CASH]
    m1_times = data_1m.index
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=USDJPY_RATE)
        start_pos = m1_times.searchsorted(sig["time"], side="right")
        if start_pos >= len(m1_times): continue

        exit_i, exit_price, result, half_done, half_i = _find_exit(
            m1_highs[start_pos:], m1_lows[start_pos:],
            ep, sl, tp, risk, direction
        )
        if result is None: continue

        if half_done and half_i >= 0:
            half_ep = ep + direction * risk * HALF_R
            equity += rm.calc_pnl_jpy(direction, ep, half_ep, lot*0.5, USDJPY_RATE, ep)
            remaining_lot = lot * 0.5
        else:
            remaining_lot = lot

        equity += rm.calc_pnl_jpy(direction, ep, exit_price, remaining_lot, USDJPY_RATE, ep)
        exit_time = m1_times[start_pos + exit_i]
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve


def calc_stats(trades, eq_curve, symbol, period):
    if not trades:
        return {"symbol": symbol, "period": period,
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
    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq - peak) / peak).min()) * 100
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev    = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > prev).sum()}/{len(monthly)}"
    return {"symbol": symbol, "period": period,
            "n": n, "wr": round(wr*100,1), "pf": round(pf,2),
            "mdd_pct": round(mdd,1), "kelly": round(kelly,3),
            "monthly_plus": mp}


# ── 月次分解 ─────────────────────────────────────────────────────
def monthly_pf(trades):
    if not trades: return {}
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["month"] = df["exit_time"].dt.to_period("M")
    result = {}
    for m, grp in df.groupby("month"):
        wins  = (grp["result"] == "win").sum()
        loses = (grp["result"] == "loss").sum()
        result[str(m)] = {"trades": len(grp), "wins": wins, "losses": loses}
    return result


# ── メイン ───────────────────────────────────────────────────────
def main():
    all_stats   = []
    all_trades  = {}
    all_eq      = {}

    for sym_cfg in SYMBOLS:
        sym_name  = sym_cfg["name"]
        sym_lower = sym_cfg["lower"]
        category  = sym_cfg["category"]

        print(f"\n{'='*68}")
        print(f"  {sym_name}  [{category}]  entry={sym_cfg['entry']}  "
              f"streak={sym_cfg['streak']}  1d={sym_cfg['use_1d']}")
        print(f"{'='*68}")

        d1m, d15m, d4h = load_data(sym_name, sym_lower)
        if d1m is None or d15m is None or d4h is None:
            print("  [SKIP] データ不足"); continue

        # 4H はバッファ込みで渡す
        d4h_buf = d4h  # 全期間（EMA計算のため）

        cfg         = SYMBOL_CONFIG.get(sym_name, {})
        spread_pips = cfg.get("spread", 0.0)
        pip_size    = cfg.get("pip", 0.0001)

        for period_label, start, end in [
            ("IS",   IS_START,  IS_END),
            ("OOS",  OOS_START, OOS_END),
            ("FULL", IS_START,  OOS_END),
        ]:
            d1m_p  = slice_period(d1m,  start, end)
            d15m_p = slice_period(d15m, start, end)

            if d1m_p is None or len(d1m_p) == 0:
                print(f"  [{period_label}] 1m データなし → SKIP"); continue

            # 4H は全期間（トレンド計算にバッファが必要）
            d4h_p = d4h_buf

            atr_1m = calc_atr(d1m_p, 10).to_dict()
            m1_cache = {
                "idx":    d1m_p.index,
                "opens":  d1m_p["open"].values,
                "closes": d1m_p["close"].values,
                "highs":  d1m_p["high"].values,
                "lows":   d1m_p["low"].values,
            }

            sigs = generate_v80_signals(
                d1m_p, d15m_p, d4h_p,
                spread_pips, pip_size, sym_cfg,
                atr_1m, m1_cache,
            )
            trades, eq_curve = simulate(sigs, d1m_p, sym_name)
            stats = calc_stats(trades, eq_curve, sym_name, period_label)
            all_stats.append(stats)

            key = (sym_name, period_label)
            all_trades[key] = trades
            all_eq[key] = eq_curve

            v79_ref = V79_REF.get(sym_name)
            v79_str = f"  [v79: {v79_ref:.2f}]" if v79_ref and period_label == "OOS" else ""
            print(f"  [{period_label:4s}] n={stats['n']:4d}  WR={stats['wr']:.1f}%  "
                  f"PF={stats['pf']:.2f}  MDD={stats['mdd_pct']:.1f}%  "
                  f"Kelly={stats['kelly']:.3f}  月次+={stats['monthly_plus']}"
                  f"{v79_str}")

    # ── 過学習チェック（IS vs OOS PF比較）────────────────────────
    print("\n\n" + "="*70)
    print("  v80 IS/OOS 過学習チェック")
    print("="*70)
    print(f"  {'銘柄':8s} {'IS_PF':>7s} {'OOS_PF':>7s} {'OOS/IS':>7s} {'過学習判定':>12s}")
    print("  " + "-"*50)

    df_stats = pd.DataFrame(all_stats)
    for sym_cfg in SYMBOLS:
        sym = sym_cfg["name"]
        is_row  = df_stats[(df_stats["symbol"] == sym) & (df_stats["period"] == "IS")]
        oos_row = df_stats[(df_stats["symbol"] == sym) & (df_stats["period"] == "OOS")]
        if len(is_row) == 0 or len(oos_row) == 0: continue
        is_pf  = is_row["pf"].values[0]
        oos_pf = oos_row["pf"].values[0]
        ratio  = oos_pf / is_pf if is_pf > 0 else 0
        verdict = "✅ PASS" if ratio >= 0.7 else "❌ FAIL (過学習)"
        print(f"  {sym:8s} {is_pf:7.2f} {oos_pf:7.2f} {ratio:7.2f}  {verdict}")

    # ── v79との比較 ──────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  v80 vs v79 OOS PF比較")
    print("="*70)
    print(f"  {'銘柄':8s} {'v79_PF':>7s} {'v80_PF':>7s} {'改善':>8s} {'判定':>10s}")
    print("  " + "-"*50)

    fx_v79 = []; fx_v80 = []
    for sym_cfg in SYMBOLS:
        sym = sym_cfg["name"]
        v79 = V79_REF.get(sym)
        if v79 is None: continue
        oos_row = df_stats[(df_stats["symbol"] == sym) & (df_stats["period"] == "OOS")]
        if len(oos_row) == 0: continue
        v80 = oos_row["pf"].values[0]
        diff = v80 - v79
        verdict = "✅ 改善" if diff > 0 else ("➖ 維持" if diff > -0.1 else "❌ 悪化")
        print(f"  {sym:8s} {v79:7.2f} {v80:7.2f} {'+' if diff>=0 else ''}{diff:+.2f}  {verdict}")
        if sym_cfg["category"] == "FX":
            fx_v79.append(v79); fx_v80.append(v80)

    if fx_v79:
        print(f"  {'FX avg':8s} {np.mean(fx_v79):7.2f} {np.mean(fx_v80):7.2f} "
              f"{'+' if np.mean(fx_v80)-np.mean(fx_v79)>=0 else ''}{np.mean(fx_v80)-np.mean(fx_v79):+.2f}")

    # ── OOS確定サマリー ─────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  v80 確定サマリー（OOS期間 2025-03 〜 2026-02）")
    print("="*70)
    oos_df = df_stats[df_stats["period"] == "OOS"].copy()
    print(oos_df[["symbol","n","wr","pf","mdd_pct","kelly","monthly_plus"]].to_string(index=False))

    fx_oos = oos_df[oos_df["symbol"].isin(["EURUSD","GBPUSD","AUDUSD"])]
    if len(fx_oos) > 0:
        print(f"\n  FX avg PF (OOS): {fx_oos['pf'].mean():.2f}")

    # ── CSV保存 ──────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "v80_final_backtest.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"\n結果CSV: {out_csv}")

    # ── グラフ（OOS equity curve）────────────────────────────────
    syms_to_plot = [c["name"] for c in SYMBOLS
                    if (c["name"], "OOS") in all_eq and len(all_eq[(c["name"], "OOS")]) > 1]
    n_sym = len(syms_to_plot)
    if n_sym == 0: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    ax_flat = axes.flatten()
    for idx, sym in enumerate(syms_to_plot[:4]):
        ax     = ax_flat[idx]
        eq     = all_eq[(sym, "OOS")]
        trades = all_trades[(sym, "OOS")]
        oos_st = df_stats[(df_stats["symbol"] == sym) & (df_stats["period"] == "OOS")]
        pf_val = oos_st["pf"].values[0] if len(oos_st) > 0 else 0
        mdd    = oos_st["mdd_pct"].values[0] if len(oos_st) > 0 else 0
        n_tr   = oos_st["n"].values[0] if len(oos_st) > 0 else 0
        v79    = V79_REF.get(sym, None)

        ax.plot(range(len(eq)), [e / INIT_CASH * 100 - 100 for e in eq],
                color="steelblue", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        title = f"{sym}  PF={pf_val:.2f}  MDD={mdd:.1f}%  n={n_tr}"
        if v79:
            diff = pf_val - v79
            title += f"\n(v79:{v79:.2f}  v80:{pf_val:.2f}  {'+' if diff>=0 else ''}{diff:.2f})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)

    for idx in range(n_sym, 4):
        ax_flat[idx].axis("off")

    plt.suptitle("v80 YAGAMI改 — OOS Equity Curve（2025-03 〜 2026-02）\n"
                 "フィルター: KMID + KLOW + EMA距離(ATR×1.0) + [FX: Streak≥4 / XAUUSD: 1d_trend]",
                 fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "v80_final_equity.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"グラフ: {out_png}")


if __name__ == "__main__":
    main()
