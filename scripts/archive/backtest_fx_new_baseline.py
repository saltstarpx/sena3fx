"""
backtest_fx_new_baseline.py
===============================
FX: +EMA距離採用後の新ベースラインへの「次の足し算」検証

【新FXベースライン（v80候補）】
  lean_baseline + EMA距離（ATR×1.0）
  = KMID + KLOW + Streak≥4 + EMA距離 + E1エントリー
  セッション/ADX/確認足 なし

【追加候補】
  B1: +ATR拡張（1H）- XAUUSD で最大の改善（+0.84）→ FXでも有効か？
  B2: +1H傾き         - 前回FXで1/3のみ改善 → 新ベースラインでは変わるか？
  B3: +Streak≥5       - Streak基準を4→5に引き上げ（選択性強化）
  B4: +Streak≥6       - さらに厳格化（過学習リスクあり）
  B5: +ATR拡張+傾き   - 複合フィルター（ATR+1H slope）

【期間】OOS: 2025-03-03 〜 2026-02-27
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

INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
MAX_LOOKAHEAD = 20_000

A1_EMA_DIST_MIN   = 1.0
A2_EMA_SLOPE_BARS = 3
A4_ATR_PERIOD     = 5
A3_DEFAULT_TOL    = 0.30
E1_MAX_WAIT_MIN   = 5

OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# (label, streak_min, use_atr_expand, use_slope)
# 新ベースラインは全て EMA距離 ON（lean + EMA_dist）
VARIANTS = [
    ("new_baseline",   4, False, False),  # lean + EMA_dist
    ("+ATR_expand",    4, True,  False),
    ("+1H_slope",      4, False, True),
    ("+Streak5",       5, False, False),
    ("+Streak6",       6, False, False),
    ("+ATR+slope",     4, True,  True),
]

SYMBOLS = [
    {"name": "EURUSD", "lower": "eurusd"},
    {"name": "GBPUSD", "lower": "gbpusd"},
    {"name": "AUDUSD", "lower": "audusd"},
]


def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])


def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def build_4h(df4h):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df


def build_1h(data_15m):
    df = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    df["atr"]     = calc_atr(df, 14)
    df["ema20"]   = df["close"].ewm(span=20, adjust=False).mean()
    df["atr_avg"] = df["atr"].rolling(A4_ATR_PERIOD).mean()
    return df


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


def check_1h_ema_slope(data_1h, signal_time, direction):
    h1b = data_1h[data_1h.index < signal_time]
    if len(h1b) < A2_EMA_SLOPE_BARS + 1: return False
    ema_vals = h1b["ema20"].iloc[-(A2_EMA_SLOPE_BARS + 1):]
    slope = ema_vals.iloc[-1] - ema_vals.iloc[0]
    return slope > 0 if direction == 1 else slope < 0


def check_atr_expand(data_1h, signal_time):
    h1b = data_1h[data_1h.index < signal_time]
    if len(h1b) < 2: return False
    latest = h1b.iloc[-1]
    atr_now = latest["atr"]; atr_avg = latest["atr_avg"]
    if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg <= 0: return False
    return atr_now > atr_avg


def pick_entry_1m(signal_time, direction, spread, m1_cache):
    m1_idx = m1_cache["idx"]
    start  = m1_idx.searchsorted(signal_time, side="left")
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


def generate_signals(data_1m, data_15m, data_4h,
                     spread_pips, pip_size, streak_min,
                     use_atr_expand, use_slope,
                     atr_1m=None, m1_cache=None):
    spread   = spread_pips * pip_size
    data_4h  = build_4h(data_4h)
    data_1h  = build_1h(data_15m)

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
    min_idx    = max(2, streak_min)

    for i in range(min_idx, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak_min, 2): continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # Streak（FX: Streak≥streak_min）
        recent = h4_before["trend"].iloc[-streak_min:].values
        if not all(t == trend for t in recent): continue

        if not check_kmid(h4_latest, trend): continue
        if not check_klow(h4_latest): continue

        # EMA距離（新ベースラインで常時ON）
        if not check_ema_dist(h4_latest): continue

        # 追加フィルター
        if use_slope and not check_1h_ema_slope(data_1h, h1_ct, trend): continue
        if use_atr_expand and not check_atr_expand(data_1h, h1_ct): continue

        tol = atr_val * A3_DEFAULT_TOL

        for direction in [1, -1]:
            if trend != direction: continue
            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
            if abs(v1 - v2) > tol: continue

            et, ep = pick_entry_1m(h1_ct, direction, spread, m1_cache)
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
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


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


def calc_stats(trades, eq_curve, symbol, label):
    if not trades:
        return {"symbol": symbol, "variant": label,
                "n": 0, "wr": 0.0, "pf": 0.0, "mdd_pct": 0.0, "monthly_plus": "0/0"}
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
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev    = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > prev).sum()}/{len(monthly)}"
    return {"symbol": symbol, "variant": label,
            "n": n, "wr": round(wr*100,1), "pf": round(pf,2),
            "mdd_pct": round(mdd,1), "monthly_plus": mp}


def main():
    all_results = []

    for sym_cfg in SYMBOLS:
        sym_name  = sym_cfg["name"]
        sym_lower = sym_cfg["lower"]
        ohlc_dir  = os.path.join(DATA_DIR, "ohlc")

        def _load(tf):
            p = os.path.join(ohlc_dir, f"{sym_name}_{tf}.csv")
            if os.path.exists(p): return load_csv(p)
            return load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_{tf}.csv"))

        d1m  = _load("1m")
        d15m = _load("15m")
        d4h  = _load("4h")
        if d1m is None or d15m is None or d4h is None:
            print(f"[SKIP] {sym_name}: データ不足"); continue

        d1m_oos  = slice_period(d1m,  OOS_START, OOS_END)
        d15m_oos = slice_period(d15m, OOS_START, OOS_END)
        d4h_oos  = slice_period(d4h,  OOS_START, OOS_END)
        if d1m_oos is None or len(d1m_oos) == 0:
            print(f"[SKIP] {sym_name}: OOS 1m なし"); continue

        cfg         = SYMBOL_CONFIG.get(sym_name, {})
        spread_pips = cfg.get("spread", 0.0)
        pip_size    = cfg.get("pip", 0.0001)

        atr_1m = calc_atr(d1m_oos, 10).to_dict()
        m1_cache = {
            "idx":    d1m_oos.index,
            "opens":  d1m_oos["open"].values,
            "closes": d1m_oos["close"].values,
            "highs":  d1m_oos["high"].values,
            "lows":   d1m_oos["low"].values,
        }

        print(f"\n{'='*60}")
        print(f"  {sym_name}  [FX 新ベースライン検証]")
        print(f"{'='*60}")

        baseline_pf = None
        for (label, streak_min, use_atr_expand, use_slope) in VARIANTS:
            sigs = generate_signals(
                d1m_oos, d15m_oos, d4h_oos,
                spread_pips, pip_size,
                streak_min, use_atr_expand, use_slope,
                atr_1m, m1_cache,
            )
            trades, eq_curve = simulate(sigs, d1m_oos, sym_name)
            stats = calc_stats(trades, eq_curve, sym_name, label)
            all_results.append(stats)

            if label == "new_baseline":
                baseline_pf = stats["pf"]
                print(f"  [{label:14s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                      f"PF={stats['pf']:.2f}  MDD={stats['mdd_pct']:.1f}%  月次+={stats['monthly_plus']}")
            else:
                diff    = stats["pf"] - baseline_pf if baseline_pf else 0
                sign    = "+" if diff >= 0 else ""
                verdict = "  ✅ 追加採用候補" if diff >= 0 else "  ❌ 逆効果"
                print(f"  [{label:14s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                      f"PF={stats['pf']:.2f}  ({sign}{diff:.2f}){verdict}")

    # サマリー
    df_res = pd.DataFrame(all_results)
    print("\n\n" + "="*65)
    print("  FX 新ベースライン（lean+EMA距離）への追加フィルター")
    print("="*65)

    pf_table = df_res.pivot_table(
        index="symbol", columns="variant", values="pf", aggfunc="first"
    )
    vorder = [v[0] for v in VARIANTS]
    avail  = [v for v in vorder if v in pf_table.columns]
    pf_table = pf_table[avail]
    bl_df = pf_table["new_baseline"]

    print("\n■ PF変化（new_baseline比）")
    diff_table = pf_table.copy()
    for sym in diff_table.index:
        diff_table.loc[sym] = diff_table.loc[sym] - bl_df.get(sym, 0)
    print(diff_table.drop(columns=["new_baseline"], errors="ignore").to_string())

    fx_syms = [c["name"] for c in SYMBOLS]
    print("\n■ カテゴリ判定（FX 2/3基準）")
    for (vname, *_) in VARIANTS:
        if vname == "new_baseline": continue
        vdf = df_res[(df_res["symbol"].isin(fx_syms)) & (df_res["variant"] == vname)]
        improved = sum(
            row["pf"] >= bl_df.get(row["symbol"], 0)
            for _, row in vdf.iterrows()
        )
        total    = len(vdf)
        avg_diff = (vdf.set_index("symbol")["pf"] - bl_df[bl_df.index.isin(fx_syms)]).mean()
        sign     = "+" if avg_diff >= 0 else ""
        verdict  = "✅ 採用候補" if improved >= 2 else "❌ 逆効果"
        print(f"  {vname:14s}: {improved}/{total}銘柄改善  avg_diff={sign}{avg_diff:.2f}  → {verdict}")

    # グラフ
    n_sym = len(fx_syms)
    fig, axes = plt.subplots(1, n_sym, figsize=(6*n_sym, 5), squeeze=False)
    for ai, sym in enumerate(fx_syms):
        ax     = axes[0][ai]
        sym_df = df_res[df_res["symbol"] == sym]
        bl_pf  = sym_df[sym_df["variant"] == "new_baseline"]["pf"].values[0]
        plot_df = sym_df[sym_df["variant"] != "new_baseline"]
        colors  = ["green" if p >= bl_pf else "tomato" for p in plot_df["pf"]]
        bars    = ax.bar(plot_df["variant"], plot_df["pf"], color=colors, alpha=0.85)
        ax.axhline(bl_pf, color="navy", linestyle="--", linewidth=1.2,
                   label=f"new_baseline={bl_pf:.2f}")
        ax.axhline(2.0, color="red", linestyle=":", linewidth=0.8, label="PF=2.0")
        ax.set_title(sym, fontsize=12)
        ax.set_ylabel("PF (OOS)")
        ax.set_ylim(0, max(plot_df["pf"].max(), bl_pf) * 1.25)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
        for b, pf_v in zip(bars, plot_df["pf"]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f"{pf_v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("FX 新ベースライン（lean+EMA距離）への追加フィルター効果\n"
                 "緑=新baseline以上 / 赤=以下", fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "fx_new_baseline_additions.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nグラフ保存: {out_png}")

    out_csv = os.path.join(OUT_DIR, "fx_new_baseline_additions.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"CSV保存:   {out_csv}")


if __name__ == "__main__":
    main()
