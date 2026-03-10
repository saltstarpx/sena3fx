"""
backtest_3sym_adaptive_chart.py
================================
NZDUSD / AUDUSD / XAUUSD 3銘柄 アダプティブリスク バックテスト + 可視化

【アダプティブリスク】
  - 初期: 2%  勝ち→+0.5%  負け→-0.5%  [2%〜3%]
  - MIN_RISK_PIPS=3（縮退シグナル防止バグ修正済み）

【コスト】
  FX  (NZDUSD/AUDUSD): $5/100,000通貨/片道
  金属 (XAUUSD)       : $5/100oz/片道

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31
  OOS: 2025-06-01 〜 2026-02-28
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 期間設定 ────────────────────────────────────────────────────
IS_START  = "2025-01-01"
IS_END    = "2025-05-31"
OOS_START = "2025-06-01"
OOS_END   = "2026-02-28"

# ── 資金設定 ─────────────────────────────────────────────────────
INIT_CASH   = 1_000_000
RR_RATIO    = 2.5
HALF_R      = 1.0
USDJPY_RATE = 150.0

# ── アダプティブリスク ────────────────────────────────────────────
RISK_INIT = 0.02
RISK_MIN  = 0.02
RISK_MAX  = 0.03
RISK_STEP = 0.005

# ── フィルター定数 ────────────────────────────────────────────────
KLOW_THR       = 0.0015
EMA_DIST_MIN   = 1.0
PATTERN_TOL    = 0.30
E2_SPIKE_ATR   = 2.0
E2_WINDOW_BARS = 3
MAX_LOOKAHEAD  = 5_000
MIN_RISK_PIPS  = 3

# ── 銘柄別設定 ────────────────────────────────────────────────────
SYM_CONFIGS = {
    "NZDUSD": {"spread_pips": 0.3,  "metals": False, "color": "#8b5cf6", "label": "NZDUSD"},
    "AUDUSD": {"spread_pips": 0.0,  "metals": False, "color": "#22c55e", "label": "AUDUSD"},
    "XAUUSD": {"spread_pips": 5.2,  "metals": True,  "color": "#d97706", "label": "XAUUSD"},
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── ユーティリティ ────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def load_sym(sym):
    syml = sym.lower()
    is15  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_15m.csv"))
    oos15 = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_15m.csv"))
    is4h  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_4h.csv"))
    oos4h = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_4h.csv"))
    d15 = pd.concat([is15, oos15]).sort_index() if (is15 is not None and oos15 is not None) else is15 or oos15
    d4h = pd.concat([is4h, oos4h]).sort_index() if (is4h is not None and oos4h is not None) else is4h or oos4h
    if d15 is not None: d15 = d15[~d15.index.duplicated(keep="first")]
    if d4h is not None: d4h = d4h[~d4h.index.duplicated(keep="first")]
    return d15, d4h


def slice_period(df, start, end):
    if df is None or len(df) == 0: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()


def build_4h(df4h):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    d1 = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna(subset=["open", "close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1


def build_1h(df15m):
    df = df15m.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


def check_kmid(bar, direction):
    return (direction == 1  and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])

def check_klow(bar):
    o, l = bar["open"], bar["low"]
    return (min(bar["open"], bar["close"]) - l) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * EMA_DIST_MIN


def pick_e2_15m(signal_time, direction, spread_price, atr_15m_d, m15c):
    idx = m15c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = min(s + E2_WINDOW_BARS, len(idx))
    for i in range(s, e):
        bar_range = m15c["highs"][i] - m15c["lows"][i]
        atr_val   = atr_15m_d.get(idx[i], np.nan)
        if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR:
            continue
        ep = m15c["opens"][i] + (spread_price if direction == 1 else -spread_price)
        return idx[i], ep
    return None, None


def generate_signals(d15m_period, d4h_full, spread_price, min_risk, atr_15m_d, m15c):
    d4h, d1d = build_4h(d4h_full)
    d1h = build_1h(d15m_period)
    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0: continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < 2: continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)): continue
        trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]

        d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0: continue
        if d1_before.iloc[-1]["trend1d"] != trend: continue

        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue

        tol = atr_1h * PATTERN_TOL
        direction = trend
        if direction == 1:  v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:               v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        et, ep = pick_e2_15m(h1_ct, direction, spread_price, atr_15m_d, m15c)
        if et is None or et in used: continue

        raw = ep - spread_price if direction == 1 else ep + spread_price
        if direction == 1: sl = min(v1, v2) - atr_1h * 0.15; risk = raw - sl
        else:              sl = max(v1, v2) + atr_1h * 0.15; risk = sl - raw
        if min_risk <= risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half = ep + direction * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl: return i, sl, "loss", False
            if h  >= tp: return i, tp, "win",  False
            if h  >= half:
                be = ep
                for j in range(i+1, lim):
                    if lows[j]  <= be: return j, be, "win",  True
                    if highs[j] >= tp: return j, tp, "win",  True
                return -1, None, None, True
        else:
            if h  >= sl: return i, sl, "loss", False
            if lo <= tp: return i, tp, "win",  False
            if lo <= half:
                be = ep
                for j in range(i+1, lim):
                    if highs[j] >= be: return j, be, "win",  True
                    if lows[j]  <= tp: return j, tp, "win",  True
                return -1, None, None, True
    return -1, None, None, False


def calc_commission(lot, metals=False):
    """往復手数料（JPY）"""
    if metals:
        return (lot / 100) * 5.0 * 2 * USDJPY_RATE      # $5/100oz/片道
    else:
        return (lot / 100_000) * 5.0 * 2 * USDJPY_RATE  # $5/100,000通貨/片道


def simulate_adaptive(sym, signals, d15m, metals=False):
    """アダプティブリスク シミュレーション（エクイティ曲線付き）"""
    if not signals:
        return [], INIT_CASH, 0, 0, []

    current_risk = RISK_INIT
    rm   = RiskManager(sym, risk_pct=current_risk)
    m15t = d15m.index
    m15h = d15m["high"].values
    m15l = d15m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []; equity_curve = [(m15t[0], INIT_CASH)]
    total_commission = 0.0

    for sig in signals:
        rm.risk_pct = current_risk
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0: continue

        commission = calc_commission(lot, metals)
        equity -= commission
        total_commission += commission

        sp = m15t.searchsorted(sig["time"], side="right")
        if sp >= len(m15t): continue

        ei, xp, result, half_done = _find_exit(
            m15h[sp:], m15l[sp:],
            sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl; rem = lot * 0.5
        else:
            rem = lot

        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total_pnl = half_pnl + pnl

        if total_pnl > 0: current_risk = round(min(current_risk + RISK_STEP, RISK_MAX), 4)
        else:             current_risk = round(max(current_risk - RISK_STEP, RISK_MIN), 4)

        exit_time = m15t[min(sp + ei, len(m15t)-1)] if ei >= 0 else sig["time"]
        trades.append({"time": sig["time"], "exit_time": exit_time,
                       "result": result, "pnl": total_pnl,
                       "commission": commission, "risk_used": rm.risk_pct,
                       "equity": equity})
        equity_curve.append((exit_time, equity))
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, total_commission, equity_curve


def run_sym(sym, period="oos"):
    cfg = SYM_CONFIGS[sym]
    pip = SYMBOL_CONFIG[sym]["pip"]
    spread_pr = cfg["spread_pips"] * pip
    min_risk  = pip * MIN_RISK_PIPS
    metals    = cfg["metals"]

    d15m_full, d4h_full = load_sym(sym)
    if d15m_full is None or d4h_full is None:
        return None

    start, end = (OOS_START, OOS_END) if period == "oos" else (IS_START, IS_END)
    d15m = slice_period(d15m_full, start, end)
    if d15m is None or len(d15m) == 0: return None

    atr_15m = calc_atr(d15m, 10).to_dict()
    m15c    = {"idx": d15m.index, "opens": d15m["open"].values,
               "highs": d15m["high"].values, "lows": d15m["low"].values}
    sigs    = generate_signals(d15m, d4h_full, spread_pr, min_risk, atr_15m, m15c)
    trades, final_eq, mdd, total_comm, eq_curve = simulate_adaptive(sym, sigs, d15m, metals)

    if not trades:
        return None

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]["pnl"]; los = df[df["pnl"] < 0]["pnl"]
    pf   = wins.sum() / abs(los.sum()) if len(los) > 0 else float("inf")
    wr   = len(wins) / len(df)

    # 月次集計
    df["ym"] = df["time"].dt.strftime("%Y-%m")
    monthly = df.groupby("ym")["pnl"].sum()

    # MDD系列
    eq_df = pd.DataFrame(eq_curve, columns=["time", "equity"]).set_index("time")
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"]   = (eq_df["peak"] - eq_df["equity"]) / eq_df["peak"] * 100

    return {
        "sym": sym, "n": len(df), "wr": wr, "pf": pf, "mdd": mdd,
        "final_eq": final_eq, "multiplier": final_eq / INIT_CASH,
        "commission": total_comm, "monthly": monthly,
        "eq_curve": eq_curve, "eq_df": eq_df,
        "trades": df, "avg_risk": df["risk_used"].mean()
    }


# ── チャート描画 ──────────────────────────────────────────────────
def plot_results(results_oos, results_is):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#0f172a",
        "axes.facecolor":   "#1e293b",
        "axes.edgecolor":   "#334155",
        "text.color":       "#f1f5f9",
        "axes.labelcolor":  "#94a3b8",
        "xtick.color":      "#64748b",
        "ytick.color":      "#64748b",
        "grid.color":       "#1e293b",
        "grid.linewidth":   0.5,
    })

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("NZDUSD / AUDUSD / XAUUSD  アダプティブリスク 2〜3%  OOS バックテスト",
                 fontsize=15, fontweight="bold", color="#f8fafc", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06)

    ax_eq   = fig.add_subplot(gs[0, :])   # エクイティ曲線（全幅）
    ax_mon  = fig.add_subplot(gs[1, :2])  # 月次棒グラフ
    ax_mdd  = fig.add_subplot(gs[1, 2])   # MDD推移
    ax_tbl  = fig.add_subplot(gs[2, :])   # 指標テーブル
    ax_tbl.axis("off")

    colors = {s: SYM_CONFIGS[s]["color"] for s in SYM_CONFIGS}

    # ── エクイティ曲線 ──────────────────────────────────────────────
    ax_eq.set_title("エクイティ曲線（OOS: 2025-06〜2026-02）", fontsize=11, color="#cbd5e1", pad=8)
    ax_eq.set_facecolor("#0f172a")
    ax_eq.spines[:].set_color("#334155")

    combined_eq = {}
    for sym, res in results_oos.items():
        if res is None: continue
        times  = [t for t, e in res["eq_curve"]]
        equities = [e for t, e in res["eq_curve"]]
        ax_eq.plot(times, [e/10000 for e in equities],
                   color=colors[sym], linewidth=2.0, label=f"{sym}", alpha=0.9)
        for t, e in res["eq_curve"]:
            combined_eq[t] = combined_eq.get(t, 0) + (e - INIT_CASH)

    ax_eq.axhline(y=INIT_CASH/10000, color="#475569", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_eq.set_ylabel("資産（万円）", fontsize=9, color="#94a3b8")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}万"))
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.3,
                 facecolor="#1e293b", edgecolor="#475569")
    ax_eq.grid(True, alpha=0.2)
    ax_eq.tick_params(axis="x", rotation=30, labelsize=8)

    # ── 月次棒グラフ（3銘柄積み上げ） ────────────────────────────────
    ax_mon.set_title("月次損益（万円）", fontsize=10, color="#cbd5e1", pad=8)
    ax_mon.set_facecolor("#0f172a")
    ax_mon.spines[:].set_color("#334155")

    all_months = sorted(set(
        m for res in results_oos.values() if res
        for m in res["monthly"].index
    ))
    x = np.arange(len(all_months))
    bar_w = 0.25
    offsets = {"NZDUSD": -bar_w, "AUDUSD": 0, "XAUUSD": bar_w}

    for sym, res in results_oos.items():
        if res is None: continue
        vals = [res["monthly"].get(m, 0) / 10000 for m in all_months]
        bar_colors = [colors[sym] if v >= 0 else "#ef4444" for v in vals]
        bars = ax_mon.bar(x + offsets[sym], vals, bar_w - 0.02,
                          color=bar_colors, alpha=0.8, label=sym)

    ax_mon.set_xticks(x)
    ax_mon.set_xticklabels([m[5:] for m in all_months], rotation=45, fontsize=7)
    ax_mon.axhline(0, color="#475569", linewidth=0.8)
    ax_mon.set_ylabel("万円", fontsize=9, color="#94a3b8")
    ax_mon.legend(fontsize=8, framealpha=0.3, facecolor="#1e293b", edgecolor="#475569")
    ax_mon.grid(True, axis="y", alpha=0.2)

    # ── MDD推移 ──────────────────────────────────────────────────
    ax_mdd.set_title("ドローダウン推移（OOS）", fontsize=10, color="#cbd5e1", pad=8)
    ax_mdd.set_facecolor("#0f172a")
    ax_mdd.spines[:].set_color("#334155")

    for sym, res in results_oos.items():
        if res is None: continue
        eq_df = res["eq_df"]
        ax_mdd.fill_between(eq_df.index, -eq_df["dd"], 0,
                            color=colors[sym], alpha=0.3, label=sym)
        ax_mdd.plot(eq_df.index, -eq_df["dd"],
                    color=colors[sym], linewidth=1.2, alpha=0.8)

    ax_mdd.set_ylabel("DD（%）", fontsize=9, color="#94a3b8")
    ax_mdd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{-x:.0f}%"))
    ax_mdd.legend(fontsize=7, framealpha=0.3, facecolor="#1e293b", edgecolor="#475569")
    ax_mdd.grid(True, alpha=0.2)
    ax_mdd.tick_params(axis="x", rotation=30, labelsize=7)

    # ── 指標テーブル ─────────────────────────────────────────────
    headers = ["銘柄", "IS PF", "OOS PF", "勝率", "MDD",
               "トレード数", "資産倍率", "最終資産", "プラス月", "平均リスク"]
    rows = []
    for sym in ["NZDUSD", "AUDUSD", "XAUUSD"]:
        r_is  = results_is.get(sym)
        r_oos = results_oos.get(sym)
        if r_oos is None:
            rows.append([sym] + ["N/A"] * (len(headers)-1))
            continue

        plus_m = sum(1 for v in r_oos["monthly"].values if v >= 0)
        total_m = len(r_oos["monthly"])
        pf_is  = f"{r_is['pf']:.2f}" if r_is and r_is['pf'] < 99 else ("∞" if r_is else "—")
        pf_oos = f"{r_oos['pf']:.2f}" if r_oos['pf'] < 99 else "∞"

        rows.append([
            sym,
            pf_is,
            pf_oos,
            f"{r_oos['wr']*100:.1f}%",
            f"{r_oos['mdd']:.1f}%",
            f"{r_oos['n']}件",
            f"{r_oos['multiplier']:.1f}x",
            f"{r_oos['final_eq']/10000:.0f}万円",
            f"{plus_m}/{total_m}月",
            f"{r_oos['avg_risk']*100:.2f}%",
        ])

    tbl = ax_tbl.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    header_color = "#1e40af"
    row_colors   = ["#1e293b", "#162032"]
    highlight    = {"NZDUSD": "#3730a3", "AUDUSD": "#14532d", "XAUUSD": "#78350f"}

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#334155")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="#e2e8f0", fontweight="bold")
        else:
            sym_name = rows[r-1][0] if r-1 < len(rows) else ""
            cell.set_facecolor(highlight.get(sym_name, row_colors[(r-1) % 2]))
            cell.set_text_props(color="#f1f5f9")
            # PF強調
            if c == 2:
                try:
                    pf_val = float(rows[r-1][2])
                    if pf_val >= 2.0:
                        cell.set_text_props(color="#4ade80", fontweight="bold")
                    elif pf_val < 1.5:
                        cell.set_text_props(color="#f87171")
                except: pass
            # MDD強調
            if c == 4:
                try:
                    mdd_val = float(rows[r-1][4].replace("%",""))
                    if mdd_val >= 30:
                        cell.set_text_props(color="#fbbf24")
                except: pass

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "3sym_adaptive_backtest.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 グラフ保存: {out_path}")
    return out_path


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  NZDUSD / AUDUSD / XAUUSD  アダプティブリスク 2〜3%  バックテスト")
    print(f"  初期資産: {INIT_CASH:,}円  RR: {RR_RATIO}  IS: {IS_START}〜{IS_END}")
    print(f"  OOS: {OOS_START}〜{OOS_END}  MIN_RISK: {MIN_RISK_PIPS}pips")
    print("="*70)

    results_is  = {}
    results_oos = {}

    for sym in ["NZDUSD", "AUDUSD", "XAUUSD"]:
        print(f"  {sym} 計算中...", end=" ", flush=True)
        results_is[sym]  = run_sym(sym, "is")
        results_oos[sym] = run_sym(sym, "oos")
        print("完了")

    # ── テキスト出力 ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  ■ OOS 結果サマリー（アダプティブリスク 2〜3%）")
    print(f"  {'銘柄':8} {'IS PF':8} {'OOS PF':8} {'勝率':8} {'MDD':8} {'倍率':8} {'最終資産':12} {'プラス月':10}")
    print("  " + "-"*72)
    for sym in ["NZDUSD", "AUDUSD", "XAUUSD"]:
        ri = results_is[sym]; ro = results_oos[sym]
        if ro is None:
            print(f"  {sym:8} データなし"); continue
        pf_is  = f"{ri['pf']:.2f}" if ri and ri['pf']<99 else "—"
        pf_oos = f"{ro['pf']:.2f}" if ro['pf']<99 else "∞"
        plus_m = sum(1 for v in ro["monthly"].values if v >= 0)
        total_m= len(ro["monthly"])
        judge  = "✅" if ro["pf"] >= 2.0 and ro["mdd"] <= 30 else "⚠️"
        print(f"  {sym:8} {pf_is:8} {pf_oos:8} {ro['wr']*100:6.1f}%  "
              f"{ro['mdd']:6.1f}%  {ro['multiplier']:6.1f}x  "
              f"{ro['final_eq']/10000:8.0f}万円  {plus_m}/{total_m}月  {judge}")

    print("\n  ■ OOS 月次損益（万円）")
    all_months = sorted(set(
        m for ro in results_oos.values() if ro
        for m in ro["monthly"].index
    ))
    header = f"  {'月':>8}"
    for sym in ["NZDUSD","AUDUSD","XAUUSD"]: header += f" {sym:>12}"
    header += f" {'合計':>12}"
    print(header)
    print("  " + "-"*56)
    for ym in all_months:
        line = f"  {ym:>8}"
        total = 0
        for sym in ["NZDUSD","AUDUSD","XAUUSD"]:
            ro = results_oos.get(sym)
            v  = ro["monthly"].get(ym, 0) / 10000 if ro else 0
            total += v
            sign = "+" if v >= 0 else ""
            line += f" {sign}{v:>11.1f}"
        sign = "+" if total >= 0 else ""
        line += f" {sign}{total:>11.1f}"
        print(line)

    # ── グラフ ───────────────────────────────────────────────────
    print("\n  グラフ生成中...")
    out = plot_results(results_oos, results_is)
    print("  完了")


if __name__ == "__main__":
    main()
