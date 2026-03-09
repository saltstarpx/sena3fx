"""
plot_equity_curves.py
======================
Goldロジック（日足EMA20+E2）全銘柄エクイティカーブ描画

対象: EURUSD / GBPUSD / AUDUSD / XAUUSD
期間: 2025-01-01 〜 2026-02-28（IS+OOS全期間）
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
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ─────────────────────────────────────────────────────────
START     = "2025-01-01"
END       = "2026-02-28"
IS_END    = "2025-05-31"   # IS/OOS境界線

INIT_CASH    = 1_000_000
RISK_PCT     = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"]
COLORS  = {"EURUSD": "#f97316", "GBPUSD": "#eab308",
           "AUDUSD": "#22c55e", "XAUUSD": "#d97706"}

# ── データ ───────────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def slice_period(df, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def calc_atr(df, n=14):
    hl = df["high"]-df["low"]
    hc = (df["high"]-df["close"].shift()).abs()
    lc = (df["low"] -df["close"].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(n).mean()

def build_4h(df4h):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min",
                                 "close":"last","volume":"sum"}).dropna(subset=["open","close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df15m):
    df = df15m.resample("1h").agg({"open":"first","high":"max","low":"min",
                                    "close":"last","volume":"sum"}).dropna(subset=["open","close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

def check_kmid(bar, direction):
    return (direction==1 and bar["close"]>bar["open"]) or \
           (direction==-1 and bar["close"]<bar["open"])

def check_klow(bar):
    o,l = bar["open"], bar["low"]
    return (min(bar["open"],bar["close"])-l)/o < KLOW_THR if o>0 else False

def check_ema_dist(bar):
    d=abs(bar["close"]-bar["ema20"]); a=bar["atr"]
    return not pd.isna(a) and a>0 and d >= a*A1_EMA_DIST_MIN

def pick_e2(signal_time, direction, spread, atr_d, m1c):
    idx = m1c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = idx.searchsorted(signal_time+pd.Timedelta(minutes=max(2,E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        br = m1c["highs"][i]-m1c["lows"][i]
        av = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and br > av*E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i]+(spread if direction==1 else -spread)
    return None, None

def generate_signals(d1m, d15m, d4h_full, spread, atr_d, m1c):
    d4h, d1d = build_4h(d4h_full)
    d1h = build_1h(d15m)
    signals=[]; used=set()
    h1_times = d1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h<=0: continue
        h4_bef = d4h[d4h.index<h1_ct]
        if len(h4_bef)<2: continue
        h4_lat = h4_bef.iloc[-1]
        if pd.isna(h4_lat.get("atr",np.nan)): continue
        trend=h4_lat["trend"]; h4_atr=h4_lat["atr"]
        d1_bef = d1d[d1d.index.normalize()<h1_ct.normalize()]
        if len(d1_bef)==0: continue
        if d1_bef.iloc[-1]["trend1d"]!=trend: continue
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue
        tol = atr_1h*A3_DEFAULT_TOL
        direction=trend
        if direction==1: v1,v2=h1_p2["low"],h1_p1["low"]
        else:            v1,v2=h1_p2["high"],h1_p1["high"]
        if abs(v1-v2)>tol: continue
        et,ep = pick_e2(h1_ct, direction, spread, atr_d, m1c)
        if et is None or et in used: continue
        raw = ep-spread if direction==1 else ep+spread
        if direction==1: sl=min(v1,v2)-atr_1h*0.15; risk=raw-sl
        else:            sl=max(v1,v2)+atr_1h*0.15; risk=sl-raw
        if 0<risk<=h4_atr*2:
            tp=raw+direction*risk*RR_RATIO
            signals.append({"time":et,"dir":direction,"ep":ep,"sl":sl,
                            "tp":tp,"risk":risk})
            used.add(et)
    signals.sort(key=lambda x:x["time"])
    return signals

def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half=ep+direction*risk*HALF_R; lim=min(len(highs),MAX_LOOKAHEAD)
    for i in range(lim):
        h=highs[i]; lo=lows[i]
        if direction==1:
            if lo<=sl: return i,sl,"loss",False
            if h>=tp:  return i,tp,"win",False
            if h>=half:
                be=ep
                for j in range(i+1,lim):
                    if lows[j]<=be:  return j,be,"win",True
                    if highs[j]>=tp: return j,tp,"win",True
                return -1,None,None,True
        else:
            if h>=sl:  return i,sl,"loss",False
            if lo<=tp: return i,tp,"win",False
            if lo<=half:
                be=ep
                for j in range(i+1,lim):
                    if highs[j]>=be: return j,be,"win",True
                    if lows[j]<=tp:  return j,tp,"win",True
                return -1,None,None,True
    return -1,None,None,False

def simulate(signals, d1m, sym):
    rm=RiskManager(sym,risk_pct=RISK_PCT)
    m1t=d1m.index; m1h=d1m["high"].values; m1l=d1m["low"].values
    equity=INIT_CASH
    curve=[(d1m.index[0], INIT_CASH)]
    trades=[]
    peak=INIT_CASH; mdd=0.0
    for sig in signals:
        lot=rm.calc_lot(equity,sig["risk"],sig["ep"],usdjpy_rate=USDJPY_RATE)
        sp=m1t.searchsorted(sig["time"],side="right")
        if sp>=len(m1t): continue
        ei,xp,result,half_done=_find_exit(m1h[sp:],m1l[sp:],
                                           sig["ep"],sig["sl"],sig["tp"],sig["risk"],sig["dir"])
        if result is None: continue
        half_pnl=0.0
        if half_done:
            hp=sig["ep"]+sig["dir"]*sig["risk"]*HALF_R
            half_pnl=rm.calc_pnl_jpy(sig["dir"],sig["ep"],hp,lot*0.5,USDJPY_RATE,sig["ep"])
            equity+=half_pnl; rem=lot*0.5
        else: rem=lot
        pnl=rm.calc_pnl_jpy(sig["dir"],sig["ep"],xp,rem,USDJPY_RATE,sig["ep"])
        equity+=pnl
        total=half_pnl+pnl
        exit_t=m1t[sp+ei]
        curve.append((exit_t, equity))
        trades.append({"time":exit_t,"pnl":total,"result":result,"equity":equity})
        peak=max(peak,equity); mdd=max(mdd,(peak-equity)/peak*100)
    return trades, equity, mdd, curve

# ── メイン ───────────────────────────────────────────────────────
def main():
    all_curves={}; all_trades={}; all_stats={}
    is_end_ts = pd.Timestamp(IS_END, tz="UTC")

    for sym in SYMBOLS:
        print(f"  {sym} 計算中...", end=" ", flush=True)
        d1m  = slice_period(load_csv(f"{DATA_DIR}/{sym}_1m.csv"),  START, END)
        d15m = slice_period(load_csv(f"{DATA_DIR}/{sym}_15m.csv"), START, END)
        d4h  = load_csv(f"{DATA_DIR}/{sym}_4h.csv")

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"]*cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx":d1m.index,"opens":d1m["open"].values,
                  "closes":d1m["close"].values,
                  "highs":d1m["high"].values,"lows":d1m["low"].values}

        sigs   = generate_signals(d1m, d15m, d4h, spread, atr_d, m1c)
        trades, final_eq, mdd, curve = simulate(sigs, d1m, sym)

        all_curves[sym] = curve
        all_trades[sym] = trades
        wins  = [t for t in trades if t["result"]=="win"]
        loses = [t for t in trades if t["result"]=="loss"]
        gw    = sum(t["pnl"] for t in wins)
        gl    = abs(sum(t["pnl"] for t in loses))
        pf    = gw/gl if gl>0 else float("inf")
        wr    = len(wins)/len(trades) if trades else 0

        all_stats[sym] = {
            "n":len(trades),"wr":wr,"pf":pf,
            "final":final_eq,"mdd":mdd,
            "total_pnl":final_eq-INIT_CASH,
            "mult":final_eq/INIT_CASH,
        }
        print(f"完了 → {len(trades)}トレード  PF={pf:.2f}  最終={final_eq/1e6:.2f}M円")

    # ── グラフ ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [2.5, 1]})

    ax_main = axes[0]
    ax_ret  = axes[1]

    # IS/OOS境界線
    ax_main.axvline(is_end_ts, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax_ret.axvline( is_end_ts, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax_main.text(is_end_ts, ax_main.get_ylim()[0] if False else 0,
                 " IS | OOS", fontsize=8, color="gray", va="bottom")

    # 個別エクイティカーブ
    combined_index = pd.date_range(
        pd.Timestamp(START, tz="UTC"),
        pd.Timestamp(END, tz="UTC")+pd.Timedelta(days=1),
        freq="D"
    )
    combined_equity = pd.Series(0.0, index=combined_index)

    sym_series = {}
    for sym in SYMBOLS:
        curve = all_curves[sym]
        times = [t for t, _ in curve]
        eqs   = [e for _, e in curve]
        ts    = pd.Series(eqs, index=pd.DatetimeIndex(times))
        ts    = ts[~ts.index.duplicated(keep="last")]
        ts_daily = ts.resample("D").last().ffill()
        sym_series[sym] = ts_daily

        ax_main.plot(ts.index, [e/1e4 for e in ts.values],
                     color=COLORS[sym], lw=1.4, label=sym, alpha=0.9)

    # 合算エクイティ（4銘柄合計）
    all_daily = pd.DataFrame(sym_series).ffill().fillna(INIT_CASH)
    combined  = all_daily.sum(axis=1)
    ax_main.plot(combined.index, [v/1e4 for v in combined.values],
                 color="#6366f1", lw=2.5, ls="-", label="合算（4銘柄）", alpha=0.95, zorder=5)

    ax_main.axhline(INIT_CASH/1e4, color="black", lw=0.8, ls=":", alpha=0.5)
    ax_main.set_ylabel("資産（万円）", fontsize=11)
    ax_main.set_title("YAGAMI改 Goldロジック — 全銘柄エクイティカーブ\n"
                      f"({START} 〜 {END}  / 各銘柄初期100万円)",
                      fontsize=12, fontweight="bold")
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_main.grid(True, alpha=0.25)
    ax_main.text(is_end_ts, ax_main.get_ylim()[1]*0.02 if False else 100,
                 " ← IS | OOS →", fontsize=8, color="gray")

    # リターン率（個別）
    for sym in SYMBOLS:
        curve = all_curves[sym]
        times = [t for t, _ in curve]
        eqs   = [e for _, e in curve]
        rets  = [(e/INIT_CASH-1)*100 for e in eqs]
        ax_ret.plot(times, rets, color=COLORS[sym], lw=1.0, alpha=0.7, label=sym)

    ax_ret.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)
    ax_ret.set_ylabel("累積リターン(%)", fontsize=10)
    ax_ret.legend(loc="upper left", fontsize=9, ncol=4)
    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax_ret.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_ret.grid(True, alpha=0.25)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "gold_logic_equity_curves.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nグラフ保存: {out_png}")

    # ── 数字サマリー ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  最終資産サマリー（Goldロジック / 各100万円スタート）")
    print("="*70)
    print(f"  {'銘柄':8} {'トレード':>8} {'勝率':>7} {'PF':>6} {'MDD':>7} "
          f"{'最終資産':>12} {'倍率':>6} {'総損益':>12}")
    print("  " + "-"*68)

    total_final = INIT_CASH * len(SYMBOLS)
    total_pnl   = 0
    for sym in SYMBOLS:
        s = all_stats[sym]
        print(f"  {sym:8} {s['n']:>8}   {s['wr']*100:>5.1f}%  {s['pf']:>5.2f}  "
              f"{s['mdd']:>5.1f}%  {s['final']:>10,.0f}円  "
              f"{s['mult']:>5.2f}x  {s['total_pnl']:>+12,.0f}円")
        total_final += s["total_pnl"]
        total_pnl   += s["total_pnl"]

    print("  " + "-"*68)
    combined_start = INIT_CASH*len(SYMBOLS)
    print(f"  {'合算':8} {'':>8}   {'':>5}   {'':>5}   {'':>5}   "
          f"{combined_start+total_pnl:>10,.0f}円  "
          f"{(combined_start+total_pnl)/combined_start:>5.2f}x  "
          f"{total_pnl:>+12,.0f}円")
    print(f"\n  初期合計: {combined_start/1e6:.0f}M円 → "
          f"最終合計: {(combined_start+total_pnl)/1e6:.2f}M円")

if __name__ == "__main__":
    main()
