"""
backtest_v78_improvements.py
=============================
v77（バグ修正済み）に改善を段階的に追加してIS/OOSで比較するバックテスト

【比較する5バリアント】
v77  : ベース（バグ修正済み）
v78A : + 1H確認足 KLOWチェック  （backtest_7sym_vmaxに存在するが v77正式版に抜けていた）
v78B : + 時間帯フィルター UTC 5-20時
v78C : + 確認足 実体サイズ最小値 ATR×0.2
v78D : + パターン許容幅を ATR×0.3→ATR×0.2 に縮小

【対象銘柄】1mデータあり: XAUUSD / AUDUSD / EURUSD / GBPUSD / SPX500 / US30 / NAS100
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

# ── 設定 ──────────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
KLOW_THR  = 0.0015
IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# 改善バリアント設定
VARIANTS = {
    "v77":  dict(h1_klow=False, time_filter=False, body_min=False, tight_tol=False),
    "v78A": dict(h1_klow=True,  time_filter=False, body_min=False, tight_tol=False),
    "v78B": dict(h1_klow=True,  time_filter=True,  body_min=False, tight_tol=False),
    "v78C": dict(h1_klow=True,  time_filter=True,  body_min=True,  tight_tol=False),
    "v78D": dict(h1_klow=True,  time_filter=True,  body_min=True,  tight_tol=True),
}

PAIRS = {
    "XAUUSD": {"sym": "xauusd", "utc_end": 21},
    "AUDUSD": {"sym": "audusd", "utc_end": 20},
    "EURUSD": {"sym": "eurusd", "utc_end": 20},
    "GBPUSD": {"sym": "gbpusd", "utc_end": 20},
    "SPX500": {"sym": "spx500", "utc_end": 22},
    "US30":   {"sym": "us30",   "utc_end": 22},
    "NAS100": {"sym": "nas100", "utc_end": 22},
}

# ── ユーティリティ ─────────────────────────────────────────
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
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def load_1m(sym, is_period=True):
    tag = "is" if is_period else "oos"
    df = load_csv(os.path.join(DATA_DIR, f"{sym}_{tag}_1m.csv"))
    if df is None:
        df = load_csv(os.path.join(DATA_DIR, f"{sym}_1m.csv"))
    if df is None: return None
    start, end = (IS_START, IS_END) if is_period else (OOS_START, OOS_END)
    return slice_period(df, start, end)

def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df):
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

# ── シグナル生成（1Hモードのみ、改善フラグ付き） ───────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size,
                        h1_klow, time_filter, body_min, tight_tol, utc_end=20):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    tol_factor = 0.2 if tight_tol else 0.3   # 改善D: パターン許容幅縮小
    body_min_factor = 0.2                      # 改善C: 確認足実体最小値（ATR×0.2）

    signals = []; used_times = set()
    h1_times = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        # 改善B: 時間帯フィルター（UTC 5時〜utc_end時）
        if time_filter and not (5 <= h1_ct.hour < utc_end):
            continue

        # 完結済み4H足のみ取得（BUG②修正済み）
        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) == 0: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue
        trend = h4_latest["trend"]; h4_atr = h4_latest["atr"]
        tol   = atr_val * tol_factor

        for direction in [1, -1]:
            if trend != direction: continue

            if direction == 1:
                v1, v2 = h1_prev2["low"], h1_prev1["low"]
                conf_ok = h1_prev1["close"] > h1_prev1["open"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
                conf_ok = h1_prev1["close"] < h1_prev1["open"]

            if abs(v1 - v2) > tol: continue
            if not conf_ok: continue

            # 改善C: 確認足実体サイズ最小値
            if body_min:
                body = abs(h1_prev1["close"] - h1_prev1["open"])
                if body < atr_val * body_min_factor:
                    continue

            # 4H文脈足 KMID+KLOW（v77 BUG①修正済み）
            if not check_kmid_klow(h4_latest, direction): continue

            # 改善A: 1H確認足 KLOW（v77正式版に抜けていた処理）
            if h1_klow and not check_kmid_klow(h1_prev1, direction): continue

            m1w = data_1m[
                (data_1m.index >= h1_ct) &
                (data_1m.index < h1_ct + pd.Timedelta(minutes=2))
            ]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue

            raw = eb["open"]
            if direction == 1:
                sl = min(v1, v2) - atr_val * 0.15
                ep = raw + spread; risk = raw - sl
            else:
                sl = max(v1, v2) + atr_val * 0.15
                ep = raw - spread; risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({"time": et, "dir": direction,
                                 "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ───────────────────────────────────────
def simulate(signals, data_1m, symbol):
    if not signals:
        return [], [INIT_CASH]
    rm = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH; trades = []; eq_curve = [INIT_CASH]

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]; sl = sig["sl"]
        tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=150.0)
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0: continue

        half_done = False; be_sl = None; result = None
        exit_price = None; exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                cur_sl = be_sl if half_done else sl
                if bar["low"] <= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(1, ep, exit_price, lot*rem, 150.0, ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * HALF_R:
                        equity += rm.calc_pnl_jpy(1, ep, ep+risk*HALF_R, lot*0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(1, ep, tp, lot*rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["high"] >= ep + risk * HALF_R:
                    equity += rm.calc_pnl_jpy(1, ep, ep+risk*HALF_R, lot*0.5, 150.0, ep)
                    half_done = True; be_sl = ep
            else:
                cur_sl = be_sl if half_done else sl
                if bar["high"] >= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(-1, ep, exit_price, lot*rem, 150.0, ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * HALF_R:
                        equity += rm.calc_pnl_jpy(-1, ep, ep-risk*HALF_R, lot*0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(-1, ep, tp, lot*rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["low"] <= ep - risk * HALF_R:
                    equity += rm.calc_pnl_jpy(-1, ep, ep-risk*HALF_R, lot*0.5, 150.0, ep)
                    half_done = True; be_sl = ep

        if result is None: continue
        trades.append({"entry_time": sig["time"], "exit_time": exit_time,
                        "dir": direction, "ep": ep, "sl": sl, "tp": tp,
                        "exit_price": exit_price, "result": result, "equity": equity})
        eq_curve.append(equity)
    return trades, eq_curve

# ── 統計 ──────────────────────────────────────────────────
def calc_stats(trades, eq_curve, pair, variant, period):
    n_sig = len(trades)
    if not trades:
        return {"pair": pair, "variant": variant, "period": period,
                "n": 0, "wr": 0, "pf": 0, "mdd": 0, "kelly": 0,
                "monthly_plus": "0/0", "pass": False}
    df = pd.DataFrame(trades)
    wins = df[df["result"] == "win"]
    n = len(df); wr = len(wins) / n
    eq = np.array(eq_curve)
    deltas = np.diff(eq)
    gw = deltas[deltas > 0].sum()
    gl = abs(deltas[deltas < 0].sum())
    pf   = gw / gl if gl > 0 else float("inf")
    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq - peak) / peak).min()) * 100
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    mprev   = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > mprev).sum()}/{len(monthly)}"
    passed  = pf >= 2.0 and kelly >= 0.0 and mdd <= 25.0
    return {"pair": pair, "variant": variant, "period": period,
            "n": n, "wr": round(wr*100, 1), "pf": round(pf, 2),
            "mdd": round(mdd, 1), "kelly": round(kelly, 3),
            "monthly_plus": mp, "pass": passed}

# ── メイン ────────────────────────────────────────────────
print("=" * 90)
print("v77 → v78 段階的改善バックテスト（1Hモード）")
print(f"IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
print("改善A: 1H確認足KLOW | 改善B: 時間帯UTC5-20 | 改善C: 確認足実体ATR×0.2 | 改善D: 許容幅ATR×0.2")
print("=" * 90)

all_results = []

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]; utc_end = cfg["utc_end"]
    rm  = RiskManager(pair, risk_pct=RISK_PCT)
    spread = rm.spread_pips; pip = rm.pip_size

    d1m_is  = load_1m(sym, is_period=True)
    d15m_is = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv")), IS_START, IS_END)
    d4h_is  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv")),  IS_START, IS_END)
    d1m_oos  = load_1m(sym, is_period=False)
    d15m_oos = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv")), OOS_START, OOS_END)
    d4h_oos  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv")),  OOS_START, OOS_END)

    if any(d is None or len(d) == 0 for d in [d1m_is, d15m_is, d4h_is, d1m_oos, d15m_oos, d4h_oos]):
        print(f"\n{pair}: [SKIP] データ不足")
        continue

    print(f"\n{'='*70}")
    print(f"  {pair}  スプレッド:{spread}pips  時間帯フィルター上限:UTC{utc_end}時")
    print(f"  {'バリアント':<8} {'期間':<5} {'件数':>5} {'勝率':>7} {'PF':>6} {'MDD':>7} {'Kelly':>7} {'月次+':>7} {'PASS'}")
    print(f"  {'-'*65}")

    for vname, vflags in VARIANTS.items():
        for period, d1m, d15m, d4h in [("IS",  d1m_is,  d15m_is,  d4h_is),
                                         ("OOS", d1m_oos, d15m_oos, d4h_oos)]:
            sigs   = generate_signals_1h(
                d1m, d15m, d4h, spread, pip, utc_end=utc_end, **vflags)
            trades, eq = simulate(sigs, d1m, pair)
            stats = calc_stats(trades, eq, pair, vname, period)
            all_results.append(stats)
            mark = "✅" if (stats["pass"] and period == "OOS") else "  "
            print(f"  {vname:<8} {period:<5} {stats['n']:>5} "
                  f"{stats['wr']:>6.1f}% {stats['pf']:>6.2f} "
                  f"{stats['mdd']:>6.1f}% {stats['kelly']:>7.3f} "
                  f"{stats['monthly_plus']:>7}  {mark}")

# ── CSV保存 ──────────────────────────────────────────────
df_all = pd.DataFrame(all_results)
csv_path = os.path.join(OUT_DIR, "v78_improvements.csv")
df_all.to_csv(csv_path, index=False)
print(f"\n結果CSV: {csv_path}")

# ── OOS PF可視化 ──────────────────────────────────────────
oos = df_all[df_all["period"] == "OOS"].copy()
pairs_done = oos["pair"].unique().tolist()
variants   = list(VARIANTS.keys())
x          = np.arange(len(pairs_done))
colors     = ["#64748b","#22c55e","#3b82f6","#f97316","#ec4899"]

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("v77→v78 段階的改善 OOS比較\n"
             "A:1H確認足KLOW | B:+時間帯UTC5-20 | C:+確認足実体ATR×0.2 | D:+許容幅ATR×0.2",
             fontsize=12, fontweight="bold")

metrics = [("pf", "OOS PF", "プロフィットファクター", 2.0),
           ("wr", "OOS 勝率", "勝率（%）", 50.0),
           ("mdd", "OOS MDD", "最大ドローダウン（%）", 25.0),
           ("n",  "OOS トレード数", "シグナル件数", None)]

w = 0.16
for ax, (metric, title, ylabel, ref) in zip(axes.flatten(), metrics):
    for j, (v, c) in enumerate(zip(variants, colors)):
        vals = [oos[(oos["pair"]==p) & (oos["variant"]==v)][metric].values[0]
                if len(oos[(oos["pair"]==p) & (oos["variant"]==v)]) > 0 else 0
                for p in pairs_done]
        ax.bar(x + j*w, vals, w, label=v, color=c, alpha=0.85)
    if ref:
        ax.axhline(ref, color="red", linestyle="--", linewidth=1.0, label=f"基準={ref}")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(pairs_done, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
chart = os.path.join(OUT_DIR, "v78_improvements.png")
plt.savefig(chart, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート: {chart}")

# ── サマリー表（OOSのみ） ─────────────────────────────────
print("\n" + "=" * 90)
print("OOS サマリー（PF比較）")
print("=" * 90)
header = f"{'銘柄':<8} " + " ".join(f"{v:>9}" for v in variants) + "  ← 最良"
print(header); print("-" * 90)
for p in pairs_done:
    pfs = []
    for v in variants:
        r = oos[(oos["pair"]==p) & (oos["variant"]==v)]
        pfs.append(r["pf"].values[0] if len(r) > 0 else 0)
    best_v = variants[int(np.argmax(pfs))]
    line = f"{p:<8} " + " ".join(f"{pf:>9.2f}" for pf in pfs) + f"  ← {best_v}"
    print(line)

print("\n" + "=" * 90)
print("OOS PASS銘柄（PF≥2.0, Kelly≥0.0, MDD≤25%）")
print("=" * 90)
pass_rows = oos[oos["pass"] == True].copy()
if len(pass_rows) == 0:
    print("  PASSなし")
else:
    for _, r in pass_rows.sort_values("pf", ascending=False).iterrows():
        print(f"  {r['pair']:<8} {r['variant']:<6} PF={r['pf']:.2f} "
              f"WR={r['wr']:.1f}% MDD={r['mdd']:.1f}% Kelly={r['kelly']:.3f} "
              f"月次+{r['monthly_plus']}")

print("\n全処理完了。")
