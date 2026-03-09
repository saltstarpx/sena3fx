"""
14銘柄 4ヶ月バックテスト【v_max: リスク制御なし・ポジション上限20】
期間: 2025-11-01 〜 2026-02-28
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.risk_manager import AdaptiveRiskManager

INIT_CASH   = 1_000_000
BASE_RISK   = 0.02
MAX_POS     = 20
START       = "2025-11-01"
END         = "2026-02-28"
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYMBOLS = [
    "USDJPY", "AUDUSD", "EURJPY", "EURGBP",
    "EURUSD", "GBPUSD", "GBPJPY", "NZDUSD",
    "USDCAD", "USDCHF", "US30", "SPX500",
    "NAS100", "XAUUSD"
]

# ── ユーティリティ ────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path, parse_dates=[0])
    col = df.columns[0]
    if df[col].dt.tz is None:
        df[col] = df[col].dt.tz_localize("UTC")
    else:
        df[col] = df[col].dt.tz_convert("UTC")
    df = df.rename(columns={col: "time"}).set_index("time").sort_index()
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            for alt in ["Open","High","Low","Close"]:
                if alt in df.columns: df = df.rename(columns={alt: alt.lower()})
    return df

def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df.loc[s:e]

# ── シグナル生成（4H AND 1H KMID/KLOW フィルター） ────────
def check_kmid_klow(bar, direction, KMID_TH=0.3, KLOW_TH=0.001):
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    body = abs(c - o)
    rng  = h - l
    if rng < 1e-10: return False
    kmid = abs((c + o) / 2 - (h + l) / 2) / rng
    klow = body / rng
    if direction == 1:
        return kmid >= KMID_TH and klow >= KLOW_TH and c > o
    else:
        return kmid >= KMID_TH and klow >= KLOW_TH and c < o

def generate_signals(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread = spread_pips * pip_size
    RR_RATIO = 2.5
    ATR_MULT  = 0.15
    ATR_PERIOD = 14
    EMA_PERIOD = 20
    TOLERANCE  = 0.3

    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    data_4h["atr"]   = (data_4h["high"] - data_4h["low"]).rolling(ATR_PERIOD).mean()

    signals = []
    idx_4h = data_4h.index.searchsorted
    idx_1m = data_1m.index

    for i in range(2, len(data_15m)):
        bar15   = data_15m.iloc[i]
        bar15_t = data_15m.index[i]
        prev1   = data_15m.iloc[i-1]
        prev2   = data_15m.iloc[i-2]

        # 4H足参照
        pos4h = data_4h.index.searchsorted(bar15_t, side="right") - 1
        if pos4h < 1: continue
        h4_latest = data_4h.iloc[pos4h]
        ema20_val  = h4_latest["ema20"]
        atr_val    = h4_latest["atr"]
        if pd.isna(ema20_val) or pd.isna(atr_val) or atr_val <= 0: continue

        trend = 1 if h4_latest["close"] > ema20_val else -1

        # 二番底（ロング）
        if trend == 1:
            v1 = prev2["low"]; v2 = prev1["low"]
            if abs(v1 - v2) / atr_val <= TOLERANCE:
                raw_ep = bar15["open"]
                sl = min(v1, v2) - atr_val * ATR_MULT
                ep = raw_ep + spread
                risk = raw_ep - sl
                if risk <= 0 or risk > atr_val * 2: continue
                tp = raw_ep + risk * RR_RATIO
                # 反転確認（15m足）
                if prev1["close"] <= prev1["open"]: continue
                # 4H KMID/KLOW
                if not check_kmid_klow(h4_latest, 1): continue
                # 1H KMID/KLOW（15m足の前バーを1H代替として使用）
                if not check_kmid_klow(prev1, 1): continue
                signals.append({
                    "time": bar15_t, "dir": 1,
                    "ep": ep, "sl": sl, "tp": tp,
                    "raw_ep": raw_ep, "atr": atr_val
                })

        # 二番天井（ショート）
        if trend == -1:
            v1 = prev2["high"]; v2 = prev1["high"]
            if abs(v1 - v2) / atr_val <= TOLERANCE:
                raw_ep = bar15["open"]
                sl = max(v1, v2) + atr_val * ATR_MULT
                ep = raw_ep - spread
                risk = sl - raw_ep
                if risk <= 0 or risk > atr_val * 2: continue
                tp = raw_ep - risk * RR_RATIO
                if prev1["close"] >= prev1["open"]: continue
                if not check_kmid_klow(h4_latest, -1): continue
                if not check_kmid_klow(prev1, -1): continue
                signals.append({
                    "time": bar15_t, "dir": -1,
                    "ep": ep, "sl": sl, "tp": tp,
                    "raw_ep": raw_ep, "atr": atr_val
                })

    return pd.DataFrame(signals)

# ── シミュレーション ──────────────────────────────────────
def simulate_single(symbol, signals, data_1m, init_cash, base_risk):
    arm = AdaptiveRiskManager(symbol, base_risk_pct=base_risk)
    equity = init_cash
    peak   = init_cash
    trades = []

    for _, sig in signals.iterrows():
        entry_t = sig["time"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        direction = sig["dir"]
        atr_val = sig["atr"]

        dd_pct = (peak - equity) / peak if peak > 0 else 0
        result = arm.calc_lot_adaptive(equity, abs(ep - sl), ref_price=ep)
        lot, eff_risk = result[0], result[1]
        if lot <= 0: continue

        commission = arm.calc_commission_jpy(lot)
        equity -= commission

        half_tp = ep + (tp - ep) * 0.4 if direction == 1 else ep - (ep - tp) * 0.4
        be_ep   = ep

        future = data_1m.loc[entry_t:]
        if len(future) < 2: continue
        future = future.iloc[1:]

        result = "OPEN"; exit_price = None; exit_time = None
        half_done = False; remaining_lot = lot
        pnl_delta = 0.0
        pnl_timeline = []

        for bar_t, bar in future.iterrows():
            price = bar["close"]
            cur_pnl = arm.calc_pnl_jpy(lot, ep, price, direction)
            pnl_timeline.append(cur_pnl)

            if direction == 1:
                if not half_done and price >= half_tp:
                    half_lot = lot * 0.5
                    pnl_delta += arm.calc_pnl_jpy(half_lot, ep, half_tp, direction)
                    equity += arm.calc_pnl_jpy(half_lot, ep, half_tp, direction)
                    equity -= arm.calc_commission_jpy(half_lot)
                    remaining_lot = lot * 0.5
                    sl = be_ep
                    half_done = True
                if price <= sl:
                    pnl_delta += arm.calc_pnl_jpy(remaining_lot, ep, sl, direction)
                    equity += arm.calc_pnl_jpy(remaining_lot, ep, sl, direction)
                    equity -= arm.calc_commission_jpy(remaining_lot)
                    result = "BE" if half_done else "SL"
                    exit_price = sl; exit_time = bar_t; break
                if price >= tp:
                    pnl_delta += arm.calc_pnl_jpy(remaining_lot, ep, tp, direction)
                    equity += arm.calc_pnl_jpy(remaining_lot, ep, tp, direction)
                    equity -= arm.calc_commission_jpy(remaining_lot)
                    result = "TP"; exit_price = tp; exit_time = bar_t; break
            else:
                if not half_done and price <= half_tp:
                    half_lot = lot * 0.5
                    pnl_delta += arm.calc_pnl_jpy(half_lot, ep, half_tp, direction)
                    equity += arm.calc_pnl_jpy(half_lot, ep, half_tp, direction)
                    equity -= arm.calc_commission_jpy(half_lot)
                    remaining_lot = lot * 0.5
                    sl = be_ep
                    half_done = True
                if price >= sl:
                    pnl_delta += arm.calc_pnl_jpy(remaining_lot, ep, sl, direction)
                    equity += arm.calc_pnl_jpy(remaining_lot, ep, sl, direction)
                    equity -= arm.calc_commission_jpy(remaining_lot)
                    result = "BE" if half_done else "SL"
                    exit_price = sl; exit_time = bar_t; break
                if price <= tp:
                    pnl_delta += arm.calc_pnl_jpy(remaining_lot, ep, tp, direction)
                    equity += arm.calc_pnl_jpy(remaining_lot, ep, tp, direction)
                    equity -= arm.calc_commission_jpy(remaining_lot)
                    result = "TP"; exit_price = tp; exit_time = bar_t; break

        if result == "OPEN": continue
        if equity > peak: peak = equity

        trades.append({
            "symbol": symbol, "entry_time": entry_t, "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result,
            "lot": lot, "eff_risk": eff_risk,
            "pnl_delta": pnl_delta, "equity_after": equity,
            "pnl_timeline": pnl_timeline,
            "risk_jpy": eff_risk * equity,
        })

    return trades

# ── 含み益枠外フィルター（vMAX: 強制カットなし） ──────────
def apply_vmax_filter(all_trades_df, init_cash):
    if all_trades_df.empty:
        return all_trades_df, pd.Series([init_cash])

    all_trades_df = all_trades_df.copy()
    all_trades_df["entry_time"] = pd.to_datetime(all_trades_df["entry_time"], utc=True)
    all_trades_df["exit_time"]  = pd.to_datetime(all_trades_df["exit_time"],  utc=True)
    all_trades_df = all_trades_df.sort_values("entry_time").reset_index(drop=True)

    equity = float(init_cash)
    peak   = float(init_cash)
    open_pos = {}
    accepted_trades = []
    processed_entries = set()
    equity_curve = [equity]
    extra_count = 0

    all_times = sorted(set(
        list(all_trades_df["entry_time"]) + list(all_trades_df["exit_time"])
    ))

    for t in all_times:
        t = pd.Timestamp(t)
        if t.tzinfo is None: t = t.tz_localize("UTC")

        # クローズ処理
        to_close = [sym for sym, pos in open_pos.items() if pos["exit_time"] <= t]
        for sym in to_close:
            pos = open_pos.pop(sym)
            equity += pos["pnl_delta"]
            if equity > peak: peak = equity
            equity_curve.append(equity)

        # 現在の含み損益を更新（含み益ポジションは枠外）
        profitable_syms = set()
        for sym, pos in open_pos.items():
            # pnl_timelineから現在の含み損益を推定
            elapsed = (t - pos["entry_time"]).total_seconds() / 60
            tl = pos["pnl_timeline"]
            idx = min(int(elapsed), len(tl)-1) if tl else 0
            cur_pnl = tl[idx] if tl else 0
            pos["current_pnl"] = cur_pnl
            if cur_pnl > 0:
                profitable_syms.add(sym)

        # 実効ポジション数（含み益を除外）
        effective_count = len(open_pos) - len(profitable_syms)

        # エントリー処理
        entries_now = all_trades_df[all_trades_df["entry_time"] == t]
        for idx, row in entries_now.iterrows():
            if idx in processed_entries: continue
            sym = row["symbol"]
            if sym in open_pos: continue  # 同銘柄は1ポジのみ

            is_extra = sym in profitable_syms
            if not is_extra and effective_count >= MAX_POS:
                continue

            exit_time = pd.Timestamp(row["exit_time"])
            if exit_time.tzinfo is None: exit_time = exit_time.tz_localize("UTC")

            open_pos[sym] = {
                "exit_time":    exit_time,
                "entry_time":   t,
                "pnl_delta":    row["pnl_delta"],
                "pnl_timeline": row["pnl_timeline"] if isinstance(row["pnl_timeline"], list) else [],
                "current_pnl":  0.0,
            }
            if is_extra:
                extra_count += 1
            effective_count += 1
            accepted_trades.append({**row.to_dict(), "is_extra": is_extra})
            processed_entries.add(idx)

    for sym, pos in open_pos.items():
        equity += pos["pnl_delta"]
        equity_curve.append(equity)

    print(f"  追加エントリー（含み益枠外）: {extra_count}件")
    return pd.DataFrame(accepted_trades), pd.Series(equity_curve)

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"14銘柄 4ヶ月バックテスト【v_max: リスク制御なし・ポジション上限{MAX_POS}】")
print(f"期間: {START} 〜 {END}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print("=" * 70)

all_trades = []
for sym in SYMBOLS:
    sym_lower = sym.lower()
    d1m  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_1m.csv"))
    d15m = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_15m.csv"))
    d4h  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv"))
    if d1m  is None: d1m  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_1m.csv"))
    if d15m is None: d15m = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_15m.csv"))
    if d4h  is None: d4h  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_4h.csv"))
    if d1m is None or d4h is None:
        print(f"  {sym}: データなし → スキップ"); continue

    d1m  = slice_period(d1m,  START, END)
    d4h  = slice_period(d4h,  START, END)
    if d15m is not None: d15m = slice_period(d15m, START, END)

    # 15m足がない場合は1m足から15m足を生成
    if d15m is None or len(d15m) == 0:
        d15m = d1m.resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna()

    if d1m is None or len(d1m) == 0:
        print(f"  {sym}: 期間データなし → スキップ"); continue

    arm_ref = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    print(f"  {sym}: シグナル生成中... (spread={arm_ref.spread_pips}pips)")
    sigs = generate_signals(d1m, d15m, d4h, arm_ref.spread_pips, arm_ref.pip_size)
    print(f"    → シグナル{len(sigs)}件 / シミュレーション実行中...")
    trades = simulate_single(sym, sigs, d1m, INIT_CASH, BASE_RISK)

    # pnl_delta を equity_after の差分から再計算
    prev_eq = INIT_CASH
    for t in trades:
        t["pnl_delta"] = t["equity_after"] - prev_eq
        prev_eq = t["equity_after"]

    n_tp = sum(1 for t in trades if t["result"] == "TP")
    n_sl = sum(1 for t in trades if t["result"] == "SL")
    n_be = sum(1 for t in trades if t["result"] == "BE")
    wr   = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0
    print(f"    → 完了: {len(trades)}件 TP:{n_tp} SL:{n_sl} BE:{n_be} 勝率:{wr*100:.0f}%")
    all_trades.extend(trades)

if not all_trades:
    print("トレードなし。終了。"); sys.exit(0)

all_df = pd.DataFrame(all_trades)

print("\nvMAXフィルター適用中...")
accepted_df, equity_curve = apply_vmax_filter(all_df, INIT_CASH)

if accepted_df.empty:
    print("採用トレードなし。終了。"); sys.exit(0)

# ── 結果集計 ──────────────────────────────────────────────
final_equity = equity_curve.iloc[-1]
peak_eq = equity_curve.cummax().max()
dd_series = (equity_curve - equity_curve.cummax()) / equity_curve.cummax() * 100
max_dd = dd_series.min()

n_tp = (accepted_df["result"] == "TP").sum()
n_sl = (accepted_df["result"] == "SL").sum()
n_be = (accepted_df["result"] == "BE").sum()
wr   = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0
tp_pnl = accepted_df[accepted_df["result"]=="TP"]["pnl_delta"].sum()
sl_pnl = abs(accepted_df[accepted_df["result"]=="SL"]["pnl_delta"].sum())
pf = tp_pnl / sl_pnl if sl_pnl > 0 else float("inf")
ret_pct = (final_equity - INIT_CASH) / INIT_CASH * 100

print(f"\n{'='*70}")
print(f"【14銘柄 vMAX 4ヶ月結果】")
print(f"  採用トレード: {len(accepted_df)}件  TP:{n_tp} SL:{n_sl} BE:{n_be}")
print(f"  勝率(BE除く): {wr*100:.1f}%  PF: {pf:.2f}")
print(f"  最大DD: {max_dd:.1f}%")
print(f"  最終資産: {final_equity:,.0f}円  リターン: {ret_pct:+.1f}%")
print(f"{'='*70}")

# 銘柄別集計
print("\n【銘柄別パフォーマンス】")
print(f"{'銘柄':8s}  {'件数':>4s}  {'TP':>3s}  {'SL':>3s}  {'BE':>3s}  {'勝率':>5s}  {'PF':>5s}  {'PnL(万)':>8s}  {'追加':>4s}")
sym_stats = accepted_df.groupby("symbol").apply(lambda g: pd.Series({
    "count": len(g),
    "tp": (g["result"]=="TP").sum(),
    "sl": (g["result"]=="SL").sum(),
    "be": (g["result"]=="BE").sum(),
    "pnl": g["pnl_delta"].sum(),
    "extra": g["is_extra"].sum() if "is_extra" in g.columns else 0,
})).reset_index()
sym_stats["wr"] = sym_stats["tp"] / (sym_stats["tp"] + sym_stats["sl"]) * 100
sym_stats["pf"] = sym_stats.apply(lambda r: r["tp"] / r["sl"] if r["sl"] > 0 else 99, axis=1)
sym_stats = sym_stats.sort_values("pnl", ascending=False)
for _, r in sym_stats.iterrows():
    print(f"{r['symbol']:8s}  {int(r['count']):4d}  {int(r['tp']):3d}  {int(r['sl']):3d}  {int(r['be']):3d}  {r['wr']:4.0f}%  {r['pf']:5.2f}  {r['pnl']/1e4:+7.1f}万  {int(r['extra']):4d}件")

# 月次集計
print("\n【月次損益】")
accepted_df["exit_time"] = pd.to_datetime(accepted_df["exit_time"], utc=True)
accepted_df["month"] = accepted_df["exit_time"].dt.to_period("M")
monthly = accepted_df.groupby("month")["pnl_delta"].sum()
for m, pnl in monthly.items():
    print(f"  {m}: {pnl/1e4:+.1f}万円")

# CSV保存
accepted_df.to_csv(os.path.join(RESULTS_DIR, "14sym_vmax_4m_trades.csv"), index=False)

# ── チャート ──────────────────────────────────────────────
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle(f"14銘柄 vMAX 4ヶ月バックテスト ({START}〜{END})\n最終資産: {final_equity/1e4:.0f}万円 (+{ret_pct:.0f}%) | 勝率: {wr*100:.1f}% | PF: {pf:.2f} | 最大DD: {max_dd:.1f}%", fontsize=13)

# 資産曲線
ax1 = axes[0]
ax1.plot(equity_curve.values / 1e4, color="royalblue", linewidth=1.5)
ax1.axhline(INIT_CASH / 1e4, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylabel("資産（万円）"); ax1.set_title("資産曲線"); ax1.grid(alpha=0.3)

# DD曲線
ax2 = axes[1]
ax2.fill_between(range(len(dd_series)), dd_series.values, 0, color="crimson", alpha=0.4)
ax2.set_ylabel("DD (%)"); ax2.set_title("ドローダウン"); ax2.grid(alpha=0.3)

# 銘柄別PnL棒グラフ
ax3 = axes[2]
colors = ["green" if v >= 0 else "red" for v in sym_stats["pnl"].values]
bars = ax3.bar(sym_stats["symbol"], sym_stats["pnl"] / 1e4, color=colors, alpha=0.7)
for bar, (_, r) in zip(bars, sym_stats.iterrows()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{r['wr']:.0f}%", ha="center", va="bottom", fontsize=8)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("損益（万円）"); ax3.set_title("銘柄別損益（棒グラフ上部は勝率）"); ax3.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "14sym_vmax_4m_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nチャート保存: {out_path}")
