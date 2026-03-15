#!/usr/bin/env python3
"""
YAGAMI改 5試練 — ストレステスト分析
===================================
Trial 1: ホームラン依存度
Trial 2: USD集中リスク
Trial 3: USDCADの真実
Trial 4: 現実的MDD推定
Trial 5: フィルターアブレーション (別途バックテスト再実行)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
CSV_PATH = RESULTS / "backtest_portfolio_2026.csv"
OUT_PATH = RESULTS / "trial_5_challenges_report.txt"

# ── helpers ──────────────────────────────────────────────────────────
def pf(wins, losses):
    """Profit Factor from gross win / gross loss"""
    gw = wins[wins > 0].sum()
    gl = abs(losses[losses < 0].sum())
    return gw / gl if gl > 0 else float("inf")

def calc_mdd_pct(equity_series):
    """Max drawdown as % of peak"""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return dd.min() * 100  # negative %

def calc_mdd_details(equity_series):
    """Return MDD%, start idx, trough idx, recovery idx"""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    trough_idx = dd.idxmin()
    mdd_pct = dd.loc[trough_idx] * 100
    # find peak before trough
    peak_val = peak.loc[trough_idx]
    start_candidates = equity_series.loc[:trough_idx]
    start_idx = start_candidates[start_candidates == peak_val].index[-1]
    # find recovery after trough
    post = equity_series.loc[trough_idx:]
    recovery = post[post >= peak_val]
    recovery_idx = recovery.index[0] if len(recovery) > 0 else None
    return mdd_pct, start_idx, trough_idx, recovery_idx

def print_sep(title, f):
    line = "=" * 70
    f.write(f"\n{line}\n  {title}\n{line}\n\n")

# ── load data ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
df["date"] = pd.to_datetime(df["date"])
df["entry_hour"] = df["entry_time"].dt.hour
# 4H candle bucket: 0,4,8,12,16,20
df["h4_bucket"] = (df["entry_time"].dt.hour // 4) * 4
df["h4_key"] = df["entry_time"].dt.strftime("%Y-%m-%d") + "_" + df["h4_bucket"].astype(str).str.zfill(2)

TOTAL_TRADES = len(df)
TOTAL_PNL = df["pnl"].sum()
WINS = df[df["result"] == "win"]
LOSSES = df[df["result"] == "loss"]
OVERALL_PF = pf(df["pnl"], df["pnl"])
OVERALL_WR = len(WINS) / TOTAL_TRADES

report_lines = []
out = open(OUT_PATH, "w", encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════
#  試練① ホームラン依存度の実態を暴け
# ═══════════════════════════════════════════════════════════════════════
print_sep("試練① ホームラン依存度の実態を暴け", out)

# Sort trades by pnl descending
df_sorted = df.sort_values("pnl", ascending=False).reset_index(drop=True)
n_top10 = int(np.ceil(TOTAL_TRADES * 0.10))
n_top20 = int(np.ceil(TOTAL_TRADES * 0.20))

top10_trades = df_sorted.head(n_top10)
bottom_90 = df_sorted.iloc[n_top10:]

# Also top 5% and top 20%
n_top5 = int(np.ceil(TOTAL_TRADES * 0.05))
top5_trades = df_sorted.head(n_top5)
bottom_95 = df_sorted.iloc[n_top5:]
top20_trades = df_sorted.head(n_top20)
bottom_80 = df_sorted.iloc[n_top20:]

out.write(f"全トレード数: {TOTAL_TRADES}\n")
out.write(f"上位10%: {n_top10}トレード, 上位5%: {n_top5}トレード, 上位20%: {n_top20}トレード\n\n")

# Scenario comparison
scenarios = {
    "A: 全トレード": df,
    "B: 上位10%除外": bottom_90,
    "C: 上位5%除外": bottom_95,
    "D: 上位20%除外": bottom_80,
}

out.write(f"{'シナリオ':<22} {'トレード数':>8} {'総損益':>16} {'PF':>8} {'勝率':>8}\n")
out.write("-" * 66 + "\n")
for name, data in scenarios.items():
    t_pnl = data["pnl"].sum()
    t_pf = pf(data["pnl"], data["pnl"])
    t_wr = len(data[data["result"] == "win"]) / len(data) if len(data) > 0 else 0
    out.write(f"{name:<22} {len(data):>8} {t_pnl:>16,.0f} {t_pf:>8.2f} {t_wr:>7.1%}\n")

out.write(f"\n上位10%の利益占有率: {top10_trades['pnl'].sum() / TOTAL_PNL:.1%}\n")
out.write(f"上位20%の利益占有率: {top20_trades['pnl'].sum() / TOTAL_PNL:.1%}\n")
out.write(f"上位5%の利益占有率:  {top5_trades['pnl'].sum() / TOTAL_PNL:.1%}\n")

# Monthly breakdown: A vs B
out.write(f"\n--- 月次損益比較（全トレード vs 上位10%除外）---\n")
monthly_a = df.groupby("month")["pnl"].sum()
monthly_b = bottom_90.groupby("month")["pnl"].sum()
out.write(f"{'月':>10} {'A: 全':>14} {'B: 10%除外':>14} {'B マイナス月?':>14}\n")
out.write("-" * 55 + "\n")
minus_months_b = 0
for m in sorted(monthly_a.index):
    a_val = monthly_a.get(m, 0)
    b_val = monthly_b.get(m, 0)
    is_minus = "★マイナス" if b_val < 0 else ""
    if b_val < 0:
        minus_months_b += 1
    out.write(f"{m:>10} {a_val:>14,.0f} {b_val:>14,.0f} {is_minus:>14}\n")

out.write(f"\nシナリオBでマイナス月: {minus_months_b} / {len(monthly_a)} ヶ月\n")

# Months with zero home runs
monthly_top10_pnl = top10_trades.groupby("month")["pnl"].sum()
months_no_hr = [m for m in monthly_a.index if m not in monthly_top10_pnl.index]
if months_no_hr:
    avg_no_hr = monthly_a.loc[months_no_hr].mean()
    out.write(f"ホームランゼロ月の平均損益: {avg_no_hr:,.0f}\n")
else:
    out.write("全月にホームランが存在（ゼロ月なし）\n")

# Home run threshold
hr_threshold = top10_trades["pnl"].min()
out.write(f"\nホームラン基準（上位10%の最小値）: {hr_threshold:,.0f}\n")
out.write(f"ホームラン平均利益: {top10_trades['pnl'].mean():,.0f}\n")
out.write(f"通常勝ちトレード平均利益: {bottom_90[bottom_90['result']=='win']['pnl'].mean():,.0f}\n")

# Win distribution deciles
out.write(f"\n--- 利益デシル分析（勝ちトレードのみ）---\n")
wins_sorted = df[df["pnl"] > 0].sort_values("pnl", ascending=False)
total_win_pnl = wins_sorted["pnl"].sum()
n_wins = len(wins_sorted)
for i in range(10):
    start = int(n_wins * i / 10)
    end = int(n_wins * (i + 1) / 10)
    slice_pnl = wins_sorted.iloc[start:end]["pnl"].sum()
    pct = slice_pnl / total_win_pnl * 100
    out.write(f"  {i*10+1:>3}〜{(i+1)*10:>3}%: {slice_pnl:>14,.0f} ({pct:>5.1f}%)\n")

# VERDICT
pf_no_hr = pf(bottom_90["pnl"], bottom_90["pnl"])
verdict_1 = "✅ PASS" if pf_no_hr >= 1.5 else "❌ FAIL"
out.write(f"\n【判定】ホームラン除外PF = {pf_no_hr:.2f} (基準: ≥1.5) → {verdict_1}\n")
out.write(f"  ホームランなしで自立{'できる' if pf_no_hr >= 1.5 else 'できない'}\n")

# ═══════════════════════════════════════════════════════════════════════
#  試練② USD集中リスクの定量化
# ═══════════════════════════════════════════════════════════════════════
print_sep("試練② USD集中リスクの定量化", out)

# 1. Simultaneous entries by 4H candle bucket
out.write("--- 同一4Hバケットでの同時エントリー ---\n")
h4_counts = df.groupby("h4_key")["sym"].nunique()
for n_sym in range(2, 9):
    cnt = (h4_counts >= n_sym).sum()
    pct = cnt / len(h4_counts) * 100
    out.write(f"  {n_sym}銘柄以上同時: {cnt:>4}回 ({pct:>5.1f}%)\n")

# Same-day simultaneous entries
daily_sym_counts = df.groupby("date")["sym"].nunique()
out.write(f"\n--- 同日エントリー銘柄数 ---\n")
for n_sym in range(4, 9):
    cnt = (daily_sym_counts >= n_sym).sum()
    out.write(f"  {n_sym}銘柄以上: {cnt:>4}日\n")

# 2. Worst consecutive losses
out.write(f"\n--- 最大連敗分析 ---\n")
df_time = df.sort_values("exit_time").reset_index(drop=True)
max_streak = 0
cur_streak = 0
streak_pnl = 0
max_streak_pnl = 0
streak_start = None
best_streaks = []

for i, row in df_time.iterrows():
    if row["result"] == "loss":
        if cur_streak == 0:
            streak_start = row["entry_time"]
        cur_streak += 1
        streak_pnl += row["pnl"]
        if cur_streak > max_streak:
            max_streak = cur_streak
    else:
        if cur_streak >= 4:
            best_streaks.append({
                "連敗数": cur_streak,
                "損失額": streak_pnl,
                "開始": streak_start,
                "終了": df_time.iloc[i-1]["exit_time"],
                "銘柄": ", ".join(df_time.iloc[max(0,i-cur_streak):i]["sym"].unique()),
            })
        cur_streak = 0
        streak_pnl = 0

# Sort by streak length
best_streaks.sort(key=lambda x: x["連敗数"], reverse=True)
out.write(f"最大連敗数: {max_streak}\n\n")
out.write(f"{'順位':>4} {'連敗数':>6} {'損失額':>14} {'銘柄':>30} {'期間'}\n")
out.write("-" * 90 + "\n")
for i, s in enumerate(best_streaks[:10]):
    out.write(f"{i+1:>4} {s['連敗数']:>6} {s['損失額']:>14,.0f} {s['銘柄']:>30} {str(s['開始'])[:16]}〜{str(s['終了'])[:16]}\n")

# 3. Correlation: same-day losses across symbols
out.write(f"\n--- 同日損失クラスタリング ---\n")
daily_losses = df[df["result"] == "loss"].groupby("date").agg(
    loss_count=("pnl", "count"),
    loss_total=("pnl", "sum"),
    symbols=("sym", lambda x: ", ".join(sorted(x.unique()))),
    n_symbols=("sym", "nunique"),
)
worst_days = daily_losses.nlargest(10, "loss_count")
out.write(f"{'日付':>12} {'敗数':>4} {'銘柄数':>6} {'損失額':>14} {'銘柄'}\n")
out.write("-" * 80 + "\n")
for dt, row in worst_days.iterrows():
    out.write(f"{str(dt)[:10]:>12} {row['loss_count']:>4} {row['n_symbols']:>6} {row['loss_total']:>14,.0f} {row['symbols']}\n")

# 4. Entry time correlation (same entry_time hour)
out.write(f"\n--- 同一時間帯（1時間）での同時エントリー ---\n")
df["entry_hour_key"] = df["entry_time"].dt.strftime("%Y-%m-%d %H")
hourly_counts = df.groupby("entry_hour_key")["sym"].nunique()
for n in [3, 4, 5, 6]:
    cnt = (hourly_counts >= n).sum()
    out.write(f"  {n}銘柄以上同時: {cnt:>4}回\n")

# 5. Symbol pair correlation (win/loss same direction)
out.write(f"\n--- 銘柄ペア勝敗相関 ---\n")
# Pivot: date x symbol -> result (1=win, 0=loss)
pivot = df.pivot_table(index="date", columns="sym", values="pnl", aggfunc="sum")
pivot_binary = (pivot > 0).astype(int)
# Only compute for symbols with enough data
valid_syms = [c for c in pivot_binary.columns if pivot_binary[c].notna().sum() >= 20]
corr = pivot_binary[valid_syms].corr()
out.write("（日次勝敗の相関行列）\n")
out.write(corr.round(2).to_string() + "\n")

# High correlation pairs
out.write(f"\n高相関ペア（相関 > 0.3）:\n")
for i in range(len(valid_syms)):
    for j in range(i+1, len(valid_syms)):
        c = corr.iloc[i, j]
        if abs(c) > 0.3:
            out.write(f"  {valid_syms[i]:>8} - {valid_syms[j]:<8}: {c:.3f}\n")

# VERDICT
pct_4plus = (h4_counts >= 4).sum() / len(h4_counts) * 100
verdict_2 = "✅ PASS" if pct_4plus < 10 else "❌ FAIL"
out.write(f"\n【判定】4銘柄以上同時エントリー = {pct_4plus:.1f}% (基準: <10%) → {verdict_2}\n")

# ═══════════════════════════════════════════════════════════════════════
#  試練③ USDCADの3.32は本物か
# ═══════════════════════════════════════════════════════════════════════
print_sep("試練③ USDCADの3.32は本物か", out)

cad = df[df["sym"] == "USDCAD"].copy()
non_cad = df[df["sym"] != "USDCAD"].copy()

cad_pf = pf(cad["pnl"], cad["pnl"])
cad_wr = len(cad[cad["result"] == "win"]) / len(cad)
cad_pnl = cad["pnl"].sum()

out.write(f"USDCAD全体: {len(cad)}トレード, PF={cad_pf:.2f}, WR={cad_wr:.1%}, 総損益={cad_pnl:,.0f}\n\n")

# 1. IS/OOS split (use the split date from backtest_final_optimized: 2025-06-24)
# Portfolio 2026 data starts from 2026-01, so all data is OOS
# We'll split the OOS data itself into halves for additional validation
cad_sorted = cad.sort_values("entry_time")
mid = len(cad_sorted) // 2
cad_first = cad_sorted.iloc[:mid]
cad_second = cad_sorted.iloc[mid:]

out.write(f"--- IS/OOS検証（OOSデータ前半 vs 後半）---\n")
for label, data in [("前半", cad_first), ("後半", cad_second)]:
    p = pf(data["pnl"], data["pnl"])
    w = len(data[data["result"] == "win"]) / len(data)
    out.write(f"  {label}: {len(data)}トレード, PF={p:.2f}, WR={w:.1%}, 損益={data['pnl'].sum():,.0f}\n")
pf_ratio = pf(cad_second["pnl"], cad_second["pnl"]) / max(pf(cad_first["pnl"], cad_first["pnl"]), 0.01)
out.write(f"  後半/前半 PF比: {pf_ratio:.2f}\n")

# 2. Trump tariff period exclusion (Nov 2024 - Mar 2025)
# Our data is Jan-Mar 2026, so we check if any CAD-specific volatility period exists
# Instead: check monthly PF stability
out.write(f"\n--- USDCAD月次パフォーマンス ---\n")
cad_monthly = cad.groupby("month").agg(
    trades=("pnl", "count"),
    pnl=("pnl", "sum"),
    wr=("result", lambda x: (x == "win").mean()),
)
cad_monthly["pf"] = cad.groupby("month").apply(lambda g: pf(g["pnl"], g["pnl"]))
out.write(f"{'月':>10} {'トレード数':>8} {'PF':>8} {'WR':>8} {'損益':>14}\n")
out.write("-" * 52 + "\n")
for m, row in cad_monthly.iterrows():
    out.write(f"{m:>10} {row['trades']:>8.0f} {row['pf']:>8.2f} {row['wr']:>7.1%} {row['pnl']:>14,.0f}\n")

# Identify best month and recalculate without it
best_month = cad_monthly["pnl"].idxmax()
cad_no_best = cad[cad["month"] != best_month]
pf_no_best = pf(cad_no_best["pnl"], cad_no_best["pnl"])
out.write(f"\n最高月（{best_month}）除外後: PF={pf_no_best:.2f}, 損益={cad_no_best['pnl'].sum():,.0f}\n")

# 3. Worst month exclusion
worst_month = cad_monthly["pnl"].idxmin()
cad_no_worst = cad[cad["month"] != worst_month]
pf_no_worst = pf(cad_no_worst["pnl"], cad_no_worst["pnl"])
out.write(f"最低月（{worst_month}）除外後: PF={pf_no_worst:.2f}\n")

# 4. Portfolio without USDCAD
pf_with = pf(df["pnl"], df["pnl"])
pf_without = pf(non_cad["pnl"], non_cad["pnl"])
out.write(f"\n--- ポートフォリオ比較 ---\n")
out.write(f"  USDCAD込み:   PF={pf_with:.2f}, 総損益={TOTAL_PNL:,.0f}\n")
out.write(f"  USDCAD除外:   PF={pf_without:.2f}, 総損益={non_cad['pnl'].sum():,.0f}\n")
out.write(f"  USDCAD利益占有率: {cad_pnl / TOTAL_PNL:.1%}\n")

# 5. Compare USDCAD win rate by time-of-day
out.write(f"\n--- USDCAD 時間帯別勝率 ---\n")
cad_hourly = cad.groupby("entry_hour").agg(
    trades=("pnl", "count"),
    wr=("result", lambda x: (x == "win").mean()),
    pnl=("pnl", "sum"),
)
for h, row in cad_hourly.iterrows():
    if row["trades"] >= 5:
        out.write(f"  UTC {h:02d}: {row['trades']:>3}回, WR={row['wr']:.1%}, 損益={row['pnl']:>12,.0f}\n")

# VERDICT
verdict_3a = "✅ PASS" if cad_pf >= 2.0 else "❌ FAIL"
verdict_3b = "✅ PASS" if pf_no_best >= 2.0 else "❌ FAIL"
out.write(f"\n【判定】\n")
out.write(f"  OOS PF = {cad_pf:.2f} (基準: ≥2.0) → {verdict_3a}\n")
out.write(f"  最高月除外PF = {pf_no_best:.2f} (基準: ≥2.0) → {verdict_3b}\n")

# ═══════════════════════════════════════════════════════════════════════
#  試練④ 本当に耐えられるMDDを計算せよ
# ═══════════════════════════════════════════════════════════════════════
print_sep("試練④ 本当に耐えられるMDDを計算せよ", out)

# 1. Backtest MDD from equity curve
eq = df.sort_values("exit_time")["equity"].reset_index(drop=True)
mdd_pct, start_i, trough_i, recovery_i = calc_mdd_details(eq)

out.write(f"--- バックテストMDD ---\n")
out.write(f"  最大ドローダウン: {abs(mdd_pct):.2f}%\n")
out.write(f"  MDD開始: インデックス {start_i}\n")
out.write(f"  MDD底:   インデックス {trough_i}\n")
if recovery_i is not None:
    out.write(f"  回復:     インデックス {recovery_i}\n")
    out.write(f"  MDD期間: {trough_i - start_i} トレード\n")
    out.write(f"  回復期間: {recovery_i - trough_i} トレード\n")
else:
    out.write(f"  回復: 未回復\n")

# Per-symbol MDD
out.write(f"\n--- 銘柄別MDD ---\n")
for sym in sorted(df["sym"].unique()):
    sym_eq = df[df["sym"] == sym].sort_values("exit_time")["equity"]
    sym_mdd = calc_mdd_pct(sym_eq)
    out.write(f"  {sym:>8}: {abs(sym_mdd):>6.2f}%\n")

# 2. Corrected MDD
backtest_mdd = abs(mdd_pct)
slippage_factor = 0.85  # PF degradation from slippage
psych_factor = 1.5      # psychological amplification
corrected_mdd = backtest_mdd * psych_factor / slippage_factor

out.write(f"\n--- 実運用補正後MDD推定 ---\n")
out.write(f"  バックテストMDD: {backtest_mdd:.2f}%\n")
out.write(f"  スリッページ補正 (÷{slippage_factor}): {backtest_mdd/slippage_factor:.2f}%\n")
out.write(f"  心理的補正 (×{psych_factor}): {corrected_mdd:.2f}%\n")

# 3. Money-based MDD by capital and risk
out.write(f"\n--- 証拠金別 MDD金額 ---\n")
capitals = [2_000_000, 5_000_000, 10_000_000]  # JPY
risks = [0.01, 0.02]
out.write(f"{'証拠金':>12} {'リスク':>6} {'BT MDD':>12} {'補正MDD':>12}\n")
out.write("-" * 46 + "\n")
for cap in capitals:
    for risk in risks:
        bt_loss = cap * (backtest_mdd / 100)
        corr_loss = cap * (corrected_mdd / 100)
        out.write(f"  ¥{cap:>10,} {risk:.0%}  ¥{bt_loss:>10,.0f}  ¥{corr_loss:>10,.0f}\n")

# 4. MDD recovery simulation
# How many trades needed to recover from MDD bottom
avg_win = df[df["pnl"] > 0]["pnl"].mean()
avg_loss = df[df["pnl"] < 0]["pnl"].mean()
wr = OVERALL_WR
expected_per_trade = wr * avg_win + (1 - wr) * avg_loss
trades_per_month = TOTAL_TRADES / df["month"].nunique()

# For 10M capital, how long to recover from MDD
cap_10m = 10_000_000
mdd_loss = cap_10m * (backtest_mdd / 100)
corr_mdd_loss = cap_10m * (corrected_mdd / 100)
# Expected gain per trade at risk 2% of initial
avg_trade_gain = expected_per_trade  # from backtest
trades_to_recover_bt = abs(mdd_loss / avg_trade_gain) if avg_trade_gain > 0 else float("inf")
trades_to_recover_corr = abs(corr_mdd_loss / avg_trade_gain) if avg_trade_gain > 0 else float("inf")
months_to_recover_bt = trades_to_recover_bt / trades_per_month
months_to_recover_corr = trades_to_recover_corr / trades_per_month

out.write(f"\n--- MDD回復シミュレーション（証拠金1000万円基準）---\n")
out.write(f"  期待値/トレード: {expected_per_trade:,.0f}\n")
out.write(f"  月間トレード数: {trades_per_month:.0f}\n")
out.write(f"  バックテストMDD回復: {trades_to_recover_bt:.0f}トレード = {months_to_recover_bt:.1f}ヶ月\n")
out.write(f"  補正後MDD回復:       {trades_to_recover_corr:.0f}トレード = {months_to_recover_corr:.1f}ヶ月\n")

# Daily equity curve stats
out.write(f"\n--- 日次損益分布 ---\n")
daily_pnl = df.groupby("date")["pnl"].sum()
out.write(f"  日次平均損益:   {daily_pnl.mean():>12,.0f}\n")
out.write(f"  日次標準偏差:   {daily_pnl.std():>12,.0f}\n")
out.write(f"  最大日次利益:   {daily_pnl.max():>12,.0f}\n")
out.write(f"  最大日次損失:   {daily_pnl.min():>12,.0f}\n")
out.write(f"  マイナス日数:   {(daily_pnl < 0).sum()} / {len(daily_pnl)} ({(daily_pnl < 0).mean():.1%})\n")

# VERDICT
verdict_4 = "✅ PASS" if corrected_mdd < 40 and months_to_recover_corr < 3 else "❌ FAIL"
out.write(f"\n【判定】補正後MDD = {corrected_mdd:.1f}% (基準: <40%), 回復 = {months_to_recover_corr:.1f}ヶ月 (基準: <3ヶ月) → {verdict_4}\n")


# ═══════════════════════════════════════════════════════════════════════
#  試練⑤ フィルター貢献度（既存データからの推定）
# ═══════════════════════════════════════════════════════════════════════
print_sep("試練⑤ フィルター貢献度（既存データからの推定）", out)

# Read the detailed backtest comparison data
detail_csv = RESULTS / "backtest_all_symbols_detail.csv"
if detail_csv.exists():
    detail = pd.read_csv(detail_csv)
    adopted_syms = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"]
    detail_adopted = detail[detail["sym"].isin(adopted_syms)]

    out.write("--- ロジック別OOS PF比較（採用8銘柄）---\n")
    out.write(f"{'銘柄':>8} {'Logic-A PF':>12} {'Logic-B PF':>12} {'Logic-C PF':>12} {'最良':>8}\n")
    out.write("-" * 56 + "\n")
    for sym in adopted_syms:
        sym_data = detail_adopted[detail_adopted["sym"] == sym]
        pf_a = sym_data[sym_data["logic"] == "A"]["oos_pf"].values
        pf_b = sym_data[sym_data["logic"] == "B"]["oos_pf"].values
        pf_c = sym_data[sym_data["logic"] == "C"]["oos_pf"].values
        pa = pf_a[0] if len(pf_a) > 0 else 0
        pb = pf_b[0] if len(pf_b) > 0 else 0
        pc = pf_c[0] if len(pf_c) > 0 else 0
        best = max(pa, pb, pc)
        best_label = "A" if best == pa else ("B" if best == pb else "C")
        out.write(f"{sym:>8} {pa:>12.2f} {pb:>12.2f} {pc:>12.2f} {best_label:>8}\n")

    # Logic differences show filter contribution
    out.write(f"\n--- フィルター寄与度推定 ---\n")
    out.write("Logic-A (日足EMA+KMID+KLOW+EMA距離) vs Logic-C (KMID+KLOWのみ)\n")
    out.write("差分 = 日足EMA + EMA距離フィルターの寄与\n\n")
    for sym in adopted_syms:
        sym_data = detail_adopted[detail_adopted["sym"] == sym]
        pf_a = sym_data[sym_data["logic"] == "A"]["oos_pf"].values
        pf_c = sym_data[sym_data["logic"] == "C"]["oos_pf"].values
        if len(pf_a) > 0 and len(pf_c) > 0:
            diff = pf_a[0] - pf_c[0]
            out.write(f"  {sym:>8}: Logic-A={pf_a[0]:.2f}, Logic-C={pf_c[0]:.2f}, 差={diff:+.2f}\n")

    out.write(f"\nLogic-B (ADX+Streak+KMID+KLOW) vs Logic-C (KMID+KLOWのみ)\n")
    out.write("差分 = ADX + Streakフィルターの寄与\n\n")
    for sym in adopted_syms:
        sym_data = detail_adopted[detail_adopted["sym"] == sym]
        pf_b = sym_data[sym_data["logic"] == "B"]["oos_pf"].values
        pf_c = sym_data[sym_data["logic"] == "C"]["oos_pf"].values
        if len(pf_b) > 0 and len(pf_c) > 0:
            diff = pf_b[0] - pf_c[0]
            out.write(f"  {sym:>8}: Logic-B={pf_b[0]:.2f}, Logic-C={pf_c[0]:.2f}, 差={diff:+.2f}\n")

# Also compare v80 vs current from backtest_final_optimized
final_csv = RESULTS / "backtest_final_optimized.csv"
if final_csv.exists():
    final = pd.read_csv(final_csv)
    out.write(f"\n--- v80 vs 現行ロジック（同一エンジン比較）---\n")
    out.write(f"{'銘柄':>8} {'現行PF':>10} {'v80 PF':>10} {'差':>8} {'v80採用?':>10}\n")
    out.write("-" * 50 + "\n")
    for sym in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"]:
        sym_data = final[final["sym"] == sym]
        # Find the adopted logic's PF (rank=1 equivalent or verdict=採用)
        v80_data = sym_data[sym_data["logic"] == "V80"]
        non_v80 = sym_data[sym_data["logic"] != "V80"]
        if len(v80_data) > 0 and len(non_v80) > 0:
            v80_pf = v80_data.iloc[0]["oos_pf"]
            # Get the best non-v80
            best_non_v80 = non_v80.loc[non_v80["oos_pf"].idxmax()]
            current_pf = best_non_v80["oos_pf"]
            diff = v80_pf - current_pf
            adopted = "✅" if v80_pf > current_pf else "❌"
            out.write(f"{sym:>8} {current_pf:>10.2f} {v80_pf:>10.2f} {diff:>+8.2f} {adopted:>10}\n")

# indicator predictive power
ind_csv = RESULTS / "indicator_predictive_power.csv"
if ind_csv.exists():
    ind = pd.read_csv(ind_csv)
    out.write(f"\n--- 指標予測力分析（indicator_predictive_power.csv）---\n")
    out.write(ind.to_string(index=False) + "\n")

out.write(f"\n【注意】試練⑤の完全なアブレーション分析にはバックテスト再実行が必要です。\n")
out.write(f"以下のコマンドで実行可能:\n")
out.write(f"  python scripts/trial5_ablation_backtest.py\n")


# ═══════════════════════════════════════════════════════════════════════
#  総合判定
# ═══════════════════════════════════════════════════════════════════════
print_sep("5試練 総合判定", out)

verdicts = {
    "① ホームラン依存": verdict_1,
    "② USD集中リスク": verdict_2,
    "③ USDCAD真贋": verdict_3a,
    "④ MDD耐性": verdict_4,
}

out.write(f"{'試練':>20} {'結果':>10}\n")
out.write("-" * 35 + "\n")
for name, v in verdicts.items():
    out.write(f"{name:>20} {v:>10}\n")
out.write(f"{'⑤ フィルター貢献':>20} {'※要バックテスト':>10}\n")

pass_count = sum(1 for v in verdicts.values() if "PASS" in v)
out.write(f"\n合格: {pass_count}/4 (試練⑤は別途)\n")
if pass_count == 4:
    out.write("→ 4試練クリア。試練⑤のアブレーション分析を経て実弾投入を判断。\n")
else:
    failed = [k for k, v in verdicts.items() if "FAIL" in v]
    out.write(f"→ 不合格: {', '.join(failed)}\n")
    out.write("→ 不合格項目の対策を実施してから再検証すること。\n")

out.close()

# Print to stdout as well
with open(OUT_PATH, "r", encoding="utf-8") as f:
    print(f.read())

print(f"\nレポート出力先: {OUT_PATH}")
