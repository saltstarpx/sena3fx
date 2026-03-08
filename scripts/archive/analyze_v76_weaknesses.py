"""
v76 USDJPY Backtest Trade Log — Weakness Analysis
Reads results/v76_battle_usdjpy_v76_trades.csv and prints detailed findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results" / "v76_battle_usdjpy_v76_trades.csv"

df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time"])

# Derived columns
df["entry_hour"] = df["entry_time"].dt.hour
df["entry_dow"] = df["entry_time"].dt.dayofweek  # 0=Mon
df["entry_dow_name"] = df["entry_time"].dt.day_name()
df["duration_min"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
df["direction"] = df["dir"].map({1: "LONG", -1: "SHORT"})
df["month_str"] = df["month"]
df["year_month"] = df["entry_time"].dt.to_period("M")

sep = "=" * 72

# ── Overall summary ──────────────────────────────────────────────────────
print(sep)
print("V76 USDJPY WEAKNESS ANALYSIS — OVERALL SUMMARY")
print(sep)
n = len(df)
wins = df[df.result == "win"]
losses = df[df.result == "loss"]
print(f"Total trades : {n}")
print(f"Wins / Losses: {len(wins)} / {len(losses)}  (WR={len(wins)/n*100:.1f}%)")
print(f"Total PnL    : {df.pnl.sum():.2f}")
print(f"Avg win      : {wins.pnl.mean():.2f}")
print(f"Avg loss     : {losses.pnl.mean():.2f}")
print(f"Win/Loss ratio (avg): {abs(wins.pnl.mean() / losses.pnl.mean()):.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# 1. LOSING TRADE CHARACTERISTICS
# ─────────────────────────────────────────────────────────────────────────
print(sep)
print("1. LOSING TRADE CHARACTERISTICS")
print(sep)

# 1a. Hour of day
print("\n--- Losses by Entry Hour (UTC) ---")
hour_all = df.groupby("entry_hour").agg(trades=("pnl", "count"), total_pnl=("pnl", "sum"))
hour_loss = losses.groupby("entry_hour").agg(loss_count=("pnl", "count"), loss_pnl=("pnl", "sum"))
hour = hour_all.join(hour_loss, how="left").fillna(0)
hour["loss_rate"] = hour["loss_count"] / hour["trades"] * 100
hour = hour.sort_values("loss_rate", ascending=False)
print(hour.to_string())
print("\nWorst hours (loss_rate > 60% AND >= 5 trades):")
bad_hours = hour[(hour.loss_rate > 60) & (hour.trades >= 5)]
if len(bad_hours):
    print(bad_hours.to_string())
else:
    print("  (none meet threshold)")

# 1b. Day of week
print("\n--- Losses by Day of Week ---")
dow_stats = df.groupby("entry_dow_name").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
)
dow_stats["wr"] = dow_stats["wins"] / dow_stats["trades"] * 100
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_stats = dow_stats.reindex([d for d in dow_order if d in dow_stats.index])
print(dow_stats.to_string())

# 1c. Direction
print("\n--- Losses by Direction ---")
dir_stats = df.groupby("direction").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
)
dir_stats["wr"] = dir_stats["wins"] / dir_stats["trades"] * 100
print(dir_stats.to_string())

# 1d. Exit type for losses
print("\n--- Exit Type Breakdown for LOSSES ---")
print(losses.groupby("exit_type").agg(count=("pnl", "count"), avg_pnl=("pnl", "mean")).to_string())

# ─────────────────────────────────────────────────────────────────────────
# 2. MONTHLY / PERIOD PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("2. MONTHLY PERFORMANCE")
print(sep)
monthly = df.groupby("month_str").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
    max_loss=("pnl", "min"),
)
monthly["wr"] = monthly["wins"] / monthly["trades"] * 100
print(monthly.to_string())
print("\nWorst months (negative total PnL):")
bad_months = monthly[monthly.total_pnl < 0].sort_values("total_pnl")
print(bad_months.to_string() if len(bad_months) else "  (none)")

# ─────────────────────────────────────────────────────────────────────────
# 3. EXIT TYPE BREAKDOWN (wins vs losses)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("3. EXIT TYPE BREAKDOWN — WINS vs LOSSES")
print(sep)
exit_result = df.groupby(["exit_type", "result"]).agg(
    count=("pnl", "count"),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
)
print(exit_result.to_string())

print("\n--- Exit type overall ---")
exit_overall = df.groupby("exit_type").agg(
    count=("pnl", "count"),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
    wr=("result", lambda x: (x == "win").sum() / len(x) * 100),
)
print(exit_overall.to_string())

# ─────────────────────────────────────────────────────────────────────────
# 4. CONSECUTIVE LOSS CLUSTERS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("4. CONSECUTIVE LOSS CLUSTERS")
print(sep)

# Identify streaks
streaks = []
current_streak = 0
streak_start = None
for i, row in df.iterrows():
    if row["result"] == "loss":
        if current_streak == 0:
            streak_start = i
        current_streak += 1
    else:
        if current_streak >= 3:
            streaks.append({
                "start_idx": streak_start,
                "length": current_streak,
                "start_time": df.loc[streak_start, "entry_time"],
                "end_time": df.loc[streak_start + current_streak - 1, "exit_time"],
                "total_loss": df.loc[streak_start:streak_start + current_streak - 1, "pnl"].sum(),
            })
        current_streak = 0
# catch trailing streak
if current_streak >= 3:
    streaks.append({
        "start_idx": streak_start,
        "length": current_streak,
        "start_time": df.loc[streak_start, "entry_time"],
        "end_time": df.loc[streak_start + current_streak - 1, "exit_time"],
        "total_loss": df.loc[streak_start:streak_start + current_streak - 1, "pnl"].sum(),
    })

streaks_df = pd.DataFrame(streaks)
if len(streaks_df):
    streaks_df = streaks_df.sort_values("length", ascending=False)
    print(f"Clusters of 3+ consecutive losses: {len(streaks_df)}")
    print(f"Longest streak: {streaks_df.iloc[0]['length']} losses")
    print(f"Worst streak PnL: {streaks_df['total_loss'].min():.2f}")
    print()
    print(streaks_df.head(15).to_string(index=False))

    # Analyze what hours/days these clusters fall in
    print("\n--- Cluster trade details (top 5 worst) ---")
    for _, s in streaks_df.head(5).iterrows():
        si = int(s["start_idx"])
        sl = int(s["length"])
        cluster = df.loc[si:si + sl - 1, ["entry_time", "direction", "pnl", "entry_hour", "entry_dow_name"]]
        print(f"\nStreak from {s['start_time']} — {sl} losses, total={s['total_loss']:.2f}")
        print(cluster.to_string(index=False))
else:
    print("No clusters of 3+ consecutive losses found.")

# ─────────────────────────────────────────────────────────────────────────
# 5. AVERAGE PNL BY DIRECTION
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("5. PNL BY DIRECTION (LONG vs SHORT)")
print(sep)
for d in ["LONG", "SHORT"]:
    sub = df[df.direction == d]
    w = sub[sub.result == "win"]
    l = sub[sub.result == "loss"]
    print(f"\n{d}:")
    print(f"  Trades: {len(sub)}  Wins: {len(w)}  Losses: {len(l)}  WR: {len(w)/len(sub)*100:.1f}%")
    print(f"  Total PnL: {sub.pnl.sum():.2f}  Avg PnL: {sub.pnl.mean():.2f}")
    print(f"  Avg win: {w.pnl.mean():.2f}   Avg loss: {l.pnl.mean():.2f}")
    print(f"  Best trade: {sub.pnl.max():.2f}  Worst trade: {sub.pnl.min():.2f}")

    # by exit type
    print(f"  Exit type breakdown:")
    et = sub.groupby("exit_type").agg(n=("pnl", "count"), pnl=("pnl", "sum"))
    for idx, row in et.iterrows():
        print(f"    {idx}: {int(row.n)} trades, PnL={row.pnl:.2f}")

# ─────────────────────────────────────────────────────────────────────────
# 6. TRADE DURATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("6. TRADE DURATION (minutes) — WINNERS vs LOSERS")
print(sep)
for label, sub in [("WINNERS", wins), ("LOSERS", losses)]:
    dur = sub["duration_min"]
    print(f"\n{label} (n={len(sub)}):")
    print(f"  Mean duration : {dur.mean():.0f} min ({dur.mean()/60:.1f} hrs)")
    print(f"  Median duration: {dur.median():.0f} min ({dur.median()/60:.1f} hrs)")
    print(f"  Std            : {dur.std():.0f} min")
    print(f"  Min / Max      : {dur.min():.0f} / {dur.max():.0f} min")
    # Quartiles
    print(f"  25th / 75th pct: {dur.quantile(0.25):.0f} / {dur.quantile(0.75):.0f} min")

# Duration buckets
print("\n--- Duration buckets vs win rate ---")
bins = [0, 30, 60, 120, 360, 720, 1440, 5000, 100000]
labels = ["<30m", "30m-1h", "1-2h", "2-6h", "6-12h", "12-24h", "1-3d", "3d+"]
df["dur_bucket"] = pd.cut(df["duration_min"], bins=bins, labels=labels)
dur_bucket = df.groupby("dur_bucket", observed=False).agg(
    trades=("pnl", "count"),
    wr=("result", lambda x: (x == "win").sum() / max(len(x), 1) * 100),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
)
print(dur_bucket.to_string())

# ─────────────────────────────────────────────────────────────────────────
# 7. SESSION ANALYSIS (derived from entry hour)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("7. SESSION ANALYSIS (UTC hours)")
print(sep)

def get_session(h):
    if 0 <= h < 7:
        return "Asia (00-07)"
    elif 7 <= h < 13:
        return "London (07-13)"
    elif 13 <= h < 20:
        return "NY (13-20)"
    else:
        return "Late NY (20-24)"

df["session"] = df["entry_hour"].apply(get_session)
sess = df.groupby("session").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
)
sess["wr"] = sess["wins"] / sess["trades"] * 100
print(sess.to_string())

# ─────────────────────────────────────────────────────────────────────────
# 8. CROSS-TAB: Direction x Session
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("8. CROSS-TAB: Direction x Session")
print(sep)
for sess_name in ["Asia (00-07)", "London (07-13)", "NY (13-20)", "Late NY (20-24)"]:
    sub = df[df.session == sess_name]
    if len(sub) == 0:
        continue
    print(f"\n{sess_name}:")
    for d in ["LONG", "SHORT"]:
        ds = sub[sub.direction == d]
        if len(ds) == 0:
            continue
        wr = (ds.result == "win").sum() / len(ds) * 100
        print(f"  {d}: {len(ds)} trades, WR={wr:.0f}%, avg_pnl={ds.pnl.mean():.2f}, total={ds.pnl.sum():.2f}")

# ─────────────────────────────────────────────────────────────────────────
# SUMMARY OF KEY FINDINGS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + sep)
print("SUMMARY OF KEY WEAKNESSES")
print(sep)

# Find worst session
worst_sess = sess.sort_values("avg_pnl").index[0]
ws = sess.loc[worst_sess]
print(f"\n1. WORST SESSION: {worst_sess}")
print(f"   {int(ws.trades)} trades, WR={ws.wr:.0f}%, avg_pnl={ws.avg_pnl:.2f}, total_pnl={ws.total_pnl:.2f}")

# Worst day of week
worst_dow = dow_stats.sort_values("avg_pnl").index[0]
wd = dow_stats.loc[worst_dow]
print(f"\n2. WORST DAY: {worst_dow}")
print(f"   {int(wd.trades)} trades, WR={wd.wr:.0f}%, avg_pnl={wd.avg_pnl:.2f}, total_pnl={wd.total_pnl:.2f}")

# Direction bias
long_pnl = df[df.direction == "LONG"].pnl.sum()
short_pnl = df[df.direction == "SHORT"].pnl.sum()
print(f"\n3. DIRECTION BIAS: LONG total={long_pnl:.2f}, SHORT total={short_pnl:.2f}")
if long_pnl < short_pnl:
    print(f"   -> LONG underperforms SHORT by {short_pnl - long_pnl:.2f}")
else:
    print(f"   -> SHORT underperforms LONG by {long_pnl - short_pnl:.2f}")

# Worst months
if len(bad_months):
    print(f"\n4. NEGATIVE MONTHS: {', '.join(bad_months.index.tolist())}")
    for m, row in bad_months.iterrows():
        print(f"   {m}: {int(row.trades)} trades, WR={row.wr:.0f}%, total_pnl={row.total_pnl:.2f}")

# Consecutive loss info
if len(streaks_df):
    print(f"\n5. CONSECUTIVE LOSSES: {len(streaks_df)} clusters of 3+, max streak={int(streaks_df.iloc[0]['length'])}")

# Duration insight
w_med = wins.duration_min.median()
l_med = losses.duration_min.median()
print(f"\n6. DURATION: Winner median={w_med:.0f}min, Loser median={l_med:.0f}min")
if l_med < w_med:
    print(f"   -> Losers close faster (hit SL quickly)")
else:
    print(f"   -> Losers linger longer before stopping out")

print()
