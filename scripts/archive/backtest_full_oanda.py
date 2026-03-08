"""
v76 IS+OOS 統合検証スクリプト（OANDAデータ版）
IS期間:  2024年7月1日〜2025年2月28日
OOS期間: 2025年3月1日〜2026年2月28日
スプレッド: 0.4pips
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
import yagami_mtf_v76 as v76

DATA    = "/home/ubuntu/sena3fx/data"
SPREAD  = 0.4
RESULTS = "/home/ubuntu/sena3fx/results"

def load(p):
    df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# データ読み込み
print("データ読み込み中...")
is_1m  = load(f"{DATA}/usdjpy_is_1m.csv")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h  = load(f"{DATA}/usdjpy_is_4h.csv")
oos_1m  = load(f"{DATA}/usdjpy_oos_1m.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h  = load(f"{DATA}/usdjpy_oos_4h.csv")

print(f"IS  1m: {len(is_1m)}行  4h: {len(is_4h)}本")
print(f"OOS 1m: {len(oos_1m)}行  4h: {len(oos_4h)}本")

# ---- バックテストエンジン ----
def run_backtest(data_1m, data_15m, data_4h, spread=0.4, label=""):
    print(f"\n{label} シグナル生成中...", flush=True)
    signals = v76.generate_signals(data_1m, data_15m, data_4h, spread_pips=spread)
    sig_map = {s["time"]: s for s in signals}
    print(f"  シグナル数: {len(signals)}")

    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        if pos is not None:
            d      = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d

            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True

            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + sl_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "SL" if not pos["half_closed"] else "HALF+SL",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": label,
                })
                pos = None; continue

            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + tp_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "TP" if not pos["half_closed"] else "HALF+TP",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": label,
                })
                pos = None; continue

        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}

    df_trades = pd.DataFrame(trades)
    print(f"  トレード数: {len(df_trades)}")
    return df_trades

# ---- 実行 ----
df_is  = run_backtest(is_1m,  is_15m,  is_4h,  SPREAD, "IS")
df_oos = run_backtest(oos_1m, oos_15m, oos_4h, SPREAD, "OOS")

# ---- 統計計算 ----
def calc_stats(df, label):
    if df.empty:
        print(f"{label}: トレードなし"); return {}
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df) * 100
    avg_w  = wins["pnl"].mean()  if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0)
    monthly     = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"  トレード数:  {len(df)}回")
    print(f"  勝率:        {wr:.1f}%")
    print(f"  PF:          {pf:.2f}")
    print(f"  総損益:      {df['pnl'].sum():+.1f}pips")
    print(f"  平均利益:    {avg_w:+.1f}pips")
    print(f"  平均損失:    {avg_l:+.1f}pips")
    print(f"  ケリー基準:  {kelly:.3f}")
    print(f"  t統計量:     {t_stat:.4f}")
    print(f"  p値:         {p_val:.4f}  {'★p<0.05 統計的有意' if p_val < 0.05 else '(有意差なし)'}")
    print(f"  プラス月:    {plus_months}/{total_months}ヶ月")
    return dict(label=label, trades=len(df), win_rate=wr, pf=pf,
                total_pnl=df["pnl"].sum(), avg_win=avg_w, avg_loss=avg_l,
                kelly=kelly, t_stat=t_stat, p_value=p_val,
                plus_months=f"{plus_months}/{total_months}", monthly=monthly)

s_is  = calc_stats(df_is,  "IS  (2024年7月〜2025年2月) ※OANDAデータ")
s_oos = calc_stats(df_oos, "OOS (2025年3月〜2026年2月) ※OANDAデータ")
df_all = pd.concat([df_is, df_oos], ignore_index=True)
s_all = calc_stats(df_all, "IS+OOS 統合 (2024年7月〜2026年2月)")

# ---- チャート生成 ----
plt.rcParams["font.family"] = "Noto Sans CJK JP"
fig, axes = plt.subplots(3, 1, figsize=(14, 18))
fig.patch.set_facecolor("#0d0d1a")

def bar_chart(ax, monthly, title):
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, fontsize=9)
    ax.axhline(0, color="white", linewidth=0.6)
    ax.set_title(title, fontsize=11, color="white")
    ax.set_ylabel("損益 (pips)", color="white")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")

if "monthly" in s_is:
    bar_chart(axes[0], s_is["monthly"],
              f"IS期間 月別損益  {s_is['trades']}回 / PF {s_is['pf']:.2f} / 勝率{s_is['win_rate']:.1f}% / p={s_is['p_value']:.4f}")

if "monthly" in s_oos:
    bar_chart(axes[1], s_oos["monthly"],
              f"OOS期間 月別損益  {s_oos['trades']}回 / PF {s_oos['pf']:.2f} / 勝率{s_oos['win_rate']:.1f}% / p={s_oos['p_value']:.4f}")

# エクイティカーブ
df_all_sorted = df_all.sort_values("entry_time")
cumulative = df_all_sorted["pnl"].cumsum().values
axes[2].plot(range(len(cumulative)), cumulative, color="#3498db", linewidth=1.5)
axes[2].fill_between(range(len(cumulative)), cumulative, alpha=0.15, color="#3498db")
axes[2].axhline(0, color="white", linewidth=0.5, linestyle="--")
is_n = len(df_is)
if 0 < is_n < len(df_all):
    axes[2].axvline(is_n, color="#f39c12", linewidth=1.5, linestyle="--", label="IS/OOS境界")
    axes[2].legend(facecolor="#1a1a2e", labelcolor="white")
sig_label = "★統計的有意 (p<0.05)" if s_all.get("p_value", 1) < 0.05 else ""
axes[2].set_title(
    f"IS+OOS エクイティカーブ  {s_all['trades']}回 / PF {s_all['pf']:.2f} / "
    f"勝率{s_all['win_rate']:.1f}% / p={s_all['p_value']:.4f} {sig_label}",
    fontsize=11, color="white")
axes[2].set_xlabel("トレード番号", color="white")
axes[2].set_ylabel("累積損益 (pips)", color="white")
axes[2].set_facecolor("#1a1a2e")
axes[2].tick_params(colors="white")

plt.tight_layout(pad=2.0)
out_img = f"{RESULTS}/v76_full_oanda_validation.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

# トレードログ保存
df_is.to_csv(f"{RESULTS}/v76_is_oanda_trades.csv", index=False)
df_oos.to_csv(f"{RESULTS}/v76_oos_oanda_trades.csv", index=False)
df_all.to_csv(f"{RESULTS}/v76_all_oanda_trades.csv", index=False)
print("トレードログ保存完了")
