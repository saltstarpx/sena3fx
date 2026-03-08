"""
v76 アウト・オブ・サンプル検証スクリプト
OOS期間: 2025年3月1日〜2026年2月28日
IS期間:  2024年7月1日〜2025年2月6日（比較用）
スプレッド: 0.4pips（実運用想定）
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
import yagami_mtf_v76 as v76

DATA = "/home/ubuntu/sena3fx/data"
SPREAD = 0.4

# ---- データ読み込み ----
def load_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# OOSデータ
oos_1m  = load_csv(f"{DATA}/usdjpy_oos_1m.csv")
oos_15m = load_csv(f"{DATA}/usdjpy_oos_15m.csv")
oos_1h  = load_csv(f"{DATA}/usdjpy_oos_1h.csv")
oos_4h  = load_csv(f"{DATA}/usdjpy_oos_4h.csv")

# ISデータ（比較用）
is_1m  = load_csv(f"{DATA}/usdjpy_1m.csv")
is_15m = load_csv(f"{DATA}/usdjpy_15m.csv")
is_1h  = load_csv(f"{DATA}/usdjpy_1h.csv")
is_4h  = load_csv(f"{DATA}/usdjpy_4h.csv")
IS_START, IS_END = "2024-07-01", "2025-02-06"
is_1m  = is_1m[IS_START:IS_END]
is_15m = is_15m[IS_START:IS_END]
is_1h  = is_1h[IS_START:IS_END]
is_4h  = is_4h[IS_START:IS_END]

print(f"OOS 1m: {len(oos_1m)}行 ({oos_1m.index[0].date()} 〜 {oos_1m.index[-1].date()})")
print(f"IS  1m: {len(is_1m)}行 ({is_1m.index[0].date()} 〜 {is_1m.index[-1].date()})")

# ---- バックテストエンジン ----
def run_backtest(data_1m, data_15m, data_1h, data_4h, spread=0.4, label=""):
    print(f"\n{label} バックテスト実行中...", flush=True)
    signals = v76.generate_signals(data_1m, data_15m, data_4h, spread_pips=spread)
    sig_map = {s["time"]: s for s in signals}

    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        if pos is not None:
            d = pos["dir"]
            # v76: epはスプレッド込みの実際の約定価格
            # チャートレベルのraw_ep = ep - spread*dir
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep  # 建値（チャートレベル）に移動
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
                })
                pos = None; continue
        if pos is None and t in sig_map:
            s = sig_map[t]
            pos = {**s, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)

# ---- 実行 ----
df_oos = run_backtest(oos_1m, oos_15m, oos_1h, oos_4h, SPREAD, "OOS")
df_is  = run_backtest(is_1m,  is_15m,  is_1h,  is_4h,  SPREAD, "IS")

# ---- 統計計算 ----
def calc_stats(df, label):
    if df.empty:
        print(f"{label}: トレードなし")
        return {}
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df) * 100
    avg_w  = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    # t検定
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0)
    # 月別集計
    monthly = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)
    stats_dict = {
        "label": label,
        "trades": len(df),
        "win_rate": wr,
        "pf": pf,
        "total_pnl": df["pnl"].sum(),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "t_stat": t_stat,
        "p_value": p_val,
        "plus_months": f"{plus_months}/{total_months}",
        "monthly": monthly,
    }
    print(f"\n{'='*40}")
    print(f"{label}")
    print(f"  トレード数: {len(df)}回")
    print(f"  勝率:       {wr:.1f}%")
    print(f"  PF:         {pf:.2f}")
    print(f"  総損益:     {df['pnl'].sum():+.1f}pips")
    print(f"  平均利益:   {avg_w:+.1f}pips")
    print(f"  平均損失:   {avg_l:+.1f}pips")
    print(f"  t統計量:    {t_stat:.4f}")
    print(f"  p値:        {p_val:.4f} {'★有意' if p_val < 0.05 else '(有意差なし)'}")
    print(f"  プラス月:   {plus_months}/{total_months}ヶ月")
    return stats_dict

stats_is  = calc_stats(df_is,  "IS  (2024年7月〜2025年2月)")
stats_oos = calc_stats(df_oos, "OOS (2025年3月〜2026年2月)")

# ---- IS+OOS統合 ----
df_all = pd.concat([df_is, df_oos], ignore_index=True)
stats_all = calc_stats(df_all, "IS+OOS 統合 (2024年7月〜2026年2月)")

# ---- 月別損益チャート ----
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not pd.io.common.file_exists(font_path):
    font_path = None

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
plt.rcParams["font.family"] = "Noto Sans CJK JP" if font_path else "sans-serif"

# IS月別
if "monthly" in stats_is:
    m = stats_is["monthly"]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in m.values]
    axes[0].bar(range(len(m)), m.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(range(len(m)))
    axes[0].set_xticklabels(m.index, rotation=45, fontsize=9)
    axes[0].axhline(0, color="white", linewidth=0.8)
    axes[0].set_title(f"IS期間 月別損益 (2024年7月〜2025年2月)  "
                      f"計{stats_is['trades']}回 / PF {stats_is['pf']:.2f} / p={stats_is['p_value']:.3f}",
                      fontsize=11)
    axes[0].set_ylabel("損益 (pips)")
    axes[0].set_facecolor("#1a1a2e")
    axes[0].tick_params(colors="white")
    axes[0].yaxis.label.set_color("white")
    axes[0].title.set_color("white")

# OOS月別
if "monthly" in stats_oos:
    m = stats_oos["monthly"]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in m.values]
    axes[1].bar(range(len(m)), m.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(range(len(m)))
    axes[1].set_xticklabels(m.index, rotation=45, fontsize=9)
    axes[1].axhline(0, color="white", linewidth=0.8)
    axes[1].set_title(f"OOS期間 月別損益 (2025年3月〜2026年2月)  "
                      f"計{stats_oos['trades']}回 / PF {stats_oos['pf']:.2f} / p={stats_oos['p_value']:.3f}",
                      fontsize=11)
    axes[1].set_ylabel("損益 (pips)")
    axes[1].set_facecolor("#1a1a2e")
    axes[1].tick_params(colors="white")
    axes[1].yaxis.label.set_color("white")
    axes[1].title.set_color("white")

# IS+OOS エクイティカーブ
df_all_sorted = df_all.sort_values("entry_time")
cumulative = df_all_sorted["pnl"].cumsum()
axes[2].plot(range(len(cumulative)), cumulative.values, color="#3498db", linewidth=1.5)
axes[2].fill_between(range(len(cumulative)), cumulative.values, alpha=0.2, color="#3498db")
axes[2].axhline(0, color="white", linewidth=0.5, linestyle="--")
# IS/OOS境界線
is_count = len(df_is)
if is_count > 0 and is_count < len(df_all):
    axes[2].axvline(is_count, color="#f39c12", linewidth=1.5, linestyle="--", label="IS/OOS境界")
    axes[2].legend(facecolor="#1a1a2e", labelcolor="white")
axes[2].set_title(f"IS+OOS エクイティカーブ  "
                  f"計{stats_all['trades']}回 / PF {stats_all['pf']:.2f} / p={stats_all['p_value']:.4f} "
                  f"{'★p<0.05 有意' if stats_all['p_value'] < 0.05 else ''}",
                  fontsize=11)
axes[2].set_xlabel("トレード番号")
axes[2].set_ylabel("累積損益 (pips)")
axes[2].set_facecolor("#1a1a2e")
axes[2].tick_params(colors="white")
axes[2].xaxis.label.set_color("white")
axes[2].yaxis.label.set_color("white")
axes[2].title.set_color("white")

fig.patch.set_facecolor("#0d0d1a")
plt.tight_layout(pad=2.0)
out_img = "/home/ubuntu/sena3fx/results/v76_oos_validation.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

# ---- 結果CSV ----
df_oos.to_csv("/home/ubuntu/sena3fx/results/v76_oos_trades.csv", index=False)
df_all.to_csv("/home/ubuntu/sena3fx/results/v76_all_trades.csv", index=False)
print("トレードログ保存完了")
