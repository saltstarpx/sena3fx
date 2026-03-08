"""
backtest_v76_full_period.py
============================
v76ロジック / IS+OOS全期間統合バックテスト
期間: 2024-07-01 〜 2026-02-27（約1.5年）
スプレッド: 0.4pips（USDJPY標準）

IS期間:  2024-07-01 〜 2025-02-28
OOS期間: 2025-03-03 〜 2026-02-27
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# 日本語フォント設定
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# v76ロジック読み込み
sys.path.insert(0, "/home/ubuntu/sena3fx/strategies/current")
import yagami_mtf_v76 as v76

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"
SPREAD  = 0.4
RR      = 2.5
IS_END  = "2025-03-01"  # IS/OOS境界

# ============================================================
# データ読み込み・結合
# ============================================================
def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

print("データ読み込み・結合中...")
data_1m  = pd.concat([load(f"{DATA}/usdjpy_is_1m.csv"),
                      load(f"{DATA}/usdjpy_oos_1m.csv")]).sort_index()
data_15m = pd.concat([load(f"{DATA}/usdjpy_is_15m.csv"),
                      load(f"{DATA}/usdjpy_oos_15m.csv")]).sort_index()
data_4h  = pd.concat([load(f"{DATA}/usdjpy_is_4h.csv"),
                      load(f"{DATA}/usdjpy_oos_4h.csv")]).sort_index()

# 重複除去（念のため）
data_1m  = data_1m[~data_1m.index.duplicated(keep='first')]
data_15m = data_15m[~data_15m.index.duplicated(keep='first')]
data_4h  = data_4h[~data_4h.index.duplicated(keep='first')]

print(f"1分足: {len(data_1m):,}行  {data_1m.index[0]} 〜 {data_1m.index[-1]}")
print(f"15分足: {len(data_15m):,}行  {data_15m.index[0]} 〜 {data_15m.index[-1]}")
print(f"4時間足: {len(data_4h):,}行  {data_4h.index[0]} 〜 {data_4h.index[-1]}")

# ============================================================
# シグナル生成
# ============================================================
print("\nシグナル生成中（全期間）...")
signals = v76.generate_signals(data_1m, data_15m, data_4h, spread_pips=SPREAD, rr_ratio=RR)
sig_map = {s["time"]: s for s in signals}
print(f"シグナル数: {len(signals)}")

# ============================================================
# バックテストエンジン
# ============================================================
print("バックテスト実行中...")
trades = []
pos = None

for i in range(len(data_1m)):
    bar = data_1m.iloc[i]
    t   = bar.name

    if pos is not None:
        d      = pos["dir"]
        raw_ep = pos["ep"] - pos["spread"] * d
        half_tp = raw_ep + pos["risk"] * d

        # 半利確チェック
        if not pos["half_closed"]:
            if (d == 1 and bar["high"] >= half_tp) or \
               (d == -1 and bar["low"] <= half_tp):
                pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                pos["sl"] = raw_ep
                pos["half_closed"] = True

        # SL到達
        if (d == 1 and bar["low"] <= pos["sl"]) or \
           (d == -1 and bar["high"] >= pos["sl"]):
            sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
            total  = pos.get("half_pnl", 0) + sl_pnl
            period = "IS" if str(pos["entry_time"]) < IS_END else "OOS"
            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time":  t,
                "dir":        d,
                "ep":         pos["ep"],
                "sl":         pos["sl"],
                "tp":         pos["tp"],
                "pnl":        total,
                "result":     "win" if total > 0 else "loss",
                "exit_type":  "HALF+SL" if pos["half_closed"] else "SL",
                "tf":         pos.get("tf", "?"),
                "pattern":    pos.get("pattern", "?"),
                "month":      pos["entry_time"].strftime("%Y-%m"),
                "period":     period,
            })
            pos = None
            continue

        # TP到達
        if (d == 1 and bar["high"] >= pos["tp"]) or \
           (d == -1 and bar["low"] <= pos["tp"]):
            tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
            total  = pos.get("half_pnl", 0) + tp_pnl
            period = "IS" if str(pos["entry_time"]) < IS_END else "OOS"
            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time":  t,
                "dir":        d,
                "ep":         pos["ep"],
                "sl":         pos["sl"],
                "tp":         pos["tp"],
                "pnl":        total,
                "result":     "win" if total > 0 else "loss",
                "exit_type":  "HALF+TP" if pos["half_closed"] else "TP",
                "tf":         pos.get("tf", "?"),
                "pattern":    pos.get("pattern", "?"),
                "month":      pos["entry_time"].strftime("%Y-%m"),
                "period":     period,
            })
            pos = None
            continue

    if pos is None and t in sig_map:
        pos = {**sig_map[t], "entry_time": t, "half_closed": False}

df = pd.DataFrame(trades)
print(f"完了: {len(df)}トレード")

# ============================================================
# 統計計算
# ============================================================
def calc_stats(df_sub, label=""):
    if df_sub.empty:
        return {}
    wins   = df_sub[df_sub["pnl"] > 0]
    losses = df_sub[df_sub["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df_sub) * 100
    avg_w  = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df_sub["pnl"], 0)
    monthly       = df_sub.groupby("month")["pnl"].sum()
    plus_months   = (monthly > 0).sum()
    total_months  = len(monthly)
    cumulative    = df_sub.sort_values("entry_time")["pnl"].cumsum()
    rolling_max   = cumulative.cummax()
    max_dd        = (rolling_max - cumulative).max()
    sharpe        = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    return dict(
        label=label, trades=len(df_sub), win_rate=wr, pf=pf,
        total_pnl=df_sub["pnl"].sum(), avg_win=avg_w, avg_loss=avg_l,
        kelly=kelly, t_stat=t_stat, p_value=p_val,
        plus_months=f"{plus_months}/{total_months}",
        max_dd=max_dd, sharpe=sharpe, monthly=monthly,
    )

s_all = calc_stats(df, "全期間")
s_is  = calc_stats(df[df["period"] == "IS"],  "IS期間")
s_oos = calc_stats(df[df["period"] == "OOS"], "OOS期間")
s_4h  = calc_stats(df[df["tf"] == "4h"], "4時間足シグナル")
s_1h  = calc_stats(df[df["tf"] == "1h"], "1時間足シグナル")

print("\n" + "="*70)
print("バックテスト結果サマリー（v76 / spread=0.4pips / 全期間）")
print("="*70)
for s in [s_all, s_is, s_oos, s_4h, s_1h]:
    if not s:
        continue
    pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "∞"
    print(f"[{s['label']}]")
    print(f"  トレード数: {s['trades']}回  勝率: {s['win_rate']:.1f}%  PF: {pf_str}")
    print(f"  総損益: {s['total_pnl']:+.1f}pips  平均利益: {s['avg_win']:.1f}p  平均損失: {s['avg_loss']:.1f}p")
    print(f"  ケリー: {s['kelly']:.3f}  最大DD: {s['max_dd']:.1f}p  シャープ: {s['sharpe']:.2f}")
    print(f"  t統計量: {s['t_stat']:.3f}  p値: {s['p_value']:.4f}  プラス月: {s['plus_months']}")
    print()

# ============================================================
# 可視化
# ============================================================
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0d0d1a")

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=9)

C_IS  = "#3498db"   # 青: IS
C_OOS = "#e74c3c"   # 赤: OOS
C_ALL = "#2ecc71"   # 緑: 全体

# ── 1. エクイティカーブ（全期間・IS/OOS色分け）──
ax1 = fig.add_subplot(4, 2, (1, 2))

df_sorted = df.sort_values("entry_time").reset_index(drop=True)
cum_pnl = df_sorted["pnl"].cumsum()

# IS/OOS境界インデックス
is_mask  = df_sorted["period"] == "IS"
oos_mask = df_sorted["period"] == "OOS"
is_idx   = df_sorted[is_mask].index
oos_idx  = df_sorted[oos_mask].index

if len(is_idx) > 0:
    ax1.plot(is_idx, cum_pnl[is_idx], color=C_IS, linewidth=1.8,
             label=f"IS期間  PF:{s_is['pf']:.2f}  {s_is['trades']}回  {s_is['total_pnl']:+.0f}pips")
if len(oos_idx) > 0:
    # IS最終値から連続させる
    offset = cum_pnl[is_idx[-1]] if len(is_idx) > 0 else 0
    ax1.plot(oos_idx, cum_pnl[oos_idx], color=C_OOS, linewidth=1.8,
             label=f"OOS期間  PF:{s_oos['pf']:.2f}  {s_oos['trades']}回  {s_oos['total_pnl']:+.0f}pips")

if len(is_idx) > 0 and len(oos_idx) > 0:
    ax1.axvline(oos_idx[0], color="#f39c12", linewidth=1.5, linestyle="--",
                label="IS/OOS境界", alpha=0.8)

ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
ax1.set_title(f"エクイティカーブ（v76 / 全期間 / spread={SPREAD}pips）", color="white", fontsize=12, pad=8)
ax1.set_xlabel("トレード番号", color="white", fontsize=9)
ax1.set_ylabel("累積損益 (pips)", color="white", fontsize=9)
ax1.set_facecolor("#1a1a2e")
ax1.tick_params(colors="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
for spine in ax1.spines.values(): spine.set_edgecolor("#444")

# ── 2. 月別損益棒グラフ ──
ax2 = fig.add_subplot(4, 2, (3, 4))
monthly_all = df.groupby("month")["pnl"].sum()
colors_m = [C_IS if m < "2025-03" else C_OOS for m in monthly_all.index]
bars = ax2.bar(range(len(monthly_all)), monthly_all.values,
               color=colors_m, edgecolor="white", linewidth=0.3, alpha=0.85)
ax2.axhline(0, color="white", linewidth=0.6)
ax2.set_xticks(range(len(monthly_all)))
ax2.set_xticklabels(monthly_all.index, rotation=45, fontsize=7)
# 月別損益の数値を棒の上に表示
for i, (bar, val) in enumerate(zip(bars, monthly_all.values)):
    ypos = val + 20 if val >= 0 else val - 50
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:+.0f}", ha='center', va='bottom' if val >= 0 else 'top',
             color='white', fontsize=6, alpha=0.8)
styled_ax(ax2, "月別損益（青=IS / 赤=OOS）", ylabel="損益 (pips)")

# ── 3. IS vs OOS 主要指標比較 ──
ax3 = fig.add_subplot(4, 2, 5)
metrics = ["PF", "勝率(%)", "ケリー×100", "シャープ"]
is_vals  = [s_is["pf"],  s_is["win_rate"],  s_is["kelly"]*100,  s_is["sharpe"]]
oos_vals = [s_oos["pf"], s_oos["win_rate"], s_oos["kelly"]*100, s_oos["sharpe"]]
x = np.arange(len(metrics))
w = 0.35
ax3.bar(x - w/2, is_vals,  w, label=f"IS  ({s_is['trades']}回)",  color=C_IS,  alpha=0.85, edgecolor="white", linewidth=0.4)
ax3.bar(x + w/2, oos_vals, w, label=f"OOS ({s_oos['trades']}回)", color=C_OOS, alpha=0.85, edgecolor="white", linewidth=0.4)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=8)
ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax3, "IS vs OOS 主要指標比較")

# ── 4. 4時間足 vs 1時間足 ──
ax4 = fig.add_subplot(4, 2, 6)
tf_metrics = ["PF", "勝率(%)", "ケリー×100"]
h4_vals = [s_4h["pf"], s_4h["win_rate"], s_4h["kelly"]*100]
h1_vals = [s_1h["pf"], s_1h["win_rate"], s_1h["kelly"]*100]
x2 = np.arange(len(tf_metrics))
ax4.bar(x2 - w/2, h4_vals, w, label=f"4時間足 ({s_4h['trades']}回)", color="#9b59b6", alpha=0.85, edgecolor="white", linewidth=0.4)
ax4.bar(x2 + w/2, h1_vals, w, label=f"1時間足 ({s_1h['trades']}回)", color="#f39c12", alpha=0.85, edgecolor="white", linewidth=0.4)
ax4.set_xticks(x2)
ax4.set_xticklabels(tf_metrics, fontsize=8)
ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax4, "4時間足 vs 1時間足シグナル比較")

# ── 5. 損益分布ヒストグラム ──
ax5 = fig.add_subplot(4, 2, 7)
ax5.hist(df["pnl"], bins=50, color=C_ALL, edgecolor="#0d0d1a", linewidth=0.3, alpha=0.85)
ax5.axvline(0, color="white", linewidth=1.0, linestyle="--")
ax5.axvline(df["pnl"].mean(), color="#f39c12", linewidth=1.5, linestyle="-",
            label=f"平均 {df['pnl'].mean():.1f}pips")
styled_ax(ax5, "損益分布ヒストグラム（全期間）", xlabel="損益 (pips)", ylabel="頻度")
ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── 6. ドローダウン推移 ──
ax6 = fig.add_subplot(4, 2, 8)
cum_all = df_sorted["pnl"].cumsum()
rolling_max = cum_all.cummax()
drawdown = rolling_max - cum_all
ax6.fill_between(range(len(drawdown)), -drawdown.values, 0,
                 color="#e74c3c", alpha=0.6, label=f"最大DD: {drawdown.max():.0f}pips")
ax6.axhline(0, color="white", linewidth=0.5)
styled_ax(ax6, "ドローダウン推移（全期間）", xlabel="トレード番号", ylabel="ドローダウン (pips)")
ax6.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

plt.tight_layout(pad=2.5)
out_img = f"{RESULTS}/v76_full_period_backtest.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"チャート保存: {out_img}")

# ============================================================
# CSV出力
# ============================================================
# 月別サマリー
monthly_summary = df.groupby("month").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x == "win").sum()),
    pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
    period=("period", "first"),
).reset_index()
monthly_summary["win_rate"] = (monthly_summary["wins"] / monthly_summary["trades"] * 100).round(1)
monthly_summary["pnl"] = monthly_summary["pnl"].round(1)
monthly_summary["avg_pnl"] = monthly_summary["avg_pnl"].round(2)
monthly_summary.to_csv(f"{RESULTS}/v76_full_period_monthly.csv", index=False, encoding="utf-8-sig")

# 全トレード詳細
df.to_csv(f"{RESULTS}/v76_full_period_trades.csv", index=False, encoding="utf-8-sig")

# サマリー
summary_rows = []
for s in [s_all, s_is, s_oos, s_4h, s_1h]:
    if not s: continue
    summary_rows.append({
        "区分": s["label"],
        "トレード数": s["trades"],
        "勝率(%)": round(s["win_rate"], 1),
        "PF": round(s["pf"], 2) if s["pf"] != float("inf") else 999,
        "総損益(pips)": round(s["total_pnl"], 1),
        "平均利益(pips)": round(s["avg_win"], 2),
        "平均損失(pips)": round(s["avg_loss"], 2),
        "ケリー基準": round(s["kelly"], 3),
        "最大DD(pips)": round(s["max_dd"], 1),
        "月次シャープ": round(s["sharpe"], 2),
        "t統計量": round(s["t_stat"], 4),
        "p値": round(s["p_value"], 4),
        "プラス月": s["plus_months"],
    })
pd.DataFrame(summary_rows).to_csv(
    f"{RESULTS}/v76_full_period_summary.csv", index=False, encoding="utf-8-sig")

print(f"CSV保存: {RESULTS}/v76_full_period_*.csv")
print("\n完了")
