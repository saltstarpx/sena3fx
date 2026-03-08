"""
compare_v75_v76.py
==================
v75 vs v76 パフォーマンス比較分析

【v75→v76の変更点】
v75（旧実装）:
  - ep = 始値 + spread（ロング）
  - risk = ep - sl  ← スプレッドがriskに含まれる
  - tp = ep + risk × RR  ← スプレッドがTPにも影響
  結果: スプレッドを変えてもrisk/TPが連動して変わるため損益がほぼ変わらない

v76（新実装）:
  - raw_ep = 始値（チャートレベル）
  - risk = raw_ep - sl  ← スプレッドを含まない純粋なリスク幅
  - tp = raw_ep + risk × RR  ← TPもチャートレベルで固定
  - ep = raw_ep + spread（実際の約定価格）
  結果: スプレッドが増えるほど損益が悪化する（実環境に近い）

比較スプレッド: 0.2pips / 0.4pips / 0.8pips / 1.5pips
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# v75 / v76 をそれぞれロード
import importlib.util

def load_strategy(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v75 = load_strategy("/home/ubuntu/sena3fx/strategies/archive/yagami_mtf_v75.py", "v75")
v76 = load_strategy("/home/ubuntu/sena3fx/strategies/current/yagami_mtf_v76.py",  "v76")

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"

# ============================================================
# データ読み込み
# ============================================================
def load(p):
    df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

print("データ読み込み中...")
is_1m  = load(f"{DATA}/usdjpy_is_1m.csv")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h  = load(f"{DATA}/usdjpy_is_4h.csv")
oos_1m  = load(f"{DATA}/usdjpy_oos_1m.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h  = load(f"{DATA}/usdjpy_oos_4h.csv")
print(f"IS  1m: {len(is_1m):,}行  4h: {len(is_4h)}本")
print(f"OOS 1m: {len(oos_1m):,}行  4h: {len(oos_4h)}本")

# ============================================================
# バックテストエンジン（共通）
# ============================================================
def run_backtest(strategy_mod, data_1m, data_15m, data_4h, spread, label):
    print(f"  [{label}] spread={spread*100:.2f}pips シグナル生成中...", flush=True)
    signals = strategy_mod.generate_signals(data_1m, data_15m, data_4h, spread_pips=spread)
    sig_map = {s["time"]: s for s in signals}
    print(f"  [{label}] シグナル数: {len(signals)}")

    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        if pos is not None:
            d = pos["dir"]
            # 半利確ライン（1R到達）
            raw_ep = pos["ep"] - pos.get("spread", 0) * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True
            # SL
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + sl_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "SL" if not pos["half_closed"] else "HALF+SL",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "tf": pos.get("tf", "?"),
                })
                pos = None; continue
            # TP
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + tp_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "TP" if not pos["half_closed"] else "HALF+TP",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "tf": pos.get("tf", "?"),
                })
                pos = None; continue
        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)

# ============================================================
# 統計計算
# ============================================================
def calc_stats(df, label):
    if df.empty:
        return dict(label=label, trades=0, win_rate=0, pf=0,
                    total_pnl=0, avg_win=0, avg_loss=0,
                    kelly=0, t_stat=0, p_value=1.0,
                    plus_months="0/0", monthly=pd.Series(dtype=float),
                    max_dd=0, sharpe=0)
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df) * 100
    avg_w  = wins["pnl"].mean()  if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0)
    monthly      = df.groupby("month")["pnl"].sum()
    plus_months  = (monthly > 0).sum()
    total_months = len(monthly)
    # 最大ドローダウン
    cumulative = df.sort_values("entry_time")["pnl"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown = rolling_max - cumulative
    max_dd = drawdown.max()
    # シャープレシオ（月次）
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    return dict(label=label, trades=len(df), win_rate=wr, pf=pf,
                total_pnl=df["pnl"].sum(), avg_win=avg_w, avg_loss=avg_l,
                kelly=kelly, t_stat=t_stat, p_value=p_val,
                plus_months=f"{plus_months}/{total_months}", monthly=monthly,
                max_dd=max_dd, sharpe=sharpe)

# ============================================================
# 実験実行: スプレッド別 v75 vs v76
# ============================================================
spreads = [0.2, 0.4, 0.8, 1.5]
results = {}

for sp in spreads:
    print(f"\n{'='*60}")
    print(f"スプレッド {sp}pips")
    print(f"{'='*60}")

    # IS
    df_v75_is  = run_backtest(v75, is_1m,  is_15m,  is_4h,  sp, f"v75_IS_{sp}")
    df_v76_is  = run_backtest(v76, is_1m,  is_15m,  is_4h,  sp, f"v76_IS_{sp}")
    # OOS
    df_v75_oos = run_backtest(v75, oos_1m, oos_15m, oos_4h, sp, f"v75_OOS_{sp}")
    df_v76_oos = run_backtest(v76, oos_1m, oos_15m, oos_4h, sp, f"v76_OOS_{sp}")

    df_v75_all = pd.concat([df_v75_is, df_v75_oos], ignore_index=True)
    df_v76_all = pd.concat([df_v76_is, df_v76_oos], ignore_index=True)

    results[sp] = {
        "v75_is":  calc_stats(df_v75_is,  f"v75_IS_{sp}"),
        "v75_oos": calc_stats(df_v75_oos, f"v75_OOS_{sp}"),
        "v75_all": calc_stats(df_v75_all, f"v75_ALL_{sp}"),
        "v76_is":  calc_stats(df_v76_is,  f"v76_IS_{sp}"),
        "v76_oos": calc_stats(df_v76_oos, f"v76_OOS_{sp}"),
        "v76_all": calc_stats(df_v76_all, f"v76_ALL_{sp}"),
        "df_v75_all": df_v75_all,
        "df_v76_all": df_v76_all,
    }

# ============================================================
# サマリー出力
# ============================================================
print("\n\n" + "="*80)
print("比較サマリー（IS+OOS統合）")
print("="*80)
print(f"{'スプレッド':>8}  {'バージョン':>6}  {'トレード':>6}  {'勝率':>6}  {'PF':>5}  "
      f"{'総損益':>10}  {'ケリー':>6}  {'最大DD':>8}  {'シャープ':>7}  {'p値':>8}  {'プラス月':>8}")
print("-"*80)
for sp in spreads:
    for ver in ["v75", "v76"]:
        s = results[sp][f"{ver}_all"]
        pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "∞"
        print(f"{sp:>7}p  {ver:>6}  {s['trades']:>6}回  {s['win_rate']:>5.1f}%  {pf_str:>5}  "
              f"{s['total_pnl']:>+10.1f}  {s['kelly']:>6.3f}  {s['max_dd']:>8.1f}  "
              f"{s['sharpe']:>7.2f}  {s['p_value']:>8.4f}  {s['plus_months']:>8}")
    print()

# ============================================================
# 可視化
# ============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0d0d1a")

# カラーパレット
c_v75 = "#e74c3c"   # 赤: v75
c_v76 = "#2ecc71"   # 緑: v76

# --- 上段: スプレッド別PF比較 ---
ax1 = fig.add_subplot(4, 3, 1)
ax2 = fig.add_subplot(4, 3, 2)
ax3 = fig.add_subplot(4, 3, 3)

def styled_ax(ax, title):
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.yaxis.label.set_color("white")

# PF vs スプレッド
pf_v75 = [results[sp]["v75_all"]["pf"] for sp in spreads]
pf_v76 = [results[sp]["v76_all"]["pf"] for sp in spreads]
x = np.arange(len(spreads))
w = 0.35
ax1.bar(x - w/2, pf_v75, w, label="v75", color=c_v75, alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.bar(x + w/2, pf_v76, w, label="v76", color=c_v76, alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{sp}p" for sp in spreads])
ax1.set_ylabel("PF", color="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax1, "PF vs スプレッド（IS+OOS）")

# 総損益 vs スプレッド
pnl_v75 = [results[sp]["v75_all"]["total_pnl"] for sp in spreads]
pnl_v76 = [results[sp]["v76_all"]["total_pnl"] for sp in spreads]
ax2.bar(x - w/2, pnl_v75, w, label="v75", color=c_v75, alpha=0.85, edgecolor="white", linewidth=0.4)
ax2.bar(x + w/2, pnl_v76, w, label="v76", color=c_v76, alpha=0.85, edgecolor="white", linewidth=0.4)
ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{sp}p" for sp in spreads])
ax2.set_ylabel("総損益 (pips)", color="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax2, "総損益 vs スプレッド（IS+OOS）")

# ケリー基準 vs スプレッド
kelly_v75 = [results[sp]["v75_all"]["kelly"] for sp in spreads]
kelly_v76 = [results[sp]["v76_all"]["kelly"] for sp in spreads]
ax3.bar(x - w/2, kelly_v75, w, label="v75", color=c_v75, alpha=0.85, edgecolor="white", linewidth=0.4)
ax3.bar(x + w/2, kelly_v76, w, label="v76", color=c_v76, alpha=0.85, edgecolor="white", linewidth=0.4)
ax3.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels([f"{sp}p" for sp in spreads])
ax3.set_ylabel("ケリー基準", color="white")
ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax3, "ケリー基準 vs スプレッド（IS+OOS）")

# --- 中段: IS vs OOS 別PF比較（0.4pipsのみ）---
ax4 = fig.add_subplot(4, 3, 4)
ax5 = fig.add_subplot(4, 3, 5)
ax6 = fig.add_subplot(4, 3, 6)

sp_main = 0.4
r = results[sp_main]

# IS PF
bars_is = ax4.bar(["v75 IS", "v76 IS"], [r["v75_is"]["pf"], r["v76_is"]["pf"]],
                   color=[c_v75, c_v76], edgecolor="white", linewidth=0.5, width=0.4)
ax4.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, val in zip(bars_is, [r["v75_is"]["pf"], r["v76_is"]["pf"]]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}", ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax4.set_ylabel("PF", color="white")
styled_ax(ax4, f"IS期間 PF比較（spread={sp_main}p）")

# OOS PF
bars_oos = ax5.bar(["v75 OOS", "v76 OOS"], [r["v75_oos"]["pf"], r["v76_oos"]["pf"]],
                    color=[c_v75, c_v76], edgecolor="white", linewidth=0.5, width=0.4)
ax5.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, val in zip(bars_oos, [r["v75_oos"]["pf"], r["v76_oos"]["pf"]]):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}", ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax5.set_ylabel("PF", color="white")
styled_ax(ax5, f"OOS期間 PF比較（spread={sp_main}p）")

# 最大ドローダウン
bars_dd = ax6.bar(["v75", "v76"], [r["v75_all"]["max_dd"], r["v76_all"]["max_dd"]],
                   color=[c_v75, c_v76], edgecolor="white", linewidth=0.5, width=0.4)
for bar, val in zip(bars_dd, [r["v75_all"]["max_dd"], r["v76_all"]["max_dd"]]):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.0f}p", ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax6.set_ylabel("最大ドローダウン (pips)", color="white")
styled_ax(ax6, f"最大ドローダウン比較（spread={sp_main}p）")

# --- 下段: エクイティカーブ比較（0.4pips）---
ax7 = fig.add_subplot(4, 1, 3)

df_v75_sorted = r["df_v75_all"].sort_values("entry_time")
df_v76_sorted = r["df_v76_all"].sort_values("entry_time")
cum_v75 = df_v75_sorted["pnl"].cumsum().values
cum_v76 = df_v76_sorted["pnl"].cumsum().values

ax7.plot(range(len(cum_v75)), cum_v75, color=c_v75, linewidth=1.5,
         label=f"v75  PF:{r['v75_all']['pf']:.2f}  {r['v75_all']['trades']}回  +{r['v75_all']['total_pnl']:.0f}pips", alpha=0.85)
ax7.plot(range(len(cum_v76)), cum_v76, color=c_v76, linewidth=1.5,
         label=f"v76  PF:{r['v76_all']['pf']:.2f}  {r['v76_all']['trades']}回  +{r['v76_all']['total_pnl']:.0f}pips", alpha=0.85)
ax7.axhline(0, color="white", linewidth=0.5, linestyle="--")
is_n_v76 = len(r["df_v76_all"][r["df_v76_all"]["entry_time"].astype(str) < "2025-03"])
ax7.axvline(is_n_v76, color="#f39c12", linewidth=1.5, linestyle="--", label="IS/OOS境界", alpha=0.7)
ax7.set_title(f"エクイティカーブ比較（IS+OOS / spread={sp_main}pips）", color="white", fontsize=11)
ax7.set_xlabel("トレード番号", color="white")
ax7.set_ylabel("累積損益 (pips)", color="white")
ax7.set_facecolor("#1a1a2e")
ax7.tick_params(colors="white")
ax7.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
for spine in ax7.spines.values():
    spine.set_edgecolor("#444")

# --- 最下段: 月別損益比較（0.4pips）---
ax8 = fig.add_subplot(4, 1, 4)

monthly_v75 = r["v75_all"]["monthly"]
monthly_v76 = r["v76_all"]["monthly"]
all_months = sorted(set(monthly_v75.index) | set(monthly_v76.index))
v75_vals = [monthly_v75.get(m, 0) for m in all_months]
v76_vals = [monthly_v76.get(m, 0) for m in all_months]
x_m = np.arange(len(all_months))
w_m = 0.4
ax8.bar(x_m - w_m/2, v75_vals, w_m, label="v75", color=c_v75, alpha=0.75, edgecolor="white", linewidth=0.3)
ax8.bar(x_m + w_m/2, v76_vals, w_m, label="v76", color=c_v76, alpha=0.75, edgecolor="white", linewidth=0.3)
ax8.axhline(0, color="white", linewidth=0.6)
ax8.set_xticks(x_m)
ax8.set_xticklabels(all_months, rotation=45, fontsize=7)
ax8.set_title(f"月別損益比較（IS+OOS / spread={sp_main}pips）", color="white", fontsize=11)
ax8.set_ylabel("損益 (pips)", color="white")
ax8.set_facecolor("#1a1a2e")
ax8.tick_params(colors="white")
ax8.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
for spine in ax8.spines.values():
    spine.set_edgecolor("#444")

plt.tight_layout(pad=2.5)
out_img = f"{RESULTS}/v75_vs_v76_comparison.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

# ============================================================
# 詳細サマリーCSV
# ============================================================
rows = []
for sp in spreads:
    for ver in ["v75", "v76"]:
        for period in ["is", "oos", "all"]:
            s = results[sp][f"{ver}_{period}"]
            rows.append({
                "バージョン": ver,
                "スプレッド(pips)": sp,
                "期間": period.upper(),
                "トレード数": s["trades"],
                "勝率(%)": round(s["win_rate"], 1),
                "PF": round(s["pf"], 2) if s["pf"] != float("inf") else 999,
                "総損益(pips)": round(s["total_pnl"], 1),
                "平均利益(pips)": round(s["avg_win"], 2),
                "平均損失(pips)": round(s["avg_loss"], 2),
                "ケリー基準": round(s["kelly"], 3),
                "最大DD(pips)": round(s["max_dd"], 1),
                "シャープレシオ(月次)": round(s["sharpe"], 2),
                "t統計量": round(s["t_stat"], 4),
                "p値": round(s["p_value"], 4),
                "プラス月": s["plus_months"],
            })

df_summary = pd.DataFrame(rows)
out_csv = f"{RESULTS}/v75_vs_v76_summary.csv"
df_summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"サマリーCSV保存: {out_csv}")
print("\n完了")
