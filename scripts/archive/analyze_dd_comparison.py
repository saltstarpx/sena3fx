"""
analyze_dd_comparison.py
========================
IS期間（2024-12-01〜2025-02-28）とOOS期間（2025-03-03〜2025-06-30）の
最大ドローダウン詳細比較分析

【注意】
equity_after は各銘柄独立シミュレーション時の累積値のため、
複数銘柄を時系列順に並べた際の正しい資産曲線にはならない。
pnl_delta を exit_time 順に積み上げて正しい資産曲線を再構築する。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INIT_CASH = 1_000_000
OUT_DIR   = os.path.join(BASE_DIR, "results")
SYMBOLS   = ["USDJPY", "AUDUSD", "EURJPY", "EURGBP", "US30"]

# ── データ読み込み ─────────────────────────────────────────
def load_trades(path, label):
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["period"]     = label
    return df

is_df  = load_trades(os.path.join(OUT_DIR, "is_period_trades.csv"),  "IS（12〜2月）")
oos_df = load_trades(os.path.join(OUT_DIR, "multi_fast_trades.csv"), "OOS（3〜6月）")

print(f"IS期間トレード数:  {len(is_df)}件")
print(f"OOS期間トレード数: {len(oos_df)}件")

# ── 資産曲線・DD再構築（pnl_deltaを時系列順に積み上げ） ──
def build_equity_curve(df, init_cash):
    """
    pnl_delta を exit_time 順に積み上げて正しい資産曲線を構築する。
    Returns: (eq_series, df_sorted)
    """
    df_s = df.sort_values("exit_time").reset_index(drop=True)
    eq   = [init_cash]
    for delta in df_s["pnl_delta"]:
        eq.append(eq[-1] + delta)
    times = [df_s["exit_time"].iloc[0] - pd.Timedelta(hours=1)] + df_s["exit_time"].tolist()
    return pd.Series(eq, index=times), df_s

def calc_dd_series(eq_series):
    peak = eq_series.expanding().max()
    dd   = (eq_series - peak) / peak * 100
    return dd

is_eq,  is_sorted  = build_equity_curve(is_df,  INIT_CASH)
oos_eq, oos_sorted = build_equity_curve(oos_df, INIT_CASH)

is_dd  = calc_dd_series(is_eq)
oos_dd = calc_dd_series(oos_eq)

print(f"\nIS期間  最大DD: {is_dd.min():.2f}%  最終資産: {is_eq.iloc[-1]:,.0f}円")
print(f"OOS期間 最大DD: {oos_dd.min():.2f}%  最終資産: {oos_eq.iloc[-1]:,.0f}円")

# ── DD詳細分析 ────────────────────────────────────────────
def analyze_dd_detail(df_sorted, eq_series, dd_series, label):
    eq_arr    = eq_series.values
    dd_arr    = dd_series.values
    times_arr = eq_series.index

    mdd_idx   = int(np.argmin(dd_arr))
    mdd_val   = dd_arr[mdd_idx]
    mdd_time  = times_arr[mdd_idx]

    peak_idx  = int(np.argmax(eq_arr[:mdd_idx+1]))
    peak_val  = eq_arr[peak_idx]
    peak_time = times_arr[peak_idx]

    # 回復（谷以降でピーク超え）
    recovery_idx  = None
    recovery_time = None
    for i in range(mdd_idx, len(eq_arr)):
        if eq_arr[i] >= peak_val:
            recovery_idx  = i
            recovery_time = times_arr[i]
            break

    # ピーク→谷のトレード数・日数（インデックスは eq_series が trades+1 個）
    # eq_series[0] は初期値なので、トレードインデックスは eq_series インデックス - 1
    peak_trade_idx  = max(0, peak_idx - 1)
    mdd_trade_idx   = max(0, mdd_idx  - 1)
    p2t_trades = mdd_trade_idx - peak_trade_idx
    p2t_days   = (pd.Timestamp(mdd_time) - pd.Timestamp(peak_time)).total_seconds() / 86400

    if recovery_idx is not None:
        rec_trade_idx  = max(0, recovery_idx - 1)
        t2r_trades = rec_trade_idx - mdd_trade_idx
        t2r_days   = (pd.Timestamp(recovery_time) - pd.Timestamp(mdd_time)).total_seconds() / 86400
    else:
        t2r_trades = None
        t2r_days   = None

    # DD期間中のトレード（ピーク→谷）
    if peak_trade_idx < mdd_trade_idx and mdd_trade_idx <= len(df_sorted):
        dd_trades = df_sorted.iloc[peak_trade_idx:mdd_trade_idx].copy()
    else:
        dd_trades = df_sorted.iloc[:max(1, mdd_trade_idx)].copy()

    sl_by_sym = dd_trades[dd_trades["result"] == "SL"]["symbol"].value_counts()

    # 連敗ストリーク
    results = df_sorted["result"].tolist()
    max_streak = 0
    cur_streak = 0
    for r in results:
        if r == "SL":
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # 連敗ストリーク分布
    streaks = []
    cur = 0
    for r in results:
        if r == "SL":
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)

    return {
        "label":             label,
        "mdd_val":           mdd_val,
        "mdd_time":          mdd_time,
        "peak_val":          peak_val,
        "peak_time":         peak_time,
        "peak_idx":          peak_idx,
        "mdd_idx":           mdd_idx,
        "recovery_time":     recovery_time,
        "recovery_idx":      recovery_idx,
        "p2t_trades":        p2t_trades,
        "p2t_days":          p2t_days,
        "t2r_trades":        t2r_trades,
        "t2r_days":          t2r_days,
        "sl_by_sym":         sl_by_sym,
        "max_streak":        max_streak,
        "streaks":           streaks,
        "dd_trades":         dd_trades,
        "df_sorted":         df_sorted,
        "eq_series":         eq_series,
        "dd_series":         dd_series,
    }

is_d  = analyze_dd_detail(is_sorted,  is_eq,  is_dd,  "IS（12〜2月）")
oos_d = analyze_dd_detail(oos_sorted, oos_eq, oos_dd, "OOS（3〜6月）")

# ── テキスト出力 ──────────────────────────────────────────
for d in [is_d, oos_d]:
    print(f"\n{'='*60}")
    print(f"【{d['label']}】最大DD詳細")
    print(f"{'='*60}")
    print(f"  最大DD:                {d['mdd_val']:.2f}%")
    print(f"  ピーク時刻:            {d['peak_time']}")
    print(f"  谷（最大DD）時刻:      {d['mdd_time']}")
    print(f"  回復時刻:              {d['recovery_time'] or '未回復'}")
    print(f"  ピーク→谷 トレード数:  {d['p2t_trades']}件")
    print(f"  ピーク→谷 日数:        {d['p2t_days']:.1f}日")
    if d['t2r_trades'] is not None:
        print(f"  谷→回復 トレード数:    {d['t2r_trades']}件")
        print(f"  谷→回復 日数:          {d['t2r_days']:.1f}日")
    else:
        print(f"  谷→回復:               未回復（期末まで）")
    print(f"  最長連敗ストリーク:    {d['max_streak']}連敗")
    print(f"  DD期間中の銘柄別SL数:")
    for sym, cnt in d['sl_by_sym'].items():
        print(f"    {sym}: {cnt}件")

# ── 月次損益（pnl_delta ベース） ──────────────────────────
def get_monthly_pnl_by_sym(df_sorted):
    df_sorted = df_sorted.copy()
    df_sorted["month"] = df_sorted["exit_time"].dt.to_period("M")
    result = {}
    months = sorted(df_sorted["month"].unique())
    for sym in SYMBOLS:
        sub = df_sorted[df_sorted["symbol"] == sym]
        monthly = sub.groupby("month")["pnl_delta"].sum()
        result[sym] = {str(m): monthly.get(m, 0) for m in months}
    total = {str(m): df_sorted[df_sorted["month"] == m]["pnl_delta"].sum() for m in months}
    return result, {str(m): m for m in months}, total

is_monthly_sym,  is_months,  is_monthly_total  = get_monthly_pnl_by_sym(is_sorted)
oos_monthly_sym, oos_months, oos_monthly_total = get_monthly_pnl_by_sym(oos_sorted)

# ── 可視化 ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 26))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.35)

fig.suptitle(
    "IS期間 vs OOS期間  ドローダウン詳細比較分析\n"
    "銘柄: USDJPY / AUDUSD / EURJPY / EURGBP / US30",
    fontsize=14, fontweight="bold", y=0.99
)

# ── (1) 資産曲線（上段） ─────────────────────────────────
for col, d, color in [(0, is_d, "#3b82f6"), (1, oos_d, "#10b981")]:
    ax = fig.add_subplot(gs[0, col])
    eq = d["eq_series"].values
    ax.plot(range(len(eq)), eq / 1e6, color=color, linewidth=1.5)
    ax.axhline(INIT_CASH / 1e6, color="gray", linestyle="--", alpha=0.5)

    # ピーク・谷・回復をマーク
    ax.axvline(d["peak_idx"], color="orange", linestyle=":", lw=1.5, label="ピーク")
    ax.axvline(d["mdd_idx"],  color="red",    linestyle=":", lw=1.5, label=f"谷({d['mdd_val']:.1f}%)")
    if d["recovery_idx"] is not None:
        ax.axvline(d["recovery_idx"], color="green", linestyle=":", lw=1.5, label="回復")

    # DD区間シェード
    ax.axvspan(d["peak_idx"], d["mdd_idx"], alpha=0.12, color="red")

    final = eq[-1]
    ret   = (final - INIT_CASH) / INIT_CASH * 100
    ax.text(0.04, 0.93, f"最終: {final/1e6:.2f}M円\n{ret:+.1f}%",
            transform=ax.transAxes, fontsize=9, color=color,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_title(f"【{d['label']}】資産曲線", fontweight="bold", fontsize=10)
    ax.set_ylabel("資産（百万円）")
    ax.set_xlabel("トレード番号")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

# ── (2) DD推移（中段） ───────────────────────────────────
for col, d, color in [(0, is_d, "#3b82f6"), (1, oos_d, "#10b981")]:
    ax = fig.add_subplot(gs[1, col])
    dd = d["dd_series"].values
    ax.fill_between(range(len(dd)), dd, 0, color=color, alpha=0.45)
    ax.plot(range(len(dd)), dd, color=color, linewidth=0.8)

    # 最大DD注釈
    mi = d["mdd_idx"]
    offset_x = mi + max(5, len(dd)//12)
    ax.annotate(
        f"最大DD\n{dd[mi]:.2f}%\n(#{mi})",
        xy=(mi, dd[mi]),
        xytext=(min(offset_x, len(dd)-1), dd[mi] + abs(dd[mi]) * 0.3),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
    )

    # DD区間シェード
    ax.axvspan(d["peak_idx"], d["mdd_idx"], alpha=0.18, color="red",
               label=f"DD区間（{d['p2t_trades']}件/{d['p2t_days']:.0f}日）")
    if d["recovery_idx"] is not None:
        ax.axvspan(d["mdd_idx"], d["recovery_idx"], alpha=0.12, color="green",
                   label=f"回復（{d['t2r_trades']}件/{d['t2r_days']:.0f}日）")

    ax.set_title(f"【{d['label']}】ドローダウン推移", fontweight="bold", fontsize=10)
    ax.set_ylabel("DD（%）")
    ax.set_xlabel("トレード番号")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

# ── (3) 月次損益（銘柄別積み上げ棒グラフ） ──────────────
SYM_COLORS = {
    "USDJPY": "#ef4444", "AUDUSD": "#22c55e", "EURJPY": "#ec4899",
    "EURGBP": "#a855f7", "US30":   "#f59e0b",
}

for col, (monthly_sym, monthly_total, months_dict, label, color) in enumerate([
    (is_monthly_sym,  is_monthly_total,  is_months,  "IS（12〜2月）",  "#3b82f6"),
    (oos_monthly_sym, oos_monthly_total, oos_months, "OOS（3〜6月）", "#10b981"),
]):
    ax = fig.add_subplot(gs[2, col])
    months_str = list(months_dict.keys())
    x = np.arange(len(months_str))

    bottoms_pos = np.zeros(len(months_str))
    bottoms_neg = np.zeros(len(months_str))

    for sym in SYMBOLS:
        vals = np.array([monthly_sym[sym].get(m, 0) / 1e4 for m in months_str])
        pos_vals = np.where(vals >= 0, vals, 0)
        neg_vals = np.where(vals < 0,  vals, 0)
        ax.bar(x, pos_vals, bottom=bottoms_pos, color=SYM_COLORS[sym], alpha=0.8,
               label=sym, width=0.6)
        ax.bar(x, neg_vals, bottom=bottoms_neg, color=SYM_COLORS[sym], alpha=0.8,
               width=0.6)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    # 合計ラベル
    for i, m in enumerate(months_str):
        total = monthly_total.get(m, 0) / 1e4
        y_pos = bottoms_pos[i] + 0.5 if total >= 0 else bottoms_neg[i] - 2
        ax.text(i, y_pos, f"{total:+.0f}", ha="center", fontsize=8, fontweight="bold",
                color="black")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months_str, rotation=30, fontsize=8)
    ax.set_ylabel("損益（万円）")
    ax.set_title(f"【{label}】月次損益（銘柄別）", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.grid(alpha=0.3, axis="y")

# ── (4) DD期間中の銘柄別SL数 + 回復速度 ─────────────────
ax = fig.add_subplot(gs[3, 0])
all_syms = SYMBOLS
is_sl  = [is_d["sl_by_sym"].get(s,  0) for s in all_syms]
oos_sl = [oos_d["sl_by_sym"].get(s, 0) for s in all_syms]
x = np.arange(len(all_syms))
ax.bar(x - 0.2, is_sl,  width=0.38, color="#3b82f6", alpha=0.8, label="IS（12〜2月）")
ax.bar(x + 0.2, oos_sl, width=0.38, color="#10b981", alpha=0.8, label="OOS（3〜6月）")
for i, (iv, ov) in enumerate(zip(is_sl, oos_sl)):
    if iv > 0: ax.text(i - 0.2, iv + 0.05, str(iv), ha="center", fontsize=9, color="#1d4ed8")
    if ov > 0: ax.text(i + 0.2, ov + 0.05, str(ov), ha="center", fontsize=9, color="#065f46")
ax.set_xticks(x)
ax.set_xticklabels(all_syms, fontsize=9)
ax.set_ylabel("SL件数")
ax.set_title("DD期間中（ピーク→谷）の銘柄別SL数", fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ── (5) 回復速度比較 ─────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])
cats = ["ピーク→谷\nトレード数", "ピーク→谷\n日数", "谷→回復\nトレード数", "谷→回復\n日数"]
is_v  = [is_d["p2t_trades"],  is_d["p2t_days"],
         is_d["t2r_trades"]  or 0, is_d["t2r_days"]  or 0]
oos_v = [oos_d["p2t_trades"], oos_d["p2t_days"],
         oos_d["t2r_trades"] or 0, oos_d["t2r_days"] or 0]
x = np.arange(len(cats))
ax.bar(x - 0.2, is_v,  width=0.38, color="#3b82f6", alpha=0.8, label="IS（12〜2月）")
ax.bar(x + 0.2, oos_v, width=0.38, color="#10b981", alpha=0.8, label="OOS（3〜6月）")
for i, (iv, ov) in enumerate(zip(is_v, oos_v)):
    ax.text(i - 0.2, iv + 0.3, f"{iv:.0f}", ha="center", fontsize=9, color="#1d4ed8")
    ax.text(i + 0.2, ov + 0.3, f"{ov:.0f}", ha="center", fontsize=9, color="#065f46")
# 未回復ラベル
if is_d["t2r_trades"] is None:
    ax.text(2 - 0.2, 1, "未回復", ha="center", fontsize=8, color="red", rotation=90)
    ax.text(3 - 0.2, 1, "未回復", ha="center", fontsize=8, color="red", rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("件数 / 日数")
ax.set_title("最大DD 発生・回復速度比較", fontweight="bold", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ── (6) 銘柄別勝率比較（下段） ───────────────────────────
ax = fig.add_subplot(gs[4, :])

def sym_stats(df):
    stats = {}
    for sym in SYMBOLS:
        sub = df[df["symbol"] == sym]
        n_tp = (sub["result"] == "TP").sum()
        n_sl = (sub["result"] == "SL").sum()
        n_be = (sub["result"] == "BE").sum()
        wr   = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0
        stats[sym] = {"wr": wr * 100, "n": len(sub), "tp": n_tp, "sl": n_sl, "be": n_be}
    return stats

is_stats  = sym_stats(is_df)
oos_stats = sym_stats(oos_df)

x = np.arange(len(SYMBOLS))
is_wrs  = [is_stats[s]["wr"]  for s in SYMBOLS]
oos_wrs = [oos_stats[s]["wr"] for s in SYMBOLS]
is_ns   = [is_stats[s]["n"]   for s in SYMBOLS]
oos_ns  = [oos_stats[s]["n"]  for s in SYMBOLS]

ax.bar(x - 0.2, is_wrs,  width=0.38, color="#3b82f6", alpha=0.8, label="IS（12〜2月）")
ax.bar(x + 0.2, oos_wrs, width=0.38, color="#10b981", alpha=0.8, label="OOS（3〜6月）")
for i, (iw, ow, iN, oN) in enumerate(zip(is_wrs, oos_wrs, is_ns, oos_ns)):
    ax.text(i - 0.2, iw + 0.8,
            f"{iw:.0f}%\n(n={iN}\nTP:{is_stats[SYMBOLS[i]]['tp']} SL:{is_stats[SYMBOLS[i]]['sl']})",
            ha="center", fontsize=7.5, color="#1d4ed8")
    ax.text(i + 0.2, ow + 0.8,
            f"{ow:.0f}%\n(n={oN}\nTP:{oos_stats[SYMBOLS[i]]['tp']} SL:{oos_stats[SYMBOLS[i]]['sl']})",
            ha="center", fontsize=7.5, color="#065f46")

ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50%ライン")
ax.set_xticks(x)
ax.set_xticklabels(SYMBOLS, fontsize=11)
ax.set_ylabel("勝率（BE除く）%")
ax.set_ylim(0, 110)
ax.set_title("銘柄別勝率比較（IS vs OOS）", fontweight="bold", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.98])
out_path = os.path.join(OUT_DIR, "dd_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n比較チャート保存: {out_path}")

# ── 最終サマリー出力 ──────────────────────────────────────
print("\n" + "="*72)
print("【IS vs OOS 最大DD 詳細比較サマリー】")
print("="*72)
print(f"{'指標':<32} {'IS（12〜2月）':>18} {'OOS（3〜6月）':>18}")
print("-"*72)
rows = [
    ("最大DD",                    f"{is_d['mdd_val']:.2f}%",        f"{oos_d['mdd_val']:.2f}%"),
    ("ピーク時刻",                 str(is_d['peak_time'])[:16],       str(oos_d['peak_time'])[:16]),
    ("谷（最大DD）時刻",           str(is_d['mdd_time'])[:16],        str(oos_d['mdd_time'])[:16]),
    ("回復時刻",
        "未回復（期末）" if is_d['recovery_time'] is None else str(is_d['recovery_time'])[:16],
        str(oos_d['recovery_time'])[:16] if oos_d['recovery_time'] else "未回復"),
    ("ピーク→谷 トレード数",       f"{is_d['p2t_trades']}件",         f"{oos_d['p2t_trades']}件"),
    ("ピーク→谷 日数",             f"{is_d['p2t_days']:.1f}日",       f"{oos_d['p2t_days']:.1f}日"),
    ("谷→回復 トレード数",
        "未回復" if is_d['t2r_trades'] is None else f"{is_d['t2r_trades']}件",
        f"{oos_d['t2r_trades']}件" if oos_d['t2r_trades'] else "未回復"),
    ("谷→回復 日数",
        "未回復" if is_d['t2r_days'] is None else f"{is_d['t2r_days']:.1f}日",
        f"{oos_d['t2r_days']:.1f}日" if oos_d['t2r_days'] else "未回復"),
    ("最長連敗ストリーク",          f"{is_d['max_streak']}連敗",       f"{oos_d['max_streak']}連敗"),
    ("採用トレード総数",            f"{len(is_df)}件",                 f"{len(oos_df)}件"),
    ("最終資産",                   f"{is_eq.iloc[-1]/1e4:.0f}万円",   f"{oos_eq.iloc[-1]/1e4:.0f}万円"),
    ("リターン",                   f"{(is_eq.iloc[-1]-INIT_CASH)/INIT_CASH*100:+.1f}%",
                                   f"{(oos_eq.iloc[-1]-INIT_CASH)/INIT_CASH*100:+.1f}%"),
]
for lbl, iv, ov in rows:
    print(f"  {lbl:<30} {iv:>18} {ov:>18}")
print("="*72)

print("\n【DD期間中（ピーク→谷）の銘柄別SL数】")
print(f"  {'銘柄':<10} {'IS':>8} {'OOS':>8}")
print("  " + "-"*28)
for sym in SYMBOLS:
    iv = is_d['sl_by_sym'].get(sym, 0)
    ov = oos_d['sl_by_sym'].get(sym, 0)
    print(f"  {sym:<10} {iv:>8} {ov:>8}")

print("\n【銘柄別勝率比較】")
print(f"  {'銘柄':<10} {'IS勝率':>8} {'IS件数':>8} {'OOS勝率':>8} {'OOS件数':>8}")
print("  " + "-"*46)
for sym in SYMBOLS:
    print(f"  {sym:<10} {is_stats[sym]['wr']:>7.0f}% {is_stats[sym]['n']:>8} "
          f"{oos_stats[sym]['wr']:>7.0f}% {oos_stats[sym]['n']:>8}")
