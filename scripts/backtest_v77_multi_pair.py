"""
backtest_v77_multi_pair.py
==========================
v77ロジック（KMID+KLOWフィルター）で
USDJPY / EURJPY / GBPJPY の3通貨ペアを共通期間（2025/1〜2026/2）でバックテスト。

スプレッド設定:
  USDJPY: 0.4pips
  EURJPY: 0.7pips
  GBPJPY: 0.9pips
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies/current")
from yagami_mtf_v77 import generate_signals

# ─── 設定 ──────────────────────────────────────────────────────────────────
PAIRS = {
    "USDJPY": {
        "data_1m":  "/home/ubuntu/sena3fx/data/usdjpy_oos_1m.csv",
        "data_15m": "/home/ubuntu/sena3fx/data/usdjpy_oos_15m.csv",
        "data_4h":  "/home/ubuntu/sena3fx/data/usdjpy_oos_4h.csv",
        "spread":   0.4,
        "color":    "#3b82f6",
    },
    "EURJPY": {
        "data_1m":  "/home/ubuntu/sena3fx/data/eurjpy_1m.csv",
        "data_15m": "/home/ubuntu/sena3fx/data/eurjpy_15m.csv",
        "data_4h":  "/home/ubuntu/sena3fx/data/eurjpy_4h.csv",
        "spread":   0.7,
        "color":    "#10b981",
    },
    "GBPJPY": {
        "data_1m":  "/home/ubuntu/sena3fx/data/gbpjpy_1m.csv",
        "data_15m": "/home/ubuntu/sena3fx/data/gbpjpy_15m.csv",
        "data_4h":  "/home/ubuntu/sena3fx/data/gbpjpy_4h.csv",
        "spread":   0.9,
        "color":    "#f59e0b",
    },
}

# 共通期間
START_DATE = "2025-01-01"
END_DATE   = "2026-02-28"

# 半利確設定
HALF_CLOSE_R = 1.0   # 1R到達で半分決済

OUT_DIR = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 日本語フォント設定 ─────────────────────────────────────────────────────
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf",
]
jp_font = None
for f in font_candidates:
    if os.path.exists(f):
        jp_font = fm.FontProperties(fname=f)
        plt.rcParams["font.family"] = jp_font.get_name()
        break
if jp_font is None:
    plt.rcParams["font.family"] = "DejaVu Sans"


def load_data(path, start=None, end=None):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize(None)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def simulate_trades(signals, data_1m):
    """シグナルリストからトレードを模擬実行（半利確あり）"""
    trades = []
    for sig in signals:
        ep    = sig["ep"]
        sl    = sig["sl"]
        tp    = sig["tp"]
        risk  = sig["risk"]
        d     = sig["dir"]
        entry_time = sig["time"]

        half_tp = ep + d * risk * HALF_CLOSE_R  # 1R到達で半利確

        future = data_1m[data_1m.index > entry_time]
        if len(future) == 0:
            continue

        result = None
        half_done = False
        half_exit_price = None
        half_exit_time  = None

        for bar_time, bar in future.iterrows():
            lo = bar["low"]
            hi = bar["high"]

            # 半利確チェック
            if not half_done:
                if d == 1 and hi >= half_tp:
                    half_done = True
                    half_exit_price = half_tp
                    half_exit_time  = bar_time
                elif d == -1 and lo <= half_tp:
                    half_done = True
                    half_exit_price = half_tp
                    half_exit_time  = bar_time

            # SLチェック（半利確後はBEに移動）
            effective_sl = ep if half_done else sl
            if d == 1 and lo <= effective_sl:
                exit_price = effective_sl
                exit_time  = bar_time
                result = "SL"
                break
            elif d == -1 and hi >= effective_sl:
                exit_price = effective_sl
                exit_time  = bar_time
                result = "SL"
                break

            # TPチェック
            if d == 1 and hi >= tp:
                exit_price = tp
                exit_time  = bar_time
                result = "TP"
                break
            elif d == -1 and lo <= tp:
                exit_price = tp
                exit_time  = bar_time
                result = "TP"
                break

        if result is None:
            continue

        # 損益計算（pips）
        if half_done:
            pnl_half = (half_exit_price - ep) * d * 100
            pnl_rest = (exit_price - ep) * d * 100
            pnl_total = pnl_half * 0.5 + pnl_rest * 0.5
        else:
            pnl_total = (exit_price - ep) * d * 100

        trades.append({
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "dir":         "LONG" if d == 1 else "SHORT",
            "tf":          sig["tf"],
            "ep":          ep,
            "sl":          sl,
            "tp":          tp,
            "exit_price":  exit_price,
            "result":      result,
            "half_done":   half_done,
            "pnl_pips":    round(pnl_total, 2),
            "win":         pnl_total > 0,
        })

    return pd.DataFrame(trades)


def calc_stats(df):
    if len(df) == 0:
        return {}
    wins  = df[df["win"]]
    loses = df[~df["win"]]
    total_pnl = df["pnl_pips"].sum()
    win_rate  = len(wins) / len(df)
    gross_win = wins["pnl_pips"].sum()
    gross_los = abs(loses["pnl_pips"].sum())
    pf = gross_win / gross_los if gross_los > 0 else float("inf")

    # 月次
    df2 = df.copy()
    df2["month"] = df2["entry_time"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl_pips"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)

    # 最大DD
    cumsum = df["pnl_pips"].cumsum()
    running_max = cumsum.cummax()
    dd = running_max - cumsum
    max_dd = dd.max()

    # シャープ（月次）
    monthly_std = monthly.std()
    monthly_mean = monthly.mean()
    sharpe = (monthly_mean / monthly_std * np.sqrt(12)) if monthly_std > 0 else 0

    # ケリー
    if win_rate > 0 and (1 - win_rate) > 0:
        avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
        avg_los = abs(loses["pnl_pips"].mean()) if len(loses) > 0 else 1
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_los) if avg_los > 0 else 0
    else:
        kelly = 0

    # 最大連敗
    results = df["win"].tolist()
    max_consec_loss = 0
    cur = 0
    for r in results:
        if not r:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    # p値（t検定）
    t_stat, p_val = stats.ttest_1samp(df["pnl_pips"], 0)
    p_val = p_val / 2 if t_stat > 0 else 1.0

    return {
        "trades":        len(df),
        "win_rate":      win_rate,
        "pf":            pf,
        "total_pnl":     total_pnl,
        "max_dd":        max_dd,
        "sharpe":        sharpe,
        "kelly":         kelly,
        "plus_months":   plus_months,
        "total_months":  total_months,
        "max_consec_loss": max_consec_loss,
        "p_value":       p_val,
        "monthly":       monthly,
    }


# ─── メイン処理 ─────────────────────────────────────────────────────────────
all_results = {}
all_trades  = {}

for pair, cfg in PAIRS.items():
    print(f"\n{'='*50}")
    print(f"[{pair}] データ読み込み中...")
    d1m  = load_data(cfg["data_1m"],  START_DATE, END_DATE)
    d15m = load_data(cfg["data_15m"], START_DATE, END_DATE)
    d4h  = load_data(cfg["data_4h"],  START_DATE, END_DATE)
    print(f"  1m: {len(d1m)}行, 15m: {len(d15m)}行, 4h: {len(d4h)}行")

    print(f"[{pair}] シグナル生成中 (spread={cfg['spread']}pips)...")
    signals = generate_signals(d1m, d15m, d4h, spread_pips=cfg["spread"])
    print(f"  シグナル数: {len(signals)}")

    print(f"[{pair}] トレード模擬実行中...")
    trades_df = simulate_trades(signals, d1m)
    print(f"  完了トレード数: {len(trades_df)}")

    stats_dict = calc_stats(trades_df)
    all_results[pair] = stats_dict
    all_trades[pair]  = trades_df

    if len(trades_df) > 0:
        print(f"  勝率: {stats_dict['win_rate']:.1%}, PF: {stats_dict['pf']:.2f}, "
              f"総損益: {stats_dict['total_pnl']:+.1f}pips, "
              f"MDD: {stats_dict['max_dd']:.1f}pips")

# ─── 結果保存 ────────────────────────────────────────────────────────────────
print("\n[可視化] チャート生成中...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0f1117")

# タイトル
fig.suptitle("v77 Multi-Pair Backtest  USDJPY / EURJPY / GBPJPY  (2025/1〜2026/2)",
             fontsize=16, color="white", y=0.98,
             fontproperties=jp_font if jp_font else None)

# ── 1. 累積損益曲線（3ペア比較）
ax1 = fig.add_axes([0.05, 0.60, 0.55, 0.30])
ax1.set_facecolor("#1a1d27")
ax1.set_title("累積損益曲線 (pips)", color="white", fontsize=12,
              fontproperties=jp_font if jp_font else None)

for pair, cfg in PAIRS.items():
    df = all_trades[pair]
    if len(df) == 0:
        continue
    cumsum = df["pnl_pips"].cumsum().reset_index(drop=True)
    ax1.plot(cumsum.index, cumsum.values, color=cfg["color"],
             linewidth=2, label=pair)
    ax1.axhline(0, color="#555", linewidth=0.5, linestyle="--")

ax1.legend(facecolor="#1a1d27", labelcolor="white", fontsize=10)
ax1.tick_params(colors="gray")
ax1.spines[:].set_color("#333")
ax1.set_xlabel("トレード番号", color="gray", fontproperties=jp_font if jp_font else None)
ax1.set_ylabel("累積損益 (pips)", color="gray", fontproperties=jp_font if jp_font else None)

# ── 2. KPIサマリーテーブル
ax2 = fig.add_axes([0.62, 0.60, 0.36, 0.30])
ax2.set_facecolor("#1a1d27")
ax2.axis("off")
ax2.set_title("パフォーマンス比較", color="white", fontsize=12,
              fontproperties=jp_font if jp_font else None)

rows = []
for pair in PAIRS:
    s = all_results[pair]
    if not s:
        continue
    rows.append([
        pair,
        str(s["trades"]),
        f"{s['win_rate']:.1%}",
        f"{s['pf']:.2f}",
        f"{s['total_pnl']:+.0f}p",
        f"{s['max_dd']:.0f}p",
        f"{s['sharpe']:.2f}",
        f"{s['kelly']:.3f}",
        f"{s['plus_months']}/{s['total_months']}",
    ])

headers = ["ペア", "取引数", "勝率", "PF", "総損益", "MDD", "シャープ", "ケリー", "月プラス"]
table = ax2.table(
    cellText=rows,
    colLabels=headers,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#1a1d27" if r > 0 else "#2a2d3a")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r > 0 and c == 4:  # 総損益列
        val = rows[r-1][4]
        cell.set_text_props(color="#10b981" if val.startswith("+") else "#ef4444")

# ── 3. 月次損益棒グラフ（3ペア）
colors_bar = {"USDJPY": "#3b82f6", "EURJPY": "#10b981", "GBPJPY": "#f59e0b"}
for idx, (pair, cfg) in enumerate(PAIRS.items()):
    s = all_results[pair]
    if not s or "monthly" not in s:
        continue
    monthly = s["monthly"]
    ax = fig.add_axes([0.05 + idx * 0.32, 0.30, 0.28, 0.22])
    ax.set_facecolor("#1a1d27")
    ax.set_title(f"{pair} 月次損益", color="white", fontsize=10,
                 fontproperties=jp_font if jp_font else None)
    bar_colors = [colors_bar[pair] if v >= 0 else "#ef4444" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="#555", linewidth=0.5)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right",
                       fontsize=7, color="gray")
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")
    ax.set_ylabel("pips", color="gray", fontproperties=jp_font if jp_font else None)

# ── 4. 方向別・TF別勝率比較
ax4 = fig.add_axes([0.05, 0.05, 0.55, 0.18])
ax4.set_facecolor("#1a1d27")
ax4.set_title("TF別・方向別 勝率", color="white", fontsize=10,
              fontproperties=jp_font if jp_font else None)

x_labels = []
x_vals   = []
x_colors = []
x_pos    = 0
xtick_pos = []
xtick_lbl = []

for pair, cfg in PAIRS.items():
    df = all_trades[pair]
    if len(df) == 0:
        continue
    for tf in ["4h", "1h"]:
        for direction in ["LONG", "SHORT"]:
            sub = df[(df["tf"] == tf) & (df["dir"] == direction)]
            if len(sub) < 5:
                x_pos += 1
                continue
            wr = sub["win"].mean()
            lbl = f"{pair}\n{tf}\n{direction}"
            bar = ax4.bar(x_pos, wr, color=cfg["color"], alpha=0.8, width=0.7)
            ax4.text(x_pos, wr + 0.02, f"{wr:.0%}\n({len(sub)})", ha="center",
                     fontsize=7, color="white")
            xtick_pos.append(x_pos)
            xtick_lbl.append(lbl)
            x_pos += 1
    x_pos += 0.5  # ペア間スペース

ax4.axhline(0.5, color="#555", linewidth=0.8, linestyle="--")
ax4.set_xticks(xtick_pos)
ax4.set_xticklabels(xtick_lbl, fontsize=7, color="gray")
ax4.set_ylim(0, 1.1)
ax4.set_ylabel("勝率", color="gray", fontproperties=jp_font if jp_font else None)
ax4.tick_params(colors="gray")
ax4.spines[:].set_color("#333")

# ── 5. 3ペア合算の累積損益
ax5 = fig.add_axes([0.62, 0.05, 0.36, 0.18])
ax5.set_facecolor("#1a1d27")
ax5.set_title("3ペア合算 累積損益", color="white", fontsize=10,
              fontproperties=jp_font if jp_font else None)

# 全トレードを時刻順に結合
all_df_list = []
for pair, df in all_trades.items():
    if len(df) > 0:
        tmp = df[["entry_time", "pnl_pips"]].copy()
        tmp["pair"] = pair
        all_df_list.append(tmp)

if all_df_list:
    combined = pd.concat(all_df_list).sort_values("entry_time").reset_index(drop=True)
    combined_cum = combined["pnl_pips"].cumsum()
    ax5.plot(combined_cum.index, combined_cum.values, color="#a78bfa", linewidth=2)
    ax5.axhline(0, color="#555", linewidth=0.5, linestyle="--")
    ax5.fill_between(combined_cum.index, 0, combined_cum.values,
                     where=combined_cum.values >= 0, alpha=0.15, color="#10b981")
    ax5.fill_between(combined_cum.index, 0, combined_cum.values,
                     where=combined_cum.values < 0, alpha=0.15, color="#ef4444")
    total_combined = combined["pnl_pips"].sum()
    ax5.set_title(f"3ペア合算 累積損益 ({total_combined:+.0f}pips)", color="white", fontsize=10,
                  fontproperties=jp_font if jp_font else None)

ax5.tick_params(colors="gray")
ax5.spines[:].set_color("#333")
ax5.set_ylabel("pips", color="gray", fontproperties=jp_font if jp_font else None)

plt.savefig(f"{OUT_DIR}/v77_multi_pair_backtest.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"  チャート保存: {OUT_DIR}/v77_multi_pair_backtest.png")

# ─── CSV保存 ─────────────────────────────────────────────────────────────────
summary_rows = []
for pair in PAIRS:
    s = all_results[pair]
    if not s:
        continue
    summary_rows.append({
        "pair":           pair,
        "spread_pips":    PAIRS[pair]["spread"],
        "trades":         s["trades"],
        "win_rate":       round(s["win_rate"], 4),
        "pf":             round(s["pf"], 3),
        "total_pnl_pips": round(s["total_pnl"], 1),
        "max_dd_pips":    round(s["max_dd"], 1),
        "sharpe":         round(s["sharpe"], 3),
        "kelly":          round(s["kelly"], 4),
        "plus_months":    s["plus_months"],
        "total_months":   s["total_months"],
        "max_consec_loss": s["max_consec_loss"],
        "p_value":        f"{s['p_value']:.6f}",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT_DIR}/v77_multi_pair_summary.csv", index=False)
print(f"  サマリーCSV保存: {OUT_DIR}/v77_multi_pair_summary.csv")

# 各ペアのトレード詳細CSV
for pair, df in all_trades.items():
    if len(df) > 0:
        path = f"{OUT_DIR}/v77_{pair.lower()}_trades.csv"
        df.to_csv(path, index=False)
        print(f"  {pair}トレード詳細CSV保存: {path}")

# ─── 最終サマリー表示 ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("【v77 3通貨ペアバックテスト 最終結果】")
print("="*60)
for pair in PAIRS:
    s = all_results[pair]
    if not s:
        print(f"\n{pair}: データ不足")
        continue
    print(f"\n{pair} (spread={PAIRS[pair]['spread']}pips)")
    print(f"  取引数:   {s['trades']}回")
    print(f"  勝率:     {s['win_rate']:.1%}")
    print(f"  PF:       {s['pf']:.2f}")
    print(f"  総損益:   {s['total_pnl']:+.1f}pips")
    print(f"  MDD:      {s['max_dd']:.1f}pips")
    print(f"  シャープ: {s['sharpe']:.2f}")
    print(f"  ケリー:   {s['kelly']:.3f}")
    print(f"  月プラス: {s['plus_months']}/{s['total_months']}")
    print(f"  最大連敗: {s['max_consec_loss']}回")
    print(f"  p値:      {s['p_value']:.6f}")

if all_df_list:
    combined_stats = calc_stats(combined.rename(columns={"entry_time": "entry_time"}))
    total_pnl = combined["pnl_pips"].sum()
    total_trades = len(combined)
    print(f"\n【3ペア合算】")
    print(f"  合計取引数: {total_trades}回")
    print(f"  合計損益:   {total_pnl:+.1f}pips")

print("\n完了")
