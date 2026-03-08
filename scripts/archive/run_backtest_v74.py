"""
run_backtest_v74.py
===================
v74戦略バックテスト: 4時間足 + 1時間足 二番底・二番天井
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')
from yagami_mtf_v74 import generate_signals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

SYMBOL   = "USDJPY"
START    = "2024-07-01"
END      = "2025-02-06"
SPREAD   = 0.2
RR_RATIO = 2.5
OUT      = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT, exist_ok=True)


def load_data(symbol, timeframe, start, end):
    path = f"/home/ubuntu/sena3fx/data/{symbol.lower()}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df[(df.index >= start) & (df.index <= end)]


def run_backtest(data_15m, signals):
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None

    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name

        if pos is not None:
            d = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]

            # 半利確
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"] <= half_tp):
                    pnl = pos["risk"] * 100
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": pnl, "type": "HALF_TP"})
                    pos["sl"] = pos["ep"]
                    pos["half_closed"] = True

            # 損切り
            if (d == 1 and bar["low"] <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"],
                               "pnl": pnl, "type": "SL"})
                pos = None
                continue

            # 利確
            if (d == 1 and bar["high"] >= pos["tp"]) or \
               (d == -1 and bar["low"] <= pos["tp"]):
                pnl = (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["tp"],
                               "pnl": pnl, "type": "TP"})
                pos = None
                continue

        if pos is None and t in sig_map:
            s = sig_map[t]
            pos = {**s, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] トレードなし")
        return {}

    total = df["pnl"].sum()
    closed = df[df["type"].isin(["SL", "TP"])]
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    nw = len(wins); nl = len(losses)
    avg_w = wins["pnl"].mean()   if nw > 0 else 0
    avg_l = losses["pnl"].mean() if nl > 0 else 0
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if nl > 0 and losses["pnl"].sum() != 0 else float("inf")
    wr = nw / (nw + nl) * 100 if (nw + nl) > 0 else 0
    kelly = 0.0
    if avg_l != 0 and (nw + nl) > 0:
        b = abs(avg_w / avg_l)
        p = nw / (nw + nl)
        kelly = (b * p - (1 - p)) / b

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  総損益          : {total:+.2f} pips")
    print(f"  エントリー数    : {len(closed)} 回")
    print(f"  勝率            : {wr:.1f}% ({nw}勝 {nl}敗)")
    print(f"  PF              : {pf:.2f}")
    print(f"  平均利益        : {avg_w:+.2f} pips")
    print(f"  平均損失        : {avg_l:+.2f} pips")
    print(f"  ケリー基準      : {kelly:+.4f}")
    print(f"{'='*60}")

    # 時間軸別内訳
    if "tf" in df.columns:
        for tf in ["4h", "1h"]:
            tf_df = df[df["tf"] == tf]
            tf_closed = tf_df[tf_df["type"].isin(["SL", "TP"])]
            tf_wins = tf_df[tf_df["pnl"] > 0]
            tf_losses = tf_df[tf_df["pnl"] < 0]
            tf_wr = len(tf_wins) / (len(tf_wins) + len(tf_losses)) * 100 if (len(tf_wins) + len(tf_losses)) > 0 else 0
            tf_pf = tf_wins["pnl"].sum() / abs(tf_losses["pnl"].sum()) if len(tf_losses) > 0 and tf_losses["pnl"].sum() != 0 else float("inf")
            print(f"  [{tf}] {len(tf_closed)}回 | 勝率{tf_wr:.1f}% | PF{tf_pf:.2f} | 損益{tf_df['pnl'].sum():+.1f}pips")

    # 月別損益
    df2 = df.copy()
    df2["exit_time"] = pd.to_datetime(df2["exit_time"])
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly_pnl = df2.groupby("month")["pnl"].sum()
    closed2 = closed.copy()
    closed2["exit_time"] = pd.to_datetime(closed2["exit_time"])
    closed2["month"] = closed2["exit_time"].dt.to_period("M")
    monthly_cnt = closed2.groupby("month")["pnl"].count()

    print("  月別損益:")
    for m in monthly_pnl.index:
        cnt = monthly_cnt.get(m, 0)
        v = monthly_pnl[m]
        sign = "+" if v >= 0 else ""
        print(f"    {m}: {sign}{v:7.1f} pips  ({cnt}回)")

    return {"total": total, "entries": len(closed), "wr": wr, "pf": pf, "kelly": kelly}


def plot_equity(df, label, out_path):
    if df.empty:
        return
    cum = df["pnl"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(range(len(cum)), cum.values, color="#2196F3", lw=1.5)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].fill_between(range(len(cum)), cum.values, 0,
                          where=(cum.values >= 0), alpha=0.15, color="#2196F3")
    axes[0].fill_between(range(len(cum)), cum.values, 0,
                          where=(cum.values < 0), alpha=0.15, color="#F44336")
    axes[0].set_title(f"{label} - 累積損益 (pips)", fontsize=13)
    axes[0].set_ylabel("pips")
    axes[0].grid(True, alpha=0.4)

    df2 = df.copy()
    df2["exit_time"] = pd.to_datetime(df2["exit_time"])
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in monthly.values]
    axes[1].bar(range(len(monthly)), monthly.values, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(monthly)))
    axes[1].set_xticklabels([str(m) for m in monthly.index], rotation=45)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("月別損益 (pips)", fontsize=13)
    axes[1].set_ylabel("pips")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Chart: {out_path}")


if __name__ == "__main__":
    print(f"データ読み込み中... ({START} 〜 {END})")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)
    print(f"  1m足: {len(data_1m)}本, 15m足: {len(data_15m)}本, 4h足: {len(data_4h)}本")

    print("\nシグナル生成中...")
    signals = generate_signals(data_1m, data_15m, data_4h, SPREAD, RR_RATIO)
    print(f"  シグナル数: {len(signals)}本")
    db4 = sum(1 for s in signals if s["tf"] == "4h" and s["pattern"] == "double_bottom")
    dt4 = sum(1 for s in signals if s["tf"] == "4h" and s["pattern"] == "double_top")
    db1 = sum(1 for s in signals if s["tf"] == "1h" and s["pattern"] == "double_bottom")
    dt1 = sum(1 for s in signals if s["tf"] == "1h" and s["pattern"] == "double_top")
    print(f"  4h 二番底: {db4}本 / 二番天井: {dt4}本")
    print(f"  1h 二番底: {db1}本 / 二番天井: {dt1}本")

    print("\nバックテスト実行中...")
    df_trades = run_backtest(data_15m, signals)
    df_trades.to_csv(f"{OUT}/trades_v74.csv", index=False)

    stats = print_stats(df_trades, f"v74 | {SYMBOL} {START}〜{END} spread={SPREAD}pips RR={RR_RATIO}")
    plot_equity(df_trades, f"v74 (4h+1h 二番底・二番天井) RR{RR_RATIO}", f"{OUT}/v74_equity_curve.png")

    # v73との比較
    print("\n\n" + "="*60)
    print("  v73 vs v74 比較サマリー")
    print("="*60)
    try:
        df_v73 = pd.read_csv(f"{OUT}/trades_v73.csv")
        v73_total = df_v73["pnl"].sum()
        v73_closed = df_v73[df_v73["type"].isin(["SL", "TP"])]
        v73_wins = df_v73[df_v73["pnl"] > 0]
        v73_losses = df_v73[df_v73["pnl"] < 0]
        v73_wr = len(v73_wins) / (len(v73_wins) + len(v73_losses)) * 100 if (len(v73_wins) + len(v73_losses)) > 0 else 0
        v73_pf = v73_wins["pnl"].sum() / abs(v73_losses["pnl"].sum()) if len(v73_losses) > 0 else float("inf")
        print(f"  {'指標':<15} {'v73 (4hのみ)':<20} {'v74 (4h+1h)':<20}")
        print(f"  {'-'*55}")
        print(f"  {'総損益':<15} {v73_total:+.1f} pips{'':<8} {stats.get('total', 0):+.1f} pips")
        print(f"  {'エントリー数':<15} {len(v73_closed)}回{'':<15} {stats.get('entries', 0)}回")
        print(f"  {'勝率':<15} {v73_wr:.1f}%{'':<14} {stats.get('wr', 0):.1f}%")
        print(f"  {'PF':<15} {v73_pf:.2f}{'':<15} {stats.get('pf', 0):.2f}")
        print(f"  {'ケリー基準':<15} 0.4446{'':<14} {stats.get('kelly', 0):.4f}")
    except Exception as e:
        print(f"  v73データ読み込みエラー: {e}")
