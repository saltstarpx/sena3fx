"""
rr_comparison_full_analysis.py
================================
RR 2.0 / 2.5 / 3.0 の3パターンをバックテストし、
定量分析・計量分析（モンテカルロ・t検定・バルサラ）を実施する
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from scipy import stats

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

SYMBOL = "USDJPY"
START  = "2024-07-01"
END    = "2025-02-06"
SPREAD = 0.2
OUT    = "/home/ubuntu/sena3fx/results"
N_SIM  = 10000
SEED   = 42
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────
def load_data(symbol, tf, start, end):
    path = f"/home/ubuntu/sena3fx/data/{symbol.lower()}_{tf}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return df[(df.index >= start) & (df.index <= end)]

def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────
# バックテストエンジン（RR可変）
# ─────────────────────────────────────────────
def run_backtest(data_1m, signals, rr_ratio):
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name

        if pos is not None:
            d = pos["dir"]
            risk = pos["risk"]
            half_tp = pos["ep"] + risk       if d == 1 else pos["ep"] - risk
            full_tp = pos["ep"] + risk * rr_ratio if d == 1 else pos["ep"] - risk * rr_ratio

            # 半利確
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"]  <= half_tp):
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": risk * 100, "type": "HALF_TP"})
                    pos["sl"] = pos["ep"]
                    pos["half_closed"] = True

            # 損切り
            if (d == 1 and bar["low"]  <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"],
                               "pnl": pnl, "type": "SL"})
                pos = None; continue

            # 全利確
            if (d == 1 and bar["high"] >= full_tp) or \
               (d == -1 and bar["low"]  <= full_tp):
                pnl = (full_tp - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": full_tp,
                               "pnl": pnl, "type": "TP"})
                pos = None; continue

        if pos is None and t in sig_map:
            s = sig_map[t]
            pos = {**s, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# 定量分析
# ─────────────────────────────────────────────
def quantitative_stats(df, label, rr):
    if df.empty:
        return {}

    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    nw, nl = len(wins), len(losses)
    total  = df["pnl"].sum()
    avg_w  = wins["pnl"].mean()   if nw > 0 else 0
    avg_l  = abs(losses["pnl"].mean()) if nl > 0 else 0
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if nl > 0 else float("inf")
    wr     = nw / (nw + nl) if (nw + nl) > 0 else 0
    kelly  = (wr * (avg_w / avg_l) - (1 - wr)) / (avg_w / avg_l) if avg_l > 0 else 0

    # ドローダウン
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    dd   = peak - cum
    max_dd = dd.max()

    # シャープレシオ（年換算）
    pnl_arr = df["pnl"].values
    sharpe  = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(252 * 6.5)) if pnl_arr.std() > 0 else 0

    # 連勝・連敗
    results = (df["pnl"] > 0).astype(int).values
    max_win_streak = max_loss_streak = cur = 0
    cur_type = None
    for r in results:
        if r == cur_type:
            cur += 1
        else:
            cur = 1; cur_type = r
        if r == 1: max_win_streak  = max(max_win_streak,  cur)
        else:      max_loss_streak = max(max_loss_streak, cur)

    # 月別損益
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"]).dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()

    return {
        "label": label, "rr": rr,
        "total": total, "n": nw + nl, "wr": wr,
        "avg_w": avg_w, "avg_l": avg_l, "pf": pf, "kelly": kelly,
        "max_dd": max_dd, "sharpe": sharpe,
        "max_win": max_win_streak, "max_loss": max_loss_streak,
        "plus_months": plus_months, "total_months": len(monthly),
        "monthly": monthly, "pnl_arr": pnl_arr,
    }


# ─────────────────────────────────────────────
# 計量分析（t検定・モンテカルロ・バルサラ）
# ─────────────────────────────────────────────
def econometric_analysis(stats_dict):
    results = {}
    for key, s in stats_dict.items():
        pnl = s["pnl_arr"]
        n   = len(pnl)

        # t検定（平均損益がゼロより有意に大きいか）
        t_stat, p_val = stats.ttest_1samp(pnl, 0)

        # モンテカルロ（10,000回 × nトレード）
        np.random.seed(SEED)
        mc_finals = []
        mc_max_dds = []
        for _ in range(N_SIM):
            sample = np.random.choice(pnl, size=n, replace=True)
            cum = np.cumsum(sample)
            mc_finals.append(cum[-1])
            peak = np.maximum.accumulate(cum)
            mc_max_dds.append((peak - cum).max())

        mc_finals  = np.array(mc_finals)
        mc_max_dds = np.array(mc_max_dds)
        plus_rate  = np.mean(mc_finals > 0)

        # バルサラ破産確率（資金2%リスク・50%損失で破産）
        wr   = s["wr"]
        po   = s["avg_w"] / s["avg_l"] if s["avg_l"] > 0 else 0
        edge = wr * po - (1 - wr)
        risk_pct = 2.0   # 水原様が決定した2%
        ruin_pct = 50.0  # 50%損失で破産
        if edge > 0:
            z = (1 - edge) / (1 + edge)
            ruin_units = ruin_pct / risk_pct
            balsara = z ** ruin_units
        else:
            balsara = 1.0

        results[key] = {
            "t_stat": t_stat, "p_val": p_val,
            "mc_median": np.median(mc_finals),
            "mc_p5": np.percentile(mc_finals, 5),
            "mc_p95": np.percentile(mc_finals, 95),
            "mc_plus_rate": plus_rate,
            "mc_dd_median": np.median(mc_max_dds),
            "mc_dd_p95": np.percentile(mc_max_dds, 95),
            "balsara_2pct": balsara,
            "mc_finals": mc_finals,
            "mc_max_dds": mc_max_dds,
        }
    return results


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("データ読み込み中...")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)
    data_4h["atr"]   = calc_atr(data_4h)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()

    from yagami_mtf_v75 import generate_signals

    rr_list = [2.0, 2.5, 3.0]
    colors  = ["#4CAF50", "#2196F3", "#FF9800"]
    all_stats = {}
    all_trades = {}

    for rr in rr_list:
        print(f"\nRR {rr} バックテスト中...")
        signals = generate_signals(data_1m, data_15m, data_4h, SPREAD, rr)
        df = run_backtest(data_1m, signals, rr)
        df.to_csv(f"{OUT}/trades_v75_rr{str(rr).replace('.','')}.csv", index=False)
        s = quantitative_stats(df, f"RR {rr}", rr)
        all_stats[rr] = s
        all_trades[rr] = df

    # ─── 定量分析サマリー ───
    print(f"\n{'='*70}")
    print("  【定量分析サマリー】")
    print(f"{'='*70}")
    headers = ["指標", "RR 2.0", "RR 2.5", "RR 3.0"]
    rows = [
        ("総損益 (pips)",    [f"{all_stats[r]['total']:+.1f}" for r in rr_list]),
        ("エントリー数",       [f"{all_stats[r]['n']}回"        for r in rr_list]),
        ("勝率",             [f"{all_stats[r]['wr']:.1%}"      for r in rr_list]),
        ("PF",              [f"{all_stats[r]['pf']:.2f}"      for r in rr_list]),
        ("ケリー基準",        [f"{all_stats[r]['kelly']:.4f}"   for r in rr_list]),
        ("平均利益",          [f"+{all_stats[r]['avg_w']:.1f}pips" for r in rr_list]),
        ("平均損失",          [f"-{all_stats[r]['avg_l']:.1f}pips" for r in rr_list]),
        ("最大DD",           [f"{all_stats[r]['max_dd']:.1f}pips" for r in rr_list]),
        ("シャープレシオ",     [f"{all_stats[r]['sharpe']:.2f}"  for r in rr_list]),
        ("最大連勝",          [f"{all_stats[r]['max_win']}回"    for r in rr_list]),
        ("最大連敗",          [f"{all_stats[r]['max_loss']}回"   for r in rr_list]),
        ("プラス月数",        [f"{all_stats[r]['plus_months']}/{all_stats[r]['total_months']}ヶ月" for r in rr_list]),
    ]
    print(f"  {'指標':<16} {'RR 2.0':<20} {'RR 2.5':<20} {'RR 3.0':<20}")
    print(f"  {'-'*68}")
    for label, vals in rows:
        print(f"  {label:<16} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}")

    # ─── 計量分析 ───
    print(f"\n  計量分析（t検定・モンテカルロ・バルサラ）実行中...")
    eco = econometric_analysis(all_stats)

    print(f"\n{'='*70}")
    print("  【計量分析サマリー】")
    print(f"{'='*70}")
    print(f"  {'指標':<25} {'RR 2.0':<20} {'RR 2.5':<20} {'RR 3.0':<20}")
    print(f"  {'-'*75}")
    eco_rows = [
        ("t統計量",              [f"{eco[r]['t_stat']:.3f}"       for r in rr_list]),
        ("p値（有意差）",         [f"{eco[r]['p_val']:.4f}"        for r in rr_list]),
        ("統計的有意",            ["✓ 有意" if eco[r]['p_val'] < 0.05 else "△ 不十分" for r in rr_list]),
        ("MC中央値損益",          [f"{eco[r]['mc_median']:+.1f}pips" for r in rr_list]),
        ("MC 5th〜95th",        [f"{eco[r]['mc_p5']:+.0f}〜{eco[r]['mc_p95']:+.0f}" for r in rr_list]),
        ("MCプラス確率",          [f"{eco[r]['mc_plus_rate']*100:.1f}%" for r in rr_list]),
        ("MC最大DD（中央値）",     [f"{eco[r]['mc_dd_median']:.1f}pips" for r in rr_list]),
        ("MC最大DD（95th）",      [f"{eco[r]['mc_dd_p95']:.1f}pips"    for r in rr_list]),
        ("バルサラ破産確率(2%R)", [f"{eco[r]['balsara_2pct']*100:.4f}%" for r in rr_list]),
    ]
    for label, vals in eco_rows:
        print(f"  {label:<25} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}")

    # ─────────────────────────────────────────────
    # 可視化（3行×3列）
    # ─────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    # 行1: 損益曲線
    for col, (rr, color) in enumerate(zip(rr_list, colors)):
        ax = fig.add_subplot(gs[0, col])
        df = all_trades[rr]
        if not df.empty:
            cum = df["pnl"].cumsum()
            ax.plot(range(len(cum)), cum.values, color=color, lw=1.8)
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values >= 0), alpha=0.15, color=color)
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values < 0), alpha=0.15, color="#F44336")
        s = all_stats[rr]
        ax.set_title(f"損益曲線 RR {rr}\n"
                     f"総損益:{s['total']:+.0f}pips  勝率:{s['wr']:.0%}  PF:{s['pf']:.2f}", fontsize=10)
        ax.set_xlabel("トレード数"); ax.set_ylabel("累積損益 (pips)")
        ax.axhline(0, color="gray", lw=0.8, ls="--"); ax.grid(True, alpha=0.35)

    # 行2: モンテカルロ最終損益分布
    for col, (rr, color) in enumerate(zip(rr_list, colors)):
        ax = fig.add_subplot(gs[1, col])
        mc_finals = eco[rr]["mc_finals"]
        ax.hist(mc_finals, bins=80, color=color, alpha=0.7, density=True)
        ax.axvline(0,                          color="black",   lw=1.2, ls="--", label="損益ゼロ")
        ax.axvline(eco[rr]["mc_median"],       color=color,     lw=1.5, label=f"中央値:{eco[rr]['mc_median']:+.0f}")
        ax.axvline(eco[rr]["mc_p5"],           color="#F44336", lw=1.0, ls=":", label=f"5th:{eco[rr]['mc_p5']:+.0f}")
        ax.axvline(eco[rr]["mc_p95"],          color="#4CAF50", lw=1.0, ls=":", label=f"95th:{eco[rr]['mc_p95']:+.0f}")
        plus_r = eco[rr]["mc_plus_rate"]
        ax.set_title(f"MC最終損益分布 RR {rr}\n"
                     f"プラス確率:{plus_r:.1%}  p値:{eco[rr]['p_val']:.4f}", fontsize=10)
        ax.set_xlabel("最終損益 (pips)"); ax.set_ylabel("確率密度")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.35)

    # 行3左: 月別損益比較
    ax_m = fig.add_subplot(gs[2, 0])
    all_months = sorted(set(
        str(m) for rr in rr_list for m in all_stats[rr]["monthly"].index
    ))
    x = np.arange(len(all_months)); w = 0.25
    for i, (rr, color) in enumerate(zip(rr_list, colors)):
        vals = [all_stats[rr]["monthly"].get(m, 0) for m in all_months]
        ax_m.bar(x + (i - 1) * w, vals, w, color=color, alpha=0.8, label=f"RR {rr}")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([m[-5:] for m in all_months], rotation=45, fontsize=8)
    ax_m.axhline(0, color="gray", lw=0.8)
    ax_m.set_title("月別損益比較", fontsize=11)
    ax_m.set_ylabel("損益 (pips)"); ax_m.legend(fontsize=8); ax_m.grid(True, alpha=0.35, axis="y")

    # 行3中: バルサラ破産確率比較
    ax_b = fig.add_subplot(gs[2, 1])
    balsara_vals = [eco[rr]["balsara_2pct"] * 100 for rr in rr_list]
    bar_colors = ["#4CAF50" if v < 1 else ("#FF9800" if v < 5 else "#F44336") for v in balsara_vals]
    bars = ax_b.bar([f"RR {r}" for r in rr_list], balsara_vals, color=bar_colors, alpha=0.85, width=0.4)
    for bar, val in zip(bars, balsara_vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                  f"{val:.4f}%", ha="center", fontsize=10, fontweight="bold")
    ax_b.axhline(1, color="#FF9800", lw=1, ls=":", label="1%ライン")
    ax_b.axhline(5, color="#F44336", lw=1, ls=":", label="5%ライン")
    ax_b.set_title("バルサラ破産確率\n（リスク2%/トレード・資金50%損失で破産）", fontsize=10)
    ax_b.set_ylabel("破産確率 (%)"); ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.35, axis="y")

    # 行3右: ケリー基準・PF・勝率の比較レーダー的棒グラフ
    ax_k = fig.add_subplot(gs[2, 2])
    metrics = ["勝率(%)", "PF", "ケリー×10", "シャープ"]
    rr_vals = {
        rr: [
            all_stats[rr]["wr"] * 100,
            all_stats[rr]["pf"],
            all_stats[rr]["kelly"] * 10,
            all_stats[rr]["sharpe"],
        ]
        for rr in rr_list
    }
    x = np.arange(len(metrics)); w = 0.25
    for i, (rr, color) in enumerate(zip(rr_list, colors)):
        ax_k.bar(x + (i - 1) * w, rr_vals[rr], w, color=color, alpha=0.8, label=f"RR {rr}")
    ax_k.set_xticks(x); ax_k.set_xticklabels(metrics, fontsize=9)
    ax_k.set_title("主要指標比較", fontsize=11)
    ax_k.legend(fontsize=8); ax_k.grid(True, alpha=0.35, axis="y")

    fig.suptitle(
        f"v75 RR比較分析（2.0 / 2.5 / 3.0）\n"
        f"USDJPY {START}〜{END}  スプレッド{SPREAD}pips  リスク2%/トレード",
        fontsize=13
    )
    out_path = f"{OUT}/v75_rr_full_analysis.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart: {out_path}")
    print("\n  完了")
