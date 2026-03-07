import pandas as pd
import sys
sys.path.append('/home/ubuntu/sena3fx/strategies')
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yagami_mtf_v69 import generate_signals, calculate_atr


def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df


def run_backtest(symbol, start_date, end_date, spread_pips):
    data_1m  = load_data(symbol.lower(), '1m',  start_date, end_date)
    data_15m = load_data(symbol.lower(), '15m', start_date, end_date)
    data_1h  = load_data(symbol.lower(), '1h',  start_date, end_date)
    data_4h  = load_data(symbol.lower(), '4h',  start_date, end_date)

    signal_series, tp_series, sl_series, _, _ = \
        generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips)

    trades = []
    pos = 0; entry_price = 0.0; entry_time = None
    tp = 0.0; sl = 0.0; orig_risk = 0.0
    half_closed = False; half_tp = 0.0

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name

        if pos != 0:
            if not half_closed:
                if pos == 1 and bar["high"] >= half_tp:
                    trades.append({"EntryTime": entry_time, "ExitTime": t,
                        "Position": "LONG_HALF", "EntryPrice": entry_price,
                        "ExitPrice": half_tp, "PnL_pips": (half_tp - entry_price)*100,
                        "Risk_pips": orig_risk*100, "Type": "HALF_TP"})
                    sl = entry_price; half_closed = True
                elif pos == -1 and bar["low"] <= half_tp:
                    trades.append({"EntryTime": entry_time, "ExitTime": t,
                        "Position": "SHORT_HALF", "EntryPrice": entry_price,
                        "ExitPrice": half_tp, "PnL_pips": (entry_price - half_tp)*100,
                        "Risk_pips": orig_risk*100, "Type": "HALF_TP"})
                    sl = entry_price; half_closed = True

            if (pos == 1 and bar["low"] <= sl) or (pos == -1 and bar["high"] >= sl):
                pnl = (sl - entry_price)*100*pos
                trades.append({"EntryTime": entry_time, "ExitTime": t,
                    "Position": "LONG" if pos==1 else "SHORT",
                    "EntryPrice": entry_price, "ExitPrice": sl,
                    "PnL_pips": pnl, "Risk_pips": orig_risk*100, "Type": "SL"})
                pos = 0; half_closed = False; continue

            if (pos == 1 and bar["high"] >= tp) or (pos == -1 and bar["low"] <= tp):
                pnl = (tp - entry_price)*100*pos
                trades.append({"EntryTime": entry_time, "ExitTime": t,
                    "Position": "LONG" if pos==1 else "SHORT",
                    "EntryPrice": entry_price, "ExitPrice": tp,
                    "PnL_pips": pnl, "Risk_pips": orig_risk*100, "Type": "TP"})
                pos = 0; half_closed = False; continue

        if pos == 0:
            sig = signal_series.loc[t]
            if sig != 0:
                entry_price = bar["close"]; entry_time = t; pos = sig
                tp = tp_series.loc[t]; sl = sl_series.loc[t]
                orig_risk = abs(entry_price - sl); half_closed = False
                half_tp = entry_price + orig_risk if sig==1 else entry_price - orig_risk

    return pd.DataFrame(trades)


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] No trades."); return {}
    total = df["PnL_pips"].sum()
    entries = df[~df["Position"].str.contains("HALF")]
    wins = df[df["PnL_pips"] > 0]; losses = df[df["PnL_pips"] < 0]
    nw = len(wins); nl = len(losses)
    avg_w = wins["PnL_pips"].mean() if nw>0 else 0
    avg_l = losses["PnL_pips"].mean() if nl>0 else 0
    pf = wins["PnL_pips"].sum() / abs(losses["PnL_pips"].sum()) if nl>0 else np.inf
    wr = nw/(nw+nl)*100 if (nw+nl)>0 else 0
    kelly = 0.0
    if avg_l != 0:
        b = abs(avg_w/avg_l); p = nw/(nw+nl)
        kelly = (b*p - (1-p)) / b
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  総損益          : {total:+.2f} pips")
    print(f"  エントリー数    : {len(entries)} 回")
    print(f"  勝率            : {wr:.1f}% ({nw}勝 {nl}敗)")
    print(f"  PF              : {pf:.2f}")
    print(f"  平均利益        : {avg_w:+.2f} pips")
    print(f"  平均損失        : {avg_l:+.2f} pips")
    print(f"  ケリー基準      : {kelly:+.4f}")
    print(f"{'='*60}")
    return {"total": total, "entries": len(entries), "wr": wr, "pf": pf, "kelly": kelly}


def plot_equity(df, label, out_path):
    if df.empty: return
    df2 = df.copy(); df2["cum"] = df2["PnL_pips"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.plot(range(len(df2)), df2["cum"], color="#2196F3", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.fill_between(range(len(df2)), df2["cum"], 0, where=df2["cum"]>=0, alpha=0.15, color="green")
    ax.fill_between(range(len(df2)), df2["cum"], 0, where=df2["cum"]<0, alpha=0.15, color="red")
    ax.set_title(f"{label} - Cumulative PnL (pips)", fontsize=13)
    ax.set_ylabel("pips"); ax.grid(True, alpha=0.4)
    ax2 = axes[1]
    colors = ["#4CAF50" if x>0 else "#F44336" for x in df2["PnL_pips"]]
    ax2.bar(range(len(df2)), df2["PnL_pips"], color=colors, alpha=0.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Individual Trade PnL (pips)", fontsize=13)
    ax2.set_ylabel("pips"); ax2.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Chart: {out_path}")


if __name__ == "__main__":
    SYMBOL = "USDJPY"; START = "2024-07-01"; END = "2024-08-06"; SPREAD = 0.2
    OUT = "/home/ubuntu/sena3fx/results"
    os.makedirs(OUT, exist_ok=True)

    trades = run_backtest(SYMBOL, START, END, SPREAD)
    trades.to_csv(f"{OUT}/trades_v69.csv", index=False)
    print_stats(trades, f"v69 | {SYMBOL} {START}〜{END} spread={SPREAD}pips")
    plot_equity(trades, "Yagami MTF v69", f"{OUT}/v69_equity_curve.png")
