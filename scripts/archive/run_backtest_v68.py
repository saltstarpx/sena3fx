import pandas as pd
import sys
sys.path.append('/home/ubuntu/sena3fx/strategies')
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
except:
    pass

from yagami_mtf_v68 import generate_signals, calculate_atr


def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df


def run_backtest(symbol, start_date, end_date, spread_pips):
    print(f"Running v68 backtest: {symbol} {start_date}〜{end_date}, spread={spread_pips}pips")

    data_1m  = load_data(symbol.lower(), '1m',  start_date, end_date)
    data_15m = load_data(symbol.lower(), '15m', start_date, end_date)
    data_1h  = load_data(symbol.lower(), '1h',  start_date, end_date)
    data_4h  = load_data(symbol.lower(), '4h',  start_date, end_date)

    signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series = \
        generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips)

    trades = []
    current_position = 0
    entry_price = 0.0
    entry_time = None
    take_profit = 0.0
    stop_loss = 0.0
    original_risk = 0.0
    half_closed = False
    half_tp = 0.0

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name

        if current_position != 0:
            # 半利確（リスク幅の1.0倍）
            if not half_closed:
                if current_position == 1 and bar["high"] >= half_tp:
                    trades.append({
                        "EntryTime": entry_time, "ExitTime": t,
                        "Position": "LONG_HALF", "EntryPrice": entry_price,
                        "ExitPrice": half_tp,
                        "PnL_pips": (half_tp - entry_price) * 100,
                        "Risk_pips": original_risk * 100, "Type": "HALF_TP"
                    })
                    stop_loss = entry_price  # SLを建値に移動
                    half_closed = True
                elif current_position == -1 and bar["low"] <= half_tp:
                    trades.append({
                        "EntryTime": entry_time, "ExitTime": t,
                        "Position": "SHORT_HALF", "EntryPrice": entry_price,
                        "ExitPrice": half_tp,
                        "PnL_pips": (entry_price - half_tp) * 100,
                        "Risk_pips": original_risk * 100, "Type": "HALF_TP"
                    })
                    stop_loss = entry_price
                    half_closed = True

            # 損切り
            if (current_position == 1 and bar["low"] <= stop_loss) or \
               (current_position == -1 and bar["high"] >= stop_loss):
                exit_p = stop_loss
                pnl = (exit_p - entry_price) * 100 * current_position
                trades.append({
                    "EntryTime": entry_time, "ExitTime": t,
                    "Position": "LONG" if current_position == 1 else "SHORT",
                    "EntryPrice": entry_price, "ExitPrice": exit_p,
                    "PnL_pips": pnl, "Risk_pips": original_risk * 100, "Type": "SL"
                })
                current_position = 0; half_closed = False; continue

            # 利確
            if (current_position == 1 and bar["high"] >= take_profit) or \
               (current_position == -1 and bar["low"] <= take_profit):
                exit_p = take_profit
                pnl = (exit_p - entry_price) * 100 * current_position
                trades.append({
                    "EntryTime": entry_time, "ExitTime": t,
                    "Position": "LONG" if current_position == 1 else "SHORT",
                    "EntryPrice": entry_price, "ExitPrice": exit_p,
                    "PnL_pips": pnl, "Risk_pips": original_risk * 100, "Type": "TP"
                })
                current_position = 0; half_closed = False; continue

        # 新規エントリー
        if current_position == 0:
            sig = signal_series.loc[t]
            if sig != 0:
                entry_price  = bar["close"]
                entry_time   = t
                current_position = sig
                take_profit  = tp_series.loc[t]
                stop_loss    = sl_series.loc[t]
                original_risk = abs(entry_price - stop_loss)
                half_closed  = False
                half_tp = entry_price + original_risk if sig == 1 else entry_price - original_risk

    return pd.DataFrame(trades)


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] No trades.")
        return {}

    total_pnl  = df["PnL_pips"].sum()
    # エントリー単位（HALF除く）
    entries    = df[~df["Position"].str.contains("HALF")]
    wins       = df[df["PnL_pips"] > 0]
    losses     = df[df["PnL_pips"] < 0]
    n_entries  = len(entries)
    n_wins     = len(wins)
    n_losses   = len(losses)
    avg_win    = wins["PnL_pips"].mean()  if n_wins   > 0 else 0
    avg_loss   = losses["PnL_pips"].mean() if n_losses > 0 else 0
    tot_profit = wins["PnL_pips"].sum()
    tot_loss   = abs(losses["PnL_pips"].sum())
    pf         = tot_profit / tot_loss if tot_loss > 0 else np.inf

    kelly = 0.0
    if avg_loss != 0:
        b = abs(avg_win / avg_loss)
        p = n_wins / (n_wins + n_losses)
        kelly = (b * p - (1 - p)) / b

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  総損益          : {total_pnl:+.2f} pips")
    print(f"  エントリー数    : {n_entries} 回")
    print(f"  勝率            : {n_wins/(n_wins+n_losses)*100:.1f}% ({n_wins}勝 {n_losses}敗)")
    print(f"  プロフィットファクター: {pf:.2f}")
    print(f"  平均利益        : {avg_win:+.2f} pips")
    print(f"  平均損失        : {avg_loss:+.2f} pips")
    print(f"  ケリー基準      : {kelly:+.4f}")
    print(f"{'='*60}")

    return {"label": label, "total_pnl": total_pnl, "n_entries": n_entries,
            "win_rate": n_wins/(n_wins+n_losses)*100 if (n_wins+n_losses)>0 else 0,
            "pf": pf, "avg_win": avg_win, "avg_loss": avg_loss, "kelly": kelly}


def plot_equity(df, label, out_path):
    if df.empty:
        return
    df2 = df.copy()
    df2["cum_pnl"] = df2["PnL_pips"].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 損益曲線
    ax = axes[0]
    ax.plot(range(len(df2)), df2["cum_pnl"], color="#2196F3", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(df2)), df2["cum_pnl"], 0,
                    where=df2["cum_pnl"] >= 0, alpha=0.15, color="green")
    ax.fill_between(range(len(df2)), df2["cum_pnl"], 0,
                    where=df2["cum_pnl"] < 0, alpha=0.15, color="red")
    ax.set_title(f"{label} - Cumulative PnL (pips)", fontsize=13)
    ax.set_ylabel("pips")
    ax.grid(True, alpha=0.4)

    # 個別トレード損益
    ax2 = axes[1]
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in df2["PnL_pips"]]
    ax2.bar(range(len(df2)), df2["PnL_pips"], color=colors, alpha=0.8)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("Individual Trade PnL (pips)", fontsize=13)
    ax2.set_ylabel("pips")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Chart saved: {out_path}")


if __name__ == "__main__":
    SYMBOL     = "USDJPY"
    START_DATE = "2024-07-01"
    END_DATE   = "2024-08-06"
    SPREAD     = 0.2
    OUT_DIR    = "/home/ubuntu/sena3fx/results"
    os.makedirs(OUT_DIR, exist_ok=True)

    trades = run_backtest(SYMBOL, START_DATE, END_DATE, SPREAD)
    trades.to_csv(f"{OUT_DIR}/trades_v68.csv", index=False)

    stats = print_stats(trades, f"v68 | {SYMBOL} {START_DATE}〜{END_DATE} spread={SPREAD}pips")
    plot_equity(trades, "Yagami MTF v68", f"{OUT_DIR}/v68_equity_curve.png")
