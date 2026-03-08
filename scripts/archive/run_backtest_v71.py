"""
v71: ピラミッディング対応バックテスト
- 同方向シグナルなら最大3ポジまで追加エントリー可能
- 逆方向シグナルは無視（ポジション保有中）
- 各ポジションは独立してSL/TPを管理
- 成行エントリー（次の15分足始値）
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yagami_mtf_v70 import generate_signals  # シグナル生成はv70と同じ


def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    return df[(df.index >= start_date) & (df.index <= end_date)]


def run_backtest_pyramid(symbol, start_date, end_date, spread_pips, max_positions=3):
    data_1m  = load_data(symbol.lower(), '1m',  start_date, end_date)
    data_15m = load_data(symbol.lower(), '15m', start_date, end_date)
    data_1h  = load_data(symbol.lower(), '1h',  start_date, end_date)
    data_4h  = load_data(symbol.lower(), '4h',  start_date, end_date)

    signal_series, tp_series, sl_series, _, _ = \
        generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips)

    trades = []
    # positions: list of {dir, entry_price, sl, tp, half_closed, half_tp}
    positions = []

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name

        # 既存ポジションの管理
        closed_indices = []
        for j, pos in enumerate(positions):
            d = pos['dir']

            # 半利確
            if not pos['half_closed']:
                if d == 1 and bar['high'] >= pos['half_tp']:
                    trades.append({
                        "EntryTime": pos['entry_time'], "ExitTime": t,
                        "Position": "LONG_HALF", "EntryPrice": pos['entry_price'],
                        "ExitPrice": pos['half_tp'],
                        "PnL_pips": (pos['half_tp'] - pos['entry_price']) * 100,
                        "Type": "HALF_TP"
                    })
                    pos['sl'] = pos['entry_price']
                    pos['half_closed'] = True
                elif d == -1 and bar['low'] <= pos['half_tp']:
                    trades.append({
                        "EntryTime": pos['entry_time'], "ExitTime": t,
                        "Position": "SHORT_HALF", "EntryPrice": pos['entry_price'],
                        "ExitPrice": pos['half_tp'],
                        "PnL_pips": (pos['entry_price'] - pos['half_tp']) * 100,
                        "Type": "HALF_TP"
                    })
                    pos['sl'] = pos['entry_price']
                    pos['half_closed'] = True

            # 損切り
            if (d == 1 and bar['low'] <= pos['sl']) or (d == -1 and bar['high'] >= pos['sl']):
                pnl = (pos['sl'] - pos['entry_price']) * 100 * d
                trades.append({
                    "EntryTime": pos['entry_time'], "ExitTime": t,
                    "Position": "LONG" if d == 1 else "SHORT",
                    "EntryPrice": pos['entry_price'], "ExitPrice": pos['sl'],
                    "PnL_pips": pnl, "Type": "SL"
                })
                closed_indices.append(j)
                continue

            # 利確
            if (d == 1 and bar['high'] >= pos['tp']) or (d == -1 and bar['low'] <= pos['tp']):
                pnl = (pos['tp'] - pos['entry_price']) * 100 * d
                trades.append({
                    "EntryTime": pos['entry_time'], "ExitTime": t,
                    "Position": "LONG" if d == 1 else "SHORT",
                    "EntryPrice": pos['entry_price'], "ExitPrice": pos['tp'],
                    "PnL_pips": pnl, "Type": "TP"
                })
                closed_indices.append(j)

        # クローズ済みを削除（逆順で）
        for j in sorted(closed_indices, reverse=True):
            positions.pop(j)

        # 新規シグナルチェック
        sig = signal_series.loc[t]
        if sig != 0:
            # 現在のポジション方向を確認
            current_dirs = [p['dir'] for p in positions]

            # 逆方向ポジションがあれば無視
            if any(d != sig for d in current_dirs):
                continue

            # 同方向でmax_positions未満なら追加
            if len(positions) < max_positions:
                ep = bar['open']
                sl = sl_series.loc[t]
                tp = tp_series.loc[t]
                risk = abs(ep - sl)
                if risk <= 0:
                    continue
                half_tp = ep + risk if sig == 1 else ep - risk
                positions.append({
                    'dir': sig, 'entry_price': ep, 'entry_time': t,
                    'sl': sl, 'tp': tp, 'half_tp': half_tp, 'half_closed': False
                })

    # 残ポジを最終バーで強制決済
    if positions and len(data_1m) > 0:
        last_bar = data_1m.iloc[-1]
        for pos in positions:
            d = pos['dir']
            ep_close = last_bar['close']
            pnl = (ep_close - pos['entry_price']) * 100 * d
            trades.append({
                "EntryTime": pos['entry_time'], "ExitTime": last_bar.name,
                "Position": "LONG" if d == 1 else "SHORT",
                "EntryPrice": pos['entry_price'], "ExitPrice": ep_close,
                "PnL_pips": pnl, "Type": "FORCE_CLOSE"
            })

    return pd.DataFrame(trades)


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] No trades."); return
    total = df["PnL_pips"].sum()
    entries = df[~df["Position"].str.contains("HALF")]
    wins   = df[df["PnL_pips"] > 0]
    losses = df[df["PnL_pips"] < 0]
    nw = len(wins); nl = len(losses)
    avg_w = wins["PnL_pips"].mean()  if nw > 0 else 0
    avg_l = losses["PnL_pips"].mean() if nl > 0 else 0
    pf = wins["PnL_pips"].sum() / abs(losses["PnL_pips"].sum()) if nl > 0 else np.inf
    wr = nw / (nw + nl) * 100 if (nw + nl) > 0 else 0
    kelly = 0.0
    if avg_l != 0:
        b = abs(avg_w / avg_l); p = nw / (nw + nl)
        kelly = (b * p - (1 - p)) / b

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


def plot_equity(df, label, out_path):
    if df.empty: return
    df2 = df.copy(); df2["cum"] = df2["PnL_pips"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.plot(range(len(df2)), df2["cum"], color="#2196F3", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.fill_between(range(len(df2)), df2["cum"], 0, where=df2["cum"] >= 0, alpha=0.15, color="green")
    ax.fill_between(range(len(df2)), df2["cum"], 0, where=df2["cum"] < 0, alpha=0.15, color="red")
    ax.set_title(f"{label} - Cumulative PnL (pips)", fontsize=13)
    ax.set_ylabel("pips"); ax.grid(True, alpha=0.4)
    ax2 = axes[1]
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in df2["PnL_pips"]]
    ax2.bar(range(len(df2)), df2["PnL_pips"], color=colors, alpha=0.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Individual Trade PnL (pips)", fontsize=13)
    ax2.set_ylabel("pips"); ax2.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Chart: {out_path}")


if __name__ == "__main__":
    SYMBOL = "USDJPY"; START = "2024-07-01"; END = "2025-02-06"; SPREAD = 0.2
    OUT = "/home/ubuntu/sena3fx/results"
    os.makedirs(OUT, exist_ok=True)

    for max_pos in [1, 2, 3]:
        trades = run_backtest_pyramid(SYMBOL, START, END, SPREAD, max_positions=max_pos)
        trades.to_csv(f"{OUT}/trades_v71_max{max_pos}.csv", index=False)
        print_stats(trades, f"v71 (最大{max_pos}ポジ) | {SYMBOL} {START}〜{END} spread={SPREAD}pips")

    # 最大3ポジのチャートを保存
    trades3 = pd.read_csv(f"{OUT}/trades_v71_max3.csv")
    plot_equity(trades3, "Yagami MTF v71 (ピラミッディング最大3ポジ)", f"{OUT}/v71_equity_curve.png")
