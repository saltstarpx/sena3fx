"""
v72: RR比率の最適化テスト
- RR 2.5 / 3.0 の2パターンを比較
- 1ポジション固定（ピラミッディングなし）
- 成行エントリー（次の15分足始値）
- 全期間: 2024/7〜2025/2
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'


def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    return df[(df.index >= start_date) & (df.index <= end_date)]


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signals_v72(data_1m, data_15m, data_4h, spread_pips, rr_ratio):
    spread = spread_pips * 0.01
    m15_atr = calculate_atr(data_15m)
    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    signals = []
    WICK_MULT = 0.3
    volatility_threshold = spread * 3.0

    for i in range(len(data_15m) - 1):
        bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        if pd.isna(atr_val) or atr_val < volatility_threshold:
            continue

        h4_time = bar.name.floor("4h")
        if h4_time not in data_4h.index:
            continue
        trend = data_4h.loc[h4_time]["trend"]

        body_high = max(bar["open"], bar["close"])
        body_low  = min(bar["open"], bar["close"])
        lower_wick = body_low - bar["low"]
        upper_wick = bar["high"] - body_high
        wick_thr = atr_val * WICK_MULT

        long_ok  = (trend == 1)  and (lower_wick > wick_thr)
        short_ok = (trend == -1) and (upper_wick > wick_thr)
        if not long_ok and not short_ok:
            continue

        next_time = data_15m.iloc[i + 1].name
        if next_time not in data_1m.index:
            continue

        entry_bar = data_1m.loc[next_time]
        if long_ok:
            ep = entry_bar["open"] + spread
            sl = bar["low"] - atr_val * 0.2
            risk = ep - sl
            if risk <= 0: continue
            tp = ep + risk * rr_ratio
            signals.append({"time": next_time, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk})
        else:
            ep = entry_bar["open"] - spread
            sl = bar["high"] + atr_val * 0.2
            risk = sl - ep
            if risk <= 0: continue
            tp = ep - risk * rr_ratio
            signals.append({"time": next_time, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk})

    return signals


def run_backtest(data_1m, signals, rr_ratio):
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name

        if pos is not None:
            d = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]

            # 半利確
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pnl = pos["risk"] * 100
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": pnl, "type": "HALF_TP"})
                    pos["sl"] = pos["ep"]
                    pos["half_closed"] = True

            # 損切り
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"],
                               "pnl": pnl, "type": "SL"})
                pos = None; continue

            # 利確
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                pnl = (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["tp"],
                               "pnl": pnl, "type": "TP"})
                pos = None; continue

        # 新規エントリー（ポジションなし時のみ）
        if pos is None and t in sig_map:
            s = sig_map[t]
            pos = {**s, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] No trades."); return {}
    total = df["pnl"].sum()
    entries = df[df["type"].isin(["SL", "TP", "FORCE_CLOSE"])]
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    nw = len(wins); nl = len(losses)
    avg_w = wins["pnl"].mean()  if nw > 0 else 0
    avg_l = losses["pnl"].mean() if nl > 0 else 0
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if nl > 0 else np.inf
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

    # 月別
    df2 = df.copy()
    df2["exit_time"] = pd.to_datetime(df2["exit_time"])
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum().round(1)
    monthly_cnt = entries.copy()
    monthly_cnt["exit_time"] = pd.to_datetime(monthly_cnt["exit_time"])
    monthly_cnt["month"] = monthly_cnt["exit_time"].dt.to_period("M")
    mcnt = monthly_cnt.groupby("month")["pnl"].count()
    print(f"  月別損益:")
    for m in monthly.index:
        cnt = mcnt.get(m, 0)
        sign = "+" if monthly[m] >= 0 else ""
        print(f"    {m}: {sign}{monthly[m]:7.1f} pips  ({cnt}回)")
    return {"total": total, "entries": len(entries), "wr": wr, "pf": pf, "kelly": kelly}


def plot_equity_compare(df25, df30, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for df, label, color in [(df25, "RR 2.5", "#2196F3"), (df30, "RR 3.0", "#FF5722")]:
        if df.empty: continue
        cum = df["pnl"].cumsum()
        axes[0].plot(range(len(cum)), cum, color=color, lw=1.5, label=label)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_title("v72 RR比較 - 累積損益 (pips)", fontsize=13)
    axes[0].set_ylabel("pips"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # 月別比較棒グラフ
    def get_monthly(df):
        d = df.copy()
        d["exit_time"] = pd.to_datetime(d["exit_time"])
        d["month"] = d["exit_time"].dt.to_period("M")
        return d.groupby("month")["pnl"].sum()

    m25 = get_monthly(df25); m30 = get_monthly(df30)
    months = sorted(set(list(m25.index) + list(m30.index)))
    x = range(len(months))
    w = 0.35
    axes[1].bar([i - w/2 for i in x], [m25.get(m, 0) for m in months], w, label="RR 2.5", color="#2196F3", alpha=0.8)
    axes[1].bar([i + w/2 for i in x], [m30.get(m, 0) for m in months], w, label="RR 3.0", color="#FF5722", alpha=0.8)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels([str(m) for m in months], rotation=45)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("月別損益比較 (pips)", fontsize=13)
    axes[1].set_ylabel("pips"); axes[1].legend(); axes[1].grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Chart: {out_path}")


if __name__ == "__main__":
    SYMBOL = "USDJPY"; START = "2024-07-01"; END = "2025-02-06"; SPREAD = 0.2
    OUT = "/home/ubuntu/sena3fx/results"
    os.makedirs(OUT, exist_ok=True)

    data_1m  = load_data(SYMBOL.lower(), '1m',  START, END)
    data_15m = load_data(SYMBOL.lower(), '15m', START, END)
    data_4h  = load_data(SYMBOL.lower(), '4h',  START, END)

    results = {}
    dfs = {}
    for rr in [2.5, 3.0]:
        signals = generate_signals_v72(data_1m, data_15m, data_4h, SPREAD, rr)
        df = run_backtest(data_1m, signals, rr)
        df.to_csv(f"{OUT}/trades_v72_rr{str(rr).replace('.','')}.csv", index=False)
        stats = print_stats(df, f"v72 RR{rr} | {SYMBOL} {START}〜{END} spread={SPREAD}pips")
        results[rr] = stats
        dfs[rr] = df

    plot_equity_compare(dfs[2.5], dfs[3.0], f"{OUT}/v72_rr_comparison.png")
