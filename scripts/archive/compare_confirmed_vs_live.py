"""
compare_confirmed_vs_live.py
============================
15分足「確定後」vs「確定前（途中）」エントリーの差異検証

【確定後】= 15分足が閉じた直後の次の1分足始値（現在のv75）
【確定前】= 15分足の最後の1分足（14分目）の始値でエントリー
            ※「もう少し早く入れば良かった」という仮説の検証

二番底・二番天井の判定は同じロジックを使用し、
エントリータイミングだけを変えて比較する。
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


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_base_signals(data_15m, data_4h):
    """
    二番底・二番天井の判定ロジック（エントリー価格は含まない）
    更新時刻と方向だけを返す
    """
    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    candidates = []

    # 4時間足
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_ct = h4_times[i]
        h4_p1 = data_4h.iloc[i-1]; h4_p2 = data_4h.iloc[i-2]; h4_c = data_4h.iloc[i]
        atr = h4_c["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend = h4_c["trend"]; tol = atr * 0.3

        if trend == 1 and abs(h4_p2["low"]-h4_p1["low"]) <= tol and h4_p1["close"] > h4_p1["open"]:
            sl_base = min(h4_p2["low"], h4_p1["low"]) - atr * 0.15
            candidates.append({"update_time": h4_ct, "dir": 1, "sl_base": sl_base, "atr": atr, "tf": "4h"})

        if trend == -1 and abs(h4_p2["high"]-h4_p1["high"]) <= tol and h4_p1["close"] < h4_p1["open"]:
            sl_base = max(h4_p2["high"], h4_p1["high"]) + atr * 0.15
            candidates.append({"update_time": h4_ct, "dir": -1, "sl_base": sl_base, "atr": atr, "tf": "4h"})

    # 1時間足
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = data_1h.iloc[i-1]; h1_p2 = data_1h.iloc[i-2]; h1_c = data_1h.iloc[i]
        atr = h1_c["atr"]
        if pd.isna(atr) or atr <= 0: continue
        h4_before = data_4h[data_4h.index <= h1_ct]
        if len(h4_before) == 0: continue
        h4_lat = h4_before.iloc[-1]; trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]
        tol = atr * 0.3

        if trend == 1 and abs(h1_p2["low"]-h1_p1["low"]) <= tol and h1_p1["close"] > h1_p1["open"]:
            sl_base = min(h1_p2["low"], h1_p1["low"]) - atr * 0.15
            candidates.append({"update_time": h1_ct, "dir": 1, "sl_base": sl_base, "atr": h4_atr, "tf": "1h"})

        if trend == -1 and abs(h1_p2["high"]-h1_p1["high"]) <= tol and h1_p1["close"] < h1_p1["open"]:
            sl_base = max(h1_p2["high"], h1_p1["high"]) + atr * 0.15
            candidates.append({"update_time": h1_ct, "dir": -1, "sl_base": sl_base, "atr": h4_atr, "tf": "1h"})

    candidates.sort(key=lambda x: x["update_time"])
    return candidates


def build_signals(candidates, data_1m, spread_pips, rr_ratio, mode):
    """
    mode: 'confirmed' = 更新後2分以内の最初の1分足（v75）
          'live'      = 更新直前（14分目）の1分足
    """
    spread = spread_pips * 0.01
    signals = []
    used_times = set()

    for c in candidates:
        update_time = c["update_time"]
        d = c["dir"]; sl = c["sl_base"]; atr = c["atr"]

        if mode == "confirmed":
            # 更新後2分以内の最初の1分足
            end = update_time + pd.Timedelta(minutes=2)
            window = data_1m[(data_1m.index >= update_time) & (data_1m.index < end)]
            if len(window) == 0: continue
            bar = window.iloc[0]

        elif mode == "live":
            # 更新の1分前（確定直前の最後の1分足）
            pre_start = update_time - pd.Timedelta(minutes=1)
            pre_end   = update_time
            window = data_1m[(data_1m.index >= pre_start) & (data_1m.index < pre_end)]
            if len(window) == 0: continue
            bar = window.iloc[-1]  # 直前の1分足

        entry_time = bar.name
        if entry_time in used_times: continue

        if d == 1:
            ep = bar["open"] + spread
            risk = ep - sl
            if 0 < risk <= atr * 3:
                signals.append({"time": entry_time, "dir": 1, "ep": ep, "sl": sl,
                                 "tp": ep + risk * rr_ratio, "risk": risk, "tf": c["tf"]})
                used_times.add(entry_time)
        else:
            ep = bar["open"] - spread
            risk = sl - ep
            if 0 < risk <= atr * 3:
                signals.append({"time": entry_time, "dir": -1, "ep": ep, "sl": sl,
                                 "tp": ep - risk * rr_ratio, "risk": risk, "tf": c["tf"]})
                used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


def run_backtest(data_1m, signals):
    sig_map = {s["time"]: s for s in signals}
    trades = []; pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]; t = bar.name
        if pos is not None:
            d = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    trades.append({**pos,"exit_time":t,"exit_price":half_tp,"pnl":pos["risk"]*100,"type":"HALF_TP"})
                    pos["sl"] = pos["ep"]; pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos,"exit_time":t,"exit_price":pos["sl"],"pnl":pnl,"type":"SL"})
                pos = None; continue
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                pnl = (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({**pos,"exit_time":t,"exit_price":pos["tp"],"pnl":pnl,"type":"TP"})
                pos = None; continue
        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}
    return pd.DataFrame(trades)


def calc_stats(df):
    if df.empty: return {}
    total = df["pnl"].sum()
    closed = df[df["type"].isin(["SL","TP"])]
    wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    nw = len(wins); nl = len(losses)
    avg_w = wins["pnl"].mean() if nw > 0 else 0
    avg_l = losses["pnl"].mean() if nl > 0 else 0
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if nl > 0 and losses["pnl"].sum() != 0 else float("inf")
    wr = nw / (nw + nl) * 100 if (nw + nl) > 0 else 0
    kelly = 0.0
    if avg_l != 0 and (nw + nl) > 0:
        b = abs(avg_w / avg_l); p = nw / (nw + nl)
        kelly = (b * p - (1 - p)) / b
    return {"total": total, "entries": len(closed), "wr": wr, "pf": pf, "kelly": kelly,
            "avg_w": avg_w, "avg_l": avg_l}


def print_stats(stats, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  総損益: {stats['total']:+.2f}pips")
    print(f"  エントリー数: {stats['entries']}回")
    print(f"  勝率: {stats['wr']:.1f}%")
    print(f"  PF: {stats['pf']:.2f}")
    print(f"  平均利益: {stats['avg_w']:+.2f}pips / 平均損失: {stats['avg_l']:+.2f}pips")
    print(f"  ケリー基準: {stats['kelly']:+.4f}")


if __name__ == "__main__":
    print(f"データ読み込み中...")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)
    print(f"  1m: {len(data_1m)}本, 15m: {len(data_15m)}本, 4h: {len(data_4h)}本")

    print("\nシグナル候補生成中...")
    candidates = generate_base_signals(data_15m, data_4h)
    print(f"  候補数: {len(candidates)}件")

    # 確定後（v75と同じ）
    sigs_confirmed = build_signals(candidates, data_1m, SPREAD, RR_RATIO, mode="confirmed")
    df_confirmed = run_backtest(data_1m, sigs_confirmed)
    df_confirmed.to_csv(f"{OUT}/trades_confirmed.csv", index=False)
    stats_c = calc_stats(df_confirmed)
    print_stats(stats_c, "【確定後エントリー】更新後2分以内の最初の1分足始値")

    # 確定前（14分目）
    sigs_live = build_signals(candidates, data_1m, SPREAD, RR_RATIO, mode="live")
    df_live = run_backtest(data_1m, sigs_live)
    df_live.to_csv(f"{OUT}/trades_live.csv", index=False)
    stats_l = calc_stats(df_live)
    print_stats(stats_l, "【確定前エントリー】更新直前の1分足始値（14分目）")

    # 比較表
    print(f"\n\n{'='*60}")
    print("  確定後 vs 確定前 比較サマリー")
    print(f"{'='*60}")
    print(f"  {'指標':<15} {'確定後 (v75)':<20} {'確定前 (live)':<20}")
    print(f"  {'-'*55}")
    print(f"  {'総損益':<15} {stats_c['total']:+.1f}pips{'':<8} {stats_l['total']:+.1f}pips")
    print(f"  {'エントリー数':<15} {stats_c['entries']}回{'':<14} {stats_l['entries']}回")
    print(f"  {'勝率':<15} {stats_c['wr']:.1f}%{'':<13} {stats_l['wr']:.1f}%")
    print(f"  {'PF':<15} {stats_c['pf']:.2f}{'':<15} {stats_l['pf']:.2f}")
    print(f"  {'ケリー基準':<15} {stats_c['kelly']:.4f}{'':<12} {stats_l['kelly']:.4f}")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df, label, color in [
        (axes[0], df_confirmed, "確定後エントリー (v75)", "#2196F3"),
        (axes[1], df_live,      "確定前エントリー (live)", "#FF9800"),
    ]:
        if not df.empty:
            cum = df["pnl"].cumsum()
            ax.plot(range(len(cum)), cum.values, color=color, lw=1.5)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values >= 0), alpha=0.15, color=color)
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values < 0), alpha=0.15, color="#F44336")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("累積損益 (pips)")
        ax.grid(True, alpha=0.4)

    plt.suptitle("15分足確定後 vs 確定前 エントリー比較", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUT}/confirmed_vs_live_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Chart: {out_path}")
