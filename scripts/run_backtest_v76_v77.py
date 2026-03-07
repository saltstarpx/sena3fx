"""
run_backtest_v76_v77.py
=======================
v76: エントリーウィンドウを2分→5分に拡大
v77: 4h+1h+15分足の二番底・二番天井を追加
両方をv75と比較する
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


def generate_signals_v76(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=2.5):
    """v76: エントリーウィンドウを5分に拡大（4h+1h）"""
    spread = spread_pips * 0.01

    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals = []
    used_times = set()

    def find_entry(update_time, window_minutes, data_1m):
        end = update_time + pd.Timedelta(minutes=window_minutes)
        window = data_1m[(data_1m.index >= update_time) & (data_1m.index < end)]
        return window.iloc[0] if len(window) > 0 else None

    # 4時間足
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_ct = h4_times[i]
        h4_p1 = data_4h.iloc[i-1]; h4_p2 = data_4h.iloc[i-2]; h4_c = data_4h.iloc[i]
        atr = h4_c["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend = h4_c["trend"]; tol = atr * 0.3

        if trend == 1 and abs(h4_p2["low"]-h4_p1["low"]) <= tol and h4_p1["close"] > h4_p1["open"]:
            sl = min(h4_p2["low"], h4_p1["low"]) - atr * 0.15
            bar = find_entry(h4_ct, 5, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] + spread; risk = ep - sl
                if 0 < risk <= atr * 3:
                    signals.append({"time":bar.name,"dir":1,"ep":ep,"sl":sl,"tp":ep+risk*rr_ratio,"risk":risk,"tf":"4h","pattern":"double_bottom"})
                    used_times.add(bar.name)

        if trend == -1 and abs(h4_p2["high"]-h4_p1["high"]) <= tol and h4_p1["close"] < h4_p1["open"]:
            sl = max(h4_p2["high"], h4_p1["high"]) + atr * 0.15
            bar = find_entry(h4_ct, 5, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] - spread; risk = sl - ep
                if 0 < risk <= atr * 3:
                    signals.append({"time":bar.name,"dir":-1,"ep":ep,"sl":sl,"tp":ep-risk*rr_ratio,"risk":risk,"tf":"4h","pattern":"double_top"})
                    used_times.add(bar.name)

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
            sl = min(h1_p2["low"], h1_p1["low"]) - atr * 0.15
            bar = find_entry(h1_ct, 5, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] + spread; risk = ep - sl
                if 0 < risk <= h4_atr * 2:
                    signals.append({"time":bar.name,"dir":1,"ep":ep,"sl":sl,"tp":ep+risk*rr_ratio,"risk":risk,"tf":"1h","pattern":"double_bottom"})
                    used_times.add(bar.name)

        if trend == -1 and abs(h1_p2["high"]-h1_p1["high"]) <= tol and h1_p1["close"] < h1_p1["open"]:
            sl = max(h1_p2["high"], h1_p1["high"]) + atr * 0.15
            bar = find_entry(h1_ct, 5, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] - spread; risk = sl - ep
                if 0 < risk <= h4_atr * 2:
                    signals.append({"time":bar.name,"dir":-1,"ep":ep,"sl":sl,"tp":ep-risk*rr_ratio,"risk":risk,"tf":"1h","pattern":"double_top"})
                    used_times.add(bar.name)

    signals.sort(key=lambda x: x["time"])
    return signals


def generate_signals_v77(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=2.5):
    """v77: 4h+1h+15分足の二番底・二番天井（ウィンドウ2分）"""
    spread = spread_pips * 0.01

    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    data_15m_copy = data_15m.copy()
    data_15m_copy["atr"] = calculate_atr(data_15m_copy, period=14)

    signals = []
    used_times = set()

    def find_entry_1m(update_time, window_minutes, data_1m):
        end = update_time + pd.Timedelta(minutes=window_minutes)
        window = data_1m[(data_1m.index >= update_time) & (data_1m.index < end)]
        return window.iloc[0] if len(window) > 0 else None

    # 4時間足（ウィンドウ2分）
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_ct = h4_times[i]
        h4_p1 = data_4h.iloc[i-1]; h4_p2 = data_4h.iloc[i-2]; h4_c = data_4h.iloc[i]
        atr = h4_c["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend = h4_c["trend"]; tol = atr * 0.3

        if trend == 1 and abs(h4_p2["low"]-h4_p1["low"]) <= tol and h4_p1["close"] > h4_p1["open"]:
            sl = min(h4_p2["low"], h4_p1["low"]) - atr * 0.15
            bar = find_entry_1m(h4_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] + spread; risk = ep - sl
                if 0 < risk <= atr * 3:
                    signals.append({"time":bar.name,"dir":1,"ep":ep,"sl":sl,"tp":ep+risk*rr_ratio,"risk":risk,"tf":"4h","pattern":"double_bottom"})
                    used_times.add(bar.name)

        if trend == -1 and abs(h4_p2["high"]-h4_p1["high"]) <= tol and h4_p1["close"] < h4_p1["open"]:
            sl = max(h4_p2["high"], h4_p1["high"]) + atr * 0.15
            bar = find_entry_1m(h4_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] - spread; risk = sl - ep
                if 0 < risk <= atr * 3:
                    signals.append({"time":bar.name,"dir":-1,"ep":ep,"sl":sl,"tp":ep-risk*rr_ratio,"risk":risk,"tf":"4h","pattern":"double_top"})
                    used_times.add(bar.name)

    # 1時間足（ウィンドウ2分）
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
            sl = min(h1_p2["low"], h1_p1["low"]) - atr * 0.15
            bar = find_entry_1m(h1_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] + spread; risk = ep - sl
                if 0 < risk <= h4_atr * 2:
                    signals.append({"time":bar.name,"dir":1,"ep":ep,"sl":sl,"tp":ep+risk*rr_ratio,"risk":risk,"tf":"1h","pattern":"double_bottom"})
                    used_times.add(bar.name)

        if trend == -1 and abs(h1_p2["high"]-h1_p1["high"]) <= tol and h1_p1["close"] < h1_p1["open"]:
            sl = max(h1_p2["high"], h1_p1["high"]) + atr * 0.15
            bar = find_entry_1m(h1_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] - spread; risk = sl - ep
                if 0 < risk <= h4_atr * 2:
                    signals.append({"time":bar.name,"dir":-1,"ep":ep,"sl":sl,"tp":ep-risk*rr_ratio,"risk":risk,"tf":"1h","pattern":"double_top"})
                    used_times.add(bar.name)

    # 15分足（ウィンドウ2分）
    m15_times = data_15m_copy.index.tolist()
    for i in range(2, len(m15_times)):
        m15_ct = m15_times[i]
        m15_p1 = data_15m_copy.iloc[i-1]; m15_p2 = data_15m_copy.iloc[i-2]; m15_c = data_15m_copy.iloc[i]
        atr = m15_c["atr"]
        if pd.isna(atr) or atr <= 0: continue
        h4_before = data_4h[data_4h.index <= m15_ct]
        if len(h4_before) == 0: continue
        h4_lat = h4_before.iloc[-1]; trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]
        tol = atr * 0.3

        if trend == 1 and abs(m15_p2["low"]-m15_p1["low"]) <= tol and m15_p1["close"] > m15_p1["open"]:
            sl = min(m15_p2["low"], m15_p1["low"]) - atr * 0.15
            bar = find_entry_1m(m15_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] + spread; risk = ep - sl
                if 0 < risk <= h4_atr * 1.5:  # 15分足はリスク上限をより厳格に
                    signals.append({"time":bar.name,"dir":1,"ep":ep,"sl":sl,"tp":ep+risk*rr_ratio,"risk":risk,"tf":"15m","pattern":"double_bottom"})
                    used_times.add(bar.name)

        if trend == -1 and abs(m15_p2["high"]-m15_p1["high"]) <= tol and m15_p1["close"] < m15_p1["open"]:
            sl = max(m15_p2["high"], m15_p1["high"]) + atr * 0.15
            bar = find_entry_1m(m15_ct, 2, data_1m)
            if bar is not None and bar.name not in used_times:
                ep = bar["open"] - spread; risk = sl - ep
                if 0 < risk <= h4_atr * 1.5:
                    signals.append({"time":bar.name,"dir":-1,"ep":ep,"sl":sl,"tp":ep-risk*rr_ratio,"risk":risk,"tf":"15m","pattern":"double_top"})
                    used_times.add(bar.name)

    signals.sort(key=lambda x: x["time"])
    return signals


def run_backtest(data_1m, signals):
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None
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


def print_stats(df, label):
    if df.empty:
        print(f"[{label}] トレードなし"); return {}
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
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  総損益: {total:+.2f}pips | エントリー数: {len(closed)}回 | 勝率: {wr:.1f}% | PF: {pf:.2f} | ケリー: {kelly:+.4f}")
    if "tf" in df.columns:
        for tf in ["4h","1h","15m"]:
            tf_df = df[df["tf"]==tf]
            if len(tf_df) == 0: continue
            tf_closed = tf_df[tf_df["type"].isin(["SL","TP"])]
            tf_w = tf_df[tf_df["pnl"]>0]; tf_l = tf_df[tf_df["pnl"]<0]
            tf_wr = len(tf_w)/(len(tf_w)+len(tf_l))*100 if (len(tf_w)+len(tf_l))>0 else 0
            tf_pf = tf_w["pnl"].sum()/abs(tf_l["pnl"].sum()) if len(tf_l)>0 and tf_l["pnl"].sum()!=0 else float("inf")
            print(f"  [{tf}] {len(tf_closed)}回 | 勝率{tf_wr:.1f}% | PF{tf_pf:.2f} | {tf_df['pnl'].sum():+.1f}pips")
    df2 = df.copy(); df2["exit_time"] = pd.to_datetime(df2["exit_time"])
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly_pnl = df2.groupby("month")["pnl"].sum()
    closed2 = closed.copy(); closed2["exit_time"] = pd.to_datetime(closed2["exit_time"])
    closed2["month"] = closed2["exit_time"].dt.to_period("M")
    monthly_cnt = closed2.groupby("month")["pnl"].count()
    print("  月別損益:")
    for m in monthly_pnl.index:
        cnt = monthly_cnt.get(m, 0); v = monthly_pnl[m]
        print(f"    {m}: {'+' if v>=0 else ''}{v:7.1f}pips ({cnt}回)")
    return {"total":total,"entries":len(closed),"wr":wr,"pf":pf,"kelly":kelly}


def plot_equity(df, label, out_path):
    if df.empty: return
    cum = df["pnl"].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(range(len(cum)), cum.values, color="#2196F3", lw=1.5)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].fill_between(range(len(cum)), cum.values, 0, where=(cum.values>=0), alpha=0.15, color="#2196F3")
    axes[0].fill_between(range(len(cum)), cum.values, 0, where=(cum.values<0), alpha=0.15, color="#F44336")
    axes[0].set_title(f"{label} - 累積損益 (pips)", fontsize=13); axes[0].set_ylabel("pips"); axes[0].grid(True, alpha=0.4)
    df2 = df.copy(); df2["exit_time"] = pd.to_datetime(df2["exit_time"])
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    colors = ["#2196F3" if v>=0 else "#F44336" for v in monthly.values]
    axes[1].bar(range(len(monthly)), monthly.values, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(monthly)))
    axes[1].set_xticklabels([str(m) for m in monthly.index], rotation=45)
    axes[1].axhline(0, color="gray", lw=0.8); axes[1].set_title("月別損益 (pips)", fontsize=13)
    axes[1].set_ylabel("pips"); axes[1].grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f"  Chart: {out_path}")


if __name__ == "__main__":
    print(f"データ読み込み中... ({START} 〜 {END})")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)
    print(f"  1m足: {len(data_1m)}本, 15m足: {len(data_15m)}本, 4h足: {len(data_4h)}本")

    # v76: 5分ウィンドウ
    print("\n\n[v76] シグナル生成中（5分ウィンドウ）...")
    sigs_v76 = generate_signals_v76(data_1m, data_15m, data_4h, SPREAD, RR_RATIO)
    print(f"  シグナル数: {len(sigs_v76)}本")
    df_v76 = run_backtest(data_1m, sigs_v76)
    df_v76.to_csv(f"{OUT}/trades_v76.csv", index=False)
    stats_v76 = print_stats(df_v76, f"v76 | 5分ウィンドウ | {SYMBOL} spread={SPREAD}pips RR={RR_RATIO}")
    plot_equity(df_v76, "v76 (4h+1h 5分ウィンドウ)", f"{OUT}/v76_equity_curve.png")

    # v77: 15分足追加
    print("\n\n[v77] シグナル生成中（4h+1h+15m）...")
    sigs_v77 = generate_signals_v77(data_1m, data_15m, data_4h, SPREAD, RR_RATIO)
    print(f"  シグナル数: {len(sigs_v77)}本")
    db4 = sum(1 for s in sigs_v77 if s["tf"]=="4h"); dt4_c = sum(1 for s in sigs_v77 if s["tf"]=="4h" and s["pattern"]=="double_top")
    db1 = sum(1 for s in sigs_v77 if s["tf"]=="1h"); db15 = sum(1 for s in sigs_v77 if s["tf"]=="15m")
    print(f"  4h: {db4}本 / 1h: {db1}本 / 15m: {db15}本")
    df_v77 = run_backtest(data_1m, sigs_v77)
    df_v77.to_csv(f"{OUT}/trades_v77.csv", index=False)
    stats_v77 = print_stats(df_v77, f"v77 | 4h+1h+15m | {SYMBOL} spread={SPREAD}pips RR={RR_RATIO}")
    plot_equity(df_v77, "v77 (4h+1h+15m 二番底・二番天井)", f"{OUT}/v77_equity_curve.png")

    # 比較サマリー
    print("\n\n" + "="*65)
    print("  v75 vs v76 vs v77 比較サマリー")
    print("="*65)
    try:
        df_v75 = pd.read_csv(f"{OUT}/trades_v75.csv")
        v75_total = df_v75["pnl"].sum()
        v75_closed = df_v75[df_v75["type"].isin(["SL","TP"])]
        v75_wins = df_v75[df_v75["pnl"]>0]; v75_losses = df_v75[df_v75["pnl"]<0]
        v75_wr = len(v75_wins)/(len(v75_wins)+len(v75_losses))*100 if (len(v75_wins)+len(v75_losses))>0 else 0
        v75_pf = v75_wins["pnl"].sum()/abs(v75_losses["pnl"].sum()) if len(v75_losses)>0 else float("inf")
        print(f"  {'指標':<12} {'v75 (2分窓)':<18} {'v76 (5分窓)':<18} {'v77 (4h+1h+15m)':<18}")
        print(f"  {'-'*66}")
        print(f"  {'総損益':<12} {v75_total:+.0f}pips{'':<8} {stats_v76.get('total',0):+.0f}pips{'':<8} {stats_v77.get('total',0):+.0f}pips")
        print(f"  {'エントリー':<12} {len(v75_closed)}回{'':<13} {stats_v76.get('entries',0)}回{'':<13} {stats_v77.get('entries',0)}回")
        print(f"  {'勝率':<12} {v75_wr:.1f}%{'':<13} {stats_v76.get('wr',0):.1f}%{'':<13} {stats_v77.get('wr',0):.1f}%")
        print(f"  {'PF':<12} {v75_pf:.2f}{'':<14} {stats_v76.get('pf',0):.2f}{'':<14} {stats_v77.get('pf',0):.2f}")
        print(f"  {'ケリー':<12} 0.5421{'':<12} {stats_v76.get('kelly',0):.4f}{'':<10} {stats_v77.get('kelly',0):.4f}")
    except Exception as e:
        print(f"  比較エラー: {e}")
