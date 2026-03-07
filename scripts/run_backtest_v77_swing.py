"""
run_backtest_v77_swing.py
=========================
v75ロジック + 半利確後の残りを直近スイングハイ/スイングロー（壁）まで伸ばす

【変更点】
- v75: 半利確（1R）→ 残りをTP固定2.5Rで決済
- v76: 半利確（1R）→ 残りを4hEMA20まで伸ばす（壁到達1回のみで機能不全）
- v77: 半利確（1R）→ 残りを4時間足の直近スイングハイ/スイングローまで伸ばす
         壁に届かない場合は次の4時間足更新時に強制決済

【スイング検出ロジック】
- ロング: エントリー前の4時間足データから直近スイングハイ（N本前の高値の最大値）
- ショート: エントリー前の4時間足データから直近スイングロー（N本前の安値の最小値）
- 探索範囲: 直近5〜20本の4時間足（約20〜80時間）
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')
from yagami_mtf_v75 import generate_signals

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
RR_RATIO = 2.5   # 半利確のタイミング（1R到達）
SWING_LOOKBACK = 10  # 直近何本の4時間足でスイングを探すか
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


def get_swing_wall(data_4h, entry_time, direction, entry_price, lookback=SWING_LOOKBACK):
    """
    エントリー時点での直近スイングハイ/スイングローを壁として取得する

    ロング: エントリー価格より上にある直近スイングハイ
    ショート: エントリー価格より下にある直近スイングロー
    """
    h4_before = data_4h[data_4h.index < entry_time].tail(lookback)
    if len(h4_before) < 3:
        return None, None

    # 次の4時間足更新時刻（壁に届かない場合の強制決済タイミング）
    h4_after = data_4h[data_4h.index > entry_time]
    next_h4_time = h4_after.index[0] if len(h4_after) > 0 else None

    if direction == 1:  # ロング: 上のスイングハイを探す
        # スイングハイ = 前後の足より高値が高い足
        swing_highs = []
        for i in range(1, len(h4_before) - 1):
            if h4_before.iloc[i]["high"] > h4_before.iloc[i-1]["high"] and \
               h4_before.iloc[i]["high"] > h4_before.iloc[i+1]["high"]:
                swing_highs.append(h4_before.iloc[i]["high"])
        # エントリー価格より上のスイングハイのうち最も近いもの
        candidates = [h for h in swing_highs if h > entry_price]
        if candidates:
            return min(candidates), next_h4_time
        # スイングハイがない場合は直近lookback本の最高値
        wall = h4_before["high"].max()
        if wall > entry_price:
            return wall, next_h4_time

    else:  # ショート: 下のスイングローを探す
        # スイングロー = 前後の足より安値が低い足
        swing_lows = []
        for i in range(1, len(h4_before) - 1):
            if h4_before.iloc[i]["low"] < h4_before.iloc[i-1]["low"] and \
               h4_before.iloc[i]["low"] < h4_before.iloc[i+1]["low"]:
                swing_lows.append(h4_before.iloc[i]["low"])
        # エントリー価格より下のスイングローのうち最も近いもの
        candidates = [l for l in swing_lows if l < entry_price]
        if candidates:
            return max(candidates), next_h4_time
        # スイングローがない場合は直近lookback本の最安値
        wall = h4_before["low"].min()
        if wall < entry_price:
            return wall, next_h4_time

    return None, next_h4_time


def run_backtest_v77(data_1m, data_4h, signals):
    """半利確後の残りを直近スイングハイ/スイングローまで伸ばすバックテスト"""
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name

        if pos is not None:
            d = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]

            # ── 半利確 ──
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"] <= half_tp):
                    pnl = pos["risk"] * 100
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": pnl, "type": "HALF_TP"})
                    pos["sl"] = pos["ep"]  # SLを建値に移動
                    pos["half_closed"] = True

            # ── 損切り（建値に移動後） ──
            if (d == 1 and bar["low"] <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"],
                               "pnl": pnl, "type": "SL"})
                pos = None
                continue

            # ── スイング壁到達で利確 ──
            wall = pos.get("wall_tp")
            if wall is not None and pos["half_closed"]:
                if (d == 1 and bar["high"] >= wall) or \
                   (d == -1 and bar["low"] <= wall):
                    pnl = (wall - pos["ep"]) * 100 * d
                    trades.append({**pos, "exit_time": t, "exit_price": wall,
                                   "pnl": pnl, "type": "SWING_TP"})
                    pos = None
                    continue

            # ── 次の4時間足更新で強制決済 ──
            next_h4 = pos.get("next_h4_time")
            if next_h4 is not None and t >= next_h4 and pos["half_closed"]:
                exit_price = bar["open"]
                pnl = (exit_price - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": exit_price,
                               "pnl": pnl, "type": "H4_EXPIRE"})
                pos = None
                continue

            # ── 固定TP（壁が設定できなかった場合のフォールバック） ──
            if wall is None:
                tp = pos["tp"]
                if (d == 1 and bar["high"] >= tp) or \
                   (d == -1 and bar["low"] <= tp):
                    pnl = (tp - pos["ep"]) * 100 * d
                    trades.append({**pos, "exit_time": t, "exit_price": tp,
                                   "pnl": pnl, "type": "TP"})
                    pos = None
                    continue

        # ── 新規エントリー ──
        if pos is None and t in sig_map:
            s = sig_map[t]
            wall_tp, next_h4_time = get_swing_wall(data_4h, t, s["dir"], s["ep"])
            pos = {**s, "entry_time": t, "half_closed": False,
                   "wall_tp": wall_tp, "next_h4_time": next_h4_time}

    return pd.DataFrame(trades)


def run_backtest_v75(data_1m, signals):
    """v75オリジナル（固定RR2.5）比較用"""
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t = bar.name
        if pos is not None:
            d = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": pos["risk"] * 100, "type": "HALF_TP"})
                    pos["sl"] = pos["ep"]; pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"], "pnl": pnl, "type": "SL"})
                pos = None; continue
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                pnl = (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["tp"], "pnl": pnl, "type": "TP"})
                pos = None; continue
        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}
    return pd.DataFrame(trades)


def calc_stats(df, label):
    if df.empty:
        print(f"[{label}] トレードなし"); return {}
    total = df["pnl"].sum()
    closed_types = ["SL", "TP", "SWING_TP", "H4_EXPIRE"]
    closed = df[df["type"].isin(closed_types)]
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

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  総損益: {total:+.2f}pips")
    print(f"  エントリー数（決済済み）: {len(closed)}回")
    print(f"  勝率: {wr:.1f}%  PF: {pf:.2f}  ケリー: {kelly:+.4f}")
    print(f"  平均利益: {avg_w:+.2f}pips / 平均損失: {avg_l:+.2f}pips")
    if "type" in df.columns:
        print(f"  決済内訳: {dict(df['type'].value_counts())}")
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"]).dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    print(f"  月別損益:")
    for m, v in monthly.items():
        mark = "✓" if v >= 0 else "✗"
        print(f"    {mark} {m}: {v:+.1f}pips")
    return {"total": total, "entries": len(closed), "wr": wr, "pf": pf, "kelly": kelly}


if __name__ == "__main__":
    print("データ読み込み中...")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)
    data_4h["atr"]   = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    print(f"  1m: {len(data_1m)}本, 15m: {len(data_15m)}本, 4h: {len(data_4h)}本")

    print("\nシグナル生成中...")
    signals = generate_signals(data_1m, data_15m, data_4h, spread_pips=SPREAD, rr_ratio=RR_RATIO)
    print(f"  シグナル数: {len(signals)}本")

    # v75（固定RR2.5）
    df_v75 = run_backtest_v75(data_1m, signals)
    stats_v75 = calc_stats(df_v75, "v75 固定RR2.5（半利確→残り2.5R）")

    # v77（スイング壁TP）
    df_v77 = run_backtest_v77(data_1m, data_4h, signals)
    df_v77.to_csv(f"{OUT}/trades_v77_swing.csv", index=False)
    stats_v77 = calc_stats(df_v77, f"v77 スイング壁TP（直近{SWING_LOOKBACK}本の高値/安値）")

    # 比較サマリー
    print(f"\n\n{'='*60}")
    print("  v75 vs v77 比較サマリー")
    print(f"{'='*60}")
    print(f"  {'指標':<15} {'v75 固定RR2.5':<22} {'v77 スイング壁':<22}")
    print(f"  {'-'*59}")
    labels = [("total","総損益(pips)"), ("entries","エントリー数"), ("wr","勝率(%)"), ("pf","PF"), ("kelly","ケリー基準")]
    for k, lbl in labels:
        v75v = stats_v75.get(k, 0); v77v = stats_v77.get(k, 0)
        print(f"  {lbl:<15} {str(round(v75v,2)):<22} {str(round(v77v,2)):<22}")

    # 可視化（3パネル）
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    configs = [
        (axes[0], df_v75, "v75 固定RR2.5", "#2196F3"),
        (axes[1], df_v77, f"v77 スイング壁TP（直近{SWING_LOOKBACK}本）", "#FF9800"),
    ]
    for ax, df, label, color in configs:
        if not df.empty:
            cum = df["pnl"].cumsum()
            ax.plot(range(len(cum)), cum.values, color=color, lw=1.5)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values >= 0), alpha=0.15, color=color)
            ax.fill_between(range(len(cum)), cum.values, 0,
                            where=(cum.values < 0), alpha=0.15, color="#F44336")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("累積損益 (pips)")
        ax.grid(True, alpha=0.4)

    # 月別比較棒グラフ
    ax3 = axes[2]
    months_v75 = {}; months_v77 = {}
    for df, d in [(df_v75, months_v75), (df_v77, months_v77)]:
        if not df.empty:
            df2 = df.copy()
            df2["month"] = pd.to_datetime(df2["exit_time"]).dt.to_period("M")
            for m, v in df2.groupby("month")["pnl"].sum().items():
                d[str(m)] = v
    all_months = sorted(set(list(months_v75.keys()) + list(months_v77.keys())))
    x = np.arange(len(all_months)); w = 0.35
    v75_vals = [months_v75.get(m, 0) for m in all_months]
    v77_vals = [months_v77.get(m, 0) for m in all_months]
    ax3.bar(x - w/2, v75_vals, w, label="v75 固定RR2.5", color="#2196F3", alpha=0.8)
    ax3.bar(x + w/2, v77_vals, w, label="v77 スイング壁", color="#FF9800", alpha=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels([m[-5:] for m in all_months], rotation=45, fontsize=8)
    ax3.axhline(0, color="gray", lw=0.8); ax3.set_title("月別損益比較", fontsize=11)
    ax3.set_ylabel("損益 (pips)"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4, axis="y")

    plt.suptitle("v75（固定RR2.5）vs v77（直近スイング壁TP）比較", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUT}/v75_vs_v77_swing_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Chart: {out_path}")
