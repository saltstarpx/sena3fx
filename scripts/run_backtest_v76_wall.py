"""
run_backtest_v76_wall.py
========================
v75ロジック + 半利確後の残りを4時間足EMA20（壁）まで伸ばす

【変更点】
- v75: 半利確（1R）→ 残りをTP固定2.5Rで決済
- v76: 半利確（1R）→ 残りを4時間足EMA20（壁）まで伸ばす
         壁に届かない場合は次の4時間足更新時に強制決済
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
RR_RATIO = 2.5   # 半利確のタイミング（1R到達）に使用
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


def get_4h_ema20_wall(data_4h, entry_time, direction):
    """
    エントリー時点での4時間足EMA20（壁）を取得する
    ロング: EMA20より上の次の4時間足の高値付近
    ショート: EMA20より下の次の4時間足の安値付近
    """
    h4_before = data_4h[data_4h.index <= entry_time]
    if len(h4_before) == 0:
        return None, None
    h4_current = h4_before.iloc[-1]
    ema20 = h4_current["ema20"]
    # 次の4時間足更新時刻
    h4_after = data_4h[data_4h.index > entry_time]
    next_h4_time = h4_after.index[0] if len(h4_after) > 0 else None
    return ema20, next_h4_time


def run_backtest_v76(data_1m, data_4h, signals):
    """
    半利確後の残りを4時間足EMA20（壁）まで伸ばすバックテスト
    """
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

            # ── 壁（EMA20）到達で利確 ──
            wall = pos.get("wall_tp")
            if wall is not None and pos["half_closed"]:
                if (d == 1 and bar["high"] >= wall) or \
                   (d == -1 and bar["low"] <= wall):
                    pnl = (wall - pos["ep"]) * 100 * d
                    trades.append({**pos, "exit_time": t, "exit_price": wall,
                                   "pnl": pnl, "type": "WALL_TP"})
                    pos = None
                    continue

            # ── 次の4時間足更新で強制決済（壁に届かなかった場合） ──
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
            # 壁（EMA20）と次の4時間足更新時刻を取得
            ema20, next_h4_time = get_4h_ema20_wall(data_4h, t, s["dir"])
            wall_tp = None
            if ema20 is not None:
                if s["dir"] == 1:
                    # ロング: EMA20が現在価格より上なら壁として使用
                    if ema20 > s["ep"]:
                        wall_tp = ema20 - SPREAD * 0.01  # スプレッド分手前
                else:
                    # ショート: EMA20が現在価格より下なら壁として使用
                    if ema20 < s["ep"]:
                        wall_tp = ema20 + SPREAD * 0.01

            pos = {**s, "entry_time": t, "half_closed": False,
                   "wall_tp": wall_tp, "next_h4_time": next_h4_time}

    return pd.DataFrame(trades)


def run_backtest_v75(data_1m, signals):
    """v75オリジナル（固定RR2.5）"""
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
    closed = df[df["type"].isin(["SL", "TP", "WALL_TP", "H4_EXPIRE"])]
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
    print(f"  エントリー数: {len(closed)}回")
    print(f"  勝率: {wr:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  平均利益: {avg_w:+.2f}pips / 平均損失: {avg_l:+.2f}pips")
    print(f"  ケリー基準: {kelly:+.4f}")

    # 決済種別の内訳
    if "type" in df.columns:
        type_counts = df["type"].value_counts()
        print(f"  決済内訳: {dict(type_counts)}")

    # 月別
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"]).dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    print(f"  月別損益:")
    for m, v in monthly.items():
        print(f"    {m}: {v:+.1f}pips")

    return {"total": total, "entries": len(closed), "wr": wr, "pf": pf, "kelly": kelly}


if __name__ == "__main__":
    print("データ読み込み中...")
    data_1m  = load_data(SYMBOL, "1m",  START, END)
    data_15m = load_data(SYMBOL, "15m", START, END)
    data_4h  = load_data(SYMBOL, "4h",  START, END)

    # 4時間足にEMA20を追加
    data_4h["atr"]   = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()

    print(f"  1m: {len(data_1m)}本, 15m: {len(data_15m)}本, 4h: {len(data_4h)}本")

    print("\nシグナル生成中...")
    signals = generate_signals(data_1m, data_15m, data_4h, spread_pips=SPREAD, rr_ratio=RR_RATIO)
    print(f"  シグナル数: {len(signals)}本")

    # v75（固定RR2.5）
    df_v75 = run_backtest_v75(data_1m, signals)
    df_v75.to_csv(f"{OUT}/trades_v75_ref.csv", index=False)
    stats_v75 = calc_stats(df_v75, "v75 固定RR2.5（半利確→残り2.5R）")

    # v76（壁まで伸ばす）
    df_v76 = run_backtest_v76(data_1m, data_4h, signals)
    df_v76.to_csv(f"{OUT}/trades_v76_wall.csv", index=False)
    stats_v76 = calc_stats(df_v76, "v76 壁TP（半利確→残りを4hEMA20まで）")

    # 比較サマリー
    print(f"\n\n{'='*60}")
    print("  v75 vs v76 比較サマリー")
    print(f"{'='*60}")
    print(f"  {'指標':<15} {'v75 固定RR2.5':<20} {'v76 壁TP':<20}")
    print(f"  {'-'*55}")
    for k, label in [("total","総損益"), ("entries","エントリー数"), ("wr","勝率"), ("pf","PF"), ("kelly","ケリー基準")]:
        v75v = stats_v75.get(k, 0); v76v = stats_v76.get(k, 0)
        if k == "total":
            print(f"  {label:<15} {v75v:+.1f}pips{'':<8} {v76v:+.1f}pips")
        elif k in ("wr",):
            print(f"  {label:<15} {v75v:.1f}%{'':<13} {v76v:.1f}%")
        else:
            print(f"  {label:<15} {v75v:<20} {v76v:<20}")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df, label, color in [
        (axes[0], df_v75, "v75 固定RR2.5", "#2196F3"),
        (axes[1], df_v76, "v76 壁TP（4hEMA20）", "#4CAF50"),
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

    plt.suptitle("v75（固定RR2.5）vs v76（壁まで伸ばす）比較", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUT}/v75_vs_v76_wall_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Chart: {out_path}")
