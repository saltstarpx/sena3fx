"""
compare_kmid_klow_1h_vs_4h.py
==============================
1時間足パターンのKMID/KLOWフィルターを
  A) 現在の実装: 直前4H足で判定
  B) 比較対象:   直前1H足で判定
の2バージョンをUSDJPYで比較する。

期間: 2025-01-01 〜 2026-02-28（14ヶ月）
"""

import sys, os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategies" / "current"))

DATA_DIR   = Path(__file__).parent.parent / "data"
USDJPY_1M  = DATA_DIR / "usdjpy_oos_1m.csv"   # OOSデータ（2025/3〜2026/2）
# IS + OOS 両方使うため全期間データを結合
USDJPY_IS  = DATA_DIR / "usdjpy_is_1m.csv"    # ISデータ（2024/7〜2025/2）

SPREAD_PIPS = 0.4
RR_RATIO    = 2.5
KLOW_THRESHOLD = 0.0015


def calculate_atr(df, period=14):
    high_low   = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def check_kmid_klow(bar, direction):
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok = klow_ratio < KLOW_THRESHOLD
    return kmid_ok and klow_ok


def load_data():
    """USDJPYの1分足データを読み込み、4H/1H/1Mに集約する"""
    dfs = []
    for p in [USDJPY_IS, USDJPY_1M]:
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("USDJPYデータが見つかりません")
    data_1m = pd.concat(dfs).sort_index()
    data_1m = data_1m[~data_1m.index.duplicated(keep='first')]

    data_4h = data_1m.resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    data_4h["atr"]   = calculate_atr(data_4h, 14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_1m.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    data_1h["atr"] = calculate_atr(data_1h, 14)

    return data_1m, data_1h, data_4h


def generate_1h_signals(data_1m, data_1h, data_4h, use_1h_filter=False):
    """
    1時間足パターンのシグナルのみ生成。
    use_1h_filter=False → 現在の実装（直前4H足でKMID/KLOW判定）
    use_1h_filter=True  → 比較対象（直前1H足でKMID/KLOW判定）
    """
    spread = SPREAD_PIPS * 0.01
    signals = []
    used_times = set()

    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # 4時間足のトレンドと直前足を取得
        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        tolerance = atr_val * 0.3

        # フィルター対象の足を選択
        filter_bar = h1_prev1 if use_1h_filter else h4_latest

        # ロング: 二番底
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                if not check_kmid_klow(filter_bar, direction=1):
                    continue
                sl = min(low1, low2) - atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * RR_RATIO
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "tf": "1h", "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                if not check_kmid_klow(filter_bar, direction=-1):
                    continue
                sl = max(high1, high2) + atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * RR_RATIO
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "tf": "1h", "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


def simulate_trades(signals, data_1m):
    """シグナルリストからトレード結果を計算する（高速版）"""
    if not signals:
        return []

    # 1分足のhigh/lowをnumpy配列に変換
    idx   = data_1m.index.view(np.int64)  # nanoseconds since epoch
    highs = data_1m["high"].values
    lows  = data_1m["low"].values

    trades = []
    for sig in signals:
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        d  = sig["dir"]; et = sig["time"]

        # エントリー後のインデックスを取得
        et_ns = np.int64(pd.Timestamp(et).value)
        pos = np.searchsorted(idx, et_ns, side='right')
        if pos >= len(idx):
            continue

        fut_high = highs[pos:]
        fut_low  = lows[pos:]

        if d == 1:
            sl_hit = np.where(fut_low  <= sl)[0]
            tp_hit = np.where(fut_high >= tp)[0]
        else:
            sl_hit = np.where(fut_high >= sl)[0]
            tp_hit = np.where(fut_low  <= tp)[0]

        sl_i = sl_hit[0] if len(sl_hit) else np.inf
        tp_i = tp_hit[0] if len(tp_hit) else np.inf

        if sl_i == np.inf and tp_i == np.inf:
            continue  # 未決済

        if tp_i < sl_i:
            exit_type, exit_price = "TP", tp
        else:
            exit_type, exit_price = "SL", sl

        pnl_pips = (exit_price - ep) * d / 0.01
        trades.append({
            "entry_time": et, "dir": d,
            "ep": ep, "sl": sl, "tp": tp,
            "exit_type": exit_type, "exit_price": exit_price,
            "pnl_pips": pnl_pips, "win": 1 if pnl_pips > 0 else 0,
            "tf": sig["tf"], "pattern": sig["pattern"]
        })
    return trades


def summarize(trades, label):
    if not trades:
        print(f"[{label}] トレードなし")
        return
    df = pd.DataFrame(trades)
    n  = len(df)
    wr = df["win"].mean()
    wins   = df[df["pnl_pips"] > 0]["pnl_pips"].sum()
    losses = abs(df[df["pnl_pips"] < 0]["pnl_pips"].sum())
    pf     = wins / losses if losses > 0 else float("inf")
    total  = df["pnl_pips"].sum()
    avg_w  = df[df["pnl_pips"] > 0]["pnl_pips"].mean() if (df["pnl_pips"] > 0).any() else 0
    avg_l  = df[df["pnl_pips"] < 0]["pnl_pips"].mean() if (df["pnl_pips"] < 0).any() else 0

    # 月次集計
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl_pips"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)

    # MDD
    equity = df["pnl_pips"].cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    mdd = abs(dd.min())

    print(f"\n{'='*50}")
    print(f"【{label}】")
    print(f"  トレード数  : {n}")
    print(f"  勝率        : {wr:.1%}")
    print(f"  PF          : {pf:.2f}")
    print(f"  総損益      : {total:+.1f} pips")
    print(f"  平均勝ち    : {avg_w:+.1f} pips")
    print(f"  平均負け    : {avg_l:+.1f} pips")
    print(f"  MDD         : {mdd:.1f} pips")
    print(f"  プラス月    : {plus_months}/{total_months}")
    print(f"{'='*50}")
    return {"label": label, "trades": n, "wr": wr, "pf": pf,
            "total_pips": total, "mdd": mdd,
            "plus_months": plus_months, "total_months": total_months}


if __name__ == "__main__":
    print("データ読み込み中...")
    data_1m, data_1h, data_4h = load_data()
    print(f"  1M: {len(data_1m)}行 ({data_1m.index[0]} 〜 {data_1m.index[-1]})")
    print(f"  1H: {len(data_1h)}行")
    print(f"  4H: {len(data_4h)}行")

    print("\n[A] 現在の実装（1H足パターン → 直前4H足でKMID/KLOW判定）シグナル生成中...")
    sigs_a = generate_1h_signals(data_1m, data_1h, data_4h, use_1h_filter=False)
    print(f"  シグナル数: {len(sigs_a)}")

    print("\n[B] 比較対象（1H足パターン → 直前1H足でKMID/KLOW判定）シグナル生成中...")
    sigs_b = generate_1h_signals(data_1m, data_1h, data_4h, use_1h_filter=True)
    print(f"  シグナル数: {len(sigs_b)}")

    print("\nトレードシミュレーション中...")
    trades_a = simulate_trades(sigs_a, data_1m)
    trades_b = simulate_trades(sigs_b, data_1m)

    res_a = summarize(trades_a, "A: 直前4H足でKMID/KLOW（現在の実装）")
    res_b = summarize(trades_b, "B: 直前1H足でKMID/KLOW（比較対象）")

    print("\n\n【差分サマリー】")
    if res_a and res_b:
        print(f"  トレード数差  : {res_b['trades'] - res_a['trades']:+d} (B-A)")
        print(f"  勝率差        : {(res_b['wr'] - res_a['wr'])*100:+.1f}pt")
        print(f"  PF差          : {res_b['pf'] - res_a['pf']:+.2f}")
        print(f"  総損益差      : {res_b['total_pips'] - res_a['total_pips']:+.1f} pips")
        print(f"  MDD差         : {res_b['mdd'] - res_a['mdd']:+.1f} pips")
