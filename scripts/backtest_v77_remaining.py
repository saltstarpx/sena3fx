"""
残り8銘柄（USDCAD〜SPX500）のみv77（1Hベース）バックテストを実行する軽量スクリプト
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
IS_START  = "2024-07-01"; IS_END  = "2025-02-28"
OOS_START = "2025-03-03"; OOS_END = "2026-02-27"

PAIRS = {
    "USDCAD": {"spread": 2.2, "pip": 0.0001, "sym": "usdcad"},
    "USDCHF": {"spread": 1.8, "pip": 0.0001, "sym": "usdchf"},
    "NZDUSD": {"spread": 1.7, "pip": 0.0001, "sym": "nzdusd"},
    "EURJPY": {"spread": 2.0, "pip": 0.01,   "sym": "eurjpy"},
    "GBPJPY": {"spread": 3.0, "pip": 0.01,   "sym": "gbpjpy"},
    "EURGBP": {"spread": 1.7, "pip": 0.0001, "sym": "eurgbp"},
    "US30":   {"spread": 3.0, "pip": 1.0,    "sym": "us30"},
    "SPX500": {"spread": 0.5, "pip": 0.1,    "sym": "spx500"},
}

def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def slice_period(df, start, end):
    if df is None:
        return None
    return df.loc[start:end]

def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    """v77 1Hベースシグナル生成（backtest_v77_all_pairs.pyと同一ロジック）"""
    spread = spread_pips * pip_size

    # 1H足をリサンプリング
    data_1h = data_15m.resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    ).dropna()

    # 4H ATR
    tr_4h = pd.concat([
        data_4h["high"] - data_4h["low"],
        (data_4h["high"] - data_4h["close"].shift()).abs(),
        (data_4h["low"]  - data_4h["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(14).mean()

    signals = []
    bars_1h = data_1h.index.tolist()

    for i in range(2, len(bars_1h)):
        t0 = bars_1h[i - 2]
        t1 = bars_1h[i - 1]
        t2 = bars_1h[i]

        b0 = data_1h.loc[t0]
        b1 = data_1h.loc[t1]
        b2 = data_1h.loc[t2]

        # 4H EMAトレンド
        t2_4h = data_4h.index[data_4h.index <= t2]
        if len(t2_4h) < 20:
            continue
        ema20_4h = data_4h["close"].loc[:t2].ewm(span=20, adjust=False).mean().iloc[-1]
        last_4h_close = data_4h["close"].loc[:t2].iloc[-1]
        trend_up = last_4h_close > ema20_4h

        # ATR取得
        atr_vals = atr_4h.loc[:t2].dropna()
        if len(atr_vals) == 0:
            continue
        atr = atr_vals.iloc[-1]
        if atr <= 0:
            continue

        # 二番底（ロング）
        if trend_up:
            low0, low1 = b0["low"], b1["low"]
            if abs(low0 - low1) <= atr * 0.3:
                # 確認足（b2が陽線）
                if b2["close"] > b2["open"]:
                    # KMID: 直前4H足の実体が陽線
                    last_4h = data_4h.loc[:t2].iloc[-1]
                    if last_4h["close"] <= last_4h["open"]:
                        continue
                    # KLOW: 下ヒゲ比率 < 0.15%
                    body_low = min(last_4h["open"], last_4h["close"])
                    if last_4h["open"] > 0 and (body_low - last_4h["low"]) / last_4h["open"] >= 0.0015:
                        continue
                    sl = min(low0, low1) - atr * 0.15
                    risk = b2["close"] - sl
                    if risk <= 0 or risk > atr * 3:
                        continue
                    tp = b2["close"] + risk * rr_ratio
                    # エントリー: t2の次の1分足始値
                    entry_bars = data_1m.loc[t2:].iloc[1:3]
                    if len(entry_bars) == 0:
                        continue
                    ep = entry_bars.iloc[0]["open"] + spread
                    signals.append({
                        "entry_time": entry_bars.index[0],
                        "direction": "long",
                        "ep": ep, "sl": sl, "tp": tp,
                        "risk": risk, "atr": atr,
                    })

        # 二番天井（ショート）
        if not trend_up:
            high0, high1 = b0["high"], b1["high"]
            if abs(high0 - high1) <= atr * 0.3:
                if b2["close"] < b2["open"]:
                    last_4h = data_4h.loc[:t2].iloc[-1]
                    if last_4h["close"] >= last_4h["open"]:
                        continue
                    body_high = max(last_4h["open"], last_4h["close"])
                    if last_4h["open"] > 0 and (last_4h["high"] - body_high) / last_4h["open"] >= 0.0015:
                        continue
                    sl = max(high0, high1) + atr * 0.15
                    risk = sl - b2["close"]
                    if risk <= 0 or risk > atr * 3:
                        continue
                    tp = b2["close"] - risk * rr_ratio
                    entry_bars = data_1m.loc[t2:].iloc[1:3]
                    if len(entry_bars) == 0:
                        continue
                    ep = entry_bars.iloc[0]["open"] - spread
                    signals.append({
                        "entry_time": entry_bars.index[0],
                        "direction": "short",
                        "ep": ep, "sl": sl, "tp": tp,
                        "risk": risk, "atr": atr,
                    })

    return pd.DataFrame(signals)

def simulate(signals, data_1m, init_cash, risk_pct, half_r=1.0):
    if len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash])
    cash = init_cash
    equity = [cash]
    trades = []

    for _, sig in signals.iterrows():
        ep = sig["ep"]
        sl = sig["sl"]
        tp = sig["tp"]
        risk = abs(ep - sl)
        if risk <= 0:
            continue
        pos_size = INIT_CASH * risk_pct / risk  # 固定ポジションサイズ（初期資金ベース）
        direction = sig["direction"]
        entry_time = sig["entry_time"]

        # 1分足でSL/TP判定
        future = data_1m.loc[entry_time:]
        if len(future) == 0:
            continue

        half_tp = ep + risk * half_r if direction == "long" else ep - risk * half_r
        half_done = False
        exit_price = None
        exit_time = None
        result = None

        for bar_t, bar in future.iterrows():
            if direction == "long":
                if not half_done and bar["high"] >= half_tp:
                    # 半利確
                    half_done = True
                    sl = ep  # SLをBEへ
                if bar["low"] <= sl:
                    exit_price = sl
                    exit_time = bar_t
                    result = "loss" if exit_price < ep else "win"
                    break
                if bar["high"] >= tp:
                    exit_price = tp
                    exit_time = bar_t
                    result = "win"
                    break
            else:
                if not half_done and bar["low"] <= half_tp:
                    half_done = True
                    sl = ep
                if bar["high"] >= sl:
                    exit_price = sl
                    exit_time = bar_t
                    result = "loss" if exit_price > ep else "win"
                    break
                if bar["low"] <= tp:
                    exit_price = tp
                    exit_time = bar_t
                    result = "win"
                    break

        if exit_price is None:
            continue

        pnl = (exit_price - ep) * pos_size if direction == "long" else (ep - exit_price) * pos_size
        cash += pnl
        equity.append(cash)
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price,
            "result": result,
            "pnl": pnl,
            "equity": cash,
        })

    return pd.DataFrame(trades), pd.Series(equity)

def calc_stats(trades, eq_series, label):
    if len(trades) == 0:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0,
                "return_pct": 0, "mdd_pct": 0, "kelly": 0, "monthly_plus": "N/A"}
    wins  = trades[trades["result"] == "win"]
    loses = trades[trades["result"] == "loss"]
    n  = len(trades)
    wr = len(wins) / n if n > 0 else 0
    gross_win  = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(loses["pnl"].sum()) if len(loses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    eq = eq_series.values
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    mdd  = dd.min()
    ret  = (eq[-1] - eq[0]) / eq[0]
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    # 月次プラス
    if "exit_time" in trades.columns and len(trades) > 0:
        t2 = trades.copy()
        t2["exit_time"] = pd.to_datetime(t2["exit_time"], utc=True)
        t2["month"] = t2["exit_time"].dt.to_period("M")
        monthly = t2.groupby("month")["equity"].last()
        monthly_pnl = monthly.diff().fillna(monthly.iloc[0] - INIT_CASH)
        n_plus  = (monthly_pnl > 0).sum()
        n_total = len(monthly_pnl)
        monthly_str = f"{n_plus}/{n_total}"
    else:
        monthly_str = "N/A"
    return {
        "label": label, "n": n,
        "winrate": wr * 100, "pf": pf,
        "return_pct": ret * 100,
        "mdd_pct": abs(mdd) * 100,
        "kelly": kelly,
        "monthly_plus": monthly_str,
    }

# ── メイン ──────────────────────────────────────────────
for pair, cfg in PAIRS.items():
    sym    = cfg["sym"]
    spread = cfg["spread"]
    pip    = cfg["pip"]
    print(f"\n--- {pair} (スプレッド: {spread}pips) ---", flush=True)
    d1m_is   = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_1m.csv")),   IS_START,  IS_END)
    d15m_is  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv")),  IS_START,  IS_END)
    d4h_is   = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv")),   IS_START,  IS_END)
    d1m_oos  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_1m.csv")),  OOS_START, OOS_END)
    d15m_oos = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv")), OOS_START, OOS_END)
    d4h_oos  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv")),  OOS_START, OOS_END)
    if any(d is None for d in [d1m_is, d15m_is, d4h_is, d1m_oos, d15m_oos, d4h_oos]):
        print("  [SKIP] データ不足")
        continue
    for period, d1m, d15m, d4h in [
        ("IS",  d1m_is,  d15m_is,  d4h_is),
        ("OOS", d1m_oos, d15m_oos, d4h_oos),
    ]:
        sigs = generate_signals_1h(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
        trades, eq = simulate(sigs, d1m, INIT_CASH, RISK_PCT, HALF_R)
        st = calc_stats(trades, eq, f"{pair}_{period}")
        print(f"  [{period}] {st['n']}件 | 勝率{st['winrate']:.1f}% | PF{st['pf']:.2f} | "
              f"リターン+{st['return_pct']:.1f}% | MDD{st['mdd_pct']:.1f}% | "
              f"ケリー{st['kelly']:.3f} | 月次+{st['monthly_plus']}", flush=True)

print("\n完了")
