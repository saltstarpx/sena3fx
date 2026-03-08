#!/usr/bin/env python3.11
"""
backtest_strategy_comparison.py
================================
v77 / F1+F3 / Hybrid の3戦略を全銘柄で比較し、最良の組み合わせを選定する。

【3戦略の定義】
  v77    : KMID+KLOW（4H実体フィルター）+ 1H足パターンのみ + 時間制限なし
  F1+F3  : UTC5-15時制限 + 4H&1H両パターン + KLOWを方向対称チェック
  Hybrid : F1+F3構造 + v77の厳格なKLOW（下ヒゲ固定 0.15%）

【合格基準】
  PF ≥ 3.0 / WR ≥ 65% / MDD ≤ 20%(ピーク比) / Kelly ≥ 0.45 / +月 ≥ 90%

【データ】 data/ohlc/{SYMBOL}_1m.csv / _15m.csv / _4h.csv
【スプレッド】 utils/risk_manager.SYMBOL_CONFIG の実測値

【出力】
  results/strategy_comparison_summary.csv
  results/strategy_comparison.png
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from utils.risk_manager import SYMBOL_CONFIG, AdaptiveRiskManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ─────────────────────────────────────────────────────
INIT_CASH   = 1_000_000
BASE_RISK   = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
GOOD_HOURS  = list(range(5, 16))   # UTC 5〜15時 (F1フィルター)

DATA_DIR    = BASE_DIR / "data" / "ohlc"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CRITERIA = {
    "pf":          3.0,
    "win_rate":    65.0,
    "mdd_pct":     20.0,
    "kelly":       0.45,
    "plus_months": 90.0,
}

STRATEGY_NAMES = ["v77", "F1+F3", "Hybrid"]


# ─────────────────────────────────────────────────────────────
#  データ読み込み
# ─────────────────────────────────────────────────────────────
def load_ohlc(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = (df.rename(columns={ts_col: "timestamp"})
           .set_index("timestamp")
           .sort_index())
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


# ─────────────────────────────────────────────────────────────
#  指標
# ─────────────────────────────────────────────────────────────
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df


# ─────────────────────────────────────────────────────────────
#  フィルター関数（3パターン）
# ─────────────────────────────────────────────────────────────
def kmid(bar, direction: int) -> bool:
    """実体方向一致（v77 / F1+F3 共通）"""
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])

def klow_v77(bar, direction: int) -> bool:
    """v77版: 常に下ヒゲ比率で判定（方向非対称）"""
    o = bar["open"]
    if o <= 0:
        return True
    body_bottom = min(bar["open"], bar["close"])
    return (body_bottom - bar["low"]) / o < KLOW_THR

def klow_symmetric(bar, direction: int) -> bool:
    """F1+F3版: ロング→下ヒゲ、ショート→上ヒゲ（方向対称）"""
    o = bar["open"]; c = bar["close"]
    h = bar["high"]; l = bar["low"]
    if o <= 0:
        return True
    if direction == 1:
        return (min(o, c) - l) / o < KLOW_THR
    else:
        return (h - max(o, c)) / o < KLOW_THR


# ─────────────────────────────────────────────────────────────
#  シグナル生成（戦略別）
# ─────────────────────────────────────────────────────────────
def generate_signals(data_1m: pd.DataFrame, data_15m: pd.DataFrame,
                     data_4h: pd.DataFrame, spread_pips: float,
                     pip_size: float, strategy: str) -> list[dict]:
    """
    strategy: "v77" / "F1+F3" / "Hybrid"
    """
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)

    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h = add_indicators(data_1h)

    signals:    list[dict] = []
    used_times: set        = set()

    use_time_filter = (strategy in ("F1+F3", "Hybrid"))
    klow_fn = klow_v77 if strategy in ("v77", "Hybrid") else klow_symmetric

    # ── ① 4H足パターン（F1+F3 / Hybrid のみ）─────────────────
    if strategy != "v77":
        for i in range(2, len(data_4h)):
            t   = data_4h.index[i]
            cur = data_4h.iloc[i]
            p1  = data_4h.iloc[i - 1]
            p2  = data_4h.iloc[i - 2]
            atr = cur["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            if use_time_filter and t.hour not in GOOD_HOURS:
                continue
            trend = cur["trend"]
            tol   = atr * 0.3

            for direction in [1, -1]:
                if trend != direction:
                    continue
                key = "low" if direction == 1 else "high"
                v1 = p2[key]; v2 = p1[key]
                if abs(v1 - v2) > tol:
                    continue
                if direction == 1 and p1["close"] <= p1["open"]:
                    continue
                if direction == -1 and p1["close"] >= p1["open"]:
                    continue
                if not kmid(p1, direction) or not klow_fn(p1, direction):
                    continue

                # 4H足では直前1H足のトレンドも確認（F1+F3の特徴）
                h1b = data_1h[data_1h.index <= t]
                if len(h1b) == 0:
                    continue
                if h1b.iloc[-1].get("trend", 0) != direction:
                    continue

                entry_end = t + pd.Timedelta(minutes=2)
                m1w = data_1m[(data_1m.index >= t) & (data_1m.index < entry_end)]
                if len(m1w) == 0:
                    continue
                eb = m1w.iloc[0]; et = eb.name
                if et in used_times:
                    continue

                raw_ep = eb["open"]
                ep   = raw_ep + spread if direction == 1 else raw_ep - spread
                sl   = (min(v1, v2) - atr * 0.15) if direction == 1 else \
                       (max(v1, v2) + atr * 0.15)
                risk = raw_ep - sl if direction == 1 else sl - raw_ep
                tp   = raw_ep + risk * RR_RATIO if direction == 1 else \
                       raw_ep - risk * RR_RATIO

                if 0 < risk <= atr * 3:
                    signals.append({
                        "time": et, "dir": direction,
                        "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"
                    })
                    used_times.add(et)
                    break

    # ── ② 1H足パターン（全戦略共通、条件はそれぞれ異なる）──
    for i in range(2, len(data_1h)):
        t   = data_1h.index[i]
        cur = data_1h.iloc[i]
        p1  = data_1h.iloc[i - 1]
        p2  = data_1h.iloc[i - 2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        if use_time_filter and t.hour not in GOOD_HOURS:
            continue

        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0:
            continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)) or pd.isna(h4l.get("ema20", np.nan)):
            continue
        trend  = h4l["trend"]
        h4_atr = h4l["atr"]
        tol    = atr * 0.3

        for direction in [1, -1]:
            if trend != direction:
                continue
            key = "low" if direction == 1 else "high"
            v1 = p2[key]; v2 = p1[key]
            if abs(v1 - v2) > tol:
                continue
            if direction == 1 and p1["close"] <= p1["open"]:
                continue
            if direction == -1 and p1["close"] >= p1["open"]:
                continue
            if not kmid(h4l, direction) or not klow_fn(h4l, direction):
                continue

            entry_end = t + pd.Timedelta(minutes=2)
            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < entry_end)]
            if len(m1w) == 0:
                continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times:
                continue

            raw_ep = eb["open"]
            ep   = raw_ep + spread if direction == 1 else raw_ep - spread
            sl   = (min(v1, v2) - atr * 0.15) if direction == 1 else \
                   (max(v1, v2) + atr * 0.15)
            risk = raw_ep - sl if direction == 1 else sl - raw_ep
            tp   = raw_ep + risk * RR_RATIO if direction == 1 else \
                   raw_ep - risk * RR_RATIO

            if 0 < risk <= h4_atr * 2:
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"
                })
                used_times.add(et)
                break

    signals.sort(key=lambda x: x["time"])
    return signals


# ─────────────────────────────────────────────────────────────
#  シミュレーション（半利確あり）
# ─────────────────────────────────────────────────────────────
def simulate_trades(symbol: str, signals: list[dict],
                    data_1m: pd.DataFrame) -> list[dict]:
    arm   = AdaptiveRiskManager(symbol, base_risk_pct=BASE_RISK)
    equity = INIT_CASH
    trades = []
    times  = data_1m.index.values
    highs  = data_1m["high"].values
    lows   = data_1m["low"].values

    for sig in signals:
        entry_time = sig["time"]
        direction  = sig["dir"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]

        idx = np.searchsorted(times, np.datetime64(entry_time))
        if idx >= len(times):
            continue

        lot, _, _ = arm.calc_lot_adaptive(
            equity=equity, sl_distance=risk, ref_price=ep, usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        equity -= arm.calc_commission_jpy(lot, USDJPY_RATE)
        half_done = False; be_sl = None; half_pnl = 0.0
        result = None; exit_time = None; exit_price = None

        for j in range(idx + 1, len(times)):
            bh = highs[j]; bl = lows[j]; bt = times[j]
            cur_sl = be_sl if half_done else sl

            if direction == 1:
                if bl <= cur_sl:
                    exit_price = cur_sl; exit_time = bt
                    rem = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price, lot * rem, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * rem, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else \
                             ("TP" if pnl > 0 else "SL")
                    break
                if not half_done and bh >= ep + risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R, lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    half_done = True; be_sl = ep
                if bh >= tp:
                    exit_price = tp; exit_time = bt
                    pnl = arm.calc_pnl_jpy(direction, ep, tp, lot * 0.5, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    result = "TP"; break
            else:
                if bh >= cur_sl:
                    exit_price = cur_sl; exit_time = bt
                    rem = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price, lot * rem, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * rem, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else \
                             ("TP" if pnl > 0 else "SL")
                    break
                if not half_done and bl <= ep - risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R, lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    half_done = True; be_sl = ep
                if bl <= tp:
                    exit_price = tp; exit_time = bt
                    pnl = arm.calc_pnl_jpy(direction, ep, tp, lot * 0.5, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    result = "TP"; break

        if result is None:
            continue
        arm.update_peak(equity)

        pip_size = SYMBOL_CONFIG[symbol]["pip"]
        pnl_pips = (exit_price - ep) / pip_size * direction

        risk_jpy = abs(arm.calc_pnl_jpy(direction, ep, sl, lot, USDJPY_RATE, ep))
        trades.append({
            "entry_time":   entry_time,
            "dir":          direction,
            "ep":           ep,
            "sl":           sl,
            "tp":           tp,
            "exit_price":   exit_price,
            "result":       result,
            "lot":          lot,
            "pnl_pips":     pnl_pips,
            "risk_jpy":     risk_jpy,
            "equity_after": equity,
            "tf":           sig.get("tf", "?"),
        })
    return trades


# ─────────────────────────────────────────────────────────────
#  統計
# ─────────────────────────────────────────────────────────────
def calc_stats(trades: list[dict], symbol: str, strategy: str) -> dict:
    base = {"symbol": symbol, "strategy": strategy, "n": 0,
            "status": "NO_TRADES", "all_pass": False, "pass_count": 0}
    if not trades:
        return base

    df = pd.DataFrame(trades)
    wins   = df[df["result"].isin(["TP", "BE"])]
    losses = df[df["result"] == "SL"]

    n        = len(df)
    win_rate = len(wins) / n * 100
    win_pips  = wins["pnl_pips"].sum() if len(wins) > 0 else 0.0
    loss_pips = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0.0
    pf = win_pips / loss_pips if loss_pips > 0 else float("inf")

    # MDD (ピーク比 %)
    eq = df["equity_after"].values
    peak = float(INIT_CASH); mdd_jpy = 0.0
    for v in eq:
        if v > peak:
            peak = v
        mdd_jpy = max(mdd_jpy, peak - v)
    mdd_pct = mdd_jpy / peak * 100

    # 月次
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_pips"].sum()
    plus_months_pct = (monthly > 0).sum() / len(monthly) * 100
    monthly_sharpe  = monthly.mean() / monthly.std() * np.sqrt(12) \
                      if monthly.std() > 0 else 0.0

    # 最終利益（円）
    final_equity  = eq[-1]
    final_profit  = final_equity - INIT_CASH
    final_profit_pct = final_profit / INIT_CASH * 100

    # MDD円
    mdd_jpy_val = mdd_jpy  # already computed above

    # CR = 年率リターン / MDD%（Calmar Ratio）
    # データ期間を trade の entry_time から推定
    t_start = df["entry_time"].min()
    t_end   = df["entry_time"].max()
    years   = max((t_end - t_start).days / 365.25, 1/12)
    annual_ret_pct = (((final_equity / INIT_CASH) ** (1 / years)) - 1) * 100
    calmar = annual_ret_pct / mdd_pct if mdd_pct > 0 else float("inf")

    # Kelly
    avg_win  = wins["pnl_pips"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pips"].mean()) if len(losses) > 0 else 1
    p = win_rate / 100
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = p - (1 - p) / rr if rr > 0 else 0.0

    # 連敗
    consec = 0; max_consec = 0
    for r in df["result"]:
        if r == "SL": consec += 1; max_consec = max(max_consec, consec)
        else: consec = 0

    passed = {
        "pf":          pf >= CRITERIA["pf"],
        "win_rate":    win_rate >= CRITERIA["win_rate"],
        "mdd_pct":     mdd_pct <= CRITERIA["mdd_pct"],
        "kelly":       kelly >= CRITERIA["kelly"],
        "plus_months": plus_months_pct >= CRITERIA["plus_months"],
    }
    pass_count = sum(passed.values())

    # TF内訳
    tf_counts = df["tf"].value_counts().to_dict()

    return {
        "symbol":           symbol,
        "strategy":         strategy,
        "n":                n,
        "win_rate":         round(win_rate, 1),
        "pf":               round(pf, 2),
        "mdd_pct":          round(mdd_pct, 1),
        "mdd_jpy":          round(mdd_jpy_val),
        "final_profit_jpy": round(final_profit),
        "final_profit_pct": round(final_profit_pct, 1),
        "annual_ret_pct":   round(annual_ret_pct, 1),
        "calmar":           round(calmar, 2),
        "kelly":            round(kelly, 3),
        "plus_months":      round(plus_months_pct, 1),
        "monthly_sharpe":   round(monthly_sharpe, 2),
        "max_consec_loss":  max_consec,
        "avg_win_pips":     round(avg_win, 1),
        "avg_loss_pips":    round(avg_loss, 1),
        "spread_pips":      SYMBOL_CONFIG[symbol]["spread"],
        "tf_1h":            tf_counts.get("1h", 0),
        "tf_4h":            tf_counts.get("4h", 0),
        "pass_pf":          passed["pf"],
        "pass_winrate":     passed["win_rate"],
        "pass_mdd":         passed["mdd_pct"],
        "pass_kelly":       passed["kelly"],
        "pass_months":      passed["plus_months"],
        "pass_count":       pass_count,
        "all_pass":         all(passed.values()),
        "status":           "PASS" if all(passed.values()) else f"FAIL({pass_count}/5)",
        "equity_curve":     df["equity_after"].tolist(),
    }


# ─────────────────────────────────────────────────────────────
#  チャート
# ─────────────────────────────────────────────────────────────
def plot_comparison(all_results: list[dict], symbols: list[str]):
    ncols = len(symbols)
    fig, axes = plt.subplots(len(STRATEGY_NAMES), ncols,
                             figsize=(4 * ncols, 3.5 * len(STRATEGY_NAMES)),
                             squeeze=False)
    fig.suptitle("戦略比較バックテスト: v77 / F1+F3 / Hybrid", fontsize=13, fontweight="bold")

    colors = {"v77": "#ef4444", "F1+F3": "#3b82f6", "Hybrid": "#22c55e"}

    for row_i, strat in enumerate(STRATEGY_NAMES):
        for col_i, sym in enumerate(symbols):
            ax = axes[row_i][col_i]
            r  = next((x for x in all_results
                       if x["symbol"] == sym and x["strategy"] == strat), None)

            if r and r["n"] > 0:
                eq = r["equity_curve"]
                ax.plot(eq, color=colors[strat], linewidth=1.2)
                ax.axhline(y=INIT_CASH, color="gray", linestyle="--", alpha=0.4, lw=0.8)
                status_mark = "✅" if r["all_pass"] else "❌"
                profit_str = f"+{r['final_profit_jpy']/1e4:.0f}万" \
                             if r['final_profit_jpy'] >= 0 \
                             else f"{r['final_profit_jpy']/1e4:.0f}万"
                ax.set_title(f"{sym} [{strat}] {status_mark}\n"
                             f"PF={r['pf']} WR={r['win_rate']}% "
                             f"MDD={r['mdd_pct']}% CR={r['calmar']}\n"
                             f"利益:{profit_str}({r['final_profit_pct']}%)",
                             fontsize=6.5)
            else:
                ax.set_title(f"{sym} [{strat}] — N/A", fontsize=7)
                ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = RESULTS_DIR / "strategy_comparison.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"チャート保存: {out.name}")


# ─────────────────────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────────────────────
def main():
    symbols = sorted(
        p.stem.replace("_1m", "")
        for p in DATA_DIR.glob("*_1m.csv")
        if (DATA_DIR / p.stem.replace("_1m", "_15m.csv")).exists()
        and (DATA_DIR / p.stem.replace("_1m", "_4h.csv")).exists()
    )

    print("=" * 78)
    print("3戦略比較バックテスト: v77 / F1+F3 / Hybrid")
    print(f"対象銘柄 ({len(symbols)}銘柄): {symbols}")
    print("=" * 78)

    all_results: list[dict] = []

    for sym in symbols:
        cfg = SYMBOL_CONFIG.get(sym)
        if cfg is None:
            print(f"[{sym}] SYMBOL_CONFIG未登録 → スキップ")
            continue

        data_1m  = load_ohlc(DATA_DIR / f"{sym}_1m.csv")
        data_15m = load_ohlc(DATA_DIR / f"{sym}_15m.csv")
        data_4h  = load_ohlc(DATA_DIR / f"{sym}_4h.csv")
        if data_1m is None or data_15m is None or data_4h is None:
            print(f"[{sym}] データ不足 → スキップ")
            continue

        spread  = cfg["spread"]
        pip     = cfg["pip"]

        print(f"\n[{sym}] spread={spread}pips")
        for strat in STRATEGY_NAMES:
            sigs   = generate_signals(data_1m, data_15m, data_4h, spread, pip, strat)
            trades = simulate_trades(sym, sigs, data_1m)
            stats  = calc_stats(trades, sym, strat)
            all_results.append(stats)
            profit_str = f"+{stats['final_profit_jpy']/1e4:.0f}万" \
                     if stats['final_profit_jpy'] >= 0 \
                     else f"{stats['final_profit_jpy']/1e4:.0f}万"
        print(f"  {strat:<8}: N={stats['n']:>4}, "
                  f"WR={stats['win_rate']:>5.1f}%, "
                  f"PF={stats['pf']:>5.2f}, "
                  f"MDD={stats['mdd_pct']:>4.1f}%({stats['mdd_jpy']/1e4:.0f}万), "
                  f"CR={stats['calmar']:>5.2f}, "
                  f"Sharpe={stats['monthly_sharpe']:>5.2f}, "
                  f"利益={profit_str}({stats['final_profit_pct']:+.1f}%)  "
                  f"→ {stats['status']}")

    # ── 結果テーブル ──────────────────────────────────────────
    W = 105
    print("\n" + "=" * W)
    print("【合格基準】 PF≥3.0 / WR≥65% / MDD≤20% / Kelly≥0.45 / +月≥90%")
    print("CR=カルマーレシオ（年率リターン÷MDD%）  Sharpe=月次シャープレシオ（×√12）")
    print("=" * W)
    print(f"{'銘柄':<8} {'戦略':<8} {'N':>5} {'WR%':>6} {'PF':>6} "
          f"{'MDD%':>5} {'MDD(万)':>7} {'最終利益(万)':>11} {'利益%':>6} "
          f"{'CR':>6} {'Sharpe':>7} {'Kelly':>7} {'+月%':>6}  判定")
    print("-" * W)

    best_per_symbol: dict[str, dict] = {}

    for sym in symbols:
        sym_results = [r for r in all_results if r["symbol"] == sym and r["n"] > 0]
        if not sym_results:
            continue
        for r in sym_results:
            mark = "✅ PASS" if r["all_pass"] else f"❌ {r['status']}"
            pf_m  = "PF○" if r["pass_pf"]      else "PF×"
            wr_m  = "WR○" if r["pass_winrate"]  else "WR×"
            mdd_m = "MD○" if r["pass_mdd"]      else "MD×"
            k_m   = "K○"  if r["pass_kelly"]    else "K×"
            mo_m  = "M○"  if r["pass_months"]   else "M×"
            flags = f"{pf_m} {wr_m} {mdd_m} {k_m} {mo_m}"
            profit_str = f"+{r['final_profit_jpy']/1e4:.0f}" \
                         if r['final_profit_jpy'] >= 0 \
                         else f"{r['final_profit_jpy']/1e4:.0f}"
            print(f"{sym:<8} {r['strategy']:<8} {r['n']:>5} "
                  f"{r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['mdd_pct']:>4.1f}% {r['mdd_jpy']/1e4:>7.1f} "
                  f"{profit_str:>11} {r['final_profit_pct']:>+5.1f}% "
                  f"{r['calmar']:>6.2f} {r['monthly_sharpe']:>7.2f} "
                  f"{r['kelly']:>7.3f} {r['plus_months']:>5.1f}%  "
                  f"{mark}  [{flags}]")

        # 最良戦略を記録（pass_count→pf優先）
        best = max(sym_results, key=lambda x: (x["pass_count"], x["pf"]))
        best_per_symbol[sym] = best
        print()

    # ── ベスト戦略サマリー ──────────────────────────────────
    print("=" * 78)
    print("【銘柄別 推奨戦略】")
    print("=" * 78)
    approved = []
    for sym, r in best_per_symbol.items():
        mark = "✅" if r["all_pass"] else "❌"
        profit_str = f"+{r['final_profit_jpy']/1e4:.0f}万" \
                     if r['final_profit_jpy'] >= 0 \
                     else f"{r['final_profit_jpy']/1e4:.0f}万"
        print(f"{mark} {sym:<8}: {r['strategy']:<8} "
              f"PF={r['pf']}, WR={r['win_rate']}%, "
              f"MDD={r['mdd_pct']}%, CR={r['calmar']}, "
              f"Sharpe={r['monthly_sharpe']}, "
              f"利益={profit_str}({r['final_profit_pct']:+.1f}%), "
              f"N={r['n']}")
        if r["all_pass"]:
            approved.append({"symbol": sym, "strategy": r["strategy"],
                             "pf": r["pf"], "win_rate": r["win_rate"]})

    print(f"\n✅ 採用銘柄・戦略 ({len(approved)}件): "
          + ", ".join(f"{a['symbol']}({a['strategy']})" for a in approved))

    # ── CSV保存 ───────────────────────────────────────────────
    rows = [{k: v for k, v in r.items() if k != "equity_curve"}
            for r in all_results]
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "strategy_comparison_summary.csv",
        index=False, encoding="utf-8-sig")
    print(f"\nCSV保存: strategy_comparison_summary.csv")

    # ── JSON保存 ──────────────────────────────────────────────
    result_json = {
        "updated":  pd.Timestamp.now().strftime("%Y-%m-%d"),
        "criteria": CRITERIA,
        "approved": approved,
        "all_results": rows,
    }
    with open(RESULTS_DIR / "approved_universe_v77.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)
    print("JSON更新: approved_universe_v77.json")

    # ── チャート ──────────────────────────────────────────────
    plot_comparison(all_results, symbols)

    return approved


if __name__ == "__main__":
    main()
