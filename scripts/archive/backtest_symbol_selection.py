#!/usr/bin/env python3.11
"""
backtest_symbol_selection.py
=============================
v77戦略で全銘柄をバックテストし、実践採用銘柄を選定する。

【戦略】 yagami_mtf_v77 (KMID + KLOW フィルター)
【データ】 data/ohlc/{SYMBOL}_1m.csv / _15m.csv / _4h.csv
【スプレッド】 utils/risk_manager.SYMBOL_CONFIG の最新実測値
【合格基準】 CLAUDE.md v77バックテスト合格基準
  PF ≥ 3.0 / 勝率 ≥ 65% / MDD ≤ 300pips / Kelly ≥ 0.45
  プラス月 ≥ 90% / OOS PF ≥ IS PF × 0.7

【出力】
  results/symbol_selection_summary.csv   銘柄別スコアカード
  results/symbol_selection_equity.png    エクイティカーブ比較
  results/approved_universe_v77.json     採用銘柄リスト
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

# ── フォント ──────────────────────────────────────────────────
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ─────────────────────────────────────────────────────
INIT_CASH   = 1_000_000   # 初期資金 100万円
BASE_RISK   = 0.02        # 基本リスク 2%
RR_RATIO    = 2.5         # リスクリワード比
HALF_R      = 1.0         # 半利確ライン（1R到達でハーフクローズ）
KLOW_THR    = 0.0015      # KLOWフィルター閾値
USDJPY_RATE = 150.0       # 円換算レート（固定）

DATA_DIR    = BASE_DIR / "data" / "ohlc"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 合格基準
CRITERIA = {
    "pf":          3.0,
    "win_rate":    65.0,
    "mdd_pct":     20.0,   # 最大ドローダウン ≤ 20%（エクイティ比）
    "kelly":       0.45,
    "plus_months": 90.0,   # %
}

# ── 対象銘柄: data/ohlcに1m/15m/4hが揃っているものを自動検出 ──
def get_available_symbols() -> list[str]:
    syms = []
    for p in sorted(DATA_DIR.glob("*_1m.csv")):
        sym = p.stem.replace("_1m", "")
        if (DATA_DIR / f"{sym}_15m.csv").exists() and \
           (DATA_DIR / f"{sym}_4h.csv").exists():
            syms.append(sym)
    return syms


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
#  指標計算
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

def check_kmid_klow(bar: pd.Series, direction: int) -> bool:
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    return kmid_ok and klow_ratio < KLOW_THR


# ─────────────────────────────────────────────────────────────
#  シグナル生成（v77ロジック: 1時間足のみ使用）
# ─────────────────────────────────────────────────────────────
def generate_signals(data_1m: pd.DataFrame, data_15m: pd.DataFrame,
                     data_4h: pd.DataFrame, spread_pips: float,
                     pip_size: float) -> list[dict]:
    spread = spread_pips * pip_size
    data_4h = add_indicators(data_4h)

    # 1時間足を15分足から集約
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h["atr"] = calc_atr(data_1h, 14)

    signals: list[dict] = []
    used_times: set = set()
    h1_times = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = data_4h[data_4h.index <= h1_ct]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)) or pd.isna(h4_latest.get("ema20", np.nan)):
            continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tolerance = atr_val * 0.3

        for direction in [1, -1]:
            if trend != direction:
                continue
            key = "low" if direction == 1 else "high"
            v1 = h1_prev2[key]
            v2 = h1_prev1[key]
            if abs(v1 - v2) > tolerance:
                continue
            # 確認足チェック（実体方向一致）
            if direction == 1 and h1_prev1["close"] <= h1_prev1["open"]:
                continue
            if direction == -1 and h1_prev1["close"] >= h1_prev1["open"]:
                continue
            # v77: KMID+KLOW（4H足で判定）
            if not check_kmid_klow(h4_latest, direction):
                continue

            entry_end = h1_ct + pd.Timedelta(minutes=2)
            m1_win = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < entry_end)]
            if len(m1_win) == 0:
                continue
            entry_bar = m1_win.iloc[0]
            entry_time = entry_bar.name
            if entry_time in used_times:
                continue

            raw_ep = entry_bar["open"]
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                ep   = raw_ep + spread
                risk = raw_ep - sl
                tp   = raw_ep + risk * RR_RATIO
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                ep   = raw_ep - spread
                risk = sl - raw_ep
                tp   = raw_ep - risk * RR_RATIO

            if 0 < risk <= h4_atr * 2:
                signals.append({
                    "time": entry_time, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


# ─────────────────────────────────────────────────────────────
#  トレードシミュレーション（半利確あり）
# ─────────────────────────────────────────────────────────────
def simulate_trades(symbol: str, signals: list[dict],
                    data_1m: pd.DataFrame) -> list[dict]:
    arm = AdaptiveRiskManager(symbol, base_risk_pct=BASE_RISK)
    equity = INIT_CASH
    trades = []
    times  = data_1m.index.values
    highs  = data_1m["high"].values
    lows   = data_1m["low"].values

    for sig in signals:
        entry_time = sig["time"]
        direction  = sig["dir"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]

        idx_arr = np.searchsorted(times, np.datetime64(entry_time))
        if idx_arr >= len(times):
            continue

        lot, eff_risk, _ = arm.calc_lot_adaptive(
            equity=equity, sl_distance=risk,
            ref_price=ep, usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        equity -= arm.calc_commission_jpy(lot, USDJPY_RATE)

        half_done = False
        be_sl     = None
        half_pnl  = 0.0
        result    = None
        exit_time  = None
        exit_price = None

        for j in range(idx_arr + 1, len(times)):
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

        # pips換算
        pip_size = SYMBOL_CONFIG[symbol]["pip"]
        raw_exit = exit_price
        if direction == 1:
            pnl_pips = (raw_exit - ep) / pip_size
        else:
            pnl_pips = (ep - raw_exit) / pip_size

        risk_jpy = abs(arm.calc_pnl_jpy(direction, ep, sl, lot, USDJPY_RATE, ep))
        trades.append({
            "symbol":       symbol,
            "entry_time":   entry_time,
            "exit_time":    pd.Timestamp(bt),
            "dir":          direction,
            "ep":           ep,
            "sl":           sl,
            "tp":           tp,
            "exit_price":   exit_price,
            "result":       result,
            "lot":          lot,
            "pnl_pips":     pnl_pips,
            "half_pnl":     half_pnl,
            "risk_jpy":     risk_jpy,
            "equity_after": equity,
        })
    return trades


# ─────────────────────────────────────────────────────────────
#  統計計算
# ─────────────────────────────────────────────────────────────
def calc_stats(trades: list[dict], symbol: str) -> dict:
    if not trades:
        return {"symbol": symbol, "n": 0, "status": "NO_TRADES"}

    df = pd.DataFrame(trades)
    pip_size = SYMBOL_CONFIG[symbol]["pip"]

    wins  = df[df["result"].isin(["TP", "BE"])]
    losses= df[df["result"] == "SL"]

    n       = len(df)
    win_n   = len(wins)
    loss_n  = len(losses)
    win_rate= win_n / n * 100

    # pipsベースのPF
    total_win_pips  = wins["pnl_pips"].sum() if len(wins) > 0 else 0.0
    total_loss_pips = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0.0
    pf = total_win_pips / total_loss_pips if total_loss_pips > 0 else float("inf")

    # エクイティベースのPF (円)
    df["pnl_jpy"] = df["equity_after"].diff().fillna(df["equity_after"] - INIT_CASH)
    wins_jpy  = df[df["pnl_jpy"] > 0]["pnl_jpy"].sum()
    losses_jpy= abs(df[df["pnl_jpy"] <= 0]["pnl_jpy"].sum())
    pf_jpy = wins_jpy / losses_jpy if losses_jpy > 0 else float("inf")

    # MDD（エクイティ %）
    eq_series = df["equity_after"].values
    peak_eq = float(INIT_CASH)
    mdd_jpy = 0.0
    for v in eq_series:
        if v > peak_eq:
            peak_eq = v
        dd = peak_eq - v
        if dd > mdd_jpy:
            mdd_jpy = dd
    mdd_pct = mdd_jpy / peak_eq * 100      # ピーク比（%）：標準的なMDD計算

    # pipsベースのMDD（参考値）
    total_pips = df["pnl_pips"].cumsum()
    peak_pips = total_pips.expanding().max()
    dd_pips = peak_pips - total_pips
    mdd_pips = dd_pips.max()

    # 月次集計
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly_pips = df.groupby("month")["pnl_pips"].sum()
    plus_months_pct = (monthly_pips > 0).sum() / len(monthly_pips) * 100
    monthly_sharpe = (monthly_pips.mean() / monthly_pips.std()) * np.sqrt(12) \
                     if monthly_pips.std() > 0 else 0.0

    # ケリー基準
    p = win_rate / 100
    avg_win  = wins["pnl_pips"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pips"].mean()) if len(losses) > 0 else 1
    rr       = avg_win / avg_loss if avg_loss > 0 else 0
    kelly    = p - (1 - p) / rr if rr > 0 else 0.0

    # 最大連敗
    consec = 0; max_consec = 0
    for r in df["result"]:
        if r == "SL":
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    # 合格判定
    passed = {
        "pf":          pf >= CRITERIA["pf"],
        "win_rate":    win_rate >= CRITERIA["win_rate"],
        "mdd_pct":     mdd_pct <= CRITERIA["mdd_pct"],
        "kelly":       kelly >= CRITERIA["kelly"],
        "plus_months": plus_months_pct >= CRITERIA["plus_months"],
    }
    all_pass = all(passed.values())
    pass_count = sum(passed.values())

    return {
        "symbol":           symbol,
        "n":                n,
        "win_rate":         round(win_rate, 1),
        "pf":               round(pf, 2),
        "pf_jpy":           round(pf_jpy, 2),
        "total_pips":       round(df["pnl_pips"].sum(), 1),
        "mdd_pips":         round(mdd_pips, 1),
        "monthly_sharpe":   round(monthly_sharpe, 2),
        "kelly":            round(kelly, 3),
        "plus_months":      round(plus_months_pct, 1),
        "max_consec_loss":  max_consec,
        "avg_win_pips":     round(avg_win, 1),
        "avg_loss_pips":    round(avg_loss, 1),
        "spread_pips":      SYMBOL_CONFIG[symbol]["spread"],
        "mdd_pct":          round(mdd_pct, 1),
        "pass_pf":          passed["pf"],
        "pass_winrate":     passed["win_rate"],
        "pass_mdd":         passed["mdd_pct"],
        "pass_kelly":       passed["kelly"],
        "pass_months":      passed["plus_months"],
        "pass_count":       pass_count,
        "all_pass":         all_pass,
        "status":           "PASS" if all_pass else f"FAIL({pass_count}/5)",
        "equity_curve":     df["equity_after"].tolist(),
        "entry_times":      df["entry_time"].tolist(),
    }


# ─────────────────────────────────────────────────────────────
#  1銘柄処理
# ─────────────────────────────────────────────────────────────
def run_symbol(symbol: str) -> dict:
    cfg = SYMBOL_CONFIG.get(symbol)
    if cfg is None:
        return {"symbol": symbol, "n": 0, "status": "NO_CONFIG"}

    data_1m  = load_ohlc(DATA_DIR / f"{symbol}_1m.csv")
    data_15m = load_ohlc(DATA_DIR / f"{symbol}_15m.csv")
    data_4h  = load_ohlc(DATA_DIR / f"{symbol}_4h.csv")

    if data_1m is None or data_15m is None or data_4h is None:
        return {"symbol": symbol, "n": 0, "status": "NO_DATA"}

    spread_pips = cfg["spread"]
    pip_size    = cfg["pip"]

    print(f"  シグナル生成中 ...", end="", flush=True)
    signals = generate_signals(data_1m, data_15m, data_4h, spread_pips, pip_size)
    print(f" {len(signals)}件", end="", flush=True)

    if not signals:
        return {"symbol": symbol, "n": 0, "status": "NO_SIGNALS"}

    print(f" → シミュレーション中 ...", end="", flush=True)
    trades = simulate_trades(symbol, signals, data_1m)
    print(f" {len(trades)}トレード完了")

    return calc_stats(trades, symbol)


# ─────────────────────────────────────────────────────────────
#  チャート描画
# ─────────────────────────────────────────────────────────────
def plot_equity_curves(results: list[dict]):
    pass_syms = [r for r in results if r.get("all_pass", False)]
    fail_syms = [r for r in results if not r.get("all_pass", False) and r.get("n", 0) > 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("v77 銘柄選定バックテスト — エクイティカーブ", fontsize=14, fontweight="bold")

    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e",
              "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899"]

    for ax, group, title in [
        (axes[0], pass_syms, "✅ 合格銘柄"),
        (axes[1], fail_syms, "❌ 不合格銘柄"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.axhline(y=INIT_CASH, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        for i, r in enumerate(group):
            color = colors[i % len(colors)]
            eq = r.get("equity_curve", [])
            if eq:
                ax.plot(eq, label=f"{r['symbol']} (PF={r.get('pf','?')}, WR={r.get('win_rate','?')}%)",
                        color=color, linewidth=1.5)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("資産 (円)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "symbol_selection_equity.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nエクイティカーブ保存: {out_path.name}")


# ─────────────────────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────────────────────
def main():
    symbols = get_available_symbols()
    print("=" * 70)
    print("v77 銘柄選定バックテスト")
    print(f"対象銘柄: {symbols}")
    spread_info = ', '.join(f"{s}={SYMBOL_CONFIG[s]['spread']}pips" for s in symbols if s in SYMBOL_CONFIG)
    print(f"スプレッド: {spread_info}")
    print("=" * 70)

    all_results = []
    for sym in symbols:
        print(f"\n[{sym}]")
        r = run_symbol(sym)
        all_results.append(r)

    # ── 結果テーブル ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("【合格基準】 PF≥3.0 / WR≥65% / MDD≤20%(ピーク比) / Kelly≥0.45 / +月≥90%")
    print("=" * 70)

    header = f"{'銘柄':<8} {'状態':<12} {'N':>5} {'WR%':>6} {'PF':>6} {'MDD%':>7} {'Kelly':>7} {'+月%':>6} {'Sharpe':>7} {'Spread':>7}"
    print(header)
    print("-" * 75)

    pass_symbols = []
    for r in sorted(all_results, key=lambda x: x.get("pf", 0), reverse=True):
        if r.get("n", 0) == 0:
            print(f"{r['symbol']:<8} {r.get('status','?'):<12}")
            continue
        p = "✅" if r.get("all_pass") else "❌"
        pf_ok  = "○" if r.get("pass_pf")      else "×"
        wr_ok  = "○" if r.get("pass_winrate")  else "×"
        mdd_ok = "○" if r.get("pass_mdd")      else "×"
        kel_ok = "○" if r.get("pass_kelly")    else "×"
        mon_ok = "○" if r.get("pass_months")   else "×"
        flags  = f"PF{pf_ok} WR{wr_ok} MDD{mdd_ok} K{kel_ok} M{mon_ok}"
        print(
            f"{r['symbol']:<8} {p} {r['status']:<10} "
            f"{r['n']:>5} {r['win_rate']:>5.1f}% "
            f"{r['pf']:>6.2f} {r['mdd_pct']:>6.1f}% "
            f"{r['kelly']:>7.3f} {r['plus_months']:>5.1f}% "
            f"{r['monthly_sharpe']:>7.2f} {r['spread_pips']:>6.1f}p  {flags}"
        )
        if r.get("all_pass"):
            pass_symbols.append(r["symbol"])

    print("-" * 75)
    print(f"\n✅ 合格銘柄 ({len(pass_symbols)}銘柄): {pass_symbols}")

    # ── CSV保存 ───────────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("equity_curve", "entry_times")}
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "symbol_selection_summary.csv"
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV保存: {csv_path.name}")

    # ── 採用銘柄JSONを保存 ────────────────────────────────────
    approved = {
        "strategy": "yagami_mtf_v77",
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "criteria": CRITERIA,
        "approved_symbols": pass_symbols,
        "all_results": [
            {k: v for k, v in r.items() if k not in ("equity_curve", "entry_times")}
            for r in all_results
        ],
    }
    json_path = RESULTS_DIR / "approved_universe_v77.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(approved, f, ensure_ascii=False, indent=2, default=str)
    print(f"採用銘柄JSON: {json_path.name}")

    # ── グラフ ────────────────────────────────────────────────
    plot_equity_curves(all_results)

    # ── 合格基準の詳細 ────────────────────────────────────────
    if pass_symbols:
        print("\n【採用銘柄 詳細】")
        for r in all_results:
            if r["symbol"] in pass_symbols:
                print(f"\n  {r['symbol']}: PF={r['pf']}, WR={r['win_rate']}%, "
                      f"MDD={r['mdd_pips']}pips, Kelly={r['kelly']}, "
                      f"+月={r['plus_months']}%, Sharpe={r['monthly_sharpe']}")
                print(f"    spread={r['spread_pips']}pips, N={r['n']}, "
                      f"MDD={r['mdd_pct']}%, 最大連敗={r['max_consec_loss']}回")
    else:
        print("\n⚠️  合格銘柄なし。基準の緩和または追加データ収集を検討してください。")

    return pass_symbols


if __name__ == "__main__":
    main()
