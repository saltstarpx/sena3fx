"""
backtest_portfolio_2026.py
===========================
2026年〜現在のポートフォリオバックテスト（時系列イベント駆動）

【設定】
  初期資金: 65万円
  リスク: 2%（3000万超で1%）
  MAX_OPEN: 無制限（銘柄ごと1ポジ制限のみ）
  8銘柄: 本番確定ロジック

【修正】
  トレードの同時オープンを正しくシミュレーション。
  エントリー時のequityでロットを計算し、exitイベント発生時にequityを更新。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_final_optimized import (
    load_all, calc_atr,
    generate_signals, generate_signals_v80,
    SYMBOL_CONFIG, USDJPY_RATE, HALF_R, MAX_LOOKAHEAD,
    RR_RATIO, RR_RATIO_V80,
)
from utils.risk_manager import RiskManager

# ── 設定 ───────────────────────────────────────────────────────────
INIT_CASH = 650_000
RISK_THRESHOLD_JPY = 30_000_000
RISK_BELOW = 0.02
RISK_ABOVE = 0.01
START_DATE = "2026-01-01"

PORTFOLIO = [
    {"sym": "USDCAD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "XAUUSD", "logic": "A",   "rr": 2.5, "tol": 0.20},
    {"sym": "EURUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "AUDUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "GBPUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "USDCHF", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "NZDUSD", "logic": "A",   "rr": 3.0, "tol": 0.20},
    {"sym": "USDJPY", "logic": "C",   "rr": 3.0, "tol": 0.30},
]

def get_risk_pct(equity):
    return RISK_ABOVE if equity >= RISK_THRESHOLD_JPY else RISK_BELOW


def _exit_with_index(highs, lows, ep, sl, tp, risk, d):
    """_exitと同じだがexit発生バーのインデックスも返す"""
    half = ep + d * risk * HALF_R
    lim = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False, i
            if h >= tp:  return tp, "win", False, i
            if h >= half:
                for j in range(i + 1, lim):
                    if lows[j] <= ep:  return ep, "win", True, j
                    if highs[j] >= tp: return tp, "win", True, j
                return None, None, True, lim - 1
        else:
            if h >= sl:  return sl, "loss", False, i
            if lo <= tp: return tp, "win", False, i
            if lo <= half:
                for j in range(i + 1, lim):
                    if highs[j] >= ep: return ep, "win", True, j
                    if lows[j] <= tp:  return tp, "win", True, j
                return None, None, True, lim - 1
    return None, None, False, lim - 1


def main():
    print("\n" + "=" * 90)
    print("  ポートフォリオバックテスト 2026年〜（時系列イベント駆動）")
    print(f"  初期資金: ¥{INIT_CASH:,}  リスク: 2%（3000万超で1%）  MAX_OPEN: 無制限")
    print("=" * 90)

    # ── 全銘柄のシグナルを生成 + exit時刻を事前計算 ─────────────────
    # 各シグナルについて (entry_time, exit_time, sym, result, pnl_func) を持つ
    # pnl_funcは「エントリー時equityを引数に取り、PnLを返す」クロージャ

    events = []  # (time, type, data) のリスト
    m1_cache = {}

    for p in PORTFOLIO:
        sym = p["sym"]
        logic = p["logic"]
        rr = p["rr"]
        tol = p["tol"]

        print(f"\n  {sym} Logic-{logic} {rr}R tol={tol} ...", end=" ", flush=True)

        d1m, d4h = load_all(sym)
        if d1m is None:
            print("データなし"); continue

        d1m_2026 = d1m[d1m.index >= START_DATE].copy()
        if len(d1m_2026) == 0:
            print("2026年データなし"); continue

        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d = calc_atr(d1m, 10).to_dict()
        m1c = {
            "idx": d1m_2026.index,
            "opens": d1m_2026["open"].values,
            "closes": d1m_2026["close"].values,
            "highs": d1m_2026["high"].values,
            "lows": d1m_2026["low"].values,
        }
        m1_cache[sym] = m1c

        if logic == "V80":
            sigs = generate_signals_v80(d1m_2026, d4h, spread, m1c, rr=rr, tol=tol)
        else:
            sigs = generate_signals(d1m_2026, d4h, spread, logic, atr_d, m1c, rr=rr, tol=tol)

        # 各シグナルのexit時刻を事前計算
        valid = 0
        for sig in sigs:
            mc = m1c
            sp = mc["idx"].searchsorted(sig["time"], side="right")
            if sp >= len(mc["idx"]):
                continue

            xp, result, half_done, exit_offset = _exit_with_index(
                mc["highs"][sp:], mc["lows"][sp:],
                sig["ep"], sig["sl"], sig["tp"],
                sig["risk"], sig["dir"]
            )
            if result is None:
                continue

            exit_bar_idx = sp + exit_offset
            if exit_bar_idx >= len(mc["idx"]):
                exit_bar_idx = len(mc["idx"]) - 1
            exit_time = mc["idx"][exit_bar_idx]

            events.append({
                "entry_time": sig["time"],
                "exit_time": exit_time,
                "sym": sym,
                "dir": sig["dir"],
                "ep": sig["ep"],
                "sl": sig["sl"],
                "tp": sig["tp"],
                "risk": sig["risk"],
                "xp": xp,
                "result": result,
                "half_done": half_done,
            })
            valid += 1

        print(f"{valid}トレード")

    if not events:
        print("\nトレードなし。終了。")
        return

    print(f"\n  全トレード: {len(events)}件")

    # ── 時系列イベント駆動シミュレーション ─────────────────────────
    # エントリーイベントとexitイベントを時系列に並べる
    timeline = []
    for i, ev in enumerate(events):
        timeline.append((ev["entry_time"], "entry", i))
        timeline.append((ev["exit_time"], "exit", i))
    timeline.sort(key=lambda x: (x[0], 0 if x[1] == "exit" else 1))  # 同時刻はexit優先

    equity = float(INIT_CASH)
    peak = equity
    mdd = 0.0
    open_positions = {}  # trade_id -> {sym, lot, rm, entry_equity, ...}
    open_by_sym = defaultdict(int)  # 銘柄ごとオープン数
    trades = []
    rms = {p["sym"]: RiskManager(p["sym"], risk_pct=RISK_BELOW) for p in PORTFOLIO}

    for ts, etype, tid in timeline:
        ev = events[tid]

        if etype == "entry":
            # 銘柄ごと1ポジ制限
            if open_by_sym[ev["sym"]] > 0:
                continue

            risk_pct = get_risk_pct(equity)
            rm = rms[ev["sym"]]
            rm.risk_pct = risk_pct

            lot = rm.calc_lot(equity, ev["risk"], ev["ep"], usdjpy_rate=USDJPY_RATE)
            if lot <= 0:
                continue

            open_positions[tid] = {
                "lot": lot,
                "risk_pct": risk_pct,
                "entry_equity": equity,
            }
            open_by_sym[ev["sym"]] += 1

        elif etype == "exit":
            if tid not in open_positions:
                continue  # エントリーがスキップされた

            pos = open_positions.pop(tid)
            open_by_sym[ev["sym"]] -= 1
            rm = rms[ev["sym"]]
            lot = pos["lot"]

            half_pnl = 0.0
            if ev["half_done"]:
                hp = ev["ep"] + ev["dir"] * ev["risk"] * HALF_R
                half_pnl = rm.calc_pnl_jpy(ev["dir"], ev["ep"], hp, lot * 0.5, USDJPY_RATE, ev["ep"])
                equity += half_pnl
                rem = lot * 0.5
            else:
                rem = lot

            pnl = rm.calc_pnl_jpy(ev["dir"], ev["ep"], ev["xp"], rem, USDJPY_RATE, ev["ep"])
            equity += pnl
            total = half_pnl + pnl

            trades.append({
                "sym": ev["sym"],
                "result": ev["result"],
                "pnl": total,
                "equity": equity,
                "risk_pct": pos["risk_pct"],
                "entry_time": ev["entry_time"],
                "exit_time": ev["exit_time"],
                "month": ev["exit_time"].strftime("%Y-%m"),
                "open_count": len(open_positions),
            })

            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            mdd = max(mdd, dd)

    # ── 結果集計 ──────────────────────────────────────────────────
    if not trades:
        print("\n取引なし。終了。")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n
    gw = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    monthly = df.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("result", lambda x: (x == "win").sum()),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    eq = float(INIT_CASH)
    monthly_eq = []
    monthly_ret = []
    for m in monthly.index:
        ret = monthly.loc[m, "pnl"] / eq if eq > 0 else 0
        monthly_ret.append(ret)
        eq += monthly.loc[m, "pnl"]
        monthly_eq.append(eq)
    monthly["end_equity"] = monthly_eq
    monthly["return_pct"] = [r * 100 for r in monthly_ret]

    mr = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0
    plus_m = (monthly["pnl"] > 0).sum()

    avg_w = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_l = abs(losses["pnl"].mean()) if len(losses) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0

    # ── 表示 ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  ポートフォリオ結果（8銘柄合算・共通エクイティ・同時ポジ正しく反映）")
    print("=" * 90)

    print(f"""
  期間        : {df['entry_time'].min().strftime('%Y-%m-%d')} 〜 {df['exit_time'].max().strftime('%Y-%m-%d')}
  初期資金    : ¥{INIT_CASH:>14,}
  最終資金    : ¥{equity:>14,.0f}  ({equity / INIT_CASH:.1f}倍)
  トレード数  : {n}件（勝{len(wins)} / 負{len(losses)}）
  勝率        : {wr * 100:.1f}%
  PF          : {pf:.2f}
  Sharpe(月次): {sharpe:.2f}
  Kelly       : {kelly:.3f}
  MDD         : {mdd:.1f}%
  プラス月    : {plus_m}/{len(monthly)}
""")

    # 同時オープン統計
    print(f"  同時オープン: 平均 {df['open_count'].mean():.1f}件  最大 {df['open_count'].max()}件")

    # 月次詳細
    print("\n  ── 月次詳細 ──")
    print(f"  {'月':10s} {'取引数':>6s} {'勝率':>6s} {'月次PnL':>14s} {'月次R%':>8s} {'月末資産':>16s} {'リスク':>6s}")
    print("  " + "-" * 75)
    for m in monthly.index:
        row = monthly.loc[m]
        risk_at_end = "1%" if row["end_equity"] >= RISK_THRESHOLD_JPY else "2%"
        print(f"  {m:10s} {int(row['trades']):6d} {row['wr']:5.1f}% ¥{row['pnl']:>13,.0f} {row['return_pct']:>+7.1f}% ¥{row['end_equity']:>15,.0f} {risk_at_end:>5s}")

    # 銘柄別集計
    print("\n  ── 銘柄別集計 ──")
    print(f"  {'銘柄':10s} {'取引数':>6s} {'勝率':>6s} {'PF':>6s} {'PnL合計':>14s}")
    print("  " + "-" * 50)
    for sym in sorted(df["sym"].unique()):
        g = df[df["sym"] == sym]
        sw = g[g["pnl"] > 0]["pnl"].sum()
        sl_amt = abs(g[g["pnl"] < 0]["pnl"].sum())
        spf = sw / sl_amt if sl_amt > 0 else float("inf")
        swr = (g["pnl"] > 0).sum() / len(g) * 100
        print(f"  {sym:10s} {len(g):6d} {swr:5.1f}% {spf:5.2f} ¥{g['pnl'].sum():>13,.0f}")

    # 日別トレード頻度
    df["date"] = df["entry_time"].dt.date
    daily = df.groupby("date").size()
    print(f"\n  ── 日別トレード頻度 ──")
    print(f"  日平均: {daily.mean():.1f}件/日  最大: {daily.max()}件/日  最小: {daily.min()}件/日")

    # 3000万到達日
    cross_30m = df[df["equity"] >= RISK_THRESHOLD_JPY]
    if len(cross_30m) > 0:
        first_30m = cross_30m.iloc[0]
        print(f"\n  ★ 3000万到達: {first_30m['exit_time'].strftime('%Y-%m-%d')} → リスク1%に切替")

    # 1億到達
    cross_1oku = df[df["equity"] >= 100_000_000]
    if len(cross_1oku) > 0:
        first_1oku = cross_1oku.iloc[0]
        print(f"  ★ 1億到達: {first_1oku['exit_time'].strftime('%Y-%m-%d')}")

    # CSV出力
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "backtest_portfolio_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  結果CSV: {out_path}")


if __name__ == "__main__":
    main()
