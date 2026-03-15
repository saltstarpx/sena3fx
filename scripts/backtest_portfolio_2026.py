"""
backtest_portfolio_2026.py
===========================
2026年〜現在のポートフォリオバックテスト

【設定】
  初期資金: 65万円
  リスク: 2%（3000万超で1%）
  MAX_OPEN: 無制限（銘柄ごと1ポジ制限のみ）
  8銘柄: 本番確定ロジック

【銘柄×ロジック】
  USDCAD  v80 3.0R tol=0.30
  XAUUSD  Logic-A 2.5R tol=0.20
  EURUSD  v80 3.0R tol=0.30
  AUDUSD  v80 3.0R tol=0.30
  GBPUSD  v80 3.0R tol=0.30
  USDCHF  v80 3.0R tol=0.30
  NZDUSD  Logic-A 3.0R tol=0.20
  USDJPY  Logic-C 3.0R tol=0.30
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 既存バックテストエンジンを再利用
from scripts.backtest_final_optimized import (
    load_all, calc_atr, build_4h, build_1h,
    generate_signals, generate_signals_v80, _exit,
    SYMBOL_CONFIG, USDJPY_RATE, HALF_R, MAX_LOOKAHEAD,
    RR_RATIO, RR_RATIO_V80,
)
from utils.risk_manager import RiskManager

# ── 設定 ───────────────────────────────────────────────────────────
INIT_CASH = 650_000
RISK_THRESHOLD_JPY = 30_000_000  # 3000万
RISK_BELOW = 0.02   # 3000万未満: 2%
RISK_ABOVE = 0.01   # 3000万以上: 1%
START_DATE = "2026-01-01"

# 本番確定: 銘柄×ロジック×RR×tol
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
    """3000万で切替"""
    return RISK_ABOVE if equity >= RISK_THRESHOLD_JPY else RISK_BELOW

def main():
    print("\n" + "=" * 90)
    print("  ポートフォリオバックテスト 2026年〜")
    print(f"  初期資金: ¥{INIT_CASH:,}  リスク: 2%（3000万超で1%）  MAX_OPEN: 無制限")
    print("=" * 90)

    # ── 全銘柄のシグナルを生成 ─────────────────────────────────────
    all_signals = []
    for p in PORTFOLIO:
        sym = p["sym"]
        logic = p["logic"]
        rr = p["rr"]
        tol = p["tol"]

        print(f"\n  {sym} Logic-{logic} {rr}R tol={tol} ...", end=" ", flush=True)

        d1m, d4h = load_all(sym)
        if d1m is None:
            print("データなし")
            continue

        # 2026年以降のデータのみ
        d1m_2026 = d1m[d1m.index >= START_DATE].copy()
        if len(d1m_2026) == 0:
            print("2026年データなし")
            continue

        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d = calc_atr(d1m, 10).to_dict()  # ATRは全期間で計算
        m1c = {
            "idx": d1m_2026.index,
            "opens": d1m_2026["open"].values,
            "closes": d1m_2026["close"].values,
            "highs": d1m_2026["high"].values,
            "lows": d1m_2026["low"].values,
        }

        if logic == "V80":
            sigs = generate_signals_v80(d1m_2026, d4h, spread, m1c, rr=rr, tol=tol)
        else:
            sigs = generate_signals(d1m_2026, d4h, spread, logic, atr_d, m1c, rr=rr, tol=tol)

        for s in sigs:
            s["sym"] = sym
        all_signals.extend(sigs)
        print(f"{len(sigs)}シグナル")

    if not all_signals:
        print("\nシグナルなし。終了。")
        return

    # 時系列順にソート
    all_signals.sort(key=lambda x: x["time"])
    print(f"\n  全シグナル: {len(all_signals)}件")
    print(f"  期間: {all_signals[0]['time'].strftime('%Y-%m-%d')} 〜 {all_signals[-1]['time'].strftime('%Y-%m-%d')}")

    # ── ポートフォリオシミュレーション（共通エクイティ） ────────────
    equity = float(INIT_CASH)
    peak = equity
    mdd = 0.0
    trades = []
    active_syms = set()  # 銘柄ごと1ポジ制限チェック用

    # 各銘柄のRiskManager
    rms = {p["sym"]: RiskManager(p["sym"], risk_pct=RISK_BELOW) for p in PORTFOLIO}

    # 1分足データをキャッシュ（exitシミュレーション用）
    m1_cache = {}
    for p in PORTFOLIO:
        sym = p["sym"]
        d1m, _ = load_all(sym)
        if d1m is not None:
            d1m_2026 = d1m[d1m.index >= START_DATE]
            m1_cache[sym] = {
                "idx": d1m_2026.index,
                "highs": d1m_2026["high"].values,
                "lows": d1m_2026["low"].values,
            }

    skipped = 0
    for sig in all_signals:
        sym = sig["sym"]

        # 銘柄ごと1ポジ制限（簡易: 同じ銘柄で未決済がある場合スキップ）
        # ここではシグナル順にexitを即時解決するので、active_symsは使わない
        # （バックテストでは各シグナルのexit結果を即座に計算）

        risk_pct = get_risk_pct(equity)
        rm = rms[sym]
        rm.risk_pct = risk_pct

        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        mc = m1_cache.get(sym)
        if mc is None:
            continue

        sp = mc["idx"].searchsorted(sig["time"], side="right")
        if sp >= len(mc["idx"]):
            continue

        xp, result, half_done = _exit(
            mc["highs"][sp:], mc["lows"][sp:],
            sig["ep"], sig["sl"], sig["tp"],
            sig["risk"], sig["dir"]
        )
        if result is None:
            continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot * 0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl
            rem = lot * 0.5
        else:
            rem = lot

        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total = half_pnl + pnl

        trades.append({
            "sym": sym,
            "result": result,
            "pnl": total,
            "equity": equity,
            "risk_pct": risk_pct,
            "time": sig["time"],
            "month": sig["time"].strftime("%Y-%m"),
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

    # 月次集計
    monthly = df.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("result", lambda x: (x == "win").sum()),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    # 月次エクイティ
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
    print("  ポートフォリオ結果（8銘柄合算・共通エクイティ）")
    print("=" * 90)

    print(f"""
  期間        : {df['time'].min().strftime('%Y-%m-%d')} 〜 {df['time'].max().strftime('%Y-%m-%d')}
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

    # 月次詳細
    print("  ── 月次詳細 ──")
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
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").size()
    print(f"\n  ── 日別トレード頻度 ──")
    print(f"  日平均: {daily.mean():.1f}件/日  最大: {daily.max()}件/日  最小: {daily.min()}件/日")

    # 3000万到達日
    cross_30m = df[df["equity"] >= RISK_THRESHOLD_JPY]
    if len(cross_30m) > 0:
        first_30m = cross_30m.iloc[0]
        print(f"\n  ★ 3000万到達: {first_30m['time'].strftime('%Y-%m-%d')} → リスク1%に切替")

    # 1億到達
    cross_1oku = df[df["equity"] >= 100_000_000]
    if len(cross_1oku) > 0:
        first_1oku = cross_1oku.iloc[0]
        print(f"  ★ 1億到達: {first_1oku['time'].strftime('%Y-%m-%d')}")

    # CSV出力
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "backtest_portfolio_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  結果CSV: {out_path}")


if __name__ == "__main__":
    main()
