"""
backtest_compound.py
====================
複利バックテスト — 総資産1億円まで損切り1%、1億円超は固定ロット

ルール:
  - 初期資金: 68万円
  - 資産 < 1億円: 損切り = 現在資産 × 1%
  - 資産 ≥ 1億円: 損切り = 1億円 × 1% = 100万円 固定
  - 6銘柄同時運用、シグナルは時系列で処理
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH         = 680_000
COMPOUND_CAP      = 100_000_000   # 1億円で複利上限
RISK_PCT          = 0.01          # 損切り = 資産の1%
RR_RATIO          = 2.5
HALF_R            = 1.0
MAX_LOOKAHEAD     = 20_000

KLOW_THR          = 0.0015
A1_EMA_DIST_MIN   = 1.0
A3_DEFAULT_TOL    = 0.30
E1_MAX_WAIT_MIN   = 5
E2_SPIKE_ATR      = 2.0
E2_WINDOW_MIN     = 3
E0_WINDOW_MIN     = 2
ADX_MIN           = 20
STREAK_MIN        = 4

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 採用6銘柄 ────────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "label": "USDJPY (Logic-C)"},
    {"sym": "GBPUSD",  "logic": "A", "label": "GBPUSD (Logic-A)"},
    {"sym": "USDCAD",  "logic": "A", "label": "USDCAD (Logic-A)"},
    {"sym": "NZDUSD",  "logic": "A", "label": "NZDUSD (Logic-A)"},
    {"sym": "AUDUSD",  "logic": "B", "label": "AUDUSD (Logic-B)", "h4_body_ratio_min": 0.3},
    {"sym": "XAUUSD",  "logic": "A", "label": "XAUUSD (Logic-A)"},
]

# ── backtest_portfolio_680k.py から関数をインポート ──────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bt", os.path.join(BASE_DIR, "scripts", "backtest_portfolio_680k.py"))
_bt = importlib.util.module_from_spec(_spec)
_bt.__file__ = os.path.join(BASE_DIR, "scripts", "backtest_portfolio_680k.py")
_spec.loader.exec_module(_bt)

load_1m          = _bt.load_1m
calc_atr         = _bt.calc_atr
generate_signals = _bt.generate_signals
_exit            = _bt._exit


def main():
    print(f"\n{'='*80}")
    print(f"  YAGAMI改 複利ポートフォリオバックテスト")
    print(f"  初期資金 ¥{INIT_CASH:,.0f} / リスク1% / 1億円で固定ロット化")
    print(f"{'='*80}")

    # ── Phase 1: 全銘柄のシグナルを収集 ─────────────────────────────
    all_signals = []
    sym_data = {}

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        print(f"\n  {tgt['label']} ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データ未発見"); continue

        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        edm = tgt.get("ema_dist_min", A1_EMA_DIST_MIN)
        hbr = tgt.get("h4_body_ratio_min", 0.0)
        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c,
                                ema_dist_min=edm, h4_body_ratio_min=hbr)
        print(f"{len(sigs)}シグナル")

        sym_data[sym] = {"m1t": d1m.index, "m1h": d1m["high"].values, "m1l": d1m["low"].values}
        for sig in sigs:
            all_signals.append({**sig, "sym": sym})

    all_signals.sort(key=lambda x: x["time"])
    print(f"\n  統合シグナル数: {len(all_signals)}")

    # ── Phase 2: 複利シミュレーション ─────────────────────────────
    equity = float(INIT_CASH)
    trades = []
    eq_history = [{"time": all_signals[0]["time"] - pd.Timedelta(days=1), "equity": equity}]

    cap_reached = False
    cap_reached_time = None
    fixed_risk_jpy = None   # 1億円到達時のリスク額を固定

    for sig in all_signals:
        sym = sig["sym"]
        sd  = sym_data[sym]

        # ── リスク額の決定 ──────────────────────────────────────
        if equity >= COMPOUND_CAP and not cap_reached:
            cap_reached = True
            cap_reached_time = sig["time"]
            fixed_risk_jpy = COMPOUND_CAP * RISK_PCT  # 100万円固定

        if cap_reached:
            risk_jpy = fixed_risk_jpy
        else:
            risk_jpy = equity * RISK_PCT

        # ── ロット計算 ──────────────────────────────────────────
        rm = RiskManager(sym, risk_pct=RISK_PCT)
        # risk_jpyからロットを逆算: lot = risk_jpy / (sl_distance * conversion)
        # calc_lotはequity * risk_pctを使うので、equityを調整して渡す
        effective_equity = risk_jpy / RISK_PCT
        lot = rm.calc_lot(effective_equity, sig["risk"], sig["ep"], usdjpy_rate=150.0)
        if lot <= 0:
            continue

        sp = sd["m1t"].searchsorted(sig["time"], side="right")
        if sp >= len(sd["m1t"]):
            continue

        xp, result, half_done = _exit(
            sd["m1h"][sp:], sd["m1l"][sp:],
            sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"]
        )
        if result is None:
            continue

        # ── PnL計算 ──────────────────────────────────────────
        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot * 0.5, 150.0, sig["ep"])
            rem = lot * 0.5
        else:
            rem = lot

        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, 150.0, sig["ep"])
        total_pnl = half_pnl + pnl
        equity += total_pnl

        trades.append({
            "time": sig["time"], "sym": sym, "result": result,
            "pnl": total_pnl, "equity": equity, "risk_jpy": risk_jpy,
            "lot": lot, "month": sig["time"].strftime("%Y-%m"),
        })
        eq_history.append({"time": sig["time"], "equity": equity})

    if not trades:
        print("\nトレードなし"); return

    df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_history)

    # ── 統計計算 ──────────────────────────────────────────────
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    peak = INIT_CASH; mdd = 0; mdd_jpy = 0; eq = INIT_CASH
    for _, row in df.iterrows():
        eq += row["pnl"]
        peak = max(peak, eq)
        dd_pct = (peak - eq) / peak * 100
        dd_jpy = peak - eq
        if dd_pct > mdd:
            mdd = dd_pct
            mdd_jpy = dd_jpy

    monthly = df.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()

    eq_t = INIT_CASH
    monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq_t if eq_t > 0 else 0
        monthly_ret.append(ret)
        eq_t += monthly[m]
    mr = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0

    final_equity = equity

    # ── 出力 ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ■ 銘柄別トレード内訳")
    print(f"  {'銘柄':10} {'トレード':>8} {'勝率':>8} {'損益(¥)':>16}")
    print(f"  {'-'*48}")
    for sym_name in [t["sym"] for t in TARGETS]:
        sub = df[df["sym"] == sym_name]
        if sub.empty: continue
        sw = len(sub[sub["pnl"] > 0])
        print(f"  {sym_name:10} {len(sub):>8} {sw/len(sub)*100:>7.1f}% {sub['pnl'].sum():>+16,.0f}")

    print(f"\n{'='*80}")
    print(f"  ■ 複利ポートフォリオ最終結果")
    print(f"{'='*80}")
    print(f"  初期資金        : ¥{INIT_CASH:>16,.0f}")
    print(f"  最終資産        : ¥{final_equity:>16,.0f}")
    print(f"  総損益          : ¥{final_equity - INIT_CASH:>+16,.0f} ({(final_equity/INIT_CASH - 1)*100:+,.1f}%)")
    print(f"  トレード数      : {n:>10}")
    print(f"  勝率            : {wr*100:>9.1f}%")
    print(f"  最大DD          : {mdd:>9.1f}% (¥{mdd_jpy:>12,.0f})")
    print(f"  プラス月        : {plus_m}/{len(monthly)}")
    print(f"  期間            : {df.iloc[0]['time'].strftime('%Y-%m-%d')} 〜 {df.iloc[-1]['time'].strftime('%Y-%m-%d')}")
    if cap_reached:
        print(f"\n  ★ 1億円到達    : {cap_reached_time.strftime('%Y-%m-%d')}")
        print(f"  ★ 固定リスク額  : ¥{fixed_risk_jpy:>12,.0f}/トレード")

    # ── 月次損益 ──────────────────────────────────────────────
    print(f"\n  ■ 月次損益")
    eq_m = INIT_CASH
    for m in monthly.index:
        pnl_m = monthly[m]
        eq_m += pnl_m
        # 月末の平均リスク額
        m_trades = df[df["month"] == m]
        avg_risk = m_trades["risk_jpy"].mean()
        print(f"    {m} : ¥{pnl_m:>+14,.0f}  (残高 ¥{eq_m:>14,.0f})  平均リスク ¥{avg_risk:>10,.0f}")

    # ── エクイティカーブ描画 ──────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(14, 7))

    times = eq_df["time"].values
    eqs   = eq_df["equity"].values

    ax.plot(times, eqs, color="#2196F3", linewidth=1.5, label="Portfolio Equity (Compound)")
    ax.axhline(y=INIT_CASH, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=COMPOUND_CAP, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Compound Cap: ¥{COMPOUND_CAP/1e8:.0f}億")

    # MDD塗り
    peak_arr = np.maximum.accumulate(eqs)
    ax.fill_between(times, eqs, peak_arr, alpha=0.15, color="red", label="Drawdown")

    ax.set_title(
        f"YAGAMI Kai Compound — ¥{INIT_CASH/1e4:.0f}万 → ¥{final_equity/1e8:.2f}億 "
        f"(+{(final_equity/INIT_CASH-1)*100:,.0f}%)",
        fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (JPY)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"¥{x/1e8:.1f}億" if x >= 1e8 else (f"¥{x/1e4:.0f}万" if x >= 1e4 else f"¥{x:,.0f}")))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    info = (f"Trades={n}  WR={wr*100:.1f}%  MDD={mdd:.1f}%  "
            f"Risk=1% (cap ¥{COMPOUND_CAP/1e8:.0f}億)")
    ax.text(0.5, -0.15, info, transform=ax.transAxes, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "equity_curve_compound.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  エクイティカーブ保存: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
