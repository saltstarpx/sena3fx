"""
backtest_v80_non_production.py
==============================
v80ロジック（KMID+KLOW+Body）を非本番銘柄（1mデータあり）でテスト。

対象10銘柄:
  AUDJPY, CADJPY, EURGBP, EURJPY, GBPJPY, USDCHF, XAGUSD, NAS100, SPX500, US30

バリアント:
  (1) KMID+KLOW+Body 2.5R
  (2) KMID+KLOW+Body 3.0R

判定基準: OOS PF>=2.0, OOS/IS>=0.70, MDD<=25%
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "strategies", "current"))

from utils.risk_manager import RiskManager, SYMBOL_CONFIG
from yagami_mtf_v79 import generate_signals

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
RISK_PCT      = 0.02
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 対象銘柄 ─────────────────────────────────────────────────────
# (sym, 1m_path, pip_size, spread_pips)
# spread は SYMBOL_CONFIG にあればそこから取得、なければ指定値
TARGETS = [
    {"sym": "AUDJPY",  "path": os.path.join(DATA_DIR_OHLC, "AUDJPY_1m.csv"),  "pip": 0.01,   "spread": None},
    {"sym": "CADJPY",  "path": os.path.join(DATA_DIR_OHLC, "CADJPY_1m.csv"),  "pip": 0.01,   "spread": 2.0},
    {"sym": "EURGBP",  "path": os.path.join(DATA_DIR_OHLC, "EURGBP_1m.csv"),  "pip": 0.0001, "spread": None},
    {"sym": "EURJPY",  "path": os.path.join(DATA_DIR_OHLC, "EURJPY_1m.csv"),  "pip": 0.01,   "spread": None},
    {"sym": "GBPJPY",  "path": os.path.join(DATA_DIR_OHLC, "GBPJPY_1m.csv"),  "pip": 0.01,   "spread": None},
    {"sym": "USDCHF",  "path": os.path.join(DATA_DIR_OHLC, "USDCHF_1m.csv"),  "pip": 0.0001, "spread": None},
    {"sym": "XAGUSD",  "path": os.path.join(DATA_DIR_OHLC, "XAGUSD_1m.csv"),  "pip": 0.001,  "spread": None},
    {"sym": "NAS100",  "path": os.path.join(DATA_DIR, "nas100_1m.csv"),        "pip": 1.0,    "spread": None},
    {"sym": "SPX500",  "path": os.path.join(DATA_DIR, "spx500_1m.csv"),        "pip": 0.1,    "spread": None},
    {"sym": "US30",    "path": os.path.join(DATA_DIR, "us30_1m.csv"),          "pip": 1.0,    "spread": None},
]

# バリアント: (label, rr_ratio)
VARIANTS = [
    ("KMID+KLOW+Body 2.5R", 2.5),
    ("KMID+KLOW+Body 3.0R", 3.0),
]


# ── データロード ─────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df.dropna(subset=["open", "high", "low", "close"])


def resample_htf(d1m, freq):
    r = d1m.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return r


def split_is_oos(d1m, ratio=0.4):
    n = int(len(d1m) * ratio)
    ts = d1m.index[n]
    return d1m[d1m.index < ts].copy(), d1m[d1m.index >= ts].copy(), ts


# ── EXIT + シミュレーション (backtest_v80_purified.pyから移植) ──────
def _exit_with_half(highs, lows, ep, sl, tp, risk, d):
    half = ep + d * risk * HALF_R
    lim = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False
            if h  >= tp: return tp, "win",  False
            if h  >= half:
                for j in range(i+1, lim):
                    if lows[j]  <= ep: return ep, "win", True
                    if highs[j] >= tp: return tp, "win", True
                return None, None, True
        else:
            if h  >= sl: return sl, "loss", False
            if lo <= tp: return tp, "win",  False
            if lo <= half:
                for j in range(i+1, lim):
                    if highs[j] >= ep: return ep, "win", True
                    if lows[j]  <= tp: return tp, "win", True
                return None, None, True
    return None, None, False


def simulate_half(signals, d1m, sym):
    if not signals:
        return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=RISK_PCT)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result, half_done = _exit_with_half(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
        if result is None: continue
        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl; rem = lot * 0.5
        else:
            rem = lot
        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        trades.append({"result": result, "pnl": half_pnl + pnl,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)
    return trades, equity, mdd


# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 5:
        return {}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr = len(wins) / n
    gw = wins.sum(); gl = abs(loss.sum())
    pf = gw / gl if gl > 0 else float("inf")
    total_pnl = df["pnl"].sum()
    monthly = df.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()
    eq = init; monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret); eq += monthly[m]
    mr = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0
    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "total_pnl": total_pnl,
            "final_eq": eq}


# ── メイン ────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*140)
    print("  v80 非本番銘柄バックテスト: KMID+KLOW+Body × 2.5R/3.0R")
    print("  対象: AUDJPY, CADJPY, EURGBP, EURJPY, GBPJPY, USDCHF, XAGUSD, NAS100, SPX500, US30")
    print("  IS/OOS = 40/60 split, INIT_CASH=1M, risk=2%")
    print("="*140)

    all_results = []

    for tgt in TARGETS:
        sym = tgt["sym"]
        print(f"\n  ◆ {sym} ...", end=" ", flush=True)

        if not os.path.exists(tgt["path"]):
            print("データなし"); continue

        # データロード
        d1m_full = load_csv(tgt["path"])
        print(f"({len(d1m_full):,}行)", end=" ", flush=True)

        # HTF生成
        d15m = resample_htf(d1m_full, "15min")
        d4h  = resample_htf(d1m_full, "4h")

        # スプレッド取得
        pip = tgt["pip"]
        if tgt["spread"] is not None:
            spread_pips = tgt["spread"]
        elif sym in SYMBOL_CONFIG:
            spread_pips = SYMBOL_CONFIG[sym]["spread"]
        else:
            print("スプレッド不明"); continue

        # IS/OOS分割
        is_d, oos_d, split_ts = split_is_oos(d1m_full)
        is_15m  = resample_htf(is_d, "15min")
        oos_15m = resample_htf(oos_d, "15min")

        for period_label, d1m_period, d15m_period in [("IS", is_d, is_15m), ("OOS", oos_d, oos_15m)]:
            for vl, rr in VARIANTS:
                # generate_signals: v80 = KMID+KLOW (built-in) + h4_body_ratio_min=0.3
                # No additional filters (no 1d trend, no ADX, no streak)
                sigs = generate_signals(
                    data_1m=d1m_period,
                    data_15m=d15m_period,
                    data_4h=d4h,  # full 4h for indicator lookback
                    spread_pips=spread_pips,
                    rr_ratio=rr,
                    pip_size=pip,
                    h4_body_ratio_min=0.3,
                    tol_factor=0.30,
                )

                trades, eq, mdd = simulate_half(sigs, d1m_period, sym)
                st = calc_stats(trades)
                if st:
                    st["mdd"] = mdd

                all_results.append({
                    "sym": sym, "variant": vl, "period": period_label,
                    "rr": rr, "stats": st,
                })

        print("完了")

    # ══════════════════════════════════════════════════════════════
    # 結果テーブル
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*150)
    print("  ■ OOS期間 銘柄別結果")
    print(f"  {'銘柄':8} {'バリアント':26} | {'n':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'Kelly':>6} {'MDD':>7} {'月+':>5} {'総PnL':>12} {'IS PF':>7} {'OOS/IS':>7} {'判定':>6}")
    print("  " + "-"*145)

    csv_rows = []
    syms_done = set()

    for tgt in TARGETS:
        sym = tgt["sym"]
        if sym in syms_done: continue
        syms_done.add(sym)

        for vl, rr in VARIANTS:
            oos_r = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "OOS"]
            is_r  = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "IS"]
            st = oos_r[0]["stats"] if oos_r and oos_r[0]["stats"] else {}
            is_st = is_r[0]["stats"] if is_r and is_r[0]["stats"] else {}
            if not st:
                print(f"  {sym:8} {vl:26} | {'n<5 trades':>30}")
                csv_rows.append({
                    "sym": sym, "variant": vl, "rr": rr,
                    "n": 0, "wr": 0, "pf": 0, "sharpe": 0, "kelly": 0, "mdd": 0,
                    "plus_m": 0, "total_m": 0, "total_pnl": 0,
                    "is_pf": 0, "oos_is_ratio": 0, "judgment": "N/A",
                })
                continue

            is_pf = is_st.get("pf", 0) if is_st else 0
            oos_is = st["pf"] / is_pf if is_pf > 0 and st["pf"] < 99 else 0
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "inf"

            # 判定: OOS PF>=2.0, OOS/IS>=0.70, MDD<=25%
            pass_pf = st["pf"] >= 2.0
            pass_ois = oos_is >= 0.70
            pass_mdd = st["mdd"] <= 25.0
            judgment = "PASS" if (pass_pf and pass_ois and pass_mdd) else "FAIL"
            jmark = "PASS" if judgment == "PASS" else "FAIL"

            print(f"  {sym:8} {vl:26} | "
                  f"{st['n']:>4} {st['wr']*100:>5.1f}% {pf_s:>6} {st['sharpe']:>7.2f} "
                  f"{st['kelly']:>6.3f} {st['mdd']:>6.1f}% "
                  f"{st['plus_m']:>2}/{st['total_m']:<2} "
                  f"{st['total_pnl']:>11,.0f} {is_pf:>7.2f} {oos_is:>7.2f} "
                  f"{'PASS' if judgment == 'PASS' else 'FAIL':>6}")

            csv_rows.append({
                "sym": sym, "variant": vl, "rr": rr,
                "n": st["n"], "wr": round(st["wr"]*100, 1),
                "pf": round(st["pf"], 2) if st["pf"] < 99 else 999,
                "sharpe": round(st["sharpe"], 2),
                "kelly": round(st["kelly"], 3), "mdd": round(st["mdd"], 1),
                "plus_m": st["plus_m"], "total_m": st["total_m"],
                "total_pnl": round(st["total_pnl"]),
                "is_pf": round(is_pf, 2),
                "oos_is_ratio": round(oos_is, 2),
                "judgment": judgment,
            })
        print()

    # ── 判定サマリー ──────────────────────────────────────────────
    print("\n" + "="*140)
    print("  ■ 判定サマリー（OOS PF>=2.0 & OOS/IS>=0.70 & MDD<=25%）")
    print("="*140)

    for tgt in TARGETS:
        sym = tgt["sym"]
        for vl, rr in VARIANTS:
            oos_r = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "OOS"]
            is_r  = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "IS"]
            st = oos_r[0]["stats"] if oos_r and oos_r[0]["stats"] else {}
            is_st = is_r[0]["stats"] if is_r and is_r[0]["stats"] else {}
            if not st:
                print(f"  {sym:8} {vl:26} → データ不足")
                continue

            is_pf = is_st.get("pf", 0) if is_st else 0
            oos_is = st["pf"] / is_pf if is_pf > 0 and st["pf"] < 99 else 0

            pass_pf  = st["pf"] >= 2.0
            pass_ois = oos_is >= 0.70
            pass_mdd = st["mdd"] <= 25.0
            all_pass = pass_pf and pass_ois and pass_mdd

            pf_mark  = "PF OK" if pass_pf else f"PF NG({st['pf']:.2f})"
            ois_mark = "OOS/IS OK" if pass_ois else f"OOS/IS NG({oos_is:.2f})"
            mdd_mark = "MDD OK" if pass_mdd else f"MDD NG({st['mdd']:.1f}%)"

            verdict = "PASS" if all_pass else "FAIL"
            print(f"  {sym:8} {vl:26} → {verdict:4}  {pf_mark:16} {ois_mark:18} {mdd_mark:16}")

    # ── カテゴリ別平均 ────────────────────────────────────────────
    print("\n" + "="*140)
    print("  ■ カテゴリ別平均（OOS）")
    print(f"  {'カテゴリ':10} {'バリアント':26} | {'avg PF':>8} {'avg Sharpe':>11} "
          f"{'avg MDD':>9} {'avg Kelly':>10} {'avg WR':>8} {'avg n':>6}")
    print("  " + "-"*100)

    categories = {
        "FX": ["AUDJPY", "CADJPY", "EURGBP", "EURJPY", "GBPJPY", "USDCHF"],
        "METALS": ["XAGUSD"],
        "INDICES": ["NAS100", "SPX500", "US30"],
    }

    for cat, syms in categories.items():
        for vl, rr in VARIANTS:
            pfs = []; shs = []; mdds = []; kls = []; wrs = []; ns = []
            for r in all_results:
                if r["sym"] in syms and r["variant"] == vl and r["period"] == "OOS" and r["stats"]:
                    st = r["stats"]
                    if st.get("pf", 0) < 99:
                        pfs.append(st["pf"]); shs.append(st["sharpe"])
                        mdds.append(st["mdd"]); kls.append(st["kelly"])
                        wrs.append(st["wr"]*100); ns.append(st["n"])
            if pfs:
                print(f"  {cat:10} {vl:26} | {np.mean(pfs):>8.2f} {np.mean(shs):>11.2f} "
                      f"{np.mean(mdds):>8.1f}% {np.mean(kls):>10.3f} "
                      f"{np.mean(wrs):>7.1f}% {np.mean(ns):>5.0f}")

    # CSV保存
    out_path = os.path.join(OUT_DIR, "backtest_v80_non_production.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n\n  結果保存: {out_path}")
    print(f"  実行時間: {elapsed:.0f}秒")


if __name__ == "__main__":
    main()
