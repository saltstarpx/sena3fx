"""
backtest_mdd_reduction.py
=========================
MDD削減のためのtol_factor最適化 + USDJPYロジック見直し

【目的】
  既存パラメータ tol_factor（二番底/天井の許容幅）を調整するだけで
  ポートフォリオ全体のMDDを削減できるか検証する。
  新フィルター追加ではなく、パラメータ変更のみ。

【テスト内容】
  Phase 1: データロード（7銘柄）
  Phase 2: tol_factor × 5値 × 7銘柄 = 35パターンの全期間テスト
  Phase 3: USDJPY Logic-C→Logic-A 切替テスト
  Phase 4: ポートフォリオ集計（Sharpe / MDD）
  Phase 5: IS/OOS過学習チェック
  Phase 6: 推奨出力
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
RR_RATIO      = 2.5
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E0_WINDOW_MIN   = 2
ADX_MIN         = 20
STREAK_MIN      = 4

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── テスト設定 ───────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY", "logic": "C", "cat": "FX_JPY"},
    {"sym": "GBPUSD", "logic": "A", "cat": "FX_USD"},
    {"sym": "EURUSD", "logic": "C", "cat": "FX_USD"},
    {"sym": "USDCAD", "logic": "A", "cat": "FX_USD"},
    {"sym": "NZDUSD", "logic": "A", "cat": "FX_USD"},
    {"sym": "XAUUSD", "logic": "A", "cat": "METALS"},
    {"sym": "AUDUSD", "logic": "B", "cat": "FX_USD"},
]
TOL_VALUES = [0.30, 0.25, 0.20, 0.15, 0.10]
LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

# ── データロード（backtest_final_optimized.py と同一）────────────
def load_csv(path):
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def _is_lfs_pointer(path):
    """Git LFSポインタファイルかどうかを判定"""
    try:
        with open(path, 'r') as f:
            first_line = f.readline()
            return first_line.startswith('version https://git-lfs')
    except Exception:
        return False

def load_all(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p) and not _is_lfs_pointer(p):
            d1m = load_csv(p); break
    else:
        return None, None
    # IS/OOS分割ファイルを優先（LFSポインタ回避）
    p_is  = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
    p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
    if os.path.exists(p_is) and os.path.exists(p_oos):
        d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
        return d1m, d4h[~d4h.index.duplicated(keep="first")].sort_index()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
        if os.path.exists(p) and not _is_lfs_pointer(p):
            return d1m, load_csv(p)
    d4h = d1m.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return d1m, d4h

# ── インジケーター ──────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    h = df["high"]; l = df["low"]
    pdm = h.diff().clip(lower=0); mdm = (-l.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0.0; mdm[mdm < pdm] = 0.0
    atr = calc_atr(df, 1).ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dim = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)

def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calc_adx(df, 14)
    d1 = None
    if need_1d:
        d1 = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df):
    r = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    r["atr"]   = calc_atr(r, 14)
    r["ema20"] = r["close"].ewm(span=20, adjust=False).mean()
    return r

# ── エントリー ──────────────────────────────────────────────────
def pick_e0(t, sp, direction, m1c):
    idx = m1c["idx"]; s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E0_WINDOW_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

def pick_e1(t, direction, sp, m1c):
    idx = m1c["idx"]; s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E1_MAX_WAIT_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        o = m1c["opens"][i]; c = m1c["closes"][i]
        if direction == 1 and c <= o: continue
        if direction == -1 and c >= o: continue
        ni = i + 1
        if ni >= len(idx): return None, None
        return idx[ni], m1c["opens"][ni] + (sp if direction == 1 else -sp)
    return None, None

def pick_e2(t, direction, sp, atr_d, m1c):
    idx = m1c["idx"]; s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        rng = m1c["highs"][i] - m1c["lows"][i]
        av  = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and rng > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

# ── フィルター ──────────────────────────────────────────────────
def chk_kmid(b, d): return (d == 1 and b["close"] > b["open"]) or (d == -1 and b["close"] < b["open"])
def chk_klow(b): return (min(b["open"], b["close"]) - b["low"]) / b["open"] < KLOW_THR if b["open"] > 0 else False
def chk_ema(b): return not pd.isna(b["atr"]) and b["atr"] > 0 and abs(b["close"] - b["ema20"]) >= b["atr"] * A1_EMA_DIST_MIN

# ── シグナル生成（tol_factorパラメータ化）──────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol_factor=0.30):
    d4h, d1d = build_4h(d4h_full, need_1d=(logic == "A"))
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1 = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue

        h4b = d4h[d4h.index < hct]
        if len(h4b) < max(2, STREAK_MIN): continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if logic == "A":
            if d1d is None: continue
            d1b = d1d[d1d.index.normalize() < hct.normalize()]
            if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        elif logic == "B":
            if h4l.get("adx", 0) < ADX_MIN: continue
            if not all(t == trend for t in h4b["trend"].iloc[-STREAK_MIN:].values): continue

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if logic != "C" and not chk_ema(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol_factor: continue  # ★ パラメータ化

        if logic == "C":
            if d == 1 and p1["close"] <= p1["open"]: continue
            if d == -1 and p1["close"] >= p1["open"]: continue

        if logic == "A":   et, ep = pick_e2(hct, d, spread, atr_d, m1c)
        elif logic == "C": et, ep = pick_e0(hct, spread, d, m1c)
        else:              et, ep = pick_e1(hct, d, spread, m1c)

        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * RR_RATIO, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── シミュレーション ────────────────────────────────────────────
def _exit(highs, lows, ep, sl, tp, risk, d):
    half = ep + d * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
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

def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result, half_done = _exit(m1h[sp:], m1l[sp:],
                                       sig["ep"], sig["sl"], sig["tp"],
                                       sig["risk"], sig["dir"])
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
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 10: return {}
    df = pd.DataFrame(trades); n = len(df)
    wins = df[df["pnl"] > 0]["pnl"]; loss = df[df["pnl"] < 0]["pnl"]
    wr = len(wins) / n; gw = wins.sum(); gl = abs(loss.sum())
    pf = gw / gl if gl > 0 else float("inf")
    monthly = df.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()
    eq = init; rets = []
    for m in monthly.index:
        rets.append(monthly[m] / eq if eq > 0 else 0); eq += monthly[m]
    mr = np.array(rets)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0
    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "mdd": 0.0}

# ── ポートフォリオ指標 ──────────────────────────────────────────
def portfolio_sharpe(all_trades):
    combined = {}
    for sym, trades in all_trades.items():
        for t in trades:
            combined[t["month"]] = combined.get(t["month"], 0) + t["pnl"]
    if len(combined) < 2: return 0.0
    months = sorted(combined.keys())
    eq = INIT_CASH * len(all_trades); rets = []
    for m in months:
        rets.append(combined[m] / eq if eq > 0 else 0); eq += combined[m]
    mr = np.array(rets)
    return (mr.mean() / mr.std()) * np.sqrt(12) if mr.std() > 0 else 0.0

def portfolio_mdd(all_trades):
    combined = {}
    for sym, trades in all_trades.items():
        for t in trades:
            combined[t["month"]] = combined.get(t["month"], 0) + t["pnl"]
    if not combined: return 0.0
    eq = INIT_CASH * len(all_trades); peak = eq; mdd = 0.0
    for m in sorted(combined.keys()):
        eq += combined[m]
        peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak * 100)
    return mdd

# ── 実行ヘルパー ────────────────────────────────────────────────
def run_sym(d1m, d4h, sym, logic, tol):
    cfg = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_d = calc_atr(d1m, 10).to_dict()
    m1c = {"idx": d1m.index, "opens": d1m["open"].values,
           "closes": d1m["close"].values,
           "highs": d1m["high"].values, "lows": d1m["low"].values}
    sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, tol_factor=tol)
    trades, eq, mdd = simulate(sigs, d1m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd
    return st, trades

# ── メイン ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*110)
    print("  MDD削減バックテスト: tol_factor最適化 + USDJPYロジック見直し")
    print("="*110)

    # ── Phase 1: データロード ──
    print("\n  [Phase 1] データロード...")
    sym_data = {}
    for tgt in TARGETS:
        sym = tgt["sym"]
        d1m, d4h = load_all(sym)
        if d1m is None:
            print(f"    ❌ {sym}: データ未発見"); continue
        sym_data[sym] = {"d1m": d1m, "d4h": d4h, "logic": tgt["logic"], "cat": tgt["cat"]}
        print(f"    ✅ {sym}: 1m={len(d1m):,}行  4h={len(d4h):,}行")

    # ── Phase 2: 全期間 tol_factor テスト ──
    print(f"\n  [Phase 2] 全期間テスト（tol_factor × {len(TOL_VALUES)}値 × {len(sym_data)}銘柄）")
    print("  " + "-"*105)

    # {tol: {sym: (stats, trades)}}
    results = {}
    for tol in TOL_VALUES:
        results[tol] = {}
        for sym, data in sym_data.items():
            st, trades = run_sym(data["d1m"], data["d4h"], sym,
                                 data["logic"], tol)
            results[tol][sym] = (st, trades)

    # 銘柄別テーブル表示
    for sym in sym_data:
        logic = sym_data[sym]["logic"]
        print(f"\n  {sym} (Logic-{logic}: {LOGIC_NAMES[logic]})")
        print(f"    {'tol':>6} {'n':>5} {'WR':>6} {'PF':>7} {'Sharpe':>7} {'MDD':>6} {'Kelly':>7} {'月+':>5}")
        print("    " + "-"*55)
        for tol in TOL_VALUES:
            st = results[tol][sym][0]
            if not st:
                print(f"    {tol:.2f}   データ不足")
                continue
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            cur = " ←現行" if tol == 0.30 else ""
            print(f"    {tol:.2f} {st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>7} "
                  f"{st['sharpe']:>7.2f} {st['mdd']:>5.1f}% {st['kelly']:>7.3f} "
                  f"{st['plus_m']:>2}/{st['total_m']:<2}{cur}")

    # ── Phase 3: USDJPY Logic-A テスト ──
    print(f"\n  [Phase 3] USDJPY ロジック切替テスト（Logic-C vs Logic-A）")
    print("  " + "-"*105)

    usdjpy_data = sym_data.get("USDJPY")
    usdjpy_a_results = {}
    if usdjpy_data:
        print(f"\n    {'Logic':>8} {'tol':>5} {'n':>5} {'WR':>6} {'PF':>7} {'Sharpe':>7} {'MDD':>6} {'Kelly':>7} {'月+':>5}")
        print("    " + "-"*65)
        for tol in TOL_VALUES:
            # Logic-C（現行）
            st_c = results[tol]["USDJPY"][0]
            if st_c:
                pf_s = f"{st_c['pf']:.2f}" if st_c['pf'] < 99 else "∞"
                cur = " ←現行" if tol == 0.30 else ""
                print(f"    Logic-C {tol:.2f} {st_c['n']:>5} {st_c['wr']*100:>5.1f}% {pf_s:>7} "
                      f"{st_c['sharpe']:>7.2f} {st_c['mdd']:>5.1f}% {st_c['kelly']:>7.3f} "
                      f"{st_c['plus_m']:>2}/{st_c['total_m']:<2}{cur}")

            # Logic-A
            st_a, tr_a = run_sym(usdjpy_data["d1m"], usdjpy_data["d4h"],
                                  "USDJPY", "A", tol)
            usdjpy_a_results[tol] = (st_a, tr_a)
            if st_a:
                pf_s = f"{st_a['pf']:.2f}" if st_a['pf'] < 99 else "∞"
                print(f"    Logic-A {tol:.2f} {st_a['n']:>5} {st_a['wr']*100:>5.1f}% {pf_s:>7} "
                      f"{st_a['sharpe']:>7.2f} {st_a['mdd']:>5.1f}% {st_a['kelly']:>7.3f} "
                      f"{st_a['plus_m']:>2}/{st_a['total_m']:<2}")
            else:
                print(f"    Logic-A {tol:.2f}   データ不足")

    # ── Phase 4: ポートフォリオ集計 ──
    print(f"\n  [Phase 4] ポートフォリオ集計")
    print("  " + "-"*105)

    print(f"\n    {'tol':>6} {'Port.Sh':>8} {'Port.MDD':>9} {'最悪MDD銘柄':>14} {'最悪MDD':>8} {'総n':>6}")
    print("    " + "-"*55)

    best_tol = None; best_score = -999
    port_summary = []

    for tol in TOL_VALUES:
        all_trades = {sym: results[tol][sym][1] for sym in sym_data}
        p_sh = portfolio_sharpe(all_trades)
        p_mdd = portfolio_mdd(all_trades)

        worst_sym = ""; worst_mdd = 0
        total_n = 0
        for sym in sym_data:
            st = results[tol][sym][0]
            if st:
                total_n += st["n"]
                if st["mdd"] > worst_mdd:
                    worst_mdd = st["mdd"]; worst_sym = sym

        cur = " ←現行" if tol == 0.30 else ""
        print(f"    {tol:.2f} {p_sh:>8.2f} {p_mdd:>8.1f}% {worst_sym:>14} {worst_mdd:>7.1f}% {total_n:>6}{cur}")
        port_summary.append({"tol": tol, "sharpe": p_sh, "mdd": p_mdd,
                             "worst_sym": worst_sym, "worst_mdd": worst_mdd, "total_n": total_n})

        # USDJPY Logic-Aに切替えた場合のポートフォリオ
        if usdjpy_a_results.get(tol) and usdjpy_a_results[tol][0]:
            alt_trades = dict(all_trades)
            alt_trades["USDJPY"] = usdjpy_a_results[tol][1]
            alt_sh = portfolio_sharpe(alt_trades)
            alt_mdd = portfolio_mdd(alt_trades)
            alt_worst = ""; alt_wmdd = 0
            for sym in sym_data:
                if sym == "USDJPY":
                    m = usdjpy_a_results[tol][0]["mdd"]
                else:
                    m = results[tol][sym][0]["mdd"] if results[tol][sym][0] else 0
                if m > alt_wmdd:
                    alt_wmdd = m; alt_worst = sym
            print(f"      +USDJPY→A {alt_sh:>5.2f} {alt_mdd:>8.1f}% {alt_worst:>14} {alt_wmdd:>7.1f}%")

    # ── Phase 5: IS/OOS過学習チェック ──
    print(f"\n  [Phase 5] IS/OOS過学習チェック（40/60分割）")
    print("  " + "-"*105)

    print(f"\n    {'tol':>6} {'IS_Sh':>7} {'OOS_Sh':>7} {'OOS/IS':>7} {'IS_MDD':>7} {'OOS_MDD':>8} {'判定':>4}")
    print("    " + "-"*55)

    for tol in TOL_VALUES:
        is_trades_all = {}; oos_trades_all = {}
        valid = True
        for sym, data in sym_data.items():
            d1m = data["d1m"]
            n_split = int(len(d1m) * 0.4)
            ts_split = d1m.index[n_split]

            d1m_is = d1m[d1m.index < ts_split]
            d1m_oos = d1m[d1m.index >= ts_split]

            st_is, tr_is = run_sym(d1m_is, data["d4h"], sym, data["logic"], tol)
            st_oos, tr_oos = run_sym(d1m_oos, data["d4h"], sym, data["logic"], tol)

            is_trades_all[sym] = tr_is
            oos_trades_all[sym] = tr_oos

        is_sh = portfolio_sharpe(is_trades_all)
        oos_sh = portfolio_sharpe(oos_trades_all)
        is_mdd = portfolio_mdd(is_trades_all)
        oos_mdd = portfolio_mdd(oos_trades_all)
        ratio = oos_sh / is_sh if is_sh > 0 else 0
        flag = "✅" if ratio >= 0.70 else "❌"
        cur = " ←現行" if tol == 0.30 else ""
        print(f"    {tol:.2f} {is_sh:>7.2f} {oos_sh:>7.2f} {ratio:>7.2f} {is_mdd:>6.1f}% {oos_mdd:>7.1f}%  {flag}{cur}")

    # USDJPY Logic-A組み合わせのIS/OOS
    print(f"\n    USDJPY Logic-A切替時:")
    print(f"    {'tol':>6} {'IS_Sh':>7} {'OOS_Sh':>7} {'OOS/IS':>7} {'判定':>4}")
    print("    " + "-"*40)

    for tol in [0.30, 0.25, 0.20, 0.15]:
        is_trades_a = {}; oos_trades_a = {}
        for sym, data in sym_data.items():
            d1m = data["d1m"]
            n_split = int(len(d1m) * 0.4)
            ts_split = d1m.index[n_split]
            d1m_is = d1m[d1m.index < ts_split]
            d1m_oos = d1m[d1m.index >= ts_split]

            logic = "A" if sym == "USDJPY" else data["logic"]
            st_is, tr_is = run_sym(d1m_is, data["d4h"], sym, logic, tol)
            st_oos, tr_oos = run_sym(d1m_oos, data["d4h"], sym, logic, tol)
            is_trades_a[sym] = tr_is
            oos_trades_a[sym] = tr_oos

        is_sh = portfolio_sharpe(is_trades_a)
        oos_sh = portfolio_sharpe(oos_trades_a)
        ratio = oos_sh / is_sh if is_sh > 0 else 0
        flag = "✅" if ratio >= 0.70 else "❌"
        print(f"    {tol:.2f} {is_sh:>7.2f} {oos_sh:>7.2f} {ratio:>7.2f}  {flag}")

    # ── Phase 6: 推奨 ──
    print(f"\n  [Phase 6] 推奨")
    print("  " + "="*105)

    base = port_summary[0]  # tol=0.30
    print(f"\n    現行（tol=0.30）: Port.Sharpe={base['sharpe']:.2f}  Port.MDD={base['mdd']:.1f}%  "
          f"最悪={base['worst_sym']} {base['worst_mdd']:.1f}%")

    for ps in port_summary[1:]:
        d_sh = ps["sharpe"] - base["sharpe"]
        d_mdd = ps["mdd"] - base["mdd"]
        d_wmdd = ps["worst_mdd"] - base["worst_mdd"]
        print(f"    tol={ps['tol']:.2f}: Port.Sharpe={ps['sharpe']:.2f}(Δ{d_sh:+.2f})  "
              f"Port.MDD={ps['mdd']:.1f}%(Δ{d_mdd:+.1f}pp)  "
              f"最悪={ps['worst_sym']} {ps['worst_mdd']:.1f}%(Δ{d_wmdd:+.1f}pp)  "
              f"n={ps['total_n']}")

    # CSV保存
    rows = []
    for tol in TOL_VALUES:
        for sym in sym_data:
            st = results[tol][sym][0]
            if st:
                rows.append({"tol_factor": tol, "sym": sym,
                             "logic": sym_data[sym]["logic"],
                             "n": st["n"], "wr": st["wr"], "pf": st["pf"],
                             "sharpe": st["sharpe"], "kelly": st["kelly"],
                             "mdd": st["mdd"], "plus_m": st["plus_m"],
                             "total_m": st["total_m"]})
    # USDJPY Logic-A行
    for tol in TOL_VALUES:
        st_a = usdjpy_a_results.get(tol, (None, None))[0]
        if st_a:
            rows.append({"tol_factor": tol, "sym": "USDJPY", "logic": "A",
                         "n": st_a["n"], "wr": st_a["wr"], "pf": st_a["pf"],
                         "sharpe": st_a["sharpe"], "kelly": st_a["kelly"],
                         "mdd": st_a["mdd"], "plus_m": st_a["plus_m"],
                         "total_m": st_a["total_m"]})

    out = os.path.join(OUT_DIR, "backtest_mdd_reduction.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n    結果保存: {out}")
    print(f"    実行時間: {time.time()-t0:.0f}秒")
    print("="*110)

if __name__ == "__main__":
    main()
