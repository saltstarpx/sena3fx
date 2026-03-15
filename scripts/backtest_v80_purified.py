"""
backtest_v80_purified.py
========================
予測力検証に基づくフィルター精査 + RR最適化 バックテスト

【背景】
  指標予測力検証（19,054件OOS）の結果:
    ✅ KMID (r=+0.122, 符号一貫率77%) — 唯一の安定的予測力
    ✅ 4Hボディ比率 (r=+0.052, 60%) — 弱いが安定
    ⚠️ KLOW (r=-0.018, 38%) — 有意だが不安定
    ❌ ADX — IS→OOSで符号反転
    ❌ EMA距離 — IS→OOSで符号反転
    ❌ 日足EMA — 統合で逆効果
    ❌ tol距離 — 完全に無予測力

【テストバリアント】
  ①現行     : 銘柄別Logic A/B/C（全フィルター）
  ②KMID-only : KMID のみ（全銘柄E0統一）
  ③KMID+Body : KMID + 4Hボディ比率≥0.3（全銘柄E0統一）
  ④KMID+KLOW : KMID + KLOW（全銘柄E0統一）
  ⑤KMID+KLOW+Body : KMID + KLOW + 4Hボディ比率≥0.3（全銘柄E0統一）

  × 2出口: (a) 半利+2.5R現行, (b) 半利+3.0R
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
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
BODY_RATIO_MIN  = 0.3

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    {"sym": "GBPUSD",  "logic": "A", "tol": 0.30},
    {"sym": "EURUSD",  "logic": "C", "tol": 0.30},
    {"sym": "USDCAD",  "logic": "A", "tol": 0.30},
    {"sym": "NZDUSD",  "logic": "A", "tol": 0.20},
    {"sym": "XAUUSD",  "logic": "A", "tol": 0.20},
    {"sym": "AUDUSD",  "logic": "B", "tol": 0.30},
    {"sym": "USDJPY",  "logic": "C", "tol": 0.30},
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
    return df.dropna(subset=["open", "high", "low", "close"])

def load_all(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            d1m = load_csv(p); break
    else:
        return None, None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
        if os.path.exists(p):
            return d1m, load_csv(p)
    p_is  = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
    p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
    if os.path.exists(p_is) and os.path.exists(p_oos):
        d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
        return d1m, d4h[~d4h.index.duplicated(keep="first")].sort_index()
    d4h = d1m.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return d1m, d4h

def split_is_oos(d1m):
    n = int(len(d1m) * 0.4)
    ts = d1m.index[n]
    return d1m[d1m.index < ts].copy(), d1m[d1m.index >= ts].copy(), ts

# ── インジケーター ────────────────────────────────────────────────
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

# ── エントリー ────────────────────────────────────────────────────
def pick_e0(t, sp, direction, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E0_WINDOW_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

def pick_e1(t, direction, sp, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
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
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        rng = m1c["highs"][i] - m1c["lows"][i]
        av  = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and rng > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

# ── フィルター ────────────────────────────────────────────────────
def chk_kmid(b, d):
    return (d == 1 and b["close"] > b["open"]) or (d == -1 and b["close"] < b["open"])

def chk_klow(b):
    return (min(b["open"], b["close"]) - b["low"]) / b["open"] < KLOW_THR if b["open"] > 0 else False

def chk_ema(b):
    return not pd.isna(b["atr"]) and b["atr"] > 0 and abs(b["close"] - b["ema20"]) >= b["atr"] * A1_EMA_DIST_MIN

def chk_body_ratio(b, min_ratio=BODY_RATIO_MIN):
    rng = b["high"] - b["low"]
    if rng <= 0: return False
    return abs(b["close"] - b["open"]) / rng >= min_ratio


# ══════════════════════════════════════════════════════════════════
# シグナル生成: 現行（Logic A/B/C）
# ══════════════════════════════════════════════════════════════════

def generate_signals_current(d1m, d4h_full, spread, logic, atr_d, m1c, tol=0.30, rr=2.5):
    """現行の銘柄別Logic A/B/C シグナル生成"""
    d4h, d1d = build_4h(d4h_full, need_1d=(logic == "A"))
    d1h = build_1h(d1m)
    signals = []; used = set()
    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue
        h4b = d4h[d4h.index < hct]
        if len(h4b) < max(2, STREAK_MIN): continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]
        # Logic A: 日足EMA
        if logic == "A":
            if d1d is None: continue
            d1b = d1d[d1d.index.normalize() < hct.normalize()]
            if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        # Logic B: ADX + Streak
        elif logic == "B":
            if h4l.get("adx", 0) < ADX_MIN: continue
            if not all(t == trend for t in h4b["trend"].iloc[-STREAK_MIN:].values): continue
        # KMID + KLOW（全Logic共通）
        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        # EMA距離（Logic A/B のみ）
        if logic != "C" and not chk_ema(h4l): continue
        # AUDUSD 4Hボディ比率（Logic B のみ）
        if logic == "B" and not chk_body_ratio(h4l): continue
        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue
        # 確認足（Logic C のみ）
        if logic == "C":
            if d == 1 and p1["close"] <= p1["open"]: continue
            if d == -1 and p1["close"] >= p1["open"]: continue
        # エントリー方式
        if logic == "A":   et, ep = pick_e2(hct, d, spread, atr_d, m1c)
        elif logic == "C": et, ep = pick_e0(hct, spread, d, m1c)
        else:              et, ep = pick_e1(hct, d, spread, m1c)
        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])


# ══════════════════════════════════════════════════════════════════
# シグナル生成: v80（フィルター可変）
# ══════════════════════════════════════════════════════════════════

def generate_signals_v80(d1m, d4h_full, spread, m1c, tol=0.30, rr=2.5,
                          use_kmid=True, use_klow=False,
                          use_body_ratio=False):
    """
    v80シグナル生成:
    - 4H EMA20でトレンド方向決定（基盤）
    - 1H二番底/天井パターン検出（基盤）
    - フィルターはパラメータで選択
    - エントリーはE0統一（即座）
    """
    d4h, _ = build_4h(d4h_full, need_1d=False)
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue
        h4b = d4h[d4h.index < hct]
        if len(h4b) < 2: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        # ── 選択フィルター ──
        if use_kmid and not chk_kmid(h4l, trend): continue
        if use_klow and not chk_klow(h4l): continue
        if use_body_ratio and not chk_body_ratio(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

        # E0エントリー統一
        et, ep = pick_e0(hct, spread, d, m1c)
        if et is None or et in used: continue

        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])


# ══════════════════════════════════════════════════════════════════
# EXIT + シミュレーション
# ══════════════════════════════════════════════════════════════════

def _exit_with_half(highs, lows, ep, sl, tp, risk, d):
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

def simulate_half(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
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
    if len(trades) < 10: return {}
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


# ══════════════════════════════════════════════════════════════════
# バリアント定義
# ══════════════════════════════════════════════════════════════════

# (label, type, kwargs)
# type: "current" = 銘柄別Logic, "v80" = 統一ロジック
FILTER_VARIANTS = [
    ("①現行",          "current", {}),
    ("②KMID-only",     "v80", {"use_kmid": True, "use_klow": False, "use_body_ratio": False}),
    ("③KMID+Body",     "v80", {"use_kmid": True, "use_klow": False, "use_body_ratio": True}),
    ("④KMID+KLOW",     "v80", {"use_kmid": True, "use_klow": True,  "use_body_ratio": False}),
    ("⑤KMID+KLOW+Body","v80", {"use_kmid": True, "use_klow": True,  "use_body_ratio": True}),
]

RR_VARIANTS = [2.5, 3.0]


# ══════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*160)
    print("  v80バックテスト: 予測力検証に基づくフィルター精査 + RR最適化")
    print("  フィルター5種 × RR2種 = 10バリアント × 7銘柄 × IS/OOS")
    print("="*160)

    all_results = []  # [{sym, variant_label, period, stats}, ...]

    for tgt in TARGETS:
        sym = tgt["sym"]
        print(f"\n  ◆ {sym} ...", end=" ", flush=True)

        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("データなし"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m_full)
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]

        for period_label, d1m_period in [("IS", is_d), ("OOS", oos_d)]:
            atr_d = calc_atr(d1m_period, 10).to_dict()
            m1c   = {"idx": d1m_period.index, "opens": d1m_period["open"].values,
                      "closes": d1m_period["close"].values,
                      "highs":  d1m_period["high"].values, "lows": d1m_period["low"].values}

            for fv_label, fv_type, fv_kwargs in FILTER_VARIANTS:
                for rr in RR_VARIANTS:
                    vl = f"{fv_label} {rr}R"

                    if fv_type == "current":
                        sigs = generate_signals_current(
                            d1m_period, d4h_full, spread, tgt["logic"],
                            atr_d, m1c, tol=tgt["tol"], rr=rr)
                    else:
                        sigs = generate_signals_v80(
                            d1m_period, d4h_full, spread, m1c,
                            tol=tgt["tol"], rr=rr, **fv_kwargs)

                    trades, eq, mdd = simulate_half(sigs, d1m_period, sym)
                    st = calc_stats(trades)
                    if st: st["mdd"] = mdd

                    all_results.append({
                        "sym": sym, "variant": vl, "period": period_label,
                        "filter": fv_label, "rr": rr,
                        "stats": st,
                    })

        print("完了")

    # ══════════════════════════════════════════════════════════════
    # 結果テーブル
    # ══════════════════════════════════════════════════════════════

    # ── OOS銘柄別テーブル ──
    print("\n" + "="*160)
    print("  ■ OOS期間 銘柄別結果")
    print(f"  {'銘柄':8} {'バリアント':22} | {'n':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'Kelly':>6} {'MDD':>7} {'月+':>5} {'総PnL':>12} {'IS PF':>7} {'OOS/IS':>7}")
    print("  " + "-"*155)

    csv_rows = []
    syms_seen = set()

    for tgt in TARGETS:
        sym = tgt["sym"]
        if sym in syms_seen: continue
        syms_seen.add(sym)

        for fv_label, _, _ in FILTER_VARIANTS:
            for rr in RR_VARIANTS:
                vl = f"{fv_label} {rr}R"
                oos_r = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "OOS"]
                is_r  = [r for r in all_results if r["sym"] == sym and r["variant"] == vl and r["period"] == "IS"]
                st = oos_r[0]["stats"] if oos_r and oos_r[0]["stats"] else {}
                is_st = is_r[0]["stats"] if is_r and is_r[0]["stats"] else {}
                if not st: continue

                is_pf = is_st.get("pf", 0) if is_st else 0
                oos_is = st["pf"] / is_pf if is_pf > 0 and st["pf"] < 99 else 0
                pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
                marker = " ★" if fv_label == "①現行" and rr == 2.5 else ""

                print(f"  {sym:8} {vl:22} | "
                      f"{st['n']:>4} {st['wr']*100:>5.1f}% {pf_s:>6} {st['sharpe']:>7.2f} "
                      f"{st['kelly']:>6.3f} {st['mdd']:>6.1f}% "
                      f"{st['plus_m']:>2}/{st['total_m']:<2} "
                      f"{st['total_pnl']:>11,.0f} {is_pf:>7.2f} {oos_is:>7.2f}{marker}")

                csv_rows.append({
                    "sym": sym, "variant": vl, "filter": fv_label, "rr": rr,
                    "n": st["n"], "wr": round(st["wr"]*100, 1),
                    "pf": round(st["pf"], 2), "sharpe": round(st["sharpe"], 2),
                    "kelly": round(st["kelly"], 3), "mdd": round(st["mdd"], 1),
                    "plus_m": st["plus_m"], "total_m": st["total_m"],
                    "total_pnl": round(st["total_pnl"]),
                    "is_pf": round(is_pf, 2), "oos_is_ratio": round(oos_is, 2),
                })
        print()

    # ── 全銘柄平均 ────────────────────────────────────────────────
    print("\n" + "="*130)
    print("  ■ 全銘柄平均比較（OOS）")
    print(f"  {'バリアント':22} | {'avg PF':>8} {'avg Sharpe':>11} {'avg MDD':>9} "
          f"{'avg Kelly':>10} {'合計PnL':>14} {'avg WR':>8} {'avg n':>6} {'OOS/IS':>7}")
    print("  " + "-"*110)

    for fv_label, _, _ in FILTER_VARIANTS:
        for rr in RR_VARIANTS:
            vl = f"{fv_label} {rr}R"
            pfs = []; shs = []; mdds = []; kls = []; pnls = []; wrs = []; ns = []; ois = []
            for r in all_results:
                if r["variant"] == vl and r["period"] == "OOS" and r["stats"]:
                    st = r["stats"]
                    if st.get("pf", 0) < 99:
                        pfs.append(st["pf"]); shs.append(st["sharpe"])
                        mdds.append(st["mdd"]); kls.append(st["kelly"])
                        pnls.append(st["total_pnl"]); wrs.append(st["wr"]*100)
                        ns.append(st["n"])
                        # IS PF
                        is_r = [x for x in all_results if x["sym"] == r["sym"] and x["variant"] == vl and x["period"] == "IS" and x["stats"]]
                        if is_r and is_r[0]["stats"].get("pf", 0) > 0:
                            ois.append(st["pf"] / is_r[0]["stats"]["pf"])
            if pfs:
                marker = " ★" if fv_label == "①現行" and rr == 2.5 else ""
                avg_ois = np.mean(ois) if ois else 0
                print(f"  {vl:22} | {np.mean(pfs):>8.2f} {np.mean(shs):>11.2f} "
                      f"{np.mean(mdds):>8.1f}% {np.mean(kls):>10.3f} "
                      f"{sum(pnls):>13,.0f} {np.mean(wrs):>7.1f}% {np.mean(ns):>5.0f} {avg_ois:>7.2f}{marker}")

    # ── 現行2.5R比 変化率 ─────────────────────────────────────────
    print("\n" + "="*130)
    print("  ■ 現行①2.5R比 変化率（OOS）")

    baseline_vl = "①現行 2.5R"
    compare_vls = [f"{fv} {rr}R" for fv, _, _ in FILTER_VARIANTS for rr in RR_VARIANTS
                   if not (fv == "①現行" and rr == 2.5)]

    for metric, label in [("pf", "PF"), ("total_pnl", "総PnL"), ("mdd", "MDD")]:
        print(f"\n  {'':8} [{label}]")
        header = "  {:8} |".format("銘柄")
        for cv in compare_vls:
            header += f" {cv:>14}"
        print(header)
        print("  " + "-"*(10 + 15*len(compare_vls)))

        for tgt in TARGETS:
            sym = tgt["sym"]
            base_r = [r for r in all_results if r["sym"] == sym and r["variant"] == baseline_vl and r["period"] == "OOS"]
            if not base_r or not base_r[0]["stats"]: continue
            base = base_r[0]["stats"].get(metric, 0)
            if base == 0: continue

            parts = f"  {sym:8} |"
            for cv in compare_vls:
                cr = [r for r in all_results if r["sym"] == sym and r["variant"] == cv and r["period"] == "OOS"]
                if cr and cr[0]["stats"]:
                    val = cr[0]["stats"].get(metric, 0)
                    diff = ((val - base) / abs(base) * 100)
                    parts += f" {diff:>+13.1f}%"
                else:
                    parts += f" {'N/A':>14}"
            print(parts)

    # ── 判定サマリー ──────────────────────────────────────────────
    print("\n" + "="*130)
    print("  ■ 判定サマリー（全銘柄平均 OOS）")
    print("="*130)

    base_avg = {}
    for r in all_results:
        if r["variant"] == baseline_vl and r["period"] == "OOS" and r["stats"]:
            for k in ["pf", "sharpe", "mdd", "kelly", "total_pnl", "n"]:
                base_avg.setdefault(k, []).append(r["stats"].get(k, 0))

    if base_avg:
        base_pf  = np.mean(base_avg["pf"])
        base_mdd = np.mean(base_avg["mdd"])
        base_sh  = np.mean(base_avg["sharpe"])
        base_pnl = sum(base_avg["total_pnl"])

        print(f"\n  現行ベースライン: avg PF={base_pf:.2f}, avg MDD={base_mdd:.1f}%, "
              f"avg Sharpe={base_sh:.2f}, 合計PnL={base_pnl:,.0f}")

        for fv_label, _, _ in FILTER_VARIANTS:
            for rr in RR_VARIANTS:
                vl = f"{fv_label} {rr}R"
                if vl == baseline_vl: continue
                vl_pfs = []; vl_mdds = []; vl_shs = []; vl_pnls = []; vl_ois = []
                for r in all_results:
                    if r["variant"] == vl and r["period"] == "OOS" and r["stats"]:
                        st = r["stats"]
                        if st.get("pf", 0) < 99:
                            vl_pfs.append(st["pf"]); vl_mdds.append(st["mdd"])
                            vl_shs.append(st["sharpe"]); vl_pnls.append(st["total_pnl"])
                            is_r = [x for x in all_results if x["sym"] == r["sym"] and x["variant"] == vl and x["period"] == "IS" and x["stats"]]
                            if is_r and is_r[0]["stats"].get("pf", 0) > 0:
                                vl_ois.append(st["pf"] / is_r[0]["stats"]["pf"])
                if not vl_pfs: continue

                avg_pf  = np.mean(vl_pfs)
                avg_mdd = np.mean(vl_mdds)
                avg_sh  = np.mean(vl_shs)
                tot_pnl = sum(vl_pnls)
                avg_ois = np.mean(vl_ois) if vl_ois else 0

                pf_chg  = (avg_pf - base_pf) / base_pf * 100
                mdd_chg = (avg_mdd - base_mdd) / base_mdd * 100
                pnl_chg = (tot_pnl - base_pnl) / abs(base_pnl) * 100

                # 判定
                verdicts = []
                if avg_pf >= base_pf * 0.95: verdicts.append("PF✅")
                else: verdicts.append("PF❌")
                if avg_mdd <= base_mdd * 1.10: verdicts.append("MDD✅")
                else: verdicts.append("MDD❌")
                if avg_ois >= 0.70: verdicts.append("IS/OOS✅")
                else: verdicts.append("IS/OOS⚠️")

                print(f"\n  {vl:22} → PF {avg_pf:.2f}({pf_chg:+.1f}%) "
                      f"MDD {avg_mdd:.1f}%({mdd_chg:+.1f}%) "
                      f"PnL {tot_pnl:,.0f}({pnl_chg:+.1f}%) "
                      f"OOS/IS={avg_ois:.2f} "
                      f"{'  '.join(verdicts)}")

    # CSV保存
    out_path = os.path.join(OUT_DIR, "backtest_v80_purified.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"\n\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
