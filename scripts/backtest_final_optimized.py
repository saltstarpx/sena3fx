"""
backtest_final_optimized.py
============================
最終確定バックテスト — 銘柄別ベストロジック × 最適リスクモード

【改善内容（前回分析より）】
  固定2%   : EURUSD/GBPUSD/AUDUSD/NZDUSD/USDCHF/XAGUSD/XAUUSD/USDJPY/USDCAD
              → Sharpeが固定2%で最高（FX系は低リスクが安定）
  適応2-3% : EURGBP / AUDJPY のみ
              → この2銘柄だけアダプティブでSharpe改善

【銘柄×ロジック×リスクモード確定組み合わせ】
  XAUUSD  Logic-A GOLDYAGAMI      固定2%   (PF=3.10 Sh=2.89→固定2%採用)
  USDJPY  Logic-C オーパーツ       固定2%   (PF=2.20 Sh=6.18)
  GBPUSD  Logic-A GOLDYAGAMI      固定2%   (PF=1.86 Sh=7.12 ← Sharpe1位)
  EURUSD  Logic-C オーパーツ       固定2%   (PF=1.81 Sh=6.18)
  USDCAD  Logic-A GOLDYAGAMI      固定2%   (PF=2.02 Sh=5.62)
  NZDUSD  Logic-A GOLDYAGAMI      固定2%   (PF=1.98 Sh=5.45)
  USDCHF  Logic-A GOLDYAGAMI      固定2%   (PF=1.78 Sh=5.26)
  AUDUSD  Logic-B ADX+Streak      固定2%   (PF=2.03 Sh=3.66)
  EURGBP  Logic-A GOLDYAGAMI      適応2-3% (PF=1.60 Sh=4.28)
  AUDJPY  Logic-A GOLDYAGAMI      適応2-3% (PF=1.73 Sh=3.53)
  XAGUSD  Logic-B ADX+Streak      固定2%   (PF=1.74 Sh=2.10)

【評価基準（シャープ導入版）】
  採用  : PF≥1.8 かつ Sharpe≥2.5 かつ MDD≤40% かつ 月次プラス≥75%
  要検討: PF≥1.5 かつ Sharpe≥1.5
  不採用: それ以下

【期間】各銘柄の最大利用可能期間 / IS:40% OOS:60%
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
RR_RATIO      = 2.5
RR_RATIO_V80  = 3.0
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E0_WINDOW_MIN   = 2
ADX_MIN         = 20
STREAK_MIN      = 4
BODY_RATIO_MIN  = 0.3

RISK_MIN  = 0.02
RISK_MAX  = 0.03
RISK_STEP = 0.005

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 銘柄設定（ベストロジック × 最適リスクモード確定） ────────────
# 現行ベスト + v80統一ロジック（全銘柄で両方テストして勝つ方を採用）
TARGETS_CURRENT = [
    {"sym": "XAUUSD",  "logic": "A", "risk": "fixed2",   "cat": "METALS",   "tol": 0.20, "note": "現行Logic-A"},
    {"sym": "GBPUSD",  "logic": "A", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.30, "note": "現行Logic-A"},
    {"sym": "USDJPY",  "logic": "C", "risk": "fixed2",   "cat": "FX_JPY",   "tol": 0.30, "note": "現行Logic-C"},
    {"sym": "EURUSD",  "logic": "C", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.30, "note": "現行Logic-C"},
    {"sym": "USDCAD",  "logic": "A", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.30, "note": "現行Logic-A"},
    {"sym": "NZDUSD",  "logic": "A", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.20, "note": "現行Logic-A"},
    {"sym": "USDCHF",  "logic": "A", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.30, "note": "現行Logic-A"},
    {"sym": "AUDUSD",  "logic": "B", "risk": "fixed2",   "cat": "FX_USD",   "tol": 0.30, "note": "現行Logic-B"},
]

TARGETS_V80 = [
    {"sym": "XAUUSD",  "logic": "V80", "risk": "fixed2",  "cat": "METALS", "tol": 0.20, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "GBPUSD",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "USDJPY",  "logic": "V80", "risk": "fixed2",  "cat": "FX_JPY", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "EURUSD",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "USDCAD",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "NZDUSD",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.20, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "USDCHF",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
    {"sym": "AUDUSD",  "logic": "V80", "risk": "fixed2",  "cat": "FX_USD", "tol": 0.30, "note": "v80 KMID+KLOW+Body 3.0R"},
]

# 現行3.0Rテスト（既存ロジックにRR=3.0だけ変更）
TARGETS_CURRENT_3R = [
    {"sym": s["sym"], "logic": s["logic"], "risk": s["risk"], "cat": s["cat"],
     "tol": s["tol"], "note": f"現行{s['logic']} 3.0R"}
    for s in TARGETS_CURRENT
]

TARGETS = TARGETS_CURRENT + TARGETS_V80 + TARGETS_CURRENT_3R

LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ", "V80": "v80統一"}
RISK_NAMES  = {"fixed2": "固定2%", "fixed3": "固定3%", "adaptive": "適応2-3%"}

# ── データロード ──────────────────────────────────────────────────
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
def chk_kmid(b, d): return (d == 1 and b["close"] > b["open"]) or (d == -1 and b["close"] < b["open"])
def chk_klow(b): return (min(b["open"], b["close"]) - b["low"]) / b["open"] < KLOW_THR if b["open"] > 0 else False
def chk_ema(b): return not pd.isna(b["atr"]) and b["atr"] > 0 and abs(b["close"] - b["ema20"]) >= b["atr"] * A1_EMA_DIST_MIN
def chk_body_ratio(b, min_ratio=BODY_RATIO_MIN):
    rng = b["high"] - b["low"]
    if rng <= 0: return False
    return abs(b["close"] - b["open"]) / rng >= min_ratio

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, rr=RR_RATIO, tol=A3_DEFAULT_TOL):
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
        # Logic B: 4Hボディ比率チェック
        if logic == "B" and not chk_body_ratio(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

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
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])


def generate_signals_v80(d1m, d4h_full, spread, m1c, rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL):
    """v80統一ロジック: KMID + KLOW + 4Hボディ比率のみ、E0エントリー、不安定フィルター全除去"""
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

        # v80フィルター: KMID + KLOW + 4Hボディ比率のみ
        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if not chk_body_ratio(h4l): continue

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

# ── シミュレーション ──────────────────────────────────────────────
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

def simulate(signals, d1m, sym, risk_mode="fixed2"):
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=RISK_MIN)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    cur_risk = RISK_MIN

    for sig in signals:
        if risk_mode == "fixed2":   r = 0.02
        elif risk_mode == "fixed3": r = 0.03
        else:                       r = cur_risk
        rm.risk_pct = r
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
            equity  += half_pnl; rem = lot * 0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total  = half_pnl + pnl
        trades.append({"result": result, "pnl": total,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

        if risk_mode == "adaptive":
            cur_risk = min(cur_risk + RISK_STEP, RISK_MAX) if result == "win" \
                       else max(cur_risk - RISK_STEP, RISK_MIN)

    return trades, equity, mdd

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 10: return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loss.sum())
    pf   = gw / gl if gl > 0 else float("inf")

    monthly = df.groupby("month")["pnl"].sum()
    plus_m  = (monthly > 0).sum()

    # シャープ（月次リターン率ベース）
    eq = init
    monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret)
        eq += monthly[m]
    mr     = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0

    # ケリー
    avg_w  = wins.mean() if len(wins) > 0 else 0
    avg_l  = abs(loss.mean()) if len(loss) > 0 else 1
    kelly  = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0

    p_val  = stats.binomtest(len(wins), n, 0.5, alternative="greater").pvalue

    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "p_val": p_val,
            "final_eq": eq}

def run(d1m, d4h, sym, logic, risk_mode, rr=RR_RATIO, tol=A3_DEFAULT_TOL):
    cfg    = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_d  = calc_atr(d1m, 10).to_dict()
    m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
              "closes": d1m["close"].values,
              "highs":  d1m["high"].values, "lows": d1m["low"].values}
    if logic == "V80":
        sigs = generate_signals_v80(d1m, d4h, spread, m1c, rr=rr, tol=tol)
    else:
        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, rr=rr, tol=tol)
    trades, final_eq, mdd = simulate(sigs, d1m, sym, risk_mode)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["final_eq"] = final_eq
    return st

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*110)
    print("  最終確定バックテスト — 銘柄別ベストロジック × 最適リスクモード")
    print("  【改善適用】固定2%優先 / EURGBP・AUDJPYのみ適応2-3%")
    print("="*110)

    results = []

    # データキャッシュ（同じ銘柄を3回ロードしない）
    data_cache = {}

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        rmode = tgt["risk"]
        lname = LOGIC_NAMES[logic]
        rname = RISK_NAMES[rmode]
        tol   = tgt.get("tol", A3_DEFAULT_TOL)
        # 現行3.0Rテストか判定
        is_3r_test = tgt in TARGETS_CURRENT_3R
        rr = RR_RATIO_V80 if (logic == "V80" or is_3r_test) else RR_RATIO
        rr_label = f" {rr}R" if rr != RR_RATIO else " 2.5R"

        print(f"\n  [{tgt['cat']}] {sym}  Logic-{logic}:{lname}{rr_label}  {rname}", end=" ... ", flush=True)

        if sym not in data_cache:
            d1m_full, d4h_full = load_all(sym)
            if d1m_full is None:
                print("データ未発見"); continue
            is_d, oos_d, split_ts = split_is_oos(d1m_full)
            data_cache[sym] = (d1m_full, d4h_full, is_d, oos_d, split_ts)

        d1m_full, d4h_full, is_d, oos_d, split_ts = data_cache[sym]
        start = d1m_full.index[0].strftime("%Y-%m-%d")
        end   = d1m_full.index[-1].strftime("%Y-%m-%d")

        is_st   = run(is_d,      d4h_full, sym, logic, rmode, rr=rr, tol=tol)
        oos_st  = run(oos_d,     d4h_full, sym, logic, rmode, rr=rr, tol=tol)
        full_st = run(d1m_full,  d4h_full, sym, logic, rmode, rr=rr, tol=tol)
        print("完了")

        results.append({
            "tgt": tgt, "sym": sym, "logic": logic, "lname": lname,
            "rmode": rmode, "rname": rname, "rr": rr, "tol": tol,
            "start": start, "end": end, "split": split_ts.strftime("%Y-%m-%d"),
            "is": is_st, "oos": oos_st, "full": full_st,
        })

    # ── OOS結果 ──────────────────────────────────────────────────
    print("\n" + "="*140)
    print("  ■ OOS期間 全結果（現行2.5R vs 現行3.0R vs v80 3.0R）")
    print(f"  {'#':>2} {'銘柄':8} {'ロジック':14} {'RR':>4} {'リスク':10} | "
          f"{'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Kelly':>6} {'MDD':>7} {'月+':>5} | "
          f"{'IS PF':>6} {'OOS/IS':>8} | {'判定':>8}")
    print("-"*140)

    def of_flag(is_pf, oos_pf):
        if not is_pf or is_pf == float("inf"): return "N/A  "
        r = oos_pf / is_pf
        return f"✅{r:.2f}" if r >= 0.70 else f"❌{r:.2f}"

    def final_verdict(pf, sh, mdd, pm, tm, of, n):
        if n < 30: return "❌データ不足"
        if "❌" in of: return "❌過学習"
        pm_r = pm / tm if tm > 0 else 0
        if pf >= 1.8 and sh >= 2.5 and mdd <= 40 and pm_r >= 0.75: return "✅採用"
        if pf >= 1.5 and sh >= 1.5: return "⚠️要検討"
        return "❌不採用"

    rows = []
    for rank, r in enumerate(
        sorted(results, key=lambda x: (x["sym"], x.get("rr", 2.5), x["logic"])), 1
    ):
        oos = r["oos"]; is_ = r["is"]
        if not oos: continue
        n   = oos.get("n", 0);     wr  = oos.get("wr", 0)
        pf  = oos.get("pf", 0);   sh  = oos.get("sharpe", 0)
        kl  = oos.get("kelly", 0); mdd = oos.get("mdd", 0)
        pm  = oos.get("plus_m", 0); tm = oos.get("total_m", 0)
        ipf = is_.get("pf", 0)    if is_ else 0
        of  = of_flag(ipf, pf)
        vd  = final_verdict(pf, sh, mdd, pm, tm, of, n)
        pf_s = f"{pf:.2f}" if pf < 99 else "∞"
        rr_s = f"{r.get('rr', 2.5):.1f}"
        print(f"  #{rank:<2} {r['sym']:8} Logic-{r['logic']}:{r['lname']:10} {rr_s:>4} {r['rname']:10} | "
              f"{n:>5} {wr*100:>5.1f}% {pf_s:>6} {sh:>7.2f} {kl:>6.3f} {mdd:>6.1f}% {pm:>2}/{tm:<2} | "
              f"{ipf:>6.2f} {of:>8} | {vd:>8}")
        rows.append({
            "rank": rank, "sym": r["sym"], "cat": r["tgt"]["cat"],
            "logic": r["logic"], "lname": r["lname"],
            "rr": r.get("rr", 2.5), "tol": r.get("tol", 0.30),
            "risk_mode": r["rmode"], "risk_name": r["rname"],
            "period": f"{r['start']}~{r['end']}", "split": r["split"],
            "oos_n": n, "oos_wr": wr, "oos_pf": pf,
            "oos_sharpe": sh, "oos_kelly": kl, "oos_mdd": mdd,
            "oos_plus_m": pm, "oos_total_m": tm,
            "is_pf": ipf, "oos_is_ratio": pf/ipf if ipf > 0 else 0,
            "overfitting": of, "verdict": vd,
            "full_pf": r["full"].get("pf", 0),
            "full_sharpe": r["full"].get("sharpe", 0),
            "full_mdd": r["full"].get("mdd", 0),
            "note": r["tgt"].get("note", ""),
        })

    # ── FULL期間 ─────────────────────────────────────────────────
    print("\n" + "="*115)
    print("  ■ 全期間（FULL）結果（参考）")
    print(f"  {'銘柄':8} {'ロジック':14} {'リスク':10} {'期間':22} | "
          f"{'PF':>6} {'Sharpe':>7} {'MDD':>7} {'月+':>7}")
    print("-"*100)
    for r in sorted(results, key=lambda x: -x["full"].get("sharpe", 0)):
        full = r["full"]
        if not full: continue
        pf_s = f"{full.get('pf',0):.2f}" if full.get('pf',0) < 99 else "∞"
        pm   = full.get("plus_m", 0); tm = full.get("total_m", 0)
        print(f"  {r['sym']:8} Logic-{r['logic']}:{r['lname']:10} {r['rname']:10} "
              f"{r['start']+'〜'+r['end']:22} | "
              f"{pf_s:>6} {full.get('sharpe',0):>7.2f} {full.get('mdd',0):>6.1f}% {pm:>2}/{tm:<2}")

    # ── カテゴリ集計 ─────────────────────────────────────────────
    print("\n" + "="*115)
    print("  ■ カテゴリ別 OOS集計")
    df_res = pd.DataFrame(rows)
    for cat in ["FX_USD", "FX_JPY", "FX_CROSS", "METALS"]:
        sub = df_res[df_res["cat"] == cat]
        if sub.empty: continue
        avg_pf = sub["oos_pf"].mean()
        avg_sh = sub["oos_sharpe"].mean()
        adopted = (sub["verdict"] == "✅採用").sum()
        syms_str = " / ".join(sub["sym"].tolist())
        print(f"  {cat:10}: avg PF={avg_pf:.2f}  avg Sharpe={avg_sh:.2f}  "
              f"採用{adopted}/{len(sub)}銘柄  [{syms_str}]")

    # ── 採用確定まとめ ────────────────────────────────────────────
    print("\n" + "="*115)
    print("  ■ 採用確定まとめ（✅採用のみ / Sharpe降順）")
    print(f"  {'銘柄':8} {'武器':24} {'OOS PF':>8} {'Sharpe':>8} {'Kelly':>7} "
          f"{'MDD':>7} {'月次':>6} {'FULL Sh':>8}")
    print("-"*85)
    adopted = [r for r in rows if r["verdict"] == "✅採用"]
    for r in sorted(adopted, key=lambda x: -x["oos_sharpe"]):
        lstr = f"Logic-{r['logic']}:{r['lname']}"
        pf_s = f"{r['oos_pf']:.2f}" if r['oos_pf'] < 99 else "∞"
        print(f"  {r['sym']:8} {lstr:24} {pf_s:>8} {r['oos_sharpe']:>8.2f} "
              f"{r['oos_kelly']:>7.3f} {r['oos_mdd']:>6.1f}% "
              f"{r['oos_plus_m']:>2}/{r['oos_total_m']:<2} "
              f"{r['full_sharpe']:>8.2f}")

    # ── 銘柄別 最強ロジック判定 ────────────────────────────────────
    print("\n" + "="*140)
    print("  ■ 銘柄別 最強ロジック判定（OOS PF × OOS/IS比 総合）")
    print(f"  {'銘柄':8} | {'現行2.5R':>12} {'現行3.0R':>12} {'v80 3.0R':>12} | {'推奨':20} {'理由'}")
    print("  " + "-"*130)

    syms_done = set()
    for r in rows:
        sym = r["sym"]
        if sym in syms_done: continue
        syms_done.add(sym)

        # 同じ銘柄の3バリアントを取得
        cur_25 = [x for x in rows if x["sym"] == sym and x["logic"] != "V80" and x.get("rr", 2.5) == 2.5]
        cur_30 = [x for x in rows if x["sym"] == sym and x["logic"] != "V80" and x.get("rr", 2.5) == 3.0]
        v80_30 = [x for x in rows if x["sym"] == sym and x["logic"] == "V80"]

        def fmt_pf(lst):
            if not lst: return "   N/A   "
            x = lst[0]
            pf = x["oos_pf"]; ois = x.get("oos_is_ratio", 0)
            of_ok = "❌" not in x.get("overfitting", "")
            return f"{pf:>5.2f}{'✅' if of_ok else '❌'}{ois:.2f}"

        # 最強判定: OOS/IS≥0.70 かつ PF最高
        candidates = []
        for label, lst in [("現行2.5R", cur_25), ("現行3.0R", cur_30), ("v80 3.0R", v80_30)]:
            if not lst: continue
            x = lst[0]
            of_ok = "❌" not in x.get("overfitting", "")
            if of_ok:
                candidates.append((label, x["oos_pf"], x.get("oos_sharpe", 0), x.get("oos_mdd", 99), x))

        if candidates:
            # PFが最も高いものを推奨
            best = max(candidates, key=lambda c: c[1])
            reason = f"PF={best[1]:.2f} Sharpe={best[2]:.2f} MDD={best[3]:.1f}%"
            rec = f"★ {best[0]}"
        else:
            rec = "判定不可"; reason = "全バリアント過学習"

        print(f"  {sym:8} | {fmt_pf(cur_25):>12} {fmt_pf(cur_30):>12} {fmt_pf(v80_30):>12} | {rec:20} {reason}")

    # ── CSV保存 ───────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "backtest_final_optimized.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  結果保存: {out}")

if __name__ == "__main__":
    main()
