"""
backtest_staged_halfprofit.py
==============================
段階的利確（ステージド・ハーフプロフィット）バックテスト比較

【現行】1R→50%決済(SL→BE) → TP(2.5R)かBEで全決済
【提案A】1R→50%決済(SL→BE) → 2.5R→25%決済(SL→1R) → 残25%はTP2=4R or SL(1R)
【提案B】同上、TP2=5R
【提案C】同上、TP2=3.5R（控えめ延長）

比較: 7採用銘柄 × IS/OOS期間
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

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 採用7銘柄（CLAUDE.md準拠） ────────────────────────────────────
TARGETS = [
    {"sym": "GBPUSD",  "logic": "A", "tol": 0.30, "note": "Logic-A GOLD"},
    {"sym": "EURUSD",  "logic": "C", "tol": 0.30, "note": "Logic-C オーパーツ"},
    {"sym": "USDCAD",  "logic": "A", "tol": 0.30, "note": "Logic-A GOLD"},
    {"sym": "NZDUSD",  "logic": "A", "tol": 0.20, "note": "Logic-A tol=0.20"},
    {"sym": "XAUUSD",  "logic": "A", "tol": 0.20, "note": "Logic-A tol=0.20"},
    {"sym": "AUDUSD",  "logic": "B", "tol": 0.30, "note": "Logic-B ADX+Streak"},
    {"sym": "USDJPY",  "logic": "C", "tol": 0.30, "note": "Logic-C オーパーツ"},
]

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

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol=0.30):
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
                            "tp": raw + d * risk * RR_RATIO, "risk": risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])

# ══════════════════════════════════════════════════════════════════
# EXIT関数: 現行 vs 段階的利確
# ══════════════════════════════════════════════════════════════════

def _exit_current(highs, lows, ep, sl, tp, risk, d):
    """現行: 1R→50%決済(SL→BE) → TP(2.5R)かBEで全決済"""
    half = ep + d * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False, None
            if h  >= tp: return tp, "win",  False, None
            if h  >= half:
                for j in range(i+1, lim):
                    if lows[j]  <= ep: return ep, "win", True, None
                    if highs[j] >= tp: return tp, "win", True, None
                return None, None, True, None
        else:
            if h  >= sl: return sl, "loss", False, None
            if lo <= tp: return tp, "win",  False, None
            if lo <= half:
                for j in range(i+1, lim):
                    if highs[j] >= ep: return ep, "win", True, None
                    if lows[j]  <= tp: return tp, "win", True, None
                return None, None, True, None
    return None, None, False, None


def _exit_staged(highs, lows, ep, sl, tp, risk, d, tp2_rr):
    """
    段階的利確:
      Stage1: 1R→50%決済, SL→BE
      Stage2: TP(2.5R)→さらに半分(25%)決済, SL→1R
      Stage3: 残り25%はTP2(tp2_rr × R)かSL(1R)で決済

    Returns: (exit_price, result, half_done, stage2_info)
      stage2_info: None or dict with stage2/stage3 details
    """
    half = ep + d * risk * HALF_R          # 1R到達価格
    sl_be = ep                              # BE = エントリー価格
    sl_1r = ep + d * risk * HALF_R          # 1R = Stage2後のSL
    tp2   = ep + d * risk * tp2_rr          # 延長TP
    lim   = min(len(highs), MAX_LOOKAHEAD)

    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False, None
            if h  >= tp: return tp, "win", False, {"stage2_hit": True, "stage2_bar": i}
            if h  >= half:
                # Stage1完了: 50%決済済み, SL→BE
                for j in range(i+1, lim):
                    if lows[j] <= sl_be:
                        # BE決済（Stage1の利益のみ）
                        return sl_be, "win", True, None
                    if highs[j] >= tp:
                        # Stage2到達: 25%決済, 残り25%は延長
                        return tp, "win", True, {"stage2_hit": True, "stage2_bar": j}
                return None, None, True, None
        else:
            if h  >= sl: return sl, "loss", False, None
            if lo <= tp: return tp, "win", False, {"stage2_hit": True, "stage2_bar": i}
            if lo <= half:
                for j in range(i+1, lim):
                    if highs[j] >= sl_be:
                        return sl_be, "win", True, None
                    if lows[j] <= tp:
                        return tp, "win", True, {"stage2_hit": True, "stage2_bar": j}
                return None, None, True, None

    return None, None, False, None


def _exit_stage3(highs, lows, ep, risk, d, tp2_rr, start_idx):
    """
    Stage3: 残り25%のポジション
    SL = 1R, TP = tp2_rr × R
    """
    sl_1r = ep + d * risk * HALF_R   # 1Rの位置がSL
    tp2   = ep + d * risk * tp2_rr   # 延長TP
    lim   = min(len(highs), MAX_LOOKAHEAD)

    for i in range(start_idx, lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl_1r: return sl_1r, "win"   # SL at 1R (still profit)
            if h  >= tp2:   return tp2,   "win"
        else:
            if h  >= sl_1r: return sl_1r, "win"
            if lo <= tp2:   return tp2,   "win"

    return None, None


# ══════════════════════════════════════════════════════════════════
# シミュレーション
# ══════════════════════════════════════════════════════════════════

def simulate_current(signals, d1m, sym):
    """現行ロジック"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        xp, result, half_done, _ = _exit_current(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl; rem = lot * 0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total  = half_pnl + pnl
        trades.append({"result": result, "pnl": total, "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd


def simulate_staged(signals, d1m, sym, tp2_rr):
    """段階的利確ロジック"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        xp, result, half_done, s2info = _exit_staged(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"],
            sig["risk"], sig["dir"], tp2_rr)
        if result is None: continue

        total_pnl = 0.0

        if not half_done and s2info and s2info.get("stage2_hit"):
            # TP直撃（半利確なし）→ Stage2処理
            # 50%をTPで決済
            pnl_50 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot*0.50, USDJPY_RATE, sig["ep"])
            # 25%をTPで決済
            pnl_25 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot*0.25, USDJPY_RATE, sig["ep"])
            equity += pnl_50 + pnl_25
            total_pnl += pnl_50 + pnl_25

            # 残り25%: Stage3
            s3_start = sp + s2info["stage2_bar"] + 1
            if s3_start < len(m1h):
                xp3, res3 = _exit_stage3(m1h, m1l, sig["ep"], sig["risk"],
                                         sig["dir"], tp2_rr, s3_start)
                if xp3 is not None:
                    pnl_s3 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp3, lot*0.25, USDJPY_RATE, sig["ep"])
                    equity += pnl_s3
                    total_pnl += pnl_s3
                else:
                    # Stage3未決済 → 1Rで計算
                    sl_1r = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
                    pnl_s3 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], sl_1r, lot*0.25, USDJPY_RATE, sig["ep"])
                    equity += pnl_s3
                    total_pnl += pnl_s3

        elif half_done and s2info and s2info.get("stage2_hit"):
            # Stage1(1R) → Stage2(2.5R) 到達
            # 50% @ 1R
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            pnl_half = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.50, USDJPY_RATE, sig["ep"])
            equity += pnl_half
            total_pnl += pnl_half

            # 25% @ TP(2.5R)
            pnl_25 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot*0.25, USDJPY_RATE, sig["ep"])
            equity += pnl_25
            total_pnl += pnl_25

            # 残り25%: Stage3
            s3_start = sp + s2info["stage2_bar"] + 1
            if s3_start < len(m1h):
                xp3, res3 = _exit_stage3(m1h, m1l, sig["ep"], sig["risk"],
                                         sig["dir"], tp2_rr, s3_start)
                if xp3 is not None:
                    pnl_s3 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp3, lot*0.25, USDJPY_RATE, sig["ep"])
                    equity += pnl_s3
                    total_pnl += pnl_s3
                else:
                    sl_1r = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
                    pnl_s3 = rm.calc_pnl_jpy(sig["dir"], sig["ep"], sl_1r, lot*0.25, USDJPY_RATE, sig["ep"])
                    equity += pnl_s3
                    total_pnl += pnl_s3

        elif half_done:
            # Stage1(1R) → BE決済（Stage2未到達）
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            pnl_half = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.50, USDJPY_RATE, sig["ep"])
            equity += pnl_half
            total_pnl += pnl_half
            # 残り50% BE決済
            pnl_be = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot*0.50, USDJPY_RATE, sig["ep"])
            equity += pnl_be
            total_pnl += pnl_be

        else:
            # SL直撃（半利確なし）
            pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot, USDJPY_RATE, sig["ep"])
            equity += pnl
            total_pnl = pnl

        trades.append({"result": result, "pnl": total_pnl,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)

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
    total_pnl = df["pnl"].sum()

    monthly = df.groupby("month")["pnl"].sum()
    plus_m  = (monthly > 0).sum()

    eq = init; monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret); eq += monthly[m]
    mr     = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0

    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0

    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "total_pnl": total_pnl,
            "final_eq": eq, "avg_win": avg_w, "avg_loss": avg_l}


def run_comparison(d1m, d4h, sym, logic, tol):
    cfg    = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_d  = calc_atr(d1m, 10).to_dict()
    m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
              "closes": d1m["close"].values,
              "highs":  d1m["high"].values, "lows": d1m["low"].values}

    sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, tol=tol)

    # 現行
    t_cur, eq_cur, mdd_cur = simulate_current(sigs, d1m, sym)
    st_cur = calc_stats(t_cur)
    if st_cur: st_cur["mdd"] = mdd_cur

    # 段階A: TP2=3.5R
    t_a, eq_a, mdd_a = simulate_staged(sigs, d1m, sym, tp2_rr=3.5)
    st_a = calc_stats(t_a)
    if st_a: st_a["mdd"] = mdd_a

    # 段階B: TP2=4R
    t_b, eq_b, mdd_b = simulate_staged(sigs, d1m, sym, tp2_rr=4.0)
    st_b = calc_stats(t_b)
    if st_b: st_b["mdd"] = mdd_b

    # 段階C: TP2=5R
    t_c, eq_c, mdd_c = simulate_staged(sigs, d1m, sym, tp2_rr=5.0)
    st_c = calc_stats(t_c)
    if st_c: st_c["mdd"] = mdd_c

    return {"current": st_cur, "staged_3.5R": st_a, "staged_4R": st_b, "staged_5R": st_c,
            "n_signals": len(sigs)}


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*120)
    print("  段階的利確（ステージド・ハーフプロフィット）バックテスト比較")
    print("  現行: 1R→50%決済→TP(2.5R)/BE")
    print("  提案: 1R→50%決済→2.5R→25%決済(SL→1R)→残25%はTP2で決済")
    print("="*120)

    all_results = []

    for tgt in TARGETS:
        sym = tgt["sym"]
        print(f"\n  {sym} ({tgt['note']}) ...", end=" ", flush=True)

        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("データなし"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m_full)

        # OOS期間で比較
        res_oos = run_comparison(oos_d, d4h_full, sym, tgt["logic"], tgt["tol"])
        res_is  = run_comparison(is_d,  d4h_full, sym, tgt["logic"], tgt["tol"])
        print(f"完了 (IS:{res_is['n_signals']}sig / OOS:{res_oos['n_signals']}sig)")

        all_results.append({"sym": sym, "note": tgt["note"],
                           "is": res_is, "oos": res_oos})

    # ── 結果テーブル ──────────────────────────────────────────────
    variants = ["current", "staged_3.5R", "staged_4R", "staged_5R"]
    v_labels = {"current": "現行(2.5R)", "staged_3.5R": "段階3.5R",
                "staged_4R": "段階4.0R", "staged_5R": "段階5.0R"}

    print("\n" + "="*130)
    print("  ■ OOS期間 比較結果")
    print(f"  {'銘柄':8} {'バリアント':14} | {'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'Kelly':>6} {'MDD':>7} {'月+':>5} {'総PnL':>12} {'平均勝':>10} {'平均負':>10}")
    print("-"*130)

    csv_rows = []
    for r in all_results:
        sym = r["sym"]
        for v in variants:
            st = r["oos"].get(v, {})
            if not st:
                print(f"  {sym:8} {v_labels[v]:14} | {'データ不足':>5}")
                continue
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            print(f"  {sym:8} {v_labels[v]:14} | "
                  f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['sharpe']:>7.2f} "
                  f"{st['kelly']:>6.3f} {st['mdd']:>6.1f}% "
                  f"{st['plus_m']:>2}/{st['total_m']:<2} "
                  f"{st['total_pnl']:>11,.0f} "
                  f"{st.get('avg_win',0):>9,.0f} {st.get('avg_loss',0):>9,.0f}")

            # IS/OOS比較
            is_st = r["is"].get(v, {})
            is_pf = is_st.get("pf", 0) if is_st else 0

            csv_rows.append({
                "sym": sym, "variant": v, "period": "OOS",
                "n": st["n"], "wr": round(st["wr"]*100, 1),
                "pf": round(st["pf"], 2), "sharpe": round(st["sharpe"], 2),
                "kelly": round(st["kelly"], 3), "mdd": round(st["mdd"], 1),
                "plus_m": st["plus_m"], "total_m": st["total_m"],
                "total_pnl": round(st["total_pnl"]),
                "is_pf": round(is_pf, 2),
                "oos_is_ratio": round(st["pf"] / is_pf, 2) if is_pf > 0 else 0,
            })
        print()  # 銘柄間の区切り

    # ── 全銘柄平均比較 ────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 全銘柄平均比較（OOS）")
    print(f"  {'バリアント':14} | {'avg PF':>8} {'avg Sharpe':>11} {'avg MDD':>9} {'avg Kelly':>10} {'合計PnL':>12}")
    print("-"*70)

    for v in variants:
        pfs = []; shs = []; mdds = []; kls = []; pnls = []
        for r in all_results:
            st = r["oos"].get(v, {})
            if st and st.get("pf", 0) < 99:
                pfs.append(st["pf"]); shs.append(st["sharpe"])
                mdds.append(st["mdd"]); kls.append(st["kelly"])
                pnls.append(st["total_pnl"])
        if pfs:
            print(f"  {v_labels[v]:14} | {np.mean(pfs):>8.2f} {np.mean(shs):>11.2f} "
                  f"{np.mean(mdds):>8.1f}% {np.mean(kls):>10.3f} {sum(pnls):>11,.0f}")

    # ── 改善率まとめ ──────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 現行比 改善率（OOS）")
    print(f"  {'銘柄':8} {'指標':>8} | {'段階3.5R':>10} {'段階4.0R':>10} {'段階5.0R':>10}")
    print("-"*60)

    for r in all_results:
        cur = r["oos"].get("current", {})
        if not cur: continue
        for metric, label in [("pf", "PF"), ("sharpe", "Sharpe"), ("total_pnl", "総PnL")]:
            base = cur.get(metric, 0)
            if base == 0: continue
            vals = []
            for v in ["staged_3.5R", "staged_4R", "staged_5R"]:
                st = r["oos"].get(v, {})
                val = st.get(metric, 0) if st else 0
                diff = ((val - base) / abs(base) * 100)
                vals.append(f"{diff:>+8.1f}%")
            print(f"  {r['sym']:8} {label:>8} | {'  '.join(vals)}")
        print()

    # CSV保存
    out_path = os.path.join(OUT_DIR, "backtest_staged_halfprofit.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
