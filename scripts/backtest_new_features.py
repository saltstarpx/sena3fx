"""
backtest_new_features.py
========================
全期間バックテスト + 新規特徴量スクリーニング

【高速化設計】
  1. 全シグナル候補を1回だけ生成し、各特徴量の判定結果をメタデータに付与
  2. 組み合わせテストはメタデータのフィルタリングのみ（再スキャン不要）

【新規特徴量（理論的根拠あり）】
  F1: KUPP       — 4H上ヒゲ比率（ロング時の反転リスク排除）
  F2: KBODY      — 4H実体/レンジ比率（優柔不断キャンドル排除）
  F3: EMA_SLOPE  — 4H EMA20の傾き（ATR正規化、トレンド加速確認）
  F4: VOL_SURGE  — 直近ボリューム/20期間平均（出来高確認）
  F5: RSI_ZONE   — 4H RSI が極端ゾーン外（逆行リスク排除）
  F6: ATR_REGIME — 短期ATR/長期ATR（ボラ拡大期のみ）
  F7: PATTERN_Q  — 二番底/天井の近さ（パターン品質）
  F8: WEEKLY_ALIGN — 週足EMA20方向一致（上位TFアライメント）
  F9: EMA_CROSS  — 4H EMA20 > EMA50（トレンド構造確認）
  F10: H1_MOM    — 直近N本の1H足が方向一致（1Hモメンタム）
"""
import os, sys, warnings, itertools, time
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

TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "cat": "FX_JPY"},
    {"sym": "GBPUSD",  "logic": "A", "cat": "FX_USD"},
    {"sym": "EURUSD",  "logic": "C", "cat": "FX_USD"},
    {"sym": "USDCAD",  "logic": "A", "cat": "FX_USD"},
    {"sym": "NZDUSD",  "logic": "A", "cat": "FX_USD"},
    {"sym": "XAUUSD",  "logic": "A", "cat": "METALS"},
    {"sym": "AUDUSD",  "logic": "B", "cat": "FX_USD"},
]
LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

# ── 新規特徴量の閾値 ──────────────────────────────────────────────
FEAT_THR = {
    "F1_kupp": 0.0015, "F2_kbody": 0.40, "F3_slope": 0.05,
    "F4_vol": 1.2, "F5_rsi_lo": 30, "F5_rsi_hi": 70,
    "F6_atr": 0.9, "F7_patq": 0.15, "F10_mom": 3,
}
FEATURE_LIST = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
FEATURE_NAMES = {
    "F1": "KUPP(上ヒゲ除去)", "F2": "KBODY(実体比率)", "F3": "EMA_SLOPE(傾き)",
    "F4": "VOL_SURGE(出来高)", "F5": "RSI_ZONE(極端回避)", "F6": "ATR_REGIME(ボラ拡大)",
    "F7": "PATTERN_Q(パターン品質)", "F8": "WEEKLY_ALIGN(週足)",
    "F9": "EMA_CROSS(EMA20/50)", "F10": "H1_MOM(1Hモメンタム)",
}

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
    d1m = None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            d1m = load_csv(p); break
    if d1m is None: return None, None

    p_is  = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
    p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
    if os.path.exists(p_is) and os.path.exists(p_oos):
        d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
        return d1m, d4h[~d4h.index.duplicated(keep="first")].sort_index()
    d4h = d1m.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return d1m, d4h

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

def calc_rsi(closes, n=14):
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def build_4h(df4h):
    df = df4h.copy()
    df["atr"]    = calc_atr(df, 14)
    df["atr5"]   = calc_atr(df, 5)
    df["ema20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["trend"]  = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]    = calc_adx(df, 14)
    df["rsi"]    = calc_rsi(df["close"], 14)
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["ema_slope"] = (df["ema20"] - df["ema20"].shift(1)) / df["atr"].replace(0, np.nan)

    # Daily
    d1 = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)

    # Weekly
    dw = df.resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    dw["ema20"]   = dw["close"].ewm(span=20, adjust=False).mean()
    dw["trend_w"] = np.where(dw["close"] > dw["ema20"], 1, -1)

    return df, d1, dw

def build_1h(df):
    r = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    r["atr"] = calc_atr(r, 14)
    return r

# ── エントリー ────────────────────────────────────────────────────
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
        av = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and rng > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

# ── シグナル生成（全メタデータ付き）──────────────────────────────
def generate_signals_with_meta(d1m, d4h_full, spread, logic, atr_d, m1c):
    """ベースラインシグナルを生成し、各特徴量の判定結果をメタデータに付与"""
    d4h, d1d, dw = build_4h(d4h_full)
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

        # ── 既存ロジック別フィルター ──
        if logic == "A":
            d1b = d1d[d1d.index.normalize() < hct.normalize()]
            if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        elif logic == "B":
            if h4l.get("adx", 0) < ADX_MIN: continue
            if not all(t == trend for t in h4b["trend"].iloc[-STREAK_MIN:].values): continue

        if not ((trend == 1 and h4l["close"] > h4l["open"]) or
                (trend == -1 and h4l["close"] < h4l["open"])): continue  # KMID
        if not ((min(h4l["open"], h4l["close"]) - h4l["low"]) / h4l["open"] < KLOW_THR
                if h4l["open"] > 0 else False): continue  # KLOW
        if logic != "C":
            if pd.isna(h4l["atr"]) or h4l["atr"] <= 0: continue
            if abs(h4l["close"] - h4l["ema20"]) < h4l["atr"] * A1_EMA_DIST_MIN: continue

        d = trend
        v1 = p2["low"] if d == 1 else p2["high"]
        v2 = p1["low"] if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * A3_DEFAULT_TOL: continue

        if logic == "C":
            if d == 1 and p1["close"] <= p1["open"]: continue
            if d == -1 and p1["close"] >= p1["open"]: continue

        # ── エントリー ──
        if logic == "A":   et, ep = pick_e2(hct, d, spread, atr_d, m1c)
        elif logic == "C": et, ep = pick_e0(hct, spread, d, m1c)
        else:              et, ep = pick_e1(hct, d, spread, m1c)

        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if not (0 < risk <= h4atr * 2): continue

        # ── 全特徴量メタデータ計算 ──
        upper_wick = h4l["high"] - max(h4l["open"], h4l["close"])
        kupp_ratio = upper_wick / h4l["open"] if h4l["open"] > 0 else 0
        rng_4h = h4l["high"] - h4l["low"]
        body_ratio = abs(h4l["close"] - h4l["open"]) / rng_4h if rng_4h > 0 else 0
        slope = h4l.get("ema_slope", 0) if not pd.isna(h4l.get("ema_slope", 0)) else 0
        vol_ma = h4l.get("vol_ma", 0)
        vol_ratio = h4l["volume"] / vol_ma if (not pd.isna(vol_ma) and vol_ma > 0) else 999
        rsi = h4l.get("rsi", 50) if not pd.isna(h4l.get("rsi", 50)) else 50
        atr5 = h4l.get("atr5", 0) if not pd.isna(h4l.get("atr5", 0)) else 0
        atr_ratio = atr5 / h4l["atr"] if h4l["atr"] > 0 else 1
        pat_dist = abs(v1 - v2) / atr1h if atr1h > 0 else 999

        wb = dw[dw.index < hct]
        weekly_ok = len(wb) > 0 and wb.iloc[-1]["trend_w"] == trend

        ema20 = h4l.get("ema20", 0)
        ema50 = h4l.get("ema50", 0)
        ema_cross_ok = (ema20 > ema50) if d == 1 else (ema20 < ema50)

        h1_mom_ok = True
        n_mom = FEAT_THR["F10_mom"]
        if i >= n_mom:
            for j in range(i-n_mom, i):
                bar = d1h.iloc[j]
                if d == 1 and bar["close"] <= bar["open"]:
                    h1_mom_ok = False; break
                if d == -1 and bar["close"] >= bar["open"]:
                    h1_mom_ok = False; break
        else:
            h1_mom_ok = False

        # F1: 上ヒゲフィルター（方向依存）
        f1_ok = (kupp_ratio < FEAT_THR["F1_kupp"]) if d == 1 else True
        f3_ok = (slope >= FEAT_THR["F3_slope"]) if d == 1 else (slope <= -FEAT_THR["F3_slope"])

        meta = {
            "F1": f1_ok,
            "F2": body_ratio >= FEAT_THR["F2_kbody"],
            "F3": f3_ok,
            "F4": vol_ratio >= FEAT_THR["F4_vol"],
            "F5": FEAT_THR["F5_rsi_lo"] <= rsi <= FEAT_THR["F5_rsi_hi"],
            "F6": atr_ratio >= FEAT_THR["F6_atr"],
            "F7": pat_dist <= FEAT_THR["F7_patq"],
            "F8": weekly_ok,
            "F9": ema_cross_ok,
            "F10": h1_mom_ok,
        }

        signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                        "tp": raw + d * risk * RR_RATIO, "risk": risk,
                        "meta": meta})
        used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── シグナルフィルタリング ─────────────────────────────────────────
def filter_signals(signals, features):
    """メタデータに基づいてシグナルをフィルタリング"""
    if not features: return signals
    return [s for s in signals if all(s["meta"].get(f, True) for f in features)]

# ── シミュレーション ──────────────────────────────────────────────
def _exit(highs, lows, ep, sl, tp, risk, d):
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

def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result, half_done = _exit(m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
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
        trades.append({"result": result, "pnl": half_pnl + pnl, "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)
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

def run_filtered(all_sigs, d1m, sym, features=None):
    sigs = filter_signals(all_sigs, features)
    trades, eq, mdd = simulate(sigs, d1m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["n_signals"] = len(sigs)
    return st, trades

# ── ポートフォリオSharpe ──────────────────────────────────────────
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

# ── メイン ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*120)
    print("  全期間バックテスト + 新規特徴量スクリーニング（採用7銘柄）")
    print("="*120)

    # ── Phase 1: データロード + シグナル生成（1回のみ）──
    print("\n  [Phase 1] データロード & ベースラインシグナル生成...")
    sym_data = {}
    all_signals = {}  # {sym: [signals_with_meta]}

    for tgt in TARGETS:
        sym = tgt["sym"]
        d1m, d4h = load_all(sym)
        if d1m is None:
            print(f"    ❌ {sym}: データ未発見"); continue
        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d = calc_atr(d1m, 10).to_dict()
        m1c = {"idx": d1m.index, "opens": d1m["open"].values,
               "closes": d1m["close"].values,
               "highs": d1m["high"].values, "lows": d1m["low"].values}

        sigs = generate_signals_with_meta(d1m, d4h, spread, tgt["logic"], atr_d, m1c)
        all_signals[sym] = sigs
        sym_data[sym] = {"d1m": d1m, "d4h": d4h, "logic": tgt["logic"], "cat": tgt["cat"]}
        print(f"    ✅ {sym}: 1m={len(d1m):,}行  シグナル={len(sigs)}件")

    # ── Phase 2: ベースライン ──
    print(f"\n  [Phase 2] ベースライン（既存ロジック、新規フィルタなし）...")
    base_stats = {}; base_trades = {}
    for sym in sym_data:
        st, trades = run_filtered(all_signals[sym], sym_data[sym]["d1m"], sym, features=None)
        base_stats[sym] = st; base_trades[sym] = trades
        if st:
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            print(f"    {sym:8} Logic-{sym_data[sym]['logic']}: "
                  f"n={st['n']:>4}  WR={st['wr']*100:.1f}%  PF={pf_s:>6}  "
                  f"Sharpe={st['sharpe']:.2f}  MDD={st['mdd']:.1f}%  "
                  f"月+={st['plus_m']}/{st['total_m']}")

    base_port = portfolio_sharpe(base_trades)
    print(f"\n    ★ ベースライン ポートフォリオSharpe = {base_port:.2f}")

    # ── Phase 3: 個別特徴量テスト（フィルタリングのみ、高速）──
    print(f"\n  [Phase 3] 個別特徴量テスト（{len(FEATURE_LIST)}種）...")
    feat_results = []
    for fn in FEATURE_LIST:
        feat_trades = {}; improved = 0
        sym_detail = {}
        for sym in sym_data:
            st, trades = run_filtered(all_signals[sym], sym_data[sym]["d1m"], sym, features={fn})
            feat_trades[sym] = trades
            sym_detail[sym] = st
            if st.get("sharpe", 0) > base_stats[sym].get("sharpe", 0): improved += 1
        port = portfolio_sharpe(feat_trades)
        delta = port - base_port
        feat_results.append({"feat": fn, "name": FEATURE_NAMES[fn], "port": port,
                             "delta": delta, "improved": improved, "sym": sym_detail})
        m = "✅" if delta > 0.1 else ("➡️" if delta > -0.1 else "❌")
        print(f"    {m} {fn:4} {FEATURE_NAMES[fn]:22}  Sharpe={port:.2f} (Δ={delta:+.2f})  改善{improved}/{len(sym_data)}銘柄")

    # ── Phase 4: 組み合わせテスト ──
    pos_feats = [f["feat"] for f in feat_results if f["delta"] > 0]
    if len(pos_feats) < 2:
        top = sorted(feat_results, key=lambda x: -x["delta"])
        pos_feats = [f["feat"] for f in top[:4]]

    print(f"\n  [Phase 4] 組み合わせテスト（有望: {pos_feats}）...")
    combo_results = []
    for r in range(2, min(len(pos_feats)+1, 5)):
        for combo in itertools.combinations(pos_feats, r):
            feat_set = set(combo)
            c_trades = {}; c_sym = {}
            for sym in sym_data:
                st, trades = run_filtered(all_signals[sym], sym_data[sym]["d1m"], sym, features=feat_set)
                c_trades[sym] = trades; c_sym[sym] = st
            port = portfolio_sharpe(c_trades)
            delta = port - base_port
            label = "+".join(combo)
            combo_results.append({"label": label, "features": feat_set, "port": port,
                                  "delta": delta, "sym": c_sym})
            m = "✅" if delta > 0.3 else ("➡️" if delta > 0 else "❌")
            print(f"    {m} {label:25}  Sharpe={port:.2f} (Δ={delta:+.2f})")

    # ── Phase 5: IS/OOS過学習チェック（top5）──
    all_cands = [(f["feat"], {f["feat"]}, f["port"], f["delta"], f["sym"]) for f in feat_results]
    all_cands += [(c["label"], c["features"], c["port"], c["delta"], c["sym"]) for c in combo_results]
    all_cands.sort(key=lambda x: -x[2])
    top5 = all_cands[:5]

    print(f"\n  [Phase 5] IS/OOS過学習チェック（top5）...")
    verified = []
    for label, feat_set, full_port, delta, _ in top5:
        is_trades = {}; oos_trades = {}
        for sym in sym_data:
            d1m = sym_data[sym]["d1m"]
            n_split = int(len(d1m) * 0.4)
            ts_split = d1m.index[n_split]

            # IS/OOS期間でシグナルを分割
            sigs_is  = [s for s in all_signals[sym] if s["time"] < ts_split]
            sigs_oos = [s for s in all_signals[sym] if s["time"] >= ts_split]

            is_filtered  = filter_signals(sigs_is, feat_set)
            oos_filtered = filter_signals(sigs_oos, feat_set)

            _, is_t   = run_filtered(is_filtered,  d1m[d1m.index < ts_split], sym, features=None)
            _, oos_t  = run_filtered(oos_filtered, d1m[d1m.index >= ts_split], sym, features=None)
            is_trades[sym] = is_t; oos_trades[sym] = oos_t

        is_port = portfolio_sharpe(is_trades)
        oos_port = portfolio_sharpe(oos_trades)
        ratio = oos_port / is_port if is_port > 0 else 0
        flag = "✅" if ratio >= 0.70 else "❌"
        print(f"    {label:30}  IS={is_port:.2f}  OOS={oos_port:.2f}  ratio={ratio:.2f} {flag}")
        verified.append({"label": label, "features": feat_set, "full": full_port,
                         "is": is_port, "oos": oos_port, "ratio": ratio,
                         "passed": ratio >= 0.70, "delta": delta})

    # ── 最終レポート ──────────────────────────────────────────────
    print("\n" + "="*120)
    print("  ■ 最終レポート")
    print("="*120)

    print(f"\n  【ベースライン】ポートフォリオSharpe = {base_port:.2f}")
    print(f"  {'銘柄':8} {'ロジック':14} {'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MDD':>6} {'月+':>5}")
    print("  " + "-"*65)
    for sym in sym_data:
        st = base_stats[sym]
        if not st: continue
        pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
        print(f"  {sym:8} Logic-{sym_data[sym]['logic']}:{LOGIC_NAMES[sym_data[sym]['logic']]:10} "
              f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['sharpe']:>7.2f} "
              f"{st['mdd']:>5.1f}% {st['plus_m']:>2}/{st['total_m']:<2}")

    print(f"\n  【個別特徴量ランキング】")
    print(f"  {'#':>3} {'特徴量':28} {'Sharpe':>8} {'Δ':>7} {'改善':>4}")
    print("  " + "-"*55)
    for rank, f in enumerate(sorted(feat_results, key=lambda x: -x["delta"]), 1):
        print(f"  #{rank:>2} {f['feat']:4} {f['name']:22} {f['port']:>8.2f} {f['delta']:>+7.2f} {f['improved']:>2}/{len(sym_data)}")

    print(f"\n  【組み合わせランキング（top5）】")
    for rank, c in enumerate(sorted(combo_results, key=lambda x: -x["port"])[:5], 1):
        print(f"  #{rank} {c['label']:25}  Sharpe={c['port']:.2f} (Δ={c['delta']:+.2f})")

    print(f"\n  【IS/OOS検証結果】")
    best_passed = [v for v in verified if v["passed"]]
    if best_passed:
        best = max(best_passed, key=lambda x: x["oos"])
        print(f"  ★ 最優秀: {best['label']}")
        print(f"    Full Sharpe = {best['full']:.2f} (Δ={best['delta']:+.2f})")
        print(f"    IS = {best['is']:.2f}  OOS = {best['oos']:.2f}  OOS/IS = {best['ratio']:.2f} ✅")

        # 銘柄別比較
        print(f"\n    {'銘柄':8} {'Base':>8} {'New':>8} {'Δ':>7}")
        print("    " + "-"*35)
        for cand in all_cands:
            if cand[0] == best["label"]:
                for sym in sym_data:
                    b_sh = base_stats[sym].get("sharpe", 0)
                    n_sh = cand[4].get(sym, {}).get("sharpe", 0)
                    d = n_sh - b_sh
                    m = "↑" if d > 0.1 else ("↓" if d < -0.1 else "→")
                    print(f"    {sym:8} {b_sh:>8.2f} {n_sh:>8.2f} {d:>+7.2f} {m}")
                break
    else:
        print("  過学習チェックをパスした候補なし")
        for v in verified:
            print(f"    {v['label']:30}  Full={v['full']:.2f}  OOS/IS={v['ratio']:.2f}")

    # CSV保存
    rows = []
    for f in feat_results:
        rows.append({"type": "single", "label": f["feat"], "name": f["name"],
                     "port_sharpe": f["port"], "delta": f["delta"], "improved": f["improved"]})
    for c in combo_results:
        rows.append({"type": "combo", "label": c["label"], "port_sharpe": c["port"], "delta": c["delta"]})
    for v in verified:
        rows.append({"type": "verified", "label": v["label"], "full_sharpe": v["full"],
                     "is_sharpe": v["is"], "oos_sharpe": v["oos"], "ratio": v["ratio"], "passed": v["passed"]})

    out = os.path.join(OUT_DIR, "backtest_new_features.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  結果保存: {out}")
    print(f"  実行時間: {time.time() - t0:.0f}秒")

if __name__ == "__main__":
    main()
