"""
backtest_portfolio_integration.py
===================================
ポートフォリオ統合分析 — 採用7銘柄の同時運用シミュレーション

【AIフィードバックへの改善対応】
  指摘: 同時運用時の合成DDと相関が未確認
  確認項目:
    ✅ 日次リターン相関マトリクス（銘柄間相関）
    ✅ 同日シグナル発生率（シグナルのクラスタリング）
    ✅ 同方向エクスポージャ偏り（ロング/ショート比率）
    ✅ 全銘柄同時運用時PF/Sharpe/MDD（ポートフォリオ合成指標）
    ✅ 最大連敗のクラスター性（連敗が特定期間に集中するか）
    ✅ 月単位での収益源の偏り（月次寄与率）
    ✅ 採用基準の明文化（OOS PF≥1.30 / Sharpe≥2.0 / OOS/IS≥0.70 / 月次+≥70% / MDD≤25%）
    ✅ リスクステージング（0.5% → 1.0% → 1.5-2.0%）
    ✅ 優先上位4銘柄サブセット（USDJPY/GBPUSD/EURUSD/USDCAD）

【採用7銘柄（実運用優先順位順）】
  1位 USDJPY  Logic-C オーパーツYAGAMI  固定2%  (Sharpe=6.18, 再現性最高)
  2位 GBPUSD  Logic-A GOLDYAGAMI       固定2%  (Sharpe=7.12, PF=1.86)
  3位 EURUSD  Logic-C オーパーツYAGAMI  固定2%  (Sharpe=6.18, PF=1.81)
  4位 USDCAD  Logic-A GOLDYAGAMI       固定2%  (Sharpe=5.62, PF=2.02)
  5位 NZDUSD  Logic-A GOLDYAGAMI       固定2%  (Sharpe=5.45, PF=1.98)
  6位 XAUUSD  Logic-A GOLDYAGAMI       固定2%  (Sharpe=3.42, 別カテゴリ)
  7位 AUDUSD  Logic-B ADX+Streak       固定2%  (Sharpe=3.66, 月次安定性要確認)

【固定採用基準（AIフィードバック反映）】
  OOS PF   ≥ 1.30
  OOS Sharpe ≥ 2.0 (年率換算)
  OOS/IS PF ≥ 0.70 (過学習チェック)
  月次プラス ≥ 70%
  MDD      ≤ 25%

【リスクステージング（フォワードテスト用）】
  Phase1: 0.5% / 銘柄（上位4銘柄のみ）    → 3ヶ月確認
  Phase2: 1.0% / 銘柄（上位4〜7銘柄）     → 3ヶ月確認
  Phase3: 1.5〜2.0% / 銘柄（全採用銘柄）   → 本格運用
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

# ── 採用基準（固定・明文化）─────────────────────────────────────
CRITERIA = {
    "oos_pf_min":   1.30,   # OOS プロフィットファクター
    "sharpe_min":   2.00,   # 年率シャープレシオ
    "oos_is_min":   0.70,   # OOS/IS PF比率（過学習チェック）
    "plus_m_min":   0.70,   # 月次プラス率
    "mdd_max":     25.0,    # 最大ドローダウン%
}

# ── リスクステージング（フォワードテスト用）──────────────────
RISK_STAGES = [
    {"phase": 1, "risk_pct": 0.005, "syms": ["USDJPY","GBPUSD","EURUSD","USDCAD"],
     "desc": "上位4銘柄のみ・低リスク検証（3ヶ月）"},
    {"phase": 2, "risk_pct": 0.010, "syms": ["USDJPY","GBPUSD","EURUSD","USDCAD","NZDUSD","XAUUSD","AUDUSD"],
     "desc": "全7銘柄・中リスク（3ヶ月）"},
    {"phase": 3, "risk_pct": 0.020, "syms": ["USDJPY","GBPUSD","EURUSD","USDCAD","NZDUSD","XAUUSD","AUDUSD"],
     "desc": "全7銘柄・本格運用（継続）"},
]

# ── 採用7銘柄（実運用優先順位順）────────────────────────────────
ADOPTED = [
    {"sym": "USDJPY", "logic": "C", "risk": "fixed2", "cat": "FX_JPY",   "priority": 1},
    {"sym": "GBPUSD", "logic": "A", "risk": "fixed2", "cat": "FX_USD",   "priority": 2},
    {"sym": "EURUSD", "logic": "C", "risk": "fixed2", "cat": "FX_USD",   "priority": 3},
    {"sym": "USDCAD", "logic": "A", "risk": "fixed2", "cat": "FX_USD",   "priority": 4},
    {"sym": "NZDUSD", "logic": "A", "risk": "fixed2", "cat": "FX_USD",   "priority": 5},
    {"sym": "XAUUSD", "logic": "A", "risk": "fixed2", "cat": "METALS",   "priority": 6},
    {"sym": "AUDUSD", "logic": "B", "risk": "fixed2", "cat": "FX_USD",   "priority": 7},
]

TOP4 = [t for t in ADOPTED if t["priority"] <= 4]

LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

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
        if os.path.exists(p):
            d1m = load_csv(p); break
    if d1m is None:
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
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c):
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
        if abs(v1 - v2) > atr1h * A3_DEFAULT_TOL: continue

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

# ── シミュレーション（トレード詳細記録付き）──────────────────────
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

def simulate_detailed(signals, d1m, sym, risk_pct=0.02):
    """トレードごとの日次PNLを追跡するシミュレーション"""
    if not signals:
        return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=risk_pct)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        rm.risk_pct = risk_pct
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
        total_pnl = half_pnl + pnl

        trades.append({
            "time":   sig["time"],
            "date":   sig["time"].date(),
            "month":  sig["time"].strftime("%Y-%m"),
            "dir":    sig["dir"],
            "result": result,
            "pnl":    total_pnl,
            "equity": equity,
        })
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd

# ── 統計計算 ─────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 10:
        return {}
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

    p_val = stats.binomtest(len(wins), n, 0.5, alternative="greater").pvalue

    return {
        "n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
        "plus_m": plus_m, "total_m": len(monthly), "p_val": p_val,
        "final_eq": equity if 'equity' in dir() else init,
        "monthly": monthly,
    }

def check_criteria(st_oos, st_is):
    """固定採用基準チェック"""
    if not st_oos or not st_is:
        return {}
    pf     = st_oos.get("pf", 0)
    sharpe = st_oos.get("sharpe", 0)
    mdd    = st_oos.get("mdd", 0)
    plus_m = st_oos.get("plus_m", 0)
    total_m = st_oos.get("total_m", 1)
    is_pf  = st_is.get("pf", 0)
    ois    = pf / is_pf if is_pf > 0 else 0

    chk = {
        "pf_ok":    pf >= CRITERIA["oos_pf_min"],
        "sharpe_ok": sharpe >= CRITERIA["sharpe_min"],
        "ois_ok":   ois >= CRITERIA["ois_is_min"] if "ois_is_min" in CRITERIA else ois >= CRITERIA["oos_is_min"],
        "pm_ok":    (plus_m / total_m) >= CRITERIA["plus_m_min"] if total_m > 0 else False,
        "mdd_ok":   mdd <= CRITERIA["mdd_max"],
    }
    chk["all_pass"] = all(chk.values())
    chk["ois_ratio"] = ois
    return chk

# ── ポートフォリオ分析 ────────────────────────────────────────────
def portfolio_analysis(sym_trades_dict, label="全7銘柄"):
    """複数銘柄のトレードデータを統合してポートフォリオ分析"""
    if not sym_trades_dict:
        return {}

    # 日次PNL集計（銘柄別）
    daily_pnl = {}
    for sym, trades in sym_trades_dict.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        daily = df.groupby("date")["pnl"].sum()
        daily_pnl[sym] = daily

    if not daily_pnl:
        return {}

    # 全銘柄の日次PNLを結合
    df_daily = pd.DataFrame(daily_pnl).fillna(0)
    df_daily["portfolio"] = df_daily.sum(axis=1)

    # ── 相関行列（日次リターン）────────────────────────────
    syms_available = list(daily_pnl.keys())
    if len(syms_available) >= 2:
        corr_matrix = df_daily[syms_available].corr()
    else:
        corr_matrix = None

    # ── ポートフォリオ合成指標 ────────────────────────────
    all_trades = []
    for sym, trades in sym_trades_dict.items():
        for t in trades:
            all_trades.append(dict(t, sym=sym))
    if not all_trades:
        return {}

    df_all = pd.DataFrame(all_trades).sort_values("time")

    # 合成PF
    wins_pnl = df_all[df_all["pnl"] > 0]["pnl"].sum()
    loss_pnl = abs(df_all[df_all["pnl"] < 0]["pnl"].sum())
    port_pf  = wins_pnl / loss_pnl if loss_pnl > 0 else float("inf")

    # 合成エクイティカーブ（累積PNL）
    port_equity = INIT_CASH + df_daily["portfolio"].cumsum()
    port_peak   = port_equity.cummax()
    port_dd     = (port_peak - port_equity) / port_peak * 100
    port_mdd    = port_dd.max()

    # 月次Sharpe（ポートフォリオ合計）
    df_daily_dt = df_daily.copy()
    df_daily_dt.index = pd.to_datetime(df_daily_dt.index)
    monthly_port = df_daily_dt["portfolio"].resample("ME").sum()
    eq = INIT_CASH
    monthly_ret = []
    for m, v in monthly_port.items():
        ret = v / eq if eq > 0 else 0
        monthly_ret.append(ret)
        eq += v
    mr = np.array(monthly_ret)
    port_sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0
    port_plus_m = (monthly_port > 0).sum()

    # ── 同日シグナル発生率 ────────────────────────────────
    sig_dates = df_all.groupby("date")["sym"].nunique()
    multi_sig_days = (sig_dates >= 2).sum()
    multi_sig_rate = multi_sig_days / len(sig_dates) * 100 if len(sig_dates) > 0 else 0

    # ── 同方向エクスポージャ ──────────────────────────────
    # 同日に全シグナルが同方向（ロング/ショート）の日の割合
    same_dir_days = 0
    total_multi_days = 0
    for date, grp in df_all.groupby("date"):
        if len(grp) >= 2:
            total_multi_days += 1
            dirs = grp["dir"].unique()
            if len(dirs) == 1:
                same_dir_days += 1
    same_dir_rate = same_dir_days / total_multi_days * 100 if total_multi_days > 0 else 0

    # ── 最大連敗のクラスター性 ────────────────────────────
    # 連敗が月をまたがず集中しているか確認
    df_loss = df_all[df_all["result"] == "loss"].copy()
    df_loss["month"] = df_loss["time"].dt.strftime("%Y-%m")
    loss_by_month    = df_loss.groupby("month").size()
    loss_max_month   = loss_by_month.max() if len(loss_by_month) > 0 else 0
    loss_concentration = loss_max_month / max(len(df_loss), 1) * 100

    # ── 月次寄与率（銘柄別） ──────────────────────────────
    monthly_by_sym = df_all.groupby(["month", "sym"])["pnl"].sum().unstack(fill_value=0)
    monthly_total  = monthly_by_sym.sum(axis=1)
    monthly_contrib = monthly_by_sym.div(monthly_total.replace(0, np.nan), axis=0) * 100

    # 銘柄間で月次寄与が極端に偏っていないか（最大寄与率の平均）
    max_contrib_avg = monthly_contrib.abs().max(axis=1).mean() if len(monthly_contrib) > 0 else 0

    return {
        "label":             label,
        "port_pf":           port_pf,
        "port_sharpe":       port_sharpe,
        "port_mdd":          port_mdd,
        "port_plus_m":       port_plus_m,
        "total_months":      len(monthly_port),
        "total_trades":      len(df_all),
        "corr_matrix":       corr_matrix,
        "multi_sig_rate":    multi_sig_rate,
        "same_dir_rate":     same_dir_rate,
        "loss_concentration":loss_concentration,
        "max_contrib_avg":   max_contrib_avg,
        "monthly_contrib":   monthly_contrib,
        "df_daily":          df_daily,
        "syms":              syms_available,
    }

# ── リスクステージ別シミュレーション ─────────────────────────────
def simulate_risk_stage(sym_data_map, stage_syms, risk_pct, label):
    """指定リスク率でのポートフォリオシミュレーション"""
    sym_trades = {}
    for tgt in ADOPTED:
        sym = tgt["sym"]
        if sym not in stage_syms:
            continue
        if sym not in sym_data_map:
            continue
        d1m_oos, d4h_full = sym_data_map[sym]
        if d1m_oos is None:
            continue
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m_oos, 10).to_dict()
        m1c    = {"idx": d1m_oos.index, "opens": d1m_oos["open"].values,
                  "closes": d1m_oos["close"].values,
                  "highs":  d1m_oos["high"].values, "lows": d1m_oos["low"].values}
        sigs = generate_signals(d1m_oos, d4h_full, spread, tgt["logic"], atr_d, m1c)
        trades, _, _ = simulate_detailed(sigs, d1m_oos, sym, risk_pct=risk_pct)
        sym_trades[sym] = trades

    return portfolio_analysis(sym_trades, label=label)

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*120)
    print("  ポートフォリオ統合分析 — 採用7銘柄 同時運用シミュレーション")
    print("  【AIフィードバック改善対応版】相関・クラスタリング・合成MDD・リスクステージング")
    print("="*120)

    # ── Step 1: 各銘柄データ読み込み ─────────────────────────────
    print("\n  [Step 1] データ読み込み & 個別OOSバックテスト")
    print("-"*80)

    sym_data_map  = {}  # sym -> (d1m_oos, d4h_full)
    sym_trades_oos = {}  # sym -> trades (OOS期間)
    indiv_stats   = {}  # sym -> {"oos": ..., "is": ...}

    for tgt in ADOPTED:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        print(f"  {sym:8} Logic-{logic}:{LOGIC_NAMES[logic]:10}", end=" ... ", flush=True)

        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("データ未発見"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m_full)

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]

        # IS
        atr_is = calc_atr(is_d, 10).to_dict()
        m1c_is = {"idx": is_d.index, "opens": is_d["open"].values,
                  "closes": is_d["close"].values,
                  "highs":  is_d["high"].values, "lows": is_d["low"].values}
        sigs_is = generate_signals(is_d, d4h_full, spread, logic, atr_is, m1c_is)
        tr_is, _, mdd_is = simulate_detailed(sigs_is, is_d, sym, risk_pct=0.02)
        st_is = calc_stats(tr_is)
        if st_is: st_is["mdd"] = mdd_is

        # OOS
        atr_oos = calc_atr(oos_d, 10).to_dict()
        m1c_oos = {"idx": oos_d.index, "opens": oos_d["open"].values,
                   "closes": oos_d["close"].values,
                   "highs":  oos_d["high"].values, "lows": oos_d["low"].values}
        sigs_oos = generate_signals(oos_d, d4h_full, spread, logic, atr_oos, m1c_oos)
        tr_oos, _, mdd_oos = simulate_detailed(sigs_oos, oos_d, sym, risk_pct=0.02)
        st_oos = calc_stats(tr_oos)
        if st_oos: st_oos["mdd"] = mdd_oos

        sym_data_map[sym]   = (oos_d, d4h_full)
        sym_trades_oos[sym] = tr_oos
        indiv_stats[sym]    = {"is": st_is, "oos": st_oos, "tgt": tgt,
                               "split": split_ts}

        n   = st_oos.get("n", 0)
        pf  = st_oos.get("pf", 0)
        sh  = st_oos.get("sharpe", 0)
        mdd = st_oos.get("mdd", 0)
        print(f"OOS n={n:3d}  PF={pf:.2f}  Sharpe={sh:.2f}  MDD={mdd:.1f}%")

    # ── Step 2: 固定採用基準チェック ─────────────────────────────
    print("\n" + "="*120)
    print("  [Step 2] 固定採用基準チェック（AIフィードバック反映）")
    print(f"  基準: OOS PF≥{CRITERIA['oos_pf_min']}  Sharpe≥{CRITERIA['sharpe_min']}  "
          f"OOS/IS≥{CRITERIA['oos_is_min']}  月次+≥{CRITERIA['plus_m_min']*100:.0f}%  "
          f"MDD≤{CRITERIA['mdd_max']}%")
    print("-"*100)
    print(f"  {'優先':>4} {'銘柄':8} {'カテゴリ':10} {'OOS PF':>8} {'Sharpe':>8} "
          f"{'OOS/IS':>8} {'月次+':>7} {'MDD':>7} | {'PF':>4} {'Sh':>4} {'OI':>4} {'PM':>4} {'MD':>4} | 判定")
    print("-"*100)

    passed_syms = []
    for tgt in ADOPTED:
        sym = tgt["sym"]
        if sym not in indiv_stats:
            continue
        st_oos = indiv_stats[sym]["oos"]
        st_is  = indiv_stats[sym]["is"]
        if not st_oos:
            continue

        pf     = st_oos.get("pf", 0)
        sharpe = st_oos.get("sharpe", 0)
        mdd    = st_oos.get("mdd", 0)
        plus_m = st_oos.get("plus_m", 0)
        total_m= st_oos.get("total_m", 1)
        is_pf  = st_is.get("pf", 0) if st_is else 0
        ois    = pf / is_pf if is_pf > 0 else 0
        pm_r   = plus_m / total_m if total_m > 0 else 0

        pf_ok  = pf >= CRITERIA["oos_pf_min"]
        sh_ok  = sharpe >= CRITERIA["sharpe_min"]
        ois_ok = ois >= CRITERIA["oos_is_min"]
        pm_ok  = pm_r >= CRITERIA["plus_m_min"]
        mdd_ok = mdd <= CRITERIA["mdd_max"]
        all_ok = pf_ok and sh_ok and ois_ok and pm_ok and mdd_ok

        verdict = "✅全基準PASS" if all_ok else "⚠️一部未達"
        pf_s = f"{pf:.2f}" if pf < 99 else "∞"

        print(f"  #{tgt['priority']:<3} {sym:8} {tgt['cat']:10} {pf_s:>8} {sharpe:>8.2f} "
              f"{ois:>8.2f} {pm_r*100:>6.0f}% {mdd:>6.1f}% | "
              f"{'✅' if pf_ok else '❌':>4} {'✅' if sh_ok else '❌':>4} "
              f"{'✅' if ois_ok else '❌':>4} {'✅' if pm_ok else '❌':>4} "
              f"{'✅' if mdd_ok else '❌':>4} | {verdict}")
        if all_ok:
            passed_syms.append(sym)

    print(f"\n  固定基準PASS: {len(passed_syms)}/{len(ADOPTED)}銘柄  [{', '.join(passed_syms)}]")

    # ── Step 3: ポートフォリオ統合分析（全7銘柄） ────────────────
    print("\n" + "="*120)
    print("  [Step 3] ポートフォリオ統合分析（全7銘柄 同時運用・OOS期間）")
    print("-"*80)

    port_all  = portfolio_analysis(sym_trades_oos, label="全7銘柄")
    # 上位4銘柄
    top4_trades = {sym: sym_trades_oos[sym] for sym in [t["sym"] for t in TOP4]
                   if sym in sym_trades_oos}
    port_top4 = portfolio_analysis(top4_trades, label="上位4銘柄")
    # FX（XAUUSD除く）
    fx_syms = [t["sym"] for t in ADOPTED if t["cat"] != "METALS"]
    fx_trades = {sym: sym_trades_oos[sym] for sym in fx_syms if sym in sym_trades_oos}
    port_fx   = portfolio_analysis(fx_trades, label="FX6銘柄")

    for port in [port_all, port_top4, port_fx]:
        if not port:
            continue
        label = port["label"]
        print(f"\n  【{label}】ポートフォリオ合成指標")
        print(f"    合成 PF     = {port['port_pf']:.3f}")
        print(f"    合成 Sharpe = {port['port_sharpe']:.2f}（年率換算）")
        print(f"    合成 MDD    = {port['port_mdd']:.1f}%")
        print(f"    月次プラス  = {port['port_plus_m']}/{port['total_months']}ヶ月")
        print(f"    総トレード数 = {port['total_trades']}回")

    # ── Step 4: 相関マトリクス ────────────────────────────────────
    print("\n" + "="*120)
    print("  [Step 4] 銘柄間 日次リターン相関マトリクス（OOS期間）")
    print("  ⚠️ |相関| > 0.6 は高相関（リスク分散不十分）")
    print("-"*80)

    if port_all.get("corr_matrix") is not None:
        corr = port_all["corr_matrix"]
        syms_c = corr.columns.tolist()
        # ヘッダー
        header = f"  {'':8}" + "".join(f"{s:>9}" for s in syms_c)
        print(header)
        print("  " + "-"*80)
        for s1 in syms_c:
            row = f"  {s1:8}"
            for s2 in syms_c:
                v = corr.loc[s1, s2]
                if s1 == s2:
                    row += f"{'1.000':>9}"
                else:
                    flag = " ⚠" if abs(v) > 0.6 and s1 != s2 else ""
                    row += f"{v:>8.3f}{flag}"[:9]
            print(row)

        # 高相関ペアの警告
        print("\n  高相関ペア（|r|>0.6）:")
        high_corr_found = False
        for i, s1 in enumerate(syms_c):
            for s2 in syms_c[i+1:]:
                v = corr.loc[s1, s2]
                if abs(v) > 0.6:
                    print(f"    {s1} ↔ {s2}: r={v:.3f}  ← {'正相関（同一方向リスク）' if v > 0 else '負相関（ヘッジ効果）'}")
                    high_corr_found = True
        if not high_corr_found:
            print("    なし（全ペアで|r|≤0.6 → 分散効果あり）")

    # ── Step 5: 同日シグナル・クラスター分析 ─────────────────────
    print("\n" + "="*120)
    print("  [Step 5] シグナルクラスタリング & エクスポージャ分析")
    print("-"*80)

    for port in [port_all, port_top4]:
        if not port:
            continue
        label = port["label"]
        print(f"\n  【{label}】")
        print(f"    同日複数シグナル発生率 = {port['multi_sig_rate']:.1f}%")
        print(f"    同方向エクスポージャ率 = {port['same_dir_rate']:.1f}%  "
              f"（同日シグナルが全て同一方向の割合）")
        print(f"    連敗集中率             = {port['loss_concentration']:.1f}%  "
              f"（損失の最大月集中度）")
        print(f"    月次最大寄与率(平均)   = {port['max_contrib_avg']:.1f}%  "
              f"（特定銘柄への月次依存度）")

        # 同方向エクスポージャの評価
        if port['same_dir_rate'] > 70:
            print(f"    ⚠️  同方向集中リスク高（{port['same_dir_rate']:.0f}%の日で全銘柄同方向）")
        elif port['same_dir_rate'] > 50:
            print(f"    ⚡ 中程度の同方向集中（{port['same_dir_rate']:.0f}%）")
        else:
            print(f"    ✅ 方向分散良好（{port['same_dir_rate']:.0f}%）")

    # ── Step 6: 月次寄与率ヒートマップ（テキスト） ───────────────
    print("\n" + "="*120)
    print("  [Step 6] 月次収益寄与率（銘柄別） — OOS期間")
    print("  （各月の合計収益に対する各銘柄の寄与率%。マイナス=その月は損失）")
    print("-"*100)

    if port_all.get("monthly_contrib") is not None:
        mc = port_all["monthly_contrib"]
        syms_mc = mc.columns.tolist()
        header = f"  {'月':>10}" + "".join(f"{s:>9}" for s in syms_mc)
        print(header)
        print("  " + "-"*90)
        for month, row in mc.iterrows():
            values = ""
            for s in syms_mc:
                v = row[s]
                if pd.isna(v):
                    values += f"{'N/A':>9}"
                else:
                    values += f"{v:>8.0f}%"
            print(f"  {month:>10}{values}")

    # ── Step 7: リスクステージング分析 ───────────────────────────
    print("\n" + "="*120)
    print("  [Step 7] リスクステージング — フォワードテスト計画")
    print("-"*80)

    for stage in RISK_STAGES:
        risk_pct = stage["risk_pct"]
        stage_syms = stage["syms"]
        label = f"Phase{stage['phase']} ({risk_pct*100:.1f}%/銘柄)"
        print(f"\n  【{label}】{stage['desc']}")
        print(f"    対象銘柄: {', '.join(stage_syms)}")

        port_stage = simulate_risk_stage(sym_data_map, stage_syms, risk_pct, label)
        if port_stage:
            print(f"    合成 PF     = {port_stage['port_pf']:.3f}")
            print(f"    合成 Sharpe = {port_stage['port_sharpe']:.2f}")
            print(f"    合成 MDD    = {port_stage['port_mdd']:.1f}%")
            print(f"    月次プラス  = {port_stage['port_plus_m']}/{port_stage['total_months']}ヶ月")
            # 期待月次リターン（資産100万円×リスク率×銘柄数）
            expected_monthly = INIT_CASH * risk_pct * len(stage_syms) * 0.3  # 概算
            print(f"    期待月次収益 ≈ {expected_monthly:,.0f}円（概算）")

    # ── Step 8: 実運用優先順位まとめ ─────────────────────────────
    print("\n" + "="*120)
    print("  [Step 8] 実運用優先順位 & 導入ロードマップ")
    print("  （AIフィードバック反映: 再現性・Sharpe・OOS/IS安定性で優先度を決定）")
    print("-"*100)
    print(f"  {'優先':>4} {'銘柄':8} {'カテゴリ':10} {'ロジック武器':20} {'OOS Sharpe':>12} "
          f"{'OOS PF':>8} {'OOS/IS':>8} {'導入フェーズ'}")
    print("-"*100)

    phase_map = {
        "USDJPY": "Phase1（即導入）", "GBPUSD": "Phase1（即導入）",
        "EURUSD": "Phase1（即導入）", "USDCAD": "Phase1（即導入）",
        "NZDUSD": "Phase2（3ヶ月後）", "XAUUSD": "Phase2（別カテゴリ）",
        "AUDUSD": "Phase2（月次安定確認後）",
    }

    for tgt in ADOPTED:
        sym = tgt["sym"]
        if sym not in indiv_stats:
            continue
        st_oos = indiv_stats[sym]["oos"]
        st_is  = indiv_stats[sym]["is"]
        if not st_oos:
            continue
        pf    = st_oos.get("pf", 0)
        sharpe= st_oos.get("sharpe", 0)
        is_pf = st_is.get("pf", 0) if st_is else 0
        ois   = pf / is_pf if is_pf > 0 else 0
        lname = LOGIC_NAMES[tgt["logic"]]
        pf_s  = f"{pf:.2f}" if pf < 99 else "∞"
        phase = phase_map.get(sym, "要検討")
        print(f"  #{tgt['priority']:<3} {sym:8} {tgt['cat']:10} Logic-{tgt['logic']}:{lname:12} "
              f"{sharpe:>12.2f} {pf_s:>8} {ois:>8.2f} {phase}")

    # ── Step 9: 総括 ──────────────────────────────────────────────
    print("\n" + "="*120)
    print("  [Step 9] 総括 & 推奨アクション")
    print("-"*80)

    # ポートフォリオ合成指標の評価
    if port_all:
        print(f"\n  ■ ポートフォリオ品質評価（全7銘柄）")
        pf_score  = "✅" if port_all["port_pf"] >= 1.3 else "❌"
        sh_score  = "✅" if port_all["port_sharpe"] >= 2.0 else "❌"
        mdd_score = "✅" if port_all["port_mdd"] <= 25.0 else "⚠️"
        pm_score  = "✅" if port_all["port_plus_m"] / max(port_all["total_months"],1) >= 0.70 else "❌"
        print(f"    {pf_score} 合成PF={port_all['port_pf']:.2f}  （基準≥1.30）")
        print(f"    {sh_score} 合成Sharpe={port_all['port_sharpe']:.2f}  （基準≥2.00）")
        print(f"    {mdd_score} 合成MDD={port_all['port_mdd']:.1f}%  （基準≤25%）")
        print(f"    {pm_score} 月次プラス={port_all['port_plus_m']}/{port_all['total_months']}ヶ月")

    print(f"\n  ■ 推奨アクション")
    print(f"    Step1: USDJPY/GBPUSD/EURUSD/USDCAD の4銘柄でPhase1開始（0.5%/銘柄）")
    print(f"    Step2: 3ヶ月後にPhase2へ（0.5%→1.0%、NZDUSD/XAUUSD追加）")
    print(f"    Step3: さらに3ヶ月後にPhase3（1.0%→2.0%、AUDUSD追加）")
    print(f"    Note:  XAUUSDはFX群と別枠で管理（相関・エクスポージャ独立性のため）")
    print(f"    Note:  USD集中リスクに注意（EURUSD/GBPUSD/USDCAD/NZDUSDが相関する可能性）")

    # ── CSV保存 ───────────────────────────────────────────────────
    rows = []
    for tgt in ADOPTED:
        sym = tgt["sym"]
        if sym not in indiv_stats:
            continue
        st_oos = indiv_stats[sym]["oos"]
        st_is  = indiv_stats[sym]["is"]
        if not st_oos:
            continue
        pf    = st_oos.get("pf", 0)
        is_pf = st_is.get("pf", 0) if st_is else 0
        rows.append({
            "priority": tgt["priority"],
            "sym":      sym,
            "cat":      tgt["cat"],
            "logic":    tgt["logic"],
            "lname":    LOGIC_NAMES[tgt["logic"]],
            "oos_n":    st_oos.get("n", 0),
            "oos_wr":   st_oos.get("wr", 0),
            "oos_pf":   pf,
            "oos_sharpe": st_oos.get("sharpe", 0),
            "oos_kelly":  st_oos.get("kelly", 0),
            "oos_mdd":  st_oos.get("mdd", 0),
            "oos_plus_m": st_oos.get("plus_m", 0),
            "oos_total_m": st_oos.get("total_m", 0),
            "is_pf":    is_pf,
            "ois_ratio": pf / is_pf if is_pf > 0 else 0,
            "pf_pass":  pf >= CRITERIA["oos_pf_min"],
            "sharpe_pass": st_oos.get("sharpe", 0) >= CRITERIA["sharpe_min"],
            "ois_pass": (pf / is_pf if is_pf > 0 else 0) >= CRITERIA["oos_is_min"],
            "pm_pass":  (st_oos.get("plus_m", 0) / max(st_oos.get("total_m", 1), 1)) >= CRITERIA["plus_m_min"],
            "mdd_pass": st_oos.get("mdd", 0) <= CRITERIA["mdd_max"],
            "phase":    phase_map.get(sym, "要検討"),
        })
        if port_all:
            rows[-1].update({
                "port7_pf":     port_all["port_pf"],
                "port7_sharpe": port_all["port_sharpe"],
                "port7_mdd":    port_all["port_mdd"],
            })

    out = os.path.join(OUT_DIR, "backtest_portfolio_integration.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  結果保存: {out}")
    print("="*120)


if __name__ == "__main__":
    main()
