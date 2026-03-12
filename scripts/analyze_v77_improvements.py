"""
analyze_v77_improvements.py
============================
v77上振れ5銘柄の定量・計量分析 → フィルター改善 → IS/OOS検証

【対象】EURUSD / AUDUSD / GBPUSD / USDCHF / USDJPY
【分析軸】時間帯 / ADX / ATR比率 / EMA距離 / 方向 / 曜日
【フィルター候補】有意差が出た軸でカットオフを設定→IS/OOS再バックテスト
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import AdaptiveRiskManager, SYMBOL_CONFIG

# ── 設定 ─────────────────────────────────────────────────────────
IS_START     = "2025-01-01"
IS_END       = "2025-06-30"   # 前半6ヶ月 IS
OOS_START    = "2025-07-01"
OOS_END      = "2026-02-28"   # 後半8ヶ月 OOS
FULL_START   = "2025-01-01"
FULL_END     = "2026-02-28"

INIT_CASH    = 100.0
BASE_RISK    = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 20_000
KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    {"name": "EURUSD", "cat": "FX",  "v77_pf": 1.96},
    {"name": "AUDUSD", "cat": "FX",  "v77_pf": 2.05},
    {"name": "GBPUSD", "cat": "FX",  "v77_pf": 2.13},
    {"name": "USDCHF", "cat": "FX",  "v77_pf": 2.08},
    {"name": "USDJPY", "cat": "JPY", "v77_pf": 2.00},
]

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tc = next((c for c in ["timestamp","datetime"] if c in df.columns), df.columns[0])
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc:"timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def resample_ohlcv(df, rule):
    cols = {"open":"first","high":"max","low":"min","close":"last"}
    if "volume" in df.columns: cols["volume"] = "sum"
    return df.resample(rule).agg(cols).dropna(subset=["open","close"])

def load_all(sym):
    def _f(tf):
        p = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
        return load_csv(p) if os.path.exists(p) else None
    d1m  = _f("1m")
    d4h_raw = _f("4h"); d4h = d4h_raw if d4h_raw is not None else (resample_ohlcv(d1m,"4h") if d1m is not None else None)
    d15m_raw= _f("15m"); d15m= d15m_raw if d15m_raw is not None else (resample_ohlcv(d1m,"15min") if d1m is not None else None)
    return d1m, d15m, d4h

def slice_period(df, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    high = df["high"]; low = df["low"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm.shift().fillna(0)] = 0.0
    tr  = pd.concat([high-low,(high-df["close"].shift()).abs(),(low-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr_w   = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm_w   = plus_dm.ewm(alpha=1/n, adjust=False).mean()
    mdm_w   = minus_dm.ewm(alpha=1/n, adjust=False).mean()
    di_p = 100 * pdm_w / atr_w.replace(0, np.nan)
    di_m = 100 * mdm_w / atr_w.replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)

def build_4h(df4h):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calc_adx(df, 14)
    df["ema_dist_atr"] = (df["close"] - df["ema20"]).abs() / df["atr"].replace(0, np.nan)
    return df

def build_1h(df15m):
    cols = {"open":"first","high":"max","low":"min","close":"last"}
    if "volume" in df15m.columns: cols["volume"] = "sum"
    df = df15m.resample("1h").agg(cols).dropna(subset=["open","close"])
    df["atr"] = calc_atr(df,14)
    df["ema20"] = df["close"].ewm(span=20,adjust=False).mean()
    return df

# ── フィルター ────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction==1 and bar["close"]>bar["open"]) or \
           (direction==-1 and bar["close"]<bar["open"])

def check_klow(bar):
    o = bar["open"]
    return (min(bar["open"],bar["close"])-bar["low"])/o < KLOW_THR if o>0 else False

def check_ema_dist(bar):
    d=abs(bar["close"]-bar["ema20"]); a=bar["atr"]
    return not pd.isna(a) and a>0 and d>=a*A1_EMA_DIST_MIN

def pick_e2(signal_time, direction, spread, atr_1m_d, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(signal_time, side="left")
    e = idx.searchsorted(signal_time + pd.Timedelta(minutes=max(2,E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        br = m1c["highs"][i] - m1c["lows"][i]
        av = atr_1m_d.get(idx[i], np.nan)
        if not np.isnan(av) and br > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (spread if direction==1 else -spread)
    return None, None

# ── シグナル生成（v77 + 特徴量付き） ─────────────────────────────
def generate_signals_v77_with_features(d1m, d15m, d4h_full, spread, atr_1m_d, m1c,
                                        adx_min=None, adx_max=None,
                                        hour_ok=None, ema_dist_min=None):
    """
    adx_min/adx_max: ADXフィルター（Noneなら無効）
    hour_ok: list of OK UTC hours（Noneなら無効）
    ema_dist_min: EMA距離ATR比（A1_EMA_DIST_MINの代替）
    """
    d4h = build_4h(d4h_full)
    d1h = build_1h(d15m)

    signals = []; used = set()
    h1_times = d1h.index.tolist()
    ema_thr = ema_dist_min if ema_dist_min is not None else A1_EMA_DIST_MIN

    for i in range(2, len(h1_times)):
        h1_ct  = h1_times[i]
        h1_p1  = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0: continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < 2: continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)): continue
        trend  = h4_lat["trend"]; h4_atr = h4_lat["atr"]
        adx    = h4_lat.get("adx", 0)
        ema_da = h4_lat.get("ema_dist_atr", 0)

        # ADXフィルター
        if adx_min is not None and adx < adx_min: continue
        if adx_max is not None and adx > adx_max: continue

        # 時間帯フィルター
        if hour_ok is not None and h1_ct.hour not in hour_ok: continue

        # コアフィルター
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat):        continue
        d = abs(h4_lat["close"]-h4_lat["ema20"]); a = h4_lat["atr"]
        if pd.isna(a) or a<=0 or d < a*ema_thr: continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction==1: v1,v2 = h1_p2["low"],h1_p1["low"]
        else:            v1,v2 = h1_p2["high"],h1_p1["high"]
        if abs(v1-v2) > tol: continue

        et, ep = pick_e2(h1_ct, direction, spread, atr_1m_d, m1c)
        if et is None or et in used: continue

        raw = ep - spread if direction==1 else ep + spread
        if direction==1: sl=min(v1,v2)-atr_1h*0.15; risk=raw-sl
        else:            sl=max(v1,v2)+atr_1h*0.15; risk=sl-raw

        if 0 < risk <= h4_atr*2:
            tp = raw + direction*risk*RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep,
                            "sl": sl, "tp": tp, "risk": risk,
                            "hour": h1_ct.hour, "dow": h1_ct.weekday(),
                            "adx": adx, "ema_dist_atr": ema_da,
                            "atr_1h": atr_1h})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── 出口探索 ─────────────────────────────────────────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half = ep + direction*risk*HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h=highs[i]; lo=lows[i]
        if direction==1:
            if lo<=sl: return i,sl,"loss",False
            if h>=tp:  return i,tp,"win",False
            if h>=half:
                be=ep
                for j in range(i+1,lim):
                    if lows[j]<=be:   return j,be,"win",True
                    if highs[j]>=tp:  return j,tp,"win",True
                return -1,None,None,True
        else:
            if h>=sl:  return i,sl,"loss",False
            if lo<=tp: return i,tp,"win",False
            if lo<=half:
                be=ep
                for j in range(i+1,lim):
                    if highs[j]>=be:  return j,be,"win",True
                    if lows[j]<=tp:   return j,tp,"win",True
                return -1,None,None,True
    return -1,None,None,False

# ── シミュレーション（特徴量付き） ─────────────────────────────────
def simulate_with_features(signals, d1m, sym):
    if not signals: return pd.DataFrame(), INIT_CASH, 0

    arm  = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    m1t  = d1m.index; m1h=d1m["high"].values; m1l=d1m["low"].values
    eq   = INIT_CASH; peak=INIT_CASH; mdd=0.0; trades=[]
    arm.update_peak(eq)

    for sig in signals:
        lot, _, _ = arm.calc_lot_adaptive(eq, sig["risk"], sig["ep"], USDJPY_RATE)
        sp = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        ei,xp,result,half_done = _find_exit(m1h[sp:],m1l[sp:],
            sig["ep"],sig["sl"],sig["tp"],sig["risk"],sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"]+sig["dir"]*sig["risk"]*HALF_R
            half_pnl = arm.calc_pnl_jpy(sig["dir"],sig["ep"],hp,lot*0.5,USDJPY_RATE,sig["ep"])
            eq += half_pnl; arm.update_peak(eq); rem=lot*0.5
        else: rem=lot

        pnl  = arm.calc_pnl_jpy(sig["dir"],sig["ep"],xp,rem,USDJPY_RATE,sig["ep"])
        eq  += pnl; arm.update_peak(eq)
        total= half_pnl+pnl
        win  = 1 if total>0 else 0

        trades.append({**{k:sig[k] for k in ["time","hour","dow","adx","ema_dist_atr","atr_1h","dir"]},
                       "result":result, "pnl":total, "win":win, "equity":eq})
        peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100)

    return pd.DataFrame(trades), eq, mdd

# ── 定量分析 ─────────────────────────────────────────────────────
def chi2_test(sub_wins, sub_n, total_wins, total_n):
    """サブグループのWRと全体WRのカイ二乗検定"""
    if sub_n < 5: return 1.0, sub_wins/sub_n if sub_n>0 else 0
    exp_w   = total_wins/total_n * sub_n
    exp_l   = (total_n-total_wins)/total_n * sub_n
    obs     = [sub_wins, sub_n-sub_wins]
    exp     = [exp_w, exp_l]
    if exp_l < 1 or exp_w < 1: return 1.0, obs[0]/sub_n
    chi2, p = stats.chisquare(obs, exp)
    return p, sub_wins/sub_n

def analyze_symbol(df, sym):
    """取引履歴の定量分析、改善提案を返す"""
    if df.empty: return {}

    n_total = len(df); wr_base = df["win"].mean()
    findings = []

    # ── 1. 時間帯別分析 ──────────────────────────────────
    hour_stats = []
    for h in range(24):
        sub = df[df["hour"]==h]
        if len(sub) < 5: continue
        p, wr = chi2_test(sub["win"].sum(), len(sub), df["win"].sum(), n_total)
        hour_stats.append({"hour":h, "n":len(sub), "wr":wr, "p":p,
                           "wr_diff": wr-wr_base})

    hour_df = pd.DataFrame(hour_stats) if hour_stats else pd.DataFrame()
    if not hour_df.empty:
        bad_hours  = hour_df[(hour_df["p"]<0.1) & (hour_df["wr_diff"]<-0.05)]["hour"].tolist()
        good_hours = hour_df[(hour_df["p"]<0.1) & (hour_df["wr_diff"]>0.05)]["hour"].tolist()
        if bad_hours:
            findings.append({
                "type": "hour_exclude",
                "desc": f"UTC {sorted(bad_hours)} 帯で勝率低下（有意）",
                "bad_hours": bad_hours,
                "hour_df": hour_df
            })

    # ── 2. ADX別分析 ────────────────────────────────────
    adx_cuts = [0, 15, 20, 30, 40, 100]
    adx_stats = []
    for i in range(len(adx_cuts)-1):
        lo, hi = adx_cuts[i], adx_cuts[i+1]
        sub = df[(df["adx"]>=lo) & (df["adx"]<hi)]
        if len(sub) < 5: continue
        p, wr = chi2_test(sub["win"].sum(), len(sub), df["win"].sum(), n_total)
        adx_stats.append({"range":f"{lo}-{hi}", "adx_lo":lo, "adx_hi":hi,
                          "n":len(sub), "wr":wr, "p":p, "wr_diff":wr-wr_base})

    adx_df = pd.DataFrame(adx_stats) if adx_stats else pd.DataFrame()
    if not adx_df.empty:
        bad_adx = adx_df[(adx_df["p"]<0.1) & (adx_df["wr_diff"]<-0.05)]
        if not bad_adx.empty:
            cutoff = bad_adx["adx_hi"].min()
            findings.append({
                "type": "adx_min",
                "desc": f"ADX<{cutoff} 帯で勝率低下（有意）→ ADX≥{cutoff}フィルター推奨",
                "adx_min": cutoff,
                "adx_df": adx_df
            })

    # ── 3. EMA距離別分析 ──────────────────────────────────
    ema_cuts = [0, 1.0, 1.5, 2.0, 3.0, 99]
    ema_stats = []
    for i in range(len(ema_cuts)-1):
        lo, hi = ema_cuts[i], ema_cuts[i+1]
        sub = df[(df["ema_dist_atr"]>=lo) & (df["ema_dist_atr"]<hi)]
        if len(sub) < 5: continue
        p, wr = chi2_test(sub["win"].sum(), len(sub), df["win"].sum(), n_total)
        ema_stats.append({"range":f"{lo}-{hi}", "ema_lo":lo, "ema_hi":hi,
                          "n":len(sub), "wr":wr, "p":p, "wr_diff":wr-wr_base})

    ema_df = pd.DataFrame(ema_stats) if ema_stats else pd.DataFrame()
    if not ema_df.empty:
        bad_ema = ema_df[(ema_df["p"]<0.1) & (ema_df["wr_diff"]<-0.05)]
        good_ema = ema_df[(ema_df["p"]<0.1) & (ema_df["wr_diff"]>0.05)]
        if not bad_ema.empty:
            cutoff = bad_ema["ema_hi"].max()
            findings.append({
                "type": "ema_dist_min",
                "desc": f"EMA距離<{cutoff}ATR帯で勝率低下 → 距離≥{cutoff}ATRフィルター推奨",
                "ema_dist_min": cutoff,
                "ema_df": ema_df
            })

    # ── 4. 方向別分析 ──────────────────────────────────────
    for d, label in [(1,"Long"), (-1,"Short")]:
        sub = df[df["dir"]==d]
        if len(sub) < 10: continue
        p, wr = chi2_test(sub["win"].sum(), len(sub), df["win"].sum(), n_total)
        if p < 0.05 and wr < wr_base - 0.08:
            findings.append({
                "type": "dir_exclude",
                "desc": f"{label}方向の勝率が有意に低い（{wr*100:.1f}% vs 基準{wr_base*100:.1f}%、p={p:.3f}）",
                "exclude_dir": d
            })

    # ── 5. 曜日別分析 ──────────────────────────────────────
    dow_names = ["Mon","Tue","Wed","Thu","Fri"]
    dow_stats = []
    for d in range(5):
        sub = df[df["dow"]==d]
        if len(sub) < 5: continue
        p, wr = chi2_test(sub["win"].sum(), len(sub), df["win"].sum(), n_total)
        dow_stats.append({"dow":d, "name":dow_names[d], "n":len(sub),
                          "wr":wr, "p":p, "wr_diff":wr-wr_base})

    return {
        "n": n_total, "wr_base": wr_base,
        "findings": findings,
        "hour_df": hour_df,
        "adx_df":  adx_df,
        "ema_df":  ema_df,
        "dow_df":  pd.DataFrame(dow_stats) if dow_stats else pd.DataFrame()
    }

# ── フィルター適用バックテスト ─────────────────────────────────────
def run_filtered(sym, d1m_full, d15m_full, d4h_full, filters, start, end):
    cfg    = SYMBOL_CONFIG.get(sym, {})
    spread = cfg.get("spread",0)*cfg.get("pip",0.0001)
    d1m  = slice_period(d1m_full,  start, end)
    d15m = slice_period(d15m_full, start, end)
    d4h  = slice_period(d4h_full,  "2024-01-01", end)
    if len(d1m)==0: return {}, [], 0, 0

    atr_1m = calc_atr(d1m, 10).to_dict()
    m1c = {"idx":d1m.index, "opens":d1m["open"].values,
           "closes":d1m["close"].values, "highs":d1m["high"].values,
           "lows":d1m["low"].values}

    sigs = generate_signals_v77_with_features(
        d1m, d15m, d4h, spread, atr_1m, m1c,
        adx_min=filters.get("adx_min"),
        hour_ok=filters.get("hour_ok"),
        ema_dist_min=filters.get("ema_dist_min")
    )
    df_trades, final_eq, mdd = simulate_with_features(sigs, d1m, sym)
    if df_trades.empty: return {}, [], final_eq, mdd

    n   = len(df_trades)
    wr  = df_trades["win"].mean()
    gw  = df_trades[df_trades["pnl"]>0]["pnl"].sum()
    gl  = abs(df_trades[df_trades["pnl"]<0]["pnl"].sum())
    pf  = gw/gl if gl>0 else float("inf")
    df_trades["month"] = pd.to_datetime(df_trades["time"]).dt.to_period("M")
    monthly = df_trades.groupby("month")["pnl"].sum()
    plus_m  = (monthly>0).sum()
    avg_w   = df_trades[df_trades["pnl"]>0]["pnl"].mean() if (df_trades["pnl"]>0).any() else 0
    avg_l   = abs(df_trades[df_trades["pnl"]<0]["pnl"].mean()) if (df_trades["pnl"]<0).any() else 1
    kelly   = wr - (1-wr)/(avg_w/avg_l) if avg_l>0 else 0

    ec = df_trades[["time","equity"]].values.tolist()
    return {"n":n,"wr":wr,"pf":pf,"mdd":mdd,"kelly":kelly,
            "plus_months":plus_m,"total_months":len(monthly),
            "final_eq":final_eq,"multiple":final_eq/INIT_CASH}, ec, final_eq, mdd

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  v77上振れ銘柄 定量・計量分析 → フィルター改善 → IS/OOS検証")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print("="*72)

    all_data     = {}  # sym -> (d1m, d15m, d4h)
    all_analysis = {}  # sym -> analysis result
    all_filters  = {}  # sym -> best filter dict

    # ── フェーズ1: IS期間で全取引データ収集 → 分析 ──────────────
    print("\n【フェーズ1】IS期間データ収集 + 定量分析")
    print("-"*72)

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        d1m_full, d15m_full, d4h_full = load_all(sym)
        if d1m_full is None: continue
        all_data[sym] = (d1m_full, d15m_full, d4h_full)

        cfg    = SYMBOL_CONFIG.get(sym, {})
        spread = cfg.get("spread",0)*cfg.get("pip",0.0001)
        d1m_is  = slice_period(d1m_full,  IS_START, IS_END)
        d15m_is = slice_period(d15m_full, IS_START, IS_END)
        d4h_is  = slice_period(d4h_full,  "2024-01-01", IS_END)
        atr_1m  = calc_atr(d1m_is, 10).to_dict()
        m1c     = {"idx":d1m_is.index, "opens":d1m_is["open"].values,
                   "closes":d1m_is["close"].values, "highs":d1m_is["high"].values,
                   "lows":d1m_is["low"].values}

        print(f"\n  [{sym}] IS シグナル生成中...", end=" ", flush=True)
        sigs = generate_signals_v77_with_features(d1m_is, d15m_is, d4h_is, spread, atr_1m, m1c)
        print(f"{len(sigs)}シグナル  シミュレーション中...", end=" ", flush=True)
        df_t, feq, mdd = simulate_with_features(sigs, d1m_is, sym)
        print(f"完了（{len(df_t)}トレード）")

        if df_t.empty: continue

        # 定量分析実行
        ana = analyze_symbol(df_t, sym)
        all_analysis[sym] = ana

        wr_pct = ana["wr_base"]*100
        print(f"  基準WR: {wr_pct:.1f}%  発見: {len(ana['findings'])}件")
        for f in ana["findings"]:
            print(f"    → {f['desc']}")

        # フィルター決定（ISデータ非依存の原則: 統計的に有意なものだけ）
        filt = {}
        for finding in ana["findings"]:
            ft = finding["type"]
            if ft == "hour_exclude":
                all_hours = list(range(24))
                ok_hours  = [h for h in all_hours if h not in finding["bad_hours"]]
                filt["hour_ok"] = ok_hours
            elif ft == "adx_min":
                filt["adx_min"] = finding["adx_min"]
            elif ft == "ema_dist_min":
                filt["ema_dist_min"] = finding["ema_dist_min"]
        all_filters[sym] = filt

    # ── フェーズ2: IS/OOS でフィルター効果検証 ───────────────────
    print("\n\n【フェーズ2】IS/OOS フィルター効果検証")
    print("="*72)

    final_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        if sym not in all_data: continue
        d1m_full, d15m_full, d4h_full = all_data[sym]
        filt = all_filters.get(sym, {})
        v77_pf_base = sym_info["v77_pf"]

        print(f"\n  [{sym}]  適用フィルター: {filt if filt else 'なし（改善余地なし）'}")

        # v77ベース（フィルターなし）
        is_base,  _, _, _ = run_filtered(sym, d1m_full, d15m_full, d4h_full, {}, IS_START, IS_END)
        oos_base, _, _, _ = run_filtered(sym, d1m_full, d15m_full, d4h_full, {}, OOS_START, OOS_END)

        # v77改善版（フィルターあり）
        is_imp,  _, _, _ = run_filtered(sym, d1m_full, d15m_full, d4h_full, filt, IS_START, IS_END)
        oos_imp, ec_oos, feq, mdd = run_filtered(sym, d1m_full, d15m_full, d4h_full, filt, OOS_START, OOS_END)

        def fmt(st):
            if not st: return "N/A"
            pf_s = f"{st['pf']:.2f}" if st.get('pf',0)<99 else "∞"
            return f"n={st['n']} WR={st['wr']*100:.1f}% PF={pf_s} MDD={st['mdd']:.1f}% K={st['kelly']:.3f} {st['multiple']:.1f}x"

        print(f"    IS  base: {fmt(is_base)}")
        print(f"    IS  imp:  {fmt(is_imp)}")
        print(f"    OOS base: {fmt(oos_base)}")
        print(f"    OOS imp:  {fmt(oos_imp)}")

        passed = oos_imp and oos_imp.get("pf",0)>=2.0 and oos_imp.get("wr",0)>=0.65 \
                 and oos_imp.get("mdd",99)<=20.0 and oos_imp.get("kelly",0)>=0.45

        # IS/OOS乖離チェック
        if is_imp and oos_imp and is_imp.get("pf",0)>0:
            pf_gap = oos_imp.get("pf",0) / is_imp.get("pf",0)
            overfit = "✅" if pf_gap >= 0.7 else "⚠️過学習疑い"
            print(f"    IS/OOS PF比: {pf_gap:.2f} {overfit}")

        final_results.append({
            "sym": sym, "cat": sym_info["cat"],
            "v77_pf": v77_pf_base,
            "is_base": is_base, "oos_base": oos_base,
            "is_imp": is_imp, "oos_imp": oos_imp,
            "filters": filt, "passed": passed,
            "ec_oos": ec_oos,
            "findings": all_analysis.get(sym, {}).get("findings", [])
        })

    # ── フェーズ3: 結果サマリー ──────────────────────────────────
    print("\n\n" + "="*72)
    print("  ■ 最終サマリー（OOS比較）")
    print(f"  {'銘柄':8} {'v77 PF':>7} {'改善 PF':>8} {'変化':>6} {'WR':>6} {'Kelly':>7} {'MDD':>6} {'判定':>6}")
    print("-"*72)

    adopted = []
    for r in final_results:
        sym    = r["sym"]
        v77pf  = r["v77_pf"]
        oi     = r["oos_imp"]
        if not oi:
            print(f"  {sym:8} {v77pf:>7.2f} {'N/A':>8}"); continue
        new_pf = oi.get("pf", 0)
        diff   = new_pf - v77pf
        arrow  = "↑" if diff>0 else "↓"
        verdict= "✅採用" if r["passed"] else ("⚠️惜" if new_pf>=1.8 else "❌却下")
        if r["passed"]: adopted.append(sym)
        pf_s = f"{new_pf:.2f}" if new_pf<99 else "∞"
        print(f"  {sym:8} {v77pf:>7.2f} {pf_s:>8} {arrow}{abs(diff):>4.2f} "
              f"{oi['wr']*100:>5.1f}% {oi['kelly']:>6.3f} {oi['mdd']:>5.1f}% {verdict}")

    print()
    if adopted:
        print(f"  ✅ 採用確定: {', '.join(adopted)}")
    else:
        print("  採用基準フルPASS銘柄なし（フィルター別効果サマリー上記参照）")

    print("\n  ■ 適用フィルター一覧")
    for r in final_results:
        sym  = r["sym"]; f = r["filters"]
        desc_parts = []
        if "hour_ok" in f:
            excluded = [h for h in range(24) if h not in f["hour_ok"]]
            desc_parts.append(f"UTC {excluded} 除外")
        if "adx_min" in f:
            desc_parts.append(f"ADX≥{f['adx_min']}")
        if "ema_dist_min" in f:
            desc_parts.append(f"EMA距離≥{f['ema_dist_min']}ATR")
        filt_str = " + ".join(desc_parts) if desc_parts else "なし"
        print(f"  {sym:8}: {filt_str}")

    # ── グラフ生成 ────────────────────────────────────────────────
    print("\n  グラフ生成中...")
    _plot_analysis(final_results, all_analysis)

    # CSV保存
    rows = []
    for r in final_results:
        oi = r.get("oos_imp") or {}
        ob = r.get("oos_base") or {}
        rows.append({
            "symbol": r["sym"], "v77_pf_base": r["v77_pf"],
            "oos_base_pf": round(ob.get("pf",0),3),
            "oos_imp_pf":  round(oi.get("pf",0),3),
            "oos_imp_wr":  round(oi.get("wr",0)*100,1),
            "oos_imp_mdd": round(oi.get("mdd",0),2),
            "oos_imp_kelly": round(oi.get("kelly",0),3),
            "oos_imp_multiple": round(oi.get("multiple",0),2),
            "adopted": r["passed"],
            "filters": str(r["filters"])
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"analyze_v77_improvements.csv"), index=False)
    print(f"  → results/analyze_v77_improvements.csv")
    print("\n完了")

def _plot_analysis(final_results, all_analysis):
    n = len(final_results)
    fig = plt.figure(figsize=(20, 5*n + 3))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("v77上振れ銘柄 定量分析 + フィルター改善結果\nIS/OOS: 2025-01〜06 / 2025-07〜2026-02",
                 color="white", fontsize=13, y=0.99)

    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.7, wspace=0.4)

    for row_i, r in enumerate(final_results):
        sym  = r["sym"]
        ana  = all_analysis.get(sym, {})
        oi   = r.get("oos_imp", {}) or {}
        ob   = r.get("oos_base", {}) or {}

        # ── 左: 時間帯WR ──────────────────────────────────
        ax1 = fig.add_subplot(gs[row_i, 0])
        ax1.set_facecolor("#16213e")
        hdf = ana.get("hour_df", pd.DataFrame())
        if not hdf.empty:
            colors = ["#e74c3c" if w < ana["wr_base"]-0.05 else
                      "#2ecc71" if w > ana["wr_base"]+0.05 else "#4C9BE8"
                      for w in hdf["wr"]]
            ax1.bar(hdf["hour"], hdf["wr"]*100, color=colors, alpha=0.8)
            ax1.axhline(ana["wr_base"]*100, color="#fff", linewidth=1, linestyle="--", alpha=0.6)
            ax1.set_title(f"{sym} 時間帯別WR（IS）", color="white", fontsize=8)
            ax1.set_xlabel("UTC hour", color="#aaa", fontsize=7)
            ax1.set_ylabel("WR%", color="#aaa", fontsize=7)
        else:
            ax1.text(0.5, 0.5, "データ不足", ha="center", va="center",
                     color="#aaa", transform=ax1.transAxes)
        ax1.tick_params(colors="#aaa", labelsize=7)
        for sp in ax1.spines.values(): sp.set_color("#333")

        # ── 中: ADX別WR ────────────────────────────────────
        ax2 = fig.add_subplot(gs[row_i, 1])
        ax2.set_facecolor("#16213e")
        adf = ana.get("adx_df", pd.DataFrame())
        if not adf.empty:
            colors = ["#e74c3c" if w < ana["wr_base"]-0.05 else
                      "#2ecc71" if w > ana["wr_base"]+0.05 else "#F5A623"
                      for w in adf["wr"]]
            ax2.bar(range(len(adf)), adf["wr"]*100, color=colors, alpha=0.8)
            ax2.axhline(ana["wr_base"]*100, color="#fff", linewidth=1, linestyle="--", alpha=0.6)
            ax2.set_xticks(range(len(adf)))
            ax2.set_xticklabels(adf["range"], rotation=30, fontsize=6)
            ax2.set_title(f"{sym} ADX帯別WR（IS）", color="white", fontsize=8)
            ax2.set_ylabel("WR%", color="#aaa", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "データ不足", ha="center", va="center",
                     color="#aaa", transform=ax2.transAxes)
        ax2.tick_params(colors="#aaa", labelsize=7)
        for sp in ax2.spines.values(): sp.set_color("#333")

        # ── 右: OOS エクイティ比較（base vs imp） ─────────────
        ax3 = fig.add_subplot(gs[row_i, 2])
        ax3.set_facecolor("#16213e")
        ec_oos = r.get("ec_oos", [])
        if ec_oos:
            times    = [e[0] for e in ec_oos]
            equities = [e[1] for e in ec_oos]
            ax3.plot(times, equities, color="#2ecc71", linewidth=1.2, label="v77改善")
            ax3.axhline(INIT_CASH, color="#555", linewidth=0.8, linestyle="--")
            ax3.fill_between(times, INIT_CASH, equities,
                             where=[e>=INIT_CASH for e in equities], alpha=0.18, color="#2ecc71")
        base_pf = ob.get("pf", 0); imp_pf = oi.get("pf", 0)
        bpf_s = f"{base_pf:.2f}" if base_pf<99 else "∞"
        ipf_s = f"{imp_pf:.2f}" if imp_pf<99 else "∞"
        passed = r.get("passed", False)
        ax3.set_title(
            f"{sym} OOS {'✅' if passed else '⚠️'}  PF: {bpf_s}→{ipf_s}\n"
            f"WR={oi.get('wr',0)*100:.1f}% MDD={oi.get('mdd',0):.1f}% K={oi.get('kelly',0):.3f}",
            color="white", fontsize=8, pad=4)
        ax3.tick_params(colors="#aaa", labelsize=7)
        for sp in ax3.spines.values(): sp.set_color("#333")
        ax3.set_ylabel("資産", color="#aaa", fontsize=7)

    out_path = os.path.join(OUT_DIR, "analyze_v77_improvements.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  → {out_path}")

if __name__ == "__main__":
    main()
