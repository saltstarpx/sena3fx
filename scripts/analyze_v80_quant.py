"""
analyze_v80_quant.py
=====================
v80バックテスト結果の定量・計量分析

【分析内容】
  1. スプレッドコスト確認
  2. 月次PF安定性（ウォークフォワード）
  3. Long/Short バイアス
  4. 結果の独立性検定（自己相関 / ランテスト）
  5. 月次・時間帯別勝率（季節性・時間帯バイアス）
  6. ブートストラップ信頼区間（PF / 勝率）
  7. モンテカルロ（最終資産の分布）
  8. MDD発生タイミング分析
  9. 問題点フラグ
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 定数（v80と同一）──────────────────────────────────────────────
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
MAX_LOOKAHEAD = 20_000
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3
OOS_START = "2025-12-01"
OOS_END   = "2026-02-28"

SYMBOLS = [
    {"name": "EURUSD", "lower": "eurusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "GBPUSD", "lower": "gbpusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "AUDUSD", "lower": "audusd", "category": "FX",
     "entry": "E1", "streak": 4, "use_1d": False},
    {"name": "XAUUSD", "lower": "xauusd", "category": "METALS",
     "entry": "E2", "streak": 0, "use_1d": True},
]

# ── データロード（v80と同一）──────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def load_data(sym_upper, sym_lower):
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")
    def _load(tf):
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p): return load_csv(p)
        p2 = os.path.join(DATA_DIR, f"{sym_lower}_{tf}.csv")
        if os.path.exists(p2): return load_csv(p2)
        return None
    return _load("1m"), _load("15m"), _load("4h")

def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(period).mean()

def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if need_1d:
        d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None

def build_1h(data_15m):
    df = data_15m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

def check_kmid(bar, direction):
    return (direction==1 and bar["close"]>bar["open"]) or (direction==-1 and bar["close"]<bar["open"])

def check_klow(bar):
    o,c,l = bar["open"],bar["close"],bar["low"]
    return (min(o,c)-l)/o < KLOW_THR if o>0 else False

def check_ema_dist(h4_bar):
    dist = abs(h4_bar["close"] - h4_bar["ema20"])
    atr  = h4_bar["atr"]
    if pd.isna(atr) or atr<=0: return False
    return dist >= atr * A1_EMA_DIST_MIN

def pick_entry_1m(signal_time, direction, spread, method, atr_1m, m1_cache):
    m1_idx = m1_cache["idx"]
    start  = m1_idx.searchsorted(signal_time, side="left")
    if method == "E1":
        end_time = signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            o=m1_cache["opens"][i]; c=m1_cache["closes"][i]
            if direction==1 and c<=o: continue
            if direction==-1 and c>=o: continue
            ni=i+1
            if ni>=len(m1_idx): return None,None
            return m1_idx[ni], m1_cache["opens"][ni]+(spread if direction==1 else -spread)
        return None,None
    else:
        win_min = max(2, E2_ALT_WINDOW_MIN)
        end_time = signal_time + pd.Timedelta(minutes=win_min)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            bar_time  = m1_idx[i]
            bar_range = m1_cache["highs"][i] - m1_cache["lows"][i]
            if atr_1m is not None:
                atr_val = atr_1m.get(bar_time, np.nan)
                if not np.isnan(atr_val) and bar_range > atr_val*E2_SPIKE_ATR_MULT: continue
            return bar_time, m1_cache["opens"][i]+(spread if direction==1 else -spread)
        return None,None

def generate_v80_signals(data_1m, data_15m, data_4h, spread_pips, pip_size, sym_cfg, atr_1m=None, m1_cache=None):
    spread=spread_pips*pip_size; streak=sym_cfg["streak"]; need_1d=sym_cfg["use_1d"]; method=sym_cfg["entry"]
    data_4h, data_1d = build_4h(data_4h, need_1d)
    data_1h = build_1h(data_15m)
    if m1_cache is None:
        m1_cache={"idx":data_1m.index,"opens":data_1m["open"].values,"closes":data_1m["close"].values,"highs":data_1m["high"].values,"lows":data_1m["low"].values}
    signals=[]; used_times=set(); h1_times=data_1h.index.tolist(); min_idx=max(2,streak if streak>0 else 2)
    for i in range(min_idx,len(h1_times)):
        h1_ct=h1_times[i]; h1_prev1=data_1h.iloc[i-1]; h1_prev2=data_1h.iloc[i-2]
        atr_val=data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val<=0: continue
        h4_before=data_4h[data_4h.index<h1_ct]
        if len(h4_before)<max(streak if streak>0 else 2,2): continue
        h4_latest=h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr",np.nan)): continue
        trend=h4_latest["trend"]; h4_atr=h4_latest["atr"]
        if streak>0:
            recent=h4_before["trend"].iloc[-streak:].values
            if not all(t==trend for t in recent): continue
        if need_1d and data_1d is not None:
            d1_before=data_1d[data_1d.index.normalize()<h1_ct.normalize()]
            if len(d1_before)==0: continue
            if d1_before.iloc[-1]["trend1d"]!=trend: continue
        if not check_kmid(h4_latest,trend): continue
        if not check_klow(h4_latest): continue
        if not check_ema_dist(h4_latest): continue
        tol=atr_val*A3_DEFAULT_TOL
        for direction in [1,-1]:
            if trend!=direction: continue
            if direction==1: v1,v2=h1_prev2["low"],h1_prev1["low"]
            else: v1,v2=h1_prev2["high"],h1_prev1["high"]
            if abs(v1-v2)>tol: continue
            et,ep=pick_entry_1m(h1_ct,direction,spread,method,atr_1m,m1_cache)
            if et is None or et in used_times: continue
            raw=ep-spread if direction==1 else ep+spread
            if direction==1: sl=min(v1,v2)-atr_val*0.15; risk=raw-sl
            else: sl=max(v1,v2)+atr_val*0.15; risk=sl-raw
            if 0<risk<=h4_atr*2:
                tp=raw+direction*risk*RR_RATIO
                signals.append({"time":et,"dir":direction,"ep":ep,"sl":sl,"tp":tp,"risk":risk,"signal_time":h1_ct})
                used_times.add(et)
    signals.sort(key=lambda x:x["time"])
    return signals

def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half_price=ep+direction*risk*HALF_R; limit=min(len(highs),MAX_LOOKAHEAD)
    for i in range(limit):
        h=highs[i]; lo=lows[i]
        if direction==1:
            if lo<=sl: return i,sl,"loss",False,-1
            if h>=tp: return i,tp,"win",False,-1
            if h>=half_price:
                be_sl=ep
                for j in range(i+1,limit):
                    if lows[j]<=be_sl: return j,be_sl,"win",True,i
                    if highs[j]>=tp: return j,tp,"win",True,i
                return -1,None,None,True,i
        else:
            if h>=sl: return i,sl,"loss",False,-1
            if lo<=tp: return i,tp,"win",False,-1
            if lo<=half_price:
                be_sl=ep
                for j in range(i+1,limit):
                    if highs[j]>=be_sl: return j,be_sl,"win",True,i
                    if lows[j]<=tp: return j,tp,"win",True,i
                return -1,None,None,True,i
    return -1,None,None,False,-1

def simulate_detailed(signals, data_1m, symbol):
    """トレード詳細（エントリー時刻・方向・損益・スプレッドコスト等）を返す"""
    if not signals: return [], [INIT_CASH]
    rm=RiskManager(symbol,risk_pct=RISK_PCT)
    cfg=SYMBOL_CONFIG[symbol]
    spread_price=cfg["spread"]*cfg["pip"]
    equity=INIT_CASH; trades=[]; eq_curve=[INIT_CASH]
    m1_times=data_1m.index; m1_highs=data_1m["high"].values; m1_lows=data_1m["low"].values

    for sig in signals:
        direction=sig["dir"]; ep=sig["ep"]; sl=sig["sl"]; tp=sig["tp"]; risk=sig["risk"]
        lot=rm.calc_lot(equity,risk,ep,usdjpy_rate=USDJPY_RATE)
        start_pos=m1_times.searchsorted(sig["time"],side="right")
        if start_pos>=len(m1_times): continue
        exit_i,exit_price,result,half_done,half_i=_find_exit(m1_highs[start_pos:],m1_lows[start_pos:],ep,sl,tp,risk,direction)
        if result is None: continue

        spread_cost = spread_price * lot * USDJPY_RATE  # スプレッドコスト（円）

        half_pnl=0.0
        if half_done and half_i>=0:
            half_ep=ep+direction*risk*HALF_R
            half_pnl=rm.calc_pnl_jpy(direction,ep,half_ep,lot*0.5,USDJPY_RATE,ep)
            equity+=half_pnl
            remaining_lot=lot*0.5
        else:
            remaining_lot=lot

        final_pnl=rm.calc_pnl_jpy(direction,ep,exit_price,remaining_lot,USDJPY_RATE,ep)
        equity+=final_pnl
        total_pnl=half_pnl+final_pnl
        exit_time=m1_times[start_pos+exit_i]

        trades.append({
            "entry_time":   sig["time"],
            "signal_time":  sig["signal_time"],
            "exit_time":    exit_time,
            "dir":          direction,
            "ep":           ep,
            "exit_price":   exit_price,
            "sl":           sl,
            "tp":           tp,
            "risk_pips":    risk,
            "result":       result,
            "half_done":    half_done,
            "pnl_jpy":      total_pnl,
            "spread_cost":  spread_cost,
            "lot":          lot,
            "equity_after": equity,
        })
        eq_curve.append(equity)

    return trades, eq_curve


# ── 分析関数群 ────────────────────────────────────────────────────

def analyze_spread_cost(trades, symbol):
    """スプレッドコスト分析"""
    df=pd.DataFrame(trades)
    total_spread=df["spread_cost"].sum()
    total_pnl_gross=df["pnl_jpy"].sum()
    total_pnl_net=total_pnl_gross  # spread already embedded in ep
    avg_spread=df["spread_cost"].mean()
    return {"total_spread_jpy":total_spread,"avg_spread_jpy":avg_spread,
            "spread_pct_of_equity":total_spread/INIT_CASH*100}


def bootstrap_pf_wr(outcomes, n_boot=5000, seed=42):
    """PFと勝率のブートストラップ信頼区間"""
    rng=np.random.default_rng(seed)
    wins =[p for p in outcomes if p>0]
    loses=[abs(p) for p in outcomes if p<0]
    pf_boots=[]; wr_boots=[]
    n=len(outcomes)
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        samp=[outcomes[i] for i in idx]
        w=sum(p>0 for p in samp); l=sum(p<0 for p in samp)
        gw=sum(p for p in samp if p>0); gl=abs(sum(p for p in samp if p<0))
        wr_boots.append(w/n)
        pf_boots.append(gw/gl if gl>0 else np.inf)
    pf_finite=[x for x in pf_boots if np.isfinite(x)]
    return {
        "pf_mean":np.mean(pf_finite),"pf_ci95":(np.percentile(pf_finite,2.5),np.percentile(pf_finite,97.5)),
        "wr_mean":np.mean(wr_boots),"wr_ci95":(np.percentile(wr_boots,2.5),np.percentile(wr_boots,97.5)),
    }


def runs_test(outcomes_binary):
    """
    ランテスト: トレード結果（勝=1/負=0）の独立性検定
    H0: 結果は独立（ランダム）
    """
    n=len(outcomes_binary); n1=sum(outcomes_binary); n0=n-n1
    if n1==0 or n0==0: return {"stat":np.nan,"p":np.nan,"verdict":"計算不可"}
    runs=1
    for i in range(1,n):
        if outcomes_binary[i]!=outcomes_binary[i-1]: runs+=1
    # 期待値・分散
    mu  = 2*n1*n0/n + 1
    var = 2*n1*n0*(2*n1*n0-n) / (n**2*(n-1))
    if var<=0: return {"stat":np.nan,"p":np.nan,"verdict":"計算不可"}
    z=(runs-mu)/np.sqrt(var)
    p=2*(1-stats.norm.cdf(abs(z)))
    verdict="✅ 独立（問題なし）" if p>0.05 else ("⚠️ 連勝/連敗パターンあり" if z<0 else "⚠️ 交互パターンあり")
    return {"runs":runs,"expected":mu,"z":z,"p":p,"verdict":verdict}


def autocorr_test(pnl_series, max_lag=5):
    """自己相関検定（Ljung-Box）"""
    from statsmodels.stats.stattools import durbin_watson
    arr=np.array(pnl_series)
    dw=durbin_watson(arr)
    # Ljung-Box は手計算
    n=len(arr); acfs=[]
    mean=arr.mean(); var=((arr-mean)**2).mean()
    if var==0: return {"dw":dw,"acfs":[],"lb_p":np.nan,"verdict":"計算不可"}
    for lag in range(1,max_lag+1):
        acf=np.mean((arr[:-lag]-mean)*(arr[lag:]-mean))/var
        acfs.append(acf)
    # Ljung-Box Q統計量
    Q=n*(n+2)*sum(acf**2/(n-k) for k,acf in enumerate(acfs,1))
    p=1-stats.chi2.cdf(Q,df=max_lag)
    verdict="✅ 独立（問題なし）" if p>0.05 else "⚠️ 自己相関あり（過去の結果が次に影響）"
    return {"dw":round(dw,3),"acfs":[round(a,3) for a in acfs],"lb_Q":round(Q,2),"lb_p":round(p,4),"verdict":verdict}


def monthly_analysis(trades):
    """月次PF・勝率の分布"""
    df=pd.DataFrame(trades)
    df["month"]=pd.to_datetime(df["exit_time"],utc=True).dt.to_period("M")
    monthly=[]
    for m,grp in df.groupby("month"):
        gw=grp[grp["pnl_jpy"]>0]["pnl_jpy"].sum()
        gl=abs(grp[grp["pnl_jpy"]<0]["pnl_jpy"].sum())
        pf=gw/gl if gl>0 else np.inf
        wr=(grp["result"]=="win").mean()
        monthly.append({"month":str(m),"n":len(grp),"wr":wr,"pf":pf})
    return pd.DataFrame(monthly)


def hourly_analysis(trades):
    """時間帯別勝率（UTC）"""
    df=pd.DataFrame(trades)
    df["hour"]=pd.to_datetime(df["entry_time"],utc=True).dt.hour
    result=[]
    for h,grp in df.groupby("hour"):
        result.append({"hour":h,"n":len(grp),"wr":(grp["result"]=="win").mean()})
    return pd.DataFrame(result)


def direction_analysis(trades):
    """Long/Short バイアス分析"""
    df=pd.DataFrame(trades)
    longs =df[df["dir"]==1];  shorts=df[df["dir"]==-1]
    def stats_(grp, label):
        if len(grp)==0: return {}
        gw=grp[grp["pnl_jpy"]>0]["pnl_jpy"].sum()
        gl=abs(grp[grp["pnl_jpy"]<0]["pnl_jpy"].sum())
        return {"dir":label,"n":len(grp),"wr":(grp["result"]=="win").mean()*100,
                "pf":round(gw/gl,2) if gl>0 else np.inf}
    l=stats_(longs,"Long"); s=stats_(shorts,"Short")
    # カイ二乗検定: Long vs Short の勝率差
    if len(longs)>0 and len(shorts)>0:
        w_l=(longs["result"]=="win").sum(); l_l=(longs["result"]=="loss").sum()
        w_s=(shorts["result"]=="win").sum(); l_s=(shorts["result"]=="loss").sum()
        chi2,p,_,_=stats.chi2_contingency([[w_l,l_l],[w_s,l_s]])
        l["chi2_p"]=round(p,4); s["chi2_p"]=round(p,4)
        if p<0.05: l["bias_flag"]="⚠️ 方向バイアスあり"; s["bias_flag"]="⚠️ 方向バイアスあり"
        else: l["bias_flag"]="✅ バイアスなし"; s["bias_flag"]="✅ バイアスなし"
    return l, s


def monte_carlo_final_equity(pnl_pct_list, n_trades, n_sim=10000, seed=42):
    """モンテカルロ: ランダム並び替えで最終資産分布"""
    rng=np.random.default_rng(seed)
    arr=np.array(pnl_pct_list)  # each element: pnl/equity_before ratio
    finals=[]
    for _ in range(n_sim):
        idx=rng.integers(0,len(arr),n_trades)
        eq=INIT_CASH
        for r in arr[idx]:
            eq*=(1+r)
        finals.append(eq)
    finals=np.array(finals)
    return {
        "median":np.median(finals),"mean":np.mean(finals),
        "p5":np.percentile(finals,5),"p95":np.percentile(finals,95),
        "p1":np.percentile(finals,1),"p99":np.percentile(finals,99),
        "prob_loss":np.mean(finals<INIT_CASH)*100,
    }


def mdd_timing_analysis(eq_curve):
    """MDDがいつ発生したか"""
    eq=np.array(eq_curve)
    peak=np.maximum.accumulate(eq)
    dd=(eq-peak)/peak
    mdd_idx=np.argmin(dd)
    mdd_pct=abs(dd[mdd_idx])*100
    return {"mdd_idx":mdd_idx,"mdd_pct":round(mdd_pct,1),"position_pct":round(mdd_idx/len(eq)*100,1)}


# ── メイン ───────────────────────────────────────────────────────
def main():
    all_symbol_results={}

    for sym_cfg in SYMBOLS:
        sym_name=sym_cfg["name"]; sym_lower=sym_cfg["lower"]
        print(f"\n{'='*65}")
        print(f"  {sym_name} データ読込・シグナル生成中...")
        print(f"{'='*65}")

        d1m,d15m,d4h=load_data(sym_name,sym_lower)
        if d1m is None or d15m is None or d4h is None:
            print("  [SKIP] データ不足"); continue

        d1m_oos =slice_period(d1m, OOS_START,OOS_END)
        d15m_oos=slice_period(d15m,OOS_START,OOS_END)
        d4h_buf =d4h

        if d1m_oos is None or len(d1m_oos)==0: continue

        cfg=SYMBOL_CONFIG[sym_name]
        spread_pips=cfg.get("spread",0.0); pip_size=cfg.get("pip",0.0001)

        atr_1m=calc_atr(d1m_oos,10).to_dict()
        m1_cache={"idx":d1m_oos.index,"opens":d1m_oos["open"].values,"closes":d1m_oos["close"].values,
                  "highs":d1m_oos["high"].values,"lows":d1m_oos["low"].values}

        sigs=generate_v80_signals(d1m_oos,d15m_oos,d4h_buf,spread_pips,pip_size,sym_cfg,atr_1m,m1_cache)
        trades,eq_curve=simulate_detailed(sigs,d1m_oos,sym_name)
        if not trades: continue

        all_symbol_results[sym_name]=(trades,eq_curve)
        print(f"  トレード数: {len(trades)}")

    # ── 銘柄別分析 ──────────────────────────────────────────────
    all_flags={}
    fig=plt.figure(figsize=(22,28))
    gs=gridspec.GridSpec(len(all_symbol_results),4,figure=fig,hspace=0.55,wspace=0.4)
    row_idx=0

    for sym_name,(trades,eq_curve) in all_symbol_results.items():
        df=pd.DataFrame(trades)
        df["entry_time"]=pd.to_datetime(df["entry_time"],utc=True)
        df["exit_time"]=pd.to_datetime(df["exit_time"],utc=True)
        cfg=SYMBOL_CONFIG[sym_name]
        spread_pips=cfg["spread"]

        flags=[]
        print(f"\n\n{'━'*65}")
        print(f"  【{sym_name}】定量・計量分析")
        print(f"{'━'*65}")

        # ── 1. スプレッドコスト ──────────────────────────────
        sc=analyze_spread_cost(trades,sym_name)
        print(f"\n■ スプレッドコスト")
        print(f"  スプレッド設定: {spread_pips} pips")
        print(f"  総スプレッドコスト: {sc['total_spread_jpy']:,.0f}円  (初期証拠金の{sc['spread_pct_of_equity']:.1f}%)")
        print(f"  1トレード平均: {sc['avg_spread_jpy']:,.0f}円")

        # ── 2. 基本統計 ─────────────────────────────────────
        pnl=df["pnl_jpy"].values
        wins=pnl[pnl>0]; loses=pnl[pnl<0]
        wr=len(wins)/len(pnl)
        pf=wins.sum()/abs(loses.sum()) if len(loses)>0 else np.inf
        avg_win=wins.mean() if len(wins)>0 else 0
        avg_lose=loses.mean() if len(loses)>0 else 0
        rr=abs(avg_win/avg_lose) if avg_lose!=0 else np.inf
        print(f"\n■ 基本統計")
        print(f"  n={len(pnl)}  WR={wr*100:.1f}%  PF={pf:.2f}  avgRR={rr:.2f}")
        print(f"  avg勝={avg_win:,.0f}円  avg負={avg_lose:,.0f}円")

        # ── 3. ブートストラップ信頼区間 ─────────────────────
        boot=bootstrap_pf_wr(list(pnl))
        print(f"\n■ ブートストラップ 95%CI (n=5000)")
        print(f"  PF: {boot['pf_mean']:.2f} [{boot['pf_ci95'][0]:.2f}, {boot['pf_ci95'][1]:.2f}]")
        print(f"  WR: {boot['wr_mean']*100:.1f}% [{boot['wr_ci95'][0]*100:.1f}%, {boot['wr_ci95'][1]*100:.1f}%]")
        if boot["pf_ci95"][0] < 1.0:
            flags.append("⚠️ PF CI下限<1.0（統計的有意でない可能性）")
            print(f"  → ⚠️ CI下限が1.0を下回る → 注意")
        else:
            print(f"  → ✅ CI下限>1.0（統計的に有意）")

        # ── 4. 独立性検定 ─────────────────────────────────────
        outcomes_bin=[1 if r=="win" else 0 for r in df["result"]]
        rt=runs_test(outcomes_bin)
        print(f"\n■ ランテスト（結果の独立性）")
        print(f"  runs={rt.get('runs')}  期待={rt.get('expected',0):.1f}  z={rt.get('z',0):.2f}  p={rt.get('p',0):.4f}")
        print(f"  → {rt['verdict']}")
        if rt.get("p",1.0)<0.05:
            flags.append(f"⚠️ ランテストp={rt.get('p',0):.4f}（結果に連依存パターン）")

        ac=autocorr_test(list(pnl))
        print(f"\n■ 自己相関検定（Ljung-Box, lag=5）")
        print(f"  DW={ac['dw']}  Q={ac.get('lb_Q')}  p={ac.get('lb_p')}")
        print(f"  lag1〜5 ACF: {ac['acfs']}")
        print(f"  → {ac['verdict']}")
        if ac.get("lb_p",1.0) is not None and ac.get("lb_p",1.0)<0.05:
            flags.append(f"⚠️ 自己相関検定p={ac.get('lb_p')}（損益に時系列依存）")

        # ── 5. Long/Short バイアス ────────────────────────────
        l_stat,s_stat=direction_analysis(trades)
        print(f"\n■ Long/Short バイアス")
        print(f"  Long:  n={l_stat.get('n',0)}  WR={l_stat.get('wr',0):.1f}%  PF={l_stat.get('pf',0)}")
        print(f"  Short: n={s_stat.get('n',0)}  WR={s_stat.get('wr',0):.1f}%  PF={s_stat.get('pf',0)}")
        print(f"  カイ二乗p={l_stat.get('chi2_p')}  → {l_stat.get('bias_flag')}")
        if "⚠️" in str(l_stat.get("bias_flag","")):
            flags.append(f"⚠️ Long/Shortバイアス有意 p={l_stat.get('chi2_p')}")

        # ── 6. 月次分析 ──────────────────────────────────────
        m_df=monthly_analysis(trades)
        pf_finite=m_df[m_df["pf"]<np.inf]["pf"]
        print(f"\n■ 月次PF分布")
        print(f"  月数: {len(m_df)}  プラス月: {(m_df['pf']>1.0).sum()}/{len(m_df)}")
        print(f"  PF min={pf_finite.min():.2f}  max={pf_finite.max():.2f}  std={pf_finite.std():.2f}")
        print(f"  月次WR: {m_df['wr'].min()*100:.0f}%〜{m_df['wr'].max()*100:.0f}%")
        # 最悪月
        worst=m_df.loc[pf_finite.idxmin()]
        print(f"  最悪月: {worst['month']}  PF={worst['pf']:.2f}  n={int(worst['n'])}")
        if pf_finite.min()<0.5:
            flags.append(f"⚠️ 最悪月PF={pf_finite.min():.2f}（{worst['month']}）")

        # ── 7. 時間帯分析 ────────────────────────────────────
        h_df=hourly_analysis(trades)
        print(f"\n■ 時間帯別勝率（UTC）")
        if len(h_df)>0:
            worst_h=h_df.loc[h_df["wr"].idxmin()]
            best_h =h_df.loc[h_df["wr"].idxmax()]
            print(f"  最高: UTC{int(best_h['hour'])}時  WR={best_h['wr']*100:.0f}%  n={int(best_h['n'])}")
            print(f"  最低: UTC{int(worst_h['hour'])}時  WR={worst_h['wr']*100:.0f}%  n={int(worst_h['n'])}")
            # 全体カイ二乗
            if len(h_df)>=3:
                obs=h_df[h_df["n"]>=10]
                if len(obs)>=3:
                    exp_wr=wr
                    obs_wins=(obs["wr"]*obs["n"]).values
                    exp_wins=exp_wr*obs["n"].values
                    obs_loses=((1-obs["wr"])*obs["n"]).values
                    exp_loses=(1-exp_wr)*obs["n"].values
                    chi2_h=sum((o-e)**2/e for o,e in zip(obs_wins,exp_wins) if e>0)
                    chi2_h+=sum((o-e)**2/e for o,e in zip(obs_loses,exp_loses) if e>0)
                    p_h=1-stats.chi2.cdf(chi2_h,df=len(obs)-1)
                    print(f"  時間帯勝率の均一性カイ二乗p={p_h:.4f}", end="")
                    if p_h<0.05:
                        print(f"  ⚠️ 時間帯バイアスあり")
                        flags.append(f"⚠️ 時間帯バイアス有意 p={p_h:.4f}")
                    else:
                        print(f"  ✅ 均一")

        # ── 8. MDDタイミング ─────────────────────────────────
        mdd_t=mdd_timing_analysis(eq_curve)
        print(f"\n■ MDDタイミング")
        print(f"  最大DD={mdd_t['mdd_pct']:.1f}%  発生位置: {mdd_t['position_pct']:.0f}%地点")
        if mdd_t["position_pct"]>70:
            flags.append(f"⚠️ 後半({mdd_t['position_pct']:.0f}%地点)でMDD発生（末尾劣化の可能性）")
            print(f"  → ⚠️ 後半でMDD集中")
        elif mdd_t["position_pct"]<20:
            print(f"  → ✅ 序盤でMDD → 後半は安定")
        else:
            print(f"  → ✅ 中盤でMDD（許容範囲）")

        # ── 9. モンテカルロ ──────────────────────────────────
        pnl_pct=[p/(e-p) if (e-p)>0 else 0 for p,e in zip(df["pnl_jpy"],df["equity_after"])]
        mc=monte_carlo_final_equity(pnl_pct,len(pnl_pct))
        print(f"\n■ モンテカルロ（順序シャッフル10,000回）")
        print(f"  中央値: {mc['median']:,.0f}円  平均: {mc['mean']:,.0f}円")
        print(f"  5%ile: {mc['p5']:,.0f}円  95%ile: {mc['p95']:,.0f}円")
        print(f"  1%ile: {mc['p1']:,.0f}円  99%ile: {mc['p99']:,.0f}円")
        print(f"  損失確率（最終<元本）: {mc['prob_loss']:.1f}%")
        if mc["prob_loss"]>5:
            flags.append(f"⚠️ MC損失確率{mc['prob_loss']:.1f}%>5%")

        # ── 問題点まとめ ─────────────────────────────────────
        print(f"\n■ 問題点フラグ ({sym_name})")
        if flags:
            for f in flags: print(f"  {f}")
        else:
            print(f"  ✅ 問題点なし")
        all_flags[sym_name]=flags

        # ── グラフ ──────────────────────────────────────────
        ax0=fig.add_subplot(gs[row_idx,0])
        ax0.plot([e/INIT_CASH*100-100 for e in eq_curve],color="steelblue",lw=1.2)
        ax0.set_title(f"{sym_name} エクイティ",fontsize=9)
        ax0.set_ylabel("累積リターン%",fontsize=7)
        ax0.grid(True,alpha=0.3)

        ax1=fig.add_subplot(gs[row_idx,1])
        m_pf=m_df[m_df["pf"]<5]["pf"]
        ax1.bar(range(len(m_pf)),m_pf.values,color=["green" if p>1 else "tomato" for p in m_pf])
        ax1.axhline(1,color="gray",lw=0.8,ls="--")
        ax1.set_title(f"{sym_name} 月次PF",fontsize=9)
        ax1.set_xticks(range(len(m_df)))
        ax1.set_xticklabels([m[-5:] for m in m_df["month"]],rotation=45,fontsize=6)
        ax1.grid(True,alpha=0.3)

        ax2=fig.add_subplot(gs[row_idx,2])
        if len(h_df)>0:
            ax2.bar(h_df["hour"],h_df["wr"]*100,color="cornflowerblue",alpha=0.8)
            ax2.axhline(wr*100,color="red",lw=1.2,ls="--",label=f"avg={wr*100:.0f}%")
            ax2.set_title(f"{sym_name} 時間帯勝率(UTC)",fontsize=9)
            ax2.set_xlabel("Hour(UTC)",fontsize=7); ax2.set_ylabel("WR%",fontsize=7)
            ax2.set_ylim(0,100); ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)

        ax3=fig.add_subplot(gs[row_idx,3])
        ax3.hist(pnl,bins=40,color="steelblue",alpha=0.7,edgecolor="white")
        ax3.axvline(0,color="red",lw=1.5,ls="--")
        ax3.axvline(np.mean(pnl),color="orange",lw=1.2,ls="-",label=f"mean={np.mean(pnl):,.0f}")
        ax3.set_title(f"{sym_name} 損益分布",fontsize=9)
        ax3.set_xlabel("損益(円)",fontsize=7); ax3.legend(fontsize=7); ax3.grid(True,alpha=0.3)

        row_idx+=1

    # ── 全銘柄サマリー ────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  ■ 全銘柄 問題点サマリー")
    print(f"{'='*65}")
    any_flags=False
    for sym,flags in all_flags.items():
        if flags:
            any_flags=True
            print(f"\n  {sym}:")
            for f in flags: print(f"    {f}")
    if not any_flags:
        print("  ✅ 全銘柄で重大な問題点なし")

    plt.suptitle("v80 YAGAMI改 — 定量・計量分析\n(2025-12〜2026-02)",fontsize=12,y=0.99)
    out_png=os.path.join(OUT_DIR,"v80_quant_analysis.png")
    plt.savefig(out_png,dpi=100,bbox_inches="tight")
    print(f"\nグラフ保存: {out_png}")


if __name__=="__main__":
    main()
