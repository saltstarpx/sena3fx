"""
backtest_tol_comparison.py
==========================
tol_factor変更前後の比較: PnLエクイティカーブ + 詳細統計

変更対象: NZDUSD/XAUUSD のみ tol_factor=0.30→0.20
他5銘柄: 現行 tol_factor=0.30 維持
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# ── 銘柄設定 ─────────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY", "logic": "C"},
    {"sym": "GBPUSD", "logic": "A"},
    {"sym": "EURUSD", "logic": "C"},
    {"sym": "USDCAD", "logic": "A"},
    {"sym": "NZDUSD", "logic": "A"},
    {"sym": "XAUUSD", "logic": "A"},
    {"sym": "AUDUSD", "logic": "B"},
]

# 変更対象銘柄のみ tol=0.20、他は 0.30
TOL_BEFORE = {s["sym"]: 0.30 for s in TARGETS}
TOL_AFTER  = {s["sym"]: 0.30 for s in TARGETS}
TOL_AFTER["NZDUSD"] = 0.20
TOL_AFTER["XAUUSD"] = 0.20

LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

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

def _is_lfs(path):
    try:
        with open(path, 'r') as f: return f.readline().startswith('version https://git-lfs')
    except: return False

def load_all(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p) and not _is_lfs(p):
            d1m = load_csv(p); break
    else: return None, None
    p_is = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
    p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
    if os.path.exists(p_is) and os.path.exists(p_oos):
        d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
        return d1m, d4h[~d4h.index.duplicated(keep="first")].sort_index()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
        if os.path.exists(p) and not _is_lfs(p): return d1m, load_csv(p)
    d4h = d1m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","close"])
    return d1m, d4h

# ── インジケーター ──────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"]-df["low"]; hc = (df["high"]-df["close"].shift()).abs(); lc = (df["low"]-df["close"].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    h=df["high"]; l=df["low"]; pdm=h.diff().clip(lower=0); mdm=(-l.diff()).clip(lower=0)
    pdm[pdm<mdm]=0.0; mdm[mdm<pdm]=0.0
    atr=calc_atr(df,1).ewm(alpha=1/n,adjust=False).mean()
    dip=100*pdm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan)
    dim=100*mdm.ewm(alpha=1/n,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean().fillna(0)

def build_4h(df4h, need_1d=False):
    df=df4h.copy(); df["atr"]=calc_atr(df,14); df["ema20"]=df["close"].ewm(span=20,adjust=False).mean()
    df["trend"]=np.where(df["close"]>df["ema20"],1,-1); df["adx"]=calc_adx(df,14)
    d1=None
    if need_1d:
        d1=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","close"])
        d1["ema20"]=d1["close"].ewm(span=20,adjust=False).mean()
        d1["trend1d"]=np.where(d1["close"]>d1["ema20"],1,-1)
    return df, d1

def build_1h(df):
    r=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(subset=["open","close"])
    r["atr"]=calc_atr(r,14); r["ema20"]=r["close"].ewm(span=20,adjust=False).mean(); return r

# ── エントリー ──────────────────────────────────────────────────
def pick_e0(t,sp,d,m1c):
    idx=m1c["idx"];s=idx.searchsorted(t,side="left");e=idx.searchsorted(t+pd.Timedelta(minutes=E0_WINDOW_MIN),side="left")
    for i in range(s,min(e,len(idx))): return idx[i],m1c["opens"][i]+(sp if d==1 else -sp)
    return None,None

def pick_e1(t,d,sp,m1c):
    idx=m1c["idx"];s=idx.searchsorted(t,side="left");e=idx.searchsorted(t+pd.Timedelta(minutes=E1_MAX_WAIT_MIN),side="left")
    for i in range(s,min(e,len(idx))):
        o=m1c["opens"][i];c=m1c["closes"][i]
        if d==1 and c<=o: continue
        if d==-1 and c>=o: continue
        ni=i+1
        if ni>=len(idx): return None,None
        return idx[ni],m1c["opens"][ni]+(sp if d==1 else -sp)
    return None,None

def pick_e2(t,d,sp,atr_d,m1c):
    idx=m1c["idx"];s=idx.searchsorted(t,side="left");e=idx.searchsorted(t+pd.Timedelta(minutes=max(2,E2_WINDOW_MIN)),side="left")
    for i in range(s,min(e,len(idx))):
        rng=m1c["highs"][i]-m1c["lows"][i];av=atr_d.get(idx[i],np.nan)
        if not np.isnan(av) and rng>av*E2_SPIKE_ATR: continue
        return idx[i],m1c["opens"][i]+(sp if d==1 else -sp)
    return None,None

# ── シグナル生成 ─────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol_factor=0.30):
    d4h, d1d = build_4h(d4h_full, need_1d=(logic=="A"))
    d1h = build_1h(d1m)
    signals = []; used = set()
    h1_idx=d1h.index.values; h1_open=d1h["open"].values; h1_high=d1h["high"].values
    h1_low=d1h["low"].values; h1_close=d1h["close"].values; h1_atr=d1h["atr"].values
    h4_idx=d4h.index.values; h4_open=d4h["open"].values; h4_close=d4h["close"].values
    h4_low=d4h["low"].values; h4_high=d4h["high"].values; h4_atr=d4h["atr"].values
    h4_trend=d4h["trend"].values; h4_adx=d4h["adx"].values; h4_ema20=d4h["ema20"].values
    d1d_idx = d1d.index.values if d1d is not None else None
    d1d_trend = d1d["trend1d"].values if d1d is not None else None

    for i in range(2, len(h1_idx)):
        hct=h1_idx[i]; atr1h=h1_atr[i]
        if np.isnan(atr1h) or atr1h<=0: continue
        h4pos=np.searchsorted(h4_idx,hct,side="left")-1
        if h4pos<max(1,STREAK_MIN-1): continue
        if np.isnan(h4_atr[h4pos]): continue
        trend=h4_trend[h4pos]; h4atr=h4_atr[h4pos]
        if logic=="A":
            if d1d_idx is None: continue
            hct_date=np.datetime64(pd.Timestamp(hct).normalize().tz_localize(None))
            d1pos=np.searchsorted(d1d_idx,hct_date,side="left")-1
            if d1pos<0 or d1d_trend[d1pos]!=trend: continue
        elif logic=="B":
            if h4_adx[h4pos]<ADX_MIN: continue
            if STREAK_MIN>1:
                ok=True
                for si in range(h4pos-STREAK_MIN+1,h4pos+1):
                    if h4_trend[si]!=trend: ok=False; break
                if not ok: continue
        if trend==1 and h4_close[h4pos]<=h4_open[h4pos]: continue
        if trend==-1 and h4_close[h4pos]>=h4_open[h4pos]: continue
        op=h4_open[h4pos];cl=h4_close[h4pos];lo=h4_low[h4pos]
        if op>0 and (min(op,cl)-lo)/op>=KLOW_THR: continue
        if logic!="C":
            if h4_atr[h4pos]<=0 or abs(h4_close[h4pos]-h4_ema20[h4pos])<h4_atr[h4pos]*A1_EMA_DIST_MIN: continue
        d=trend
        v1=h1_low[i-2] if d==1 else h1_high[i-2]; v2=h1_low[i-1] if d==1 else h1_high[i-1]
        if abs(v1-v2)>atr1h*tol_factor: continue
        if logic=="C":
            if d==1 and h1_close[i-1]<=h1_open[i-1]: continue
            if d==-1 and h1_close[i-1]>=h1_open[i-1]: continue
        hct_ts=pd.Timestamp(hct,tz="UTC")
        if logic=="A":   et,ep=pick_e2(hct_ts,d,spread,atr_d,m1c)
        elif logic=="C": et,ep=pick_e0(hct_ts,spread,d,m1c)
        else:            et,ep=pick_e1(hct_ts,d,spread,m1c)
        if et is None or et in used: continue
        raw=ep-spread if d==1 else ep+spread
        sl=(min(v1,v2)-atr1h*0.15) if d==1 else (max(v1,v2)+atr1h*0.15)
        risk=(raw-sl) if d==1 else (sl-raw)
        if 0<risk<=h4atr*2:
            signals.append({"time":et,"dir":d,"ep":ep,"sl":sl,"tp":raw+d*risk*RR_RATIO,"risk":risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])

# ── Numba JIT _exit ──────────────────────────────────────────────
@njit(cache=True)
def _exit_numba(highs,lows,ep,sl,tp,risk,d,half_r,max_look):
    half=ep+d*risk*half_r; lim=min(len(highs),max_look)
    for i in range(lim):
        h=highs[i];lo=lows[i]
        if d==1:
            if lo<=sl: return sl,2,False
            if h>=tp: return tp,1,False
            if h>=half:
                for j in range(i+1,lim):
                    if lows[j]<=ep: return ep,1,True
                    if highs[j]>=tp: return tp,1,True
                return 0.0,0,True
        else:
            if h>=sl: return sl,2,False
            if lo<=tp: return tp,1,False
            if lo<=half:
                for j in range(i+1,lim):
                    if highs[j]>=ep: return ep,1,True
                    if lows[j]<=tp: return tp,1,True
                return 0.0,0,True
    return 0.0,0,False

# ── シミュレーション（トレード詳細付き）─────────────────────────
def simulate_detailed(signals, d1m, sym):
    """トレード毎の日時・PnLを記録（エクイティカーブ用）"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t=d1m.index; m1h=d1m["high"].values; m1l=d1m["low"].values
    equity=INIT_CASH; trades=[]; peak=INIT_CASH; mdd=0.0

    for sig in signals:
        lot=rm.calc_lot(equity,sig["risk"],sig["ep"],usdjpy_rate=USDJPY_RATE)
        sp=m1t.searchsorted(sig["time"],side="right")
        if sp>=len(m1t): continue
        xp,rcode,half_done=_exit_numba(m1h[sp:],m1l[sp:],sig["ep"],sig["sl"],sig["tp"],sig["risk"],sig["dir"],HALF_R,MAX_LOOKAHEAD)
        if rcode==0: continue
        result="win" if rcode==1 else "loss"
        half_pnl=0.0
        if half_done:
            hp=sig["ep"]+sig["dir"]*sig["risk"]*HALF_R
            half_pnl=rm.calc_pnl_jpy(sig["dir"],sig["ep"],hp,lot*0.5,USDJPY_RATE,sig["ep"])
            equity+=half_pnl; rem=lot*0.5
        else: rem=lot
        pnl=rm.calc_pnl_jpy(sig["dir"],sig["ep"],xp,rem,USDJPY_RATE,sig["ep"])
        equity+=pnl; total_pnl=half_pnl+pnl
        trades.append({"time":sig["time"],"result":result,"pnl":total_pnl,
                       "equity":equity,"month":sig["time"].strftime("%Y-%m"),
                       "dir":sig["dir"],"ep":sig["ep"],"sl":sig["sl"],"tp":sig["tp"]})
        peak=max(peak,equity); mdd=max(mdd,(peak-equity)/peak*100)
    return trades, equity, mdd

def calc_stats(trades):
    if len(trades)<5: return {}
    df=pd.DataFrame(trades); n=len(df)
    wins=df[df["pnl"]>0]["pnl"]; loss=df[df["pnl"]<0]["pnl"]
    wr=len(wins)/n; gw=wins.sum(); gl=abs(loss.sum())
    pf=gw/gl if gl>0 else float("inf")
    monthly=df.groupby("month")["pnl"].sum(); plus_m=(monthly>0).sum()
    eq=INIT_CASH; rets=[]
    for m in monthly.index: rets.append(monthly[m]/eq if eq>0 else 0); eq+=monthly[m]
    mr=np.array(rets)
    sharpe=(mr.mean()/mr.std())*np.sqrt(12) if len(mr)>=2 and mr.std()>0 else 0.0
    avg_w=wins.mean() if len(wins)>0 else 0; avg_l=abs(loss.mean()) if len(loss)>0 else 1
    kelly=wr-(1-wr)/(avg_w/avg_l) if avg_l>0 and avg_w>0 else 0
    max_consec_loss=0; cur=0
    for _,r in df.iterrows():
        if r["pnl"]<0: cur+=1; max_consec_loss=max(max_consec_loss,cur)
        else: cur=0
    return {"n":n,"wr":wr,"pf":pf,"sharpe":sharpe,"kelly":kelly,
            "plus_m":plus_m,"total_m":len(monthly),"mdd":0.0,
            "max_consec_loss":max_consec_loss,"gross_win":gw,"gross_loss":gl}

def run_sym(d1m, d4h, sym, logic, tol):
    cfg=SYMBOL_CONFIG[sym]; spread=cfg["spread"]*cfg["pip"]
    atr_d=calc_atr(d1m,10).to_dict()
    m1c={"idx":d1m.index,"opens":d1m["open"].values,"closes":d1m["close"].values,
         "highs":d1m["high"].values,"lows":d1m["low"].values}
    sigs=generate_signals(d1m,d4h,spread,logic,atr_d,m1c,tol_factor=tol)
    trades,eq,mdd=simulate_detailed(sigs,d1m,sym)
    st=calc_stats(trades)
    if st: st["mdd"]=mdd
    return st, trades

# ── メイン ───────────────────────────────────────────────────────
def main():
    t0=time.time()
    print("\n"+"="*110)
    print("  tol_factor変更前後 比較: NZDUSD/XAUUSD tol=0.30→0.20")
    print("="*110)

    # Numba warmup
    _exit_numba(np.array([1.0,2.0]),np.array([0.5,0.5]),1.0,0.5,2.0,0.5,1,1.0,2)

    # データロード
    print("\n  [1] データロード...")
    sym_data = {}
    for tgt in TARGETS:
        sym=tgt["sym"]; d1m,d4h=load_all(sym)
        if d1m is None: print(f"    ❌ {sym}: 未発見"); continue
        sym_data[sym]={"d1m":d1m,"d4h":d4h,"logic":tgt["logic"]}
        print(f"    ✅ {sym}: {len(d1m):,}行")

    # バックテスト（Before / After）
    print("\n  [2] バックテスト実行...")
    before = {}; after = {}
    for sym, data in sym_data.items():
        st_b, tr_b = run_sym(data["d1m"], data["d4h"], sym, data["logic"], TOL_BEFORE[sym])
        st_a, tr_a = run_sym(data["d1m"], data["d4h"], sym, data["logic"], TOL_AFTER[sym])
        before[sym] = {"stats": st_b, "trades": tr_b}
        after[sym]  = {"stats": st_a, "trades": tr_a}
    print(f"    完了 ({time.time()-t0:.0f}秒)")

    # ── 詳細統計テーブル ──
    print("\n  [3] 銘柄別 詳細統計")
    print("  "+"="*108)
    print(f"    {'銘柄':8} {'tol':>5} | {'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MDD':>6} {'Kelly':>7} "
          f"{'月+':>5} {'最大連敗':>5} | {'変化':>6}")
    print("  "+"-"*108)

    changed = ["NZDUSD", "XAUUSD"]

    for sym in sym_data:
        sb = before[sym]["stats"]; sa = after[sym]["stats"]
        is_changed = sym in changed
        tag = " ★変更" if is_changed else ""

        # Before行
        if sb:
            pf_s = f"{sb['pf']:.2f}" if sb['pf']<99 else "∞"
            print(f"    {sym:8} {TOL_BEFORE[sym]:.2f}  | {sb['n']:>5} {sb['wr']*100:>5.1f}% {pf_s:>6} "
                  f"{sb['sharpe']:>7.2f} {sb['mdd']:>5.1f}% {sb['kelly']:>7.3f} "
                  f"{sb['plus_m']:>2}/{sb['total_m']:<2}  {sb.get('max_consec_loss',0):>4}  | 変更前")

        # After行（変更対象のみ）
        if is_changed and sa:
            pf_s = f"{sa['pf']:.2f}" if sa['pf']<99 else "∞"
            d_mdd = sa['mdd'] - sb['mdd'] if sb else 0
            d_pf = sa['pf'] - sb['pf'] if sb else 0
            print(f"    {'→':8} {TOL_AFTER[sym]:.2f}  | {sa['n']:>5} {sa['wr']*100:>5.1f}% {pf_s:>6} "
                  f"{sa['sharpe']:>7.2f} {sa['mdd']:>5.1f}% {sa['kelly']:>7.3f} "
                  f"{sa['plus_m']:>2}/{sa['total_m']:<2}  {sa.get('max_consec_loss',0):>4}  | "
                  f"MDD{d_mdd:+.1f}pp PF{d_pf:+.2f}{tag}")
        print()

    # ── ポートフォリオ比較 ──
    print("  [4] ポートフォリオ統計")
    print("  "+"="*108)

    for label, data_dict in [("変更前 (全tol=0.30)", before), ("変更後 (NZD/XAU=0.20)", after)]:
        combined_m = {}
        total_n=0; total_win=0; total_loss_n=0
        for sym, d in data_dict.items():
            st = d["stats"]
            if st: total_n += st["n"]; total_win += int(st["wr"]*st["n"])
            for t in d["trades"]:
                combined_m[t["month"]] = combined_m.get(t["month"],0) + t["pnl"]
        months = sorted(combined_m.keys())
        eq=INIT_CASH*len(data_dict); rets=[]; peak=eq; mdd=0; plus_m=0
        for m in months:
            r = combined_m[m]/eq if eq>0 else 0; rets.append(r)
            eq += combined_m[m]; peak=max(peak,eq); mdd=max(mdd,(peak-eq)/peak*100)
            if combined_m[m]>0: plus_m+=1
        mr=np.array(rets)
        sh = (mr.mean()/mr.std())*np.sqrt(12) if len(mr)>=2 and mr.std()>0 else 0
        total_pnl = sum(combined_m.values())
        wr = total_win/total_n if total_n>0 else 0
        print(f"    {label}")
        print(f"      総トレード: {total_n}  ポートフォリオ勝率: {wr*100:.1f}%")
        print(f"      ポートフォリオSharpe: {sh:.2f}")
        print(f"      ポートフォリオMDD: {mdd:.2f}%")
        print(f"      月次プラス: {plus_m}/{len(months)}")
        print(f"      総損益: ¥{total_pnl:,.0f}")
        print()

    # ── PnLグラフ生成 ──
    print("  [5] PnLエクイティカーブ生成...")
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle("tol_factor変更前後 エクイティカーブ比較\nNZDUSD/XAUUSD: tol=0.30→0.20", fontsize=14, y=0.98)

    # 銘柄別エクイティカーブ（変更対象2銘柄）
    for idx, sym in enumerate(changed):
        ax = axes[idx, 0]
        tr_b = before[sym]["trades"]; tr_a = after[sym]["trades"]
        sb = before[sym]["stats"]; sa = after[sym]["stats"]

        if tr_b:
            times_b = [t["time"] for t in tr_b]; eq_b = [t["equity"] for t in tr_b]
            ax.plot(times_b, eq_b, 'b-', alpha=0.7, linewidth=1.2, label=f'tol=0.30 (PF={sb["pf"]:.2f}, MDD={sb["mdd"]:.1f}%)')
        if tr_a:
            times_a = [t["time"] for t in tr_a]; eq_a = [t["equity"] for t in tr_a]
            ax.plot(times_a, eq_a, 'r-', alpha=0.7, linewidth=1.2, label=f'tol=0.20 (PF={sa["pf"]:.2f}, MDD={sa["mdd"]:.1f}%)')

        ax.axhline(y=INIT_CASH, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{sym} エクイティカーブ', fontsize=12)
        ax.set_ylabel('Equity (¥)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

    # 銘柄別 月次PnLバー（変更対象2銘柄）
    for idx, sym in enumerate(changed):
        ax = axes[idx, 1]
        for label, trades, color in [("Before", before[sym]["trades"], "steelblue"),
                                      ("After",  after[sym]["trades"], "coral")]:
            monthly = {}
            for t in trades: monthly[t["month"]] = monthly.get(t["month"],0)+t["pnl"]
            if monthly:
                ms = sorted(monthly.keys()); vals = [monthly[m] for m in ms]
                x = np.arange(len(ms))
                w = 0.35; offset = -w/2 if label=="Before" else w/2
                ax.bar(x+offset, vals, w, label=f'{label} (tol={TOL_BEFORE[sym] if label=="Before" else TOL_AFTER[sym]})',
                       color=color, alpha=0.7)
                ax.set_xticks(x); ax.set_xticklabels(ms, rotation=45, fontsize=8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(f'{sym} 月次PnL (¥)', fontsize=12)
        ax.set_ylabel('PnL (¥)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # ポートフォリオ全体エクイティカーブ
    ax = axes[2, 0]
    for label, data_dict, color, ls in [("Before (全tol=0.30)", before, "blue", "-"),
                                         ("After (NZD/XAU=0.20)", after, "red", "-")]:
        all_trades = []
        for sym, d in data_dict.items():
            for t in d["trades"]:
                all_trades.append(t)
        all_trades.sort(key=lambda x: x["time"])
        eq = INIT_CASH * len(data_dict); eqs = []; times = []
        for t in all_trades:
            eq += t["pnl"]; eqs.append(eq); times.append(t["time"])
        if times:
            ax.plot(times, eqs, color=color, linestyle=ls, linewidth=1.5, alpha=0.8, label=label)
    ax.axhline(y=INIT_CASH*len(data_dict), color='gray', linestyle='--', alpha=0.5)
    ax.set_title('ポートフォリオ全体 エクイティカーブ', fontsize=12)
    ax.set_ylabel('Equity (¥)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

    # ポートフォリオ月次PnL
    ax = axes[2, 1]
    for label, data_dict, color in [("Before", before, "steelblue"), ("After", after, "coral")]:
        monthly = {}
        for sym, d in data_dict.items():
            for t in d["trades"]:
                monthly[t["month"]] = monthly.get(t["month"],0)+t["pnl"]
        if monthly:
            ms=sorted(monthly.keys()); vals=[monthly[m] for m in ms]
            x=np.arange(len(ms)); w=0.35; offset=-w/2 if label=="Before" else w/2
            ax.bar(x+offset, vals, w, label=label, color=color, alpha=0.7)
            ax.set_xticks(x); ax.set_xticklabels(ms, rotation=45, fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('ポートフォリオ 月次PnL (¥)', fontsize=12)
    ax.set_ylabel('PnL (¥)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # ドローダウンカーブ（変更対象2銘柄）
    for idx, sym in enumerate(changed):
        ax = axes[3, idx]
        for label, trades, color in [("Before (0.30)", before[sym]["trades"], "blue"),
                                      ("After (0.20)", after[sym]["trades"], "red")]:
            if not trades: continue
            eq=INIT_CASH; peak=eq; dds=[]; times=[]
            for t in trades:
                eq+=t["pnl"]; peak=max(peak,eq)
                dds.append((peak-eq)/peak*100); times.append(t["time"])
            ax.fill_between(times, 0, dds, alpha=0.3, color=color, label=label)
            ax.plot(times, dds, color=color, linewidth=0.8, alpha=0.7)
        ax.invert_yaxis()
        ax.set_title(f'{sym} ドローダウン (%)', fontsize=12)
        ax.set_ylabel('Drawdown (%)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT_DIR, "tol_comparison_nzdusd_xauusd.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {out_png}")

    print(f"\n    実行時間: {time.time()-t0:.0f}秒")
    print("="*110)

if __name__ == "__main__":
    main()
