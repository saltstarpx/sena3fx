"""
backtest_yoto_kento.py
======================
要検討銘柄の精査バックテスト

【目的】
  前回バックテストで「要検討（OOS PF 1.5〜2.0）」となった銘柄を
  最大期間・アダプティブリスクで再検証し、シャープレシオも含めた
  総合評価で採用/不採用を判断する。

【対象銘柄（前回ベストロジック固定）】
  EURUSD  Logic-C オーパーツYAGAMI   PF=1.81
  GBPUSD  Logic-A GOLDYAGAMI         PF=1.86
  AUDUSD  Logic-B ADX+Streak         PF=2.05（月次7/9でFAIL）
  NZDUSD  Logic-A GOLDYAGAMI         PF=1.98
  USDCHF  Logic-A GOLDYAGAMI         PF=1.78
  EURGBP  Logic-A GOLDYAGAMI         PF=1.56
  AUDJPY  Logic-A GOLDYAGAMI         PF=1.52
  XAGUSD  Logic-B ADX+Streak         PF=1.84

【リスクモード比較】
  ① 固定2%（ベースライン）
  ② 固定3%（高リスク）
  ③ アダプティブ2→3%（勝ち+0.5%/負け-0.5%, 2.0/2.5/3.0の3段階）

【評価指標】
  PF / 勝率 / MDD / シャープレシオ（月次リターン×√12換算）
  ケリー基準 / 月次プラス率 / 過学習チェック（IS/OOS比）

【期間】
  各銘柄の最大利用可能期間（データ全量）
  IS: 最初40%  /  OOS: 残り60%
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
ADX_MIN    = 20
STREAK_MIN = 4

# アダプティブリスク設定
RISK_INIT = 0.02
RISK_MIN  = 0.02
RISK_MAX  = 0.03
RISK_STEP = 0.005   # 勝ち→+0.5%, 負け→-0.5%

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 銘柄×ベストロジック定義 ──────────────────────────────────────
TARGETS = [
    # 要検討
    {"sym": "EURUSD",  "logic": "C", "cat": "FX_USD",   "prev_pf": 1.81},
    {"sym": "GBPUSD",  "logic": "A", "cat": "FX_USD",   "prev_pf": 1.86},
    {"sym": "AUDUSD",  "logic": "B", "cat": "FX_USD",   "prev_pf": 2.05},
    {"sym": "NZDUSD",  "logic": "A", "cat": "FX_USD",   "prev_pf": 1.98},
    {"sym": "USDCHF",  "logic": "A", "cat": "FX_USD",   "prev_pf": 1.78},
    {"sym": "EURGBP",  "logic": "A", "cat": "FX_CROSS", "prev_pf": 1.56},
    {"sym": "AUDJPY",  "logic": "A", "cat": "FX_JPY",   "prev_pf": 1.52},
    {"sym": "XAGUSD",  "logic": "B", "cat": "METALS",   "prev_pf": 1.84},
    # 採用済み（比較参照用）
    {"sym": "XAUUSD",  "logic": "A", "cat": "METALS",   "prev_pf": 3.10},
    {"sym": "USDJPY",  "logic": "C", "cat": "FX_JPY",   "prev_pf": 2.15},
    {"sym": "USDCAD",  "logic": "A", "cat": "FX_USD",   "prev_pf": 2.02},
]

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def load_all(sym):
    sym_lower = sym.lower()
    p1m = os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv")
    if not os.path.exists(p1m):
        p1m = os.path.join(DATA_DIR, f"{sym_lower}_1m.csv")
    if not os.path.exists(p1m):
        return None, None
    d1m = load_csv(p1m)

    p4h_all = os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv")
    p4h_is  = os.path.join(DATA_DIR, f"{sym_lower}_is_4h.csv")
    p4h_oos = os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv")
    p4h_all2 = os.path.join(DATA_DIR, f"{sym_lower}_4h.csv")

    if os.path.exists(p4h_all):
        d4h = load_csv(p4h_all)
    elif os.path.exists(p4h_all2):
        d4h = load_csv(p4h_all2)
    elif os.path.exists(p4h_is) and os.path.exists(p4h_oos):
        d4h = pd.concat([load_csv(p4h_is), load_csv(p4h_oos)])
        d4h = d4h[~d4h.index.duplicated(keep="first")].sort_index()
    else:
        d4h = d1m.resample("4h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna(subset=["open","close"])
    return d1m, d4h

def split_is_oos(d1m):
    """全期間を IS:40% / OOS:60% に分割"""
    dates = d1m.index
    split_idx = int(len(dates) * 0.4)
    split_ts  = dates[split_idx]
    is_  = d1m[d1m.index <  split_ts].copy()
    oos  = d1m[d1m.index >= split_ts].copy()
    return is_, oos, split_ts

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    h = df["high"]; l = df["low"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0.0; mdm[mdm < pdm] = 0.0
    tr   = calc_atr(df, 1)
    atr  = tr.ewm(alpha=1/n, adjust=False).mean()
    di_p = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    di_m = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna(subset=["open","close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df_in):
    df = df_in.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── エントリー ────────────────────────────────────────────────────
def pick_e0(t, spread, direction, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E0_WINDOW_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        return idx[i], m1c["opens"][i] + (spread if direction==1 else -spread)
    return None, None

def pick_e1(t, direction, spread, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E1_MAX_WAIT_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        o = m1c["opens"][i]; c = m1c["closes"][i]
        if direction==1 and c<=o: continue
        if direction==-1 and c>=o: continue
        ni = i+1
        if ni >= len(idx): return None, None
        return idx[ni], m1c["opens"][ni] + (spread if direction==1 else -spread)
    return None, None

def pick_e2(t, direction, spread, atr_d, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        rng = m1c["highs"][i] - m1c["lows"][i]
        av  = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and rng > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (spread if direction==1 else -spread)
    return None, None

# ── フィルター ────────────────────────────────────────────────────
def chk_kmid(bar, d): return (d==1 and bar["close"]>bar["open"]) or (d==-1 and bar["close"]<bar["open"])
def chk_klow(bar):
    o = bar["open"]; return (min(bar["open"],bar["close"])-bar["low"])/o < KLOW_THR if o>0 else False
def chk_ema_dist(bar):
    d = abs(bar["close"]-bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a>0 and d >= a*A1_EMA_DIST_MIN

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c):
    need_1d = (logic=="A")
    d4h, d1d = build_4h(d4h_full, need_1d=need_1d)
    d1h = build_1h(d1m)
    signals=[]; used=set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h<=0: continue

        h4_b = d4h[d4h.index < h1_ct]
        if len(h4_b) < max(2, STREAK_MIN): continue
        h4l = h4_b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if logic=="A":
            if d1d is None: continue
            d1b = d1d[d1d.index.normalize() < h1_ct.normalize()]
            if len(d1b)==0: continue
            if d1b.iloc[-1]["trend1d"] != trend: continue
        elif logic=="B":
            if h4l.get("adx",0) < ADX_MIN: continue
            if not all(t==trend for t in h4_b["trend"].iloc[-STREAK_MIN:].values): continue

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if logic!="C" and not chk_ema_dist(h4l): continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction==1: v1,v2 = h1_p2["low"],  h1_p1["low"]
        else:            v1,v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1-v2) > tol: continue

        if logic=="C":
            if direction==1 and h1_p1["close"]<=h1_p1["open"]: continue
            if direction==-1 and h1_p1["close"]>=h1_p1["open"]: continue

        if logic=="A":   et,ep = pick_e2(h1_ct, direction, spread, atr_d, m1c)
        elif logic=="C": et,ep = pick_e0(h1_ct, spread, direction, m1c)
        else:            et,ep = pick_e1(h1_ct, direction, spread, m1c)

        if et is None or et in used: continue

        raw = ep-spread if direction==1 else ep+spread
        if direction==1: sl = min(v1,v2)-atr_1h*0.15; risk=raw-sl
        else:            sl = max(v1,v2)+atr_1h*0.15; risk=sl-raw
        if 0 < risk <= h4atr*2:
            tp = raw + direction*risk*RR_RATIO
            signals.append({"time":et,"dir":direction,"ep":ep,"sl":sl,
                            "tp":tp,"risk":risk})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── 出口計算 ──────────────────────────────────────────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half = ep + direction*risk*HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h=highs[i]; lo=lows[i]
        if direction==1:
            if lo<=sl: return sl,"loss",False
            if h >=tp: return tp,"win", False
            if h >=half:
                be=ep
                for j in range(i+1, lim):
                    if lows[j] <=be: return be,"win",True
                    if highs[j]>=tp: return tp,"win",True
                return None,None,True
        else:
            if h >=sl: return sl,"loss",False
            if lo<=tp: return tp,"win", False
            if lo<=half:
                be=ep
                for j in range(i+1, lim):
                    if highs[j]>=be: return be,"win",True
                    if lows[j] <=tp: return tp,"win",True
                return None,None,True
    return None,None,False

# ── シミュレーション（リスクモード対応） ────────────────────────
def simulate(signals, d1m, sym, risk_mode="fixed2"):
    """
    risk_mode: "fixed2"=固定2%, "fixed3"=固定3%, "adaptive"=2-3%アダプティブ
    """
    if not signals: return [], INIT_CASH, 0.0

    rm = RiskManager(sym, risk_pct=RISK_INIT)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    # アダプティブリスク状態
    current_risk = RISK_INIT

    for sig in signals:
        # リスク率決定
        if risk_mode == "fixed2":   r_pct = 0.02
        elif risk_mode == "fixed3": r_pct = 0.03
        else:                       r_pct = current_risk  # adaptive

        rm.risk_pct = r_pct
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        xp, result, half_done = _find_exit(m1h[sp:], m1l[sp:],
                                            sig["ep"], sig["sl"], sig["tp"],
                                            sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"]*sig["risk"]*HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp,
                                        lot*0.5, USDJPY_RATE, sig["ep"])
            equity  += half_pnl; rem = lot*0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total  = half_pnl + pnl
        trades.append({"result":result,"pnl":total,
                       "month":sig["time"].strftime("%Y-%m")})

        peak = max(peak, equity)
        mdd  = max(mdd, (peak-equity)/peak*100)

        # アダプティブリスク更新
        if risk_mode == "adaptive":
            if result == "win":
                current_risk = min(current_risk + RISK_STEP, RISK_MAX)
            else:
                current_risk = max(current_risk - RISK_STEP, RISK_MIN)

    return trades, equity, mdd

# ── 統計計算 ──────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 10: return {}
    df     = pd.DataFrame(trades)
    n      = len(df)
    wins_s = df[df["pnl"]>0]["pnl"]
    lose_s = df[df["pnl"]<0]["pnl"]
    wr     = len(wins_s)/n
    gw     = wins_s.sum(); gl = abs(lose_s.sum())
    pf     = gw/gl if gl>0 else float("inf")

    # 月次リターン
    monthly = df.groupby("month")["pnl"].sum()
    plus_m  = (monthly>0).sum()
    total_m = len(monthly)

    # シャープレシオ（月次リターン率→年換算）
    eq_series = [init]
    for _, row in df.iterrows():
        eq_series.append(eq_series[-1]+row["pnl"])
    monthly_eq = {}
    eq_cursor  = init
    for _, row in df.iterrows():
        m = row["month"]
        if m not in monthly_eq:
            monthly_eq[m] = eq_cursor
        eq_cursor += row["pnl"]
    monthly_ret = []
    for m, start_eq in monthly_eq.items():
        pnl_m = monthly.get(m, 0)
        if start_eq > 0:
            monthly_ret.append(pnl_m / start_eq)
    if len(monthly_ret) >= 2:
        mr = np.array(monthly_ret)
        sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if mr.std()>0 else 0.0
    else:
        sharpe = 0.0

    # ケリー基準
    avg_win  = wins_s.mean() if len(wins_s)>0 else 0
    avg_loss = abs(lose_s.mean()) if len(lose_s)>0 else 1
    rr_act   = avg_win/avg_loss if avg_loss>0 else 0
    kelly    = wr - (1-wr)/rr_act if rr_act>0 else 0

    # 二項検定
    p_val = stats.binomtest(len(wins_s), n, 0.5, alternative="greater").pvalue

    return {
        "n": n, "wr": wr, "pf": pf, "sharpe": sharpe,
        "kelly": kelly, "mdd": 0.0,  # mddはsimulateから取得
        "plus_m": plus_m, "total_m": total_m,
        "p_val": p_val,
        "final_eq": eq_series[-1],
    }

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*110)
    print("  要検討銘柄 精査バックテスト（MAX期間 / シャープレシオ / アダプティブリスク）")
    print("  ベストロジック固定 / リスクモード：固定2% / 固定3% / アダプティブ2-3%")
    print("="*110)

    all_res = []

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        lname = {"A":"GOLDYAGAMI","B":"ADX+Streak","C":"オーパーツ"}[logic]

        print(f"\n  [{tgt['cat']}] {sym} (Logic-{logic}/{lname}) データロード中...", end=" ", flush=True)
        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("データ未発見 → スキップ"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m_full)
        is_pct  = len(is_d)/len(d1m_full)*100
        start_d = d1m_full.index[0].strftime("%Y-%m-%d")
        end_d   = d1m_full.index[-1].strftime("%Y-%m-%d")
        split_s = split_ts.strftime("%Y-%m-%d")
        print(f"{start_d}〜{end_d} ({len(d1m_full):,}行) / IS〜{split_s}({is_pct:.0f}%)")

        # シグナル生成（IS / OOS 別）
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]

        sym_res = {"sym": sym, "logic": logic, "lname": lname,
                   "cat": tgt["cat"], "prev_pf": tgt["prev_pf"],
                   "start": start_d, "end": end_d, "split": split_s}

        for period_name, d1m_p in [("IS", is_d), ("OOS", oos_d), ("FULL", d1m_full)]:
            print(f"    {sym} {period_name} 計算中...", end=" ", flush=True)
            atr_d = calc_atr(d1m_p, 10).to_dict()
            m1c   = {"idx":    d1m_p.index,
                     "opens":  d1m_p["open"].values,
                     "closes": d1m_p["close"].values,
                     "highs":  d1m_p["high"].values,
                     "lows":   d1m_p["low"].values}
            sigs = generate_signals(d1m_p, d4h_full, spread, logic, atr_d, m1c)

            for rmode in ["fixed2", "fixed3", "adaptive"]:
                trades, final_eq, mdd = simulate(sigs, d1m_p, sym, risk_mode=rmode)
                st = calc_stats(trades)
                if st: st["mdd"] = mdd; st["final_eq"] = final_eq
                sym_res[f"{period_name}_{rmode}"] = st
            print("完了")

        all_res.append(sym_res)

    # ── 結果表示 ─────────────────────────────────────────────────
    print("\n" + "="*110)
    print("  ■ OOS期間 詳細結果（ベストロジック / アダプティブリスク）")
    header = (f"  {'銘柄':8} {'ロジック':12} | "
              f"{'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Kelly':>6} {'MDD':>7} {'月+':>5} | "
              f"{'IS PF':>6} {'過学習':>12} | {'判定':>10}")
    print(header)
    print("-"*110)

    def of_check(is_pf, oos_pf):
        if not is_pf or is_pf==float("inf"): return "N/A"
        r = oos_pf/is_pf
        return f"✅({r:.2f})" if r>=0.70 else f"❌({r:.2f})"

    def verdict(pf, sharpe, mdd, pm, tm, of, n):
        if n < 30: return "❌データ不足"
        if "❌" in of: return "❌過学習"
        pm_rate = pm/tm if tm>0 else 0
        if pf >= 2.0 and sharpe >= 1.0 and pm_rate >= 0.80: return "✅採用"
        if pf >= 1.8 and sharpe >= 0.8 and pm_rate >= 0.78: return "✅採用"
        if pf >= 1.5 and sharpe >= 0.6: return "⚠️要検討"
        return "❌不採用"

    csv_rows = []
    for r in all_res:
        oos = r.get("OOS_adaptive", {})
        is_ = r.get("IS_adaptive",  {})
        if not oos: continue

        n      = oos.get("n", 0)
        wr     = oos.get("wr", 0)
        pf     = oos.get("pf", 0)
        sh     = oos.get("sharpe", 0)
        kl     = oos.get("kelly", 0)
        mdd    = oos.get("mdd", 0)
        pm     = oos.get("plus_m", 0)
        tm     = oos.get("total_m", 0)
        is_pf  = is_.get("pf", 0)
        of     = of_check(is_pf, pf)
        vd     = verdict(pf, sh, mdd, pm, tm, of, n)
        pf_s   = f"{pf:.2f}" if pf<99 else "∞"
        is_pf_s= f"{is_pf:.2f}" if is_pf<99 else "∞"

        print(f"  {r['sym']:8} Logic-{r['logic']}:{r['lname']:10} | "
              f"{n:>5} {wr*100:>5.1f}% {pf_s:>6} {sh:>7.2f} {kl:>6.3f} {mdd:>6.1f}% {pm:>2}/{tm:<2} | "
              f"{is_pf_s:>6} {of:>12} | {vd:>10}")

        csv_rows.append({
            "sym": r["sym"], "logic": r["logic"], "lname": r["lname"],
            "cat": r["cat"], "prev_oos_pf": r["prev_pf"],
            "period": f"{r['start']}~{r['end']}", "split": r["split"],
            **{f"oos_{k}": v for k, v in oos.items()},
            **{f"is_{k}": v for k, v in (is_ if is_ else {}).items()},
            "overfitting": of, "verdict": vd,
        })

    # ── リスクモード比較 ──────────────────────────────────────────
    print("\n" + "="*110)
    print("  ■ リスクモード比較（OOS PF / Sharpe）")
    print(f"  {'銘柄':8} {'ロジック':12} | {'固定2% PF':>10} {'固定2% Sh':>10} | "
          f"{'固定3% PF':>10} {'固定3% Sh':>10} | {'適応2-3% PF':>12} {'適応2-3% Sh':>12} | {'推奨モード':>10}")
    print("-"*110)

    for r in all_res:
        f2  = r.get("OOS_fixed2",   {}); f3  = r.get("OOS_fixed3",  {})
        adp = r.get("OOS_adaptive", {})
        if not f2 and not f3 and not adp: continue

        pf2  = f2.get("pf",0);  sh2  = f2.get("sharpe",0)
        pf3  = f3.get("pf",0);  sh3  = f3.get("sharpe",0)
        pfa  = adp.get("pf",0); sha  = adp.get("sharpe",0)

        # シャープで最良モードを選択
        best_sh   = max(sh2, sh3, sha)
        best_mode = ["固定2%","固定3%","適応2-3%"][[sh2,sh3,sha].index(best_sh)]

        pf2_s = f"{pf2:.2f}" if pf2<99 else "∞"
        pf3_s = f"{pf3:.2f}" if pf3<99 else "∞"
        pfa_s = f"{pfa:.2f}" if pfa<99 else "∞"

        print(f"  {r['sym']:8} Logic-{r['logic']}:{r['lname']:10} | "
              f"{pf2_s:>10} {sh2:>10.2f} | "
              f"{pf3_s:>10} {sh3:>10.2f} | "
              f"{pfa_s:>12} {sha:>12.2f} | {best_mode:>10}")

    # ── 最終採用ランキング ────────────────────────────────────────
    print("\n" + "="*110)
    print("  ■ 最終採用ランキング（OOS Sharpe降順 / アダプティブリスク基準）")
    print(f"  Rank  {'銘柄':8} {'カテゴリ':10} {'ロジック':16} | "
          f"{'OOS PF':>8} {'OOS Sh':>8} {'WR':>7} {'MDD':>7} {'Kelly':>7} | {'判定':>10}")
    print("-"*100)

    ranked = sorted(csv_rows, key=lambda x: -x.get("oos_sharpe",0))
    for rank, r in enumerate(ranked, 1):
        pf_s = f"{r['oos_pf']:.2f}" if r.get('oos_pf',0)<99 else "∞"
        lname_full = f"Logic-{r['logic']}:{r['lname']}"
        print(f"  #{rank:<4} {r['sym']:8} {r['cat']:10} {lname_full:16} | "
              f"{pf_s:>8} {r.get('oos_sharpe',0):>8.2f} "
              f"{r.get('oos_wr',0)*100:>6.1f}% {r.get('oos_mdd',0):>6.1f}% "
              f"{r.get('oos_kelly',0):>7.3f} | {r.get('verdict',''):>10}")

    # ── FULL期間（参考） ──────────────────────────────────────────
    print("\n" + "="*110)
    print("  ■ 全期間（FULL）Sharpe / PF（アダプティブリスク）")
    print(f"  {'銘柄':8} {'ロジック':12} | {'期間':22} | "
          f"{'FULL PF':>8} {'FULL Sh':>8} {'MDD':>7} {'月次+':>7} | {'判定':>10}")
    print("-"*95)
    for r in sorted(all_res, key=lambda x: -x.get("FULL_adaptive",{}).get("sharpe",0)):
        full = r.get("FULL_adaptive",{})
        if not full: continue
        oos  = r.get("OOS_adaptive",{})
        of   = of_check(r.get("IS_adaptive",{}).get("pf",0), oos.get("pf",0))
        vd   = verdict(oos.get("pf",0), oos.get("sharpe",0),
                       oos.get("mdd",0), oos.get("plus_m",0),
                       oos.get("total_m",0), of, oos.get("n",0))
        pf_s = f"{full.get('pf',0):.2f}" if full.get('pf',0)<99 else "∞"
        pm   = full.get("plus_m",0); tm = full.get("total_m",0)
        print(f"  {r['sym']:8} Logic-{r['logic']}:{r['lname']:10} | "
              f"{r['start']+'〜'+r['end']:22} | "
              f"{pf_s:>8} {full.get('sharpe',0):>8.2f} "
              f"{full.get('mdd',0):>6.1f}% {pm:>2}/{tm:<2} | {vd:>10}")

    # ── CSV保存 ───────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "backtest_yoto_kento.csv")
    pd.DataFrame(csv_rows).to_csv(out, index=False)
    print(f"\n  結果を保存: {out}")

if __name__ == "__main__":
    main()
