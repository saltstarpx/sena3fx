"""
backtest_lot_cap.py
====================
Exnessロット上限の影響検証バックテスト

【概要】
  資産が成長してExnessの1注文あたりロット上限に達した場合の
  パフォーマンス影響をシミュレーションする。

【Exnessロット上限】
  日中 (UTC 07:00-20:59): 200ロット/注文
  夜間 (UTC 21:00-06:59): 20ロット/注文
  FX: 1ロット = 100,000通貨
  XAUUSD: 1ロット = 100oz

【検証内容】
  1. 各初期資産レベル（100万〜10億）でロット上限ありvs無しを比較
  2. ロット上限にヒットするトレードの割合と資産閾値を特定
  3. 複利成長への影響（成長曲線の「天井効果」）を数値化
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import (
    RiskManager, SYMBOL_CONFIG,
    EXNESS_LOT_DAY, EXNESS_LOT_NIGHT, EXNESS_DAY_START, EXNESS_DAY_END,
)

# ── 定数 ─────────────────────────────────────────────────────────
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

# ── 採用7銘柄（本番構成） ────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "EURUSD",  "logic": "C", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "GBPUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "USDCAD",  "logic": "A", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "NZDUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.20},
    {"sym": "XAUUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.20},
    {"sym": "AUDUSD",  "logic": "B", "risk_pct": 0.02, "tol": 0.30},
]

LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

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
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol_factor=0.30):
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
            body = abs(h4l["close"] - h4l["open"])
            rng  = h4l["high"] - h4l["low"]
            if rng > 0 and body / rng < 0.3: continue

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if logic != "C" and not chk_ema(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol_factor: continue

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

# ── シミュレーション（ロット上限対応版） ──────────────────────────
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

def simulate(signals, d1m, sym, risk_pct, init_cash, apply_lot_cap=False):
    """
    バックテスト実行。

    Parameters
    ----------
    apply_lot_cap : bool
        True: Exnessロット上限を適用（時間帯判定あり）
        False: 上限なし（従来通り）
    """
    if not signals:
        return [], init_cash, 0.0, 0

    rm = RiskManager(sym, risk_pct=risk_pct)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = init_cash; trades = []; peak = init_cash; mdd = 0.0
    cap_count = 0  # ロット上限にヒットした回数

    for sig in signals:
        rm.risk_pct = risk_pct
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)

        # ロット上限キャップ
        if apply_lot_cap:
            entry_hour = sig["time"].hour if hasattr(sig["time"], 'hour') else None
            lot, was_capped = rm.cap_lot(lot, entry_hour_utc=entry_hour)
            if was_capped:
                cap_count += 1

        sp = m1t.searchsorted(sig["time"], side="right")
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
                       "month": sig["time"].strftime("%Y-%m"),
                       "lot": lot, "equity_at_entry": equity - total})
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, cap_count

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades, init_cash):
    if len(trades) < 5: return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loss.sum())
    pf   = gw / gl if gl > 0 else float("inf")

    monthly = df.groupby("month")["pnl"].sum()
    plus_m  = (monthly > 0).sum()

    eq = init_cash
    monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret)
        eq += monthly[m]
    mr     = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0

    avg_w  = wins.mean() if len(wins) > 0 else 0
    avg_l  = abs(loss.mean()) if len(loss) > 0 else 1
    kelly  = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0

    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "final_eq": eq}

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*120)
    print("  Exness ロット上限 影響検証バックテスト")
    print("  日中(UTC07-20): 200ロット/注文 | 夜間(UTC21-06): 20ロット/注文")
    print("="*120)

    # 初期資産レベル
    INIT_LEVELS = [
        (    1_000_000, "100万円"),
        (   10_000_000, "1,000万円"),
        (   50_000_000, "5,000万円"),
        (  100_000_000, "1億円"),
        (  500_000_000, "5億円"),
        (1_000_000_000, "10億円"),
    ]

    all_results = []

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        lname = LOGIC_NAMES[logic]
        tol   = tgt["tol"]

        print(f"\n{'─'*100}")
        print(f"  {sym}  Logic-{logic}:{lname}  tol={tol}")
        print(f"{'─'*100}")

        d1m, d4h = load_all(sym)
        if d1m is None:
            print("  データ未発見 → スキップ")
            continue

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        cs     = cfg.get("contract_size", 100_000)
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, tol_factor=tol)
        n_sigs = len(sigs)
        print(f"  シグナル数: {n_sigs}")
        if n_sigs < 5:
            print("  シグナル不足 → スキップ"); continue

        # 各初期資産レベルで上限なし vs 上限あり比較
        print(f"\n  {'初期資産':>12}  | {'--- 上限なし ---':^36} | {'--- 上限あり ---':^36} | {'差分':^24}")
        print(f"  {'':>12}  | {'最終資産':>14} {'PF':>6} {'MDD':>7} {'Sharpe':>7} | "
              f"{'最終資産':>14} {'PF':>6} {'MDD':>7} {'Sharpe':>7} | "
              f"{'ヒット':>6} {'PF差':>7} {'資産差':>9}")
        print(f"  {'-'*110}")

        for init_cash, label in INIT_LEVELS:
            # 上限なし
            tr_nc, eq_nc, mdd_nc, _ = simulate(sigs, d1m, sym, tgt["risk_pct"], init_cash, apply_lot_cap=False)
            st_nc = calc_stats(tr_nc, init_cash)

            # 上限あり
            tr_cap, eq_cap, mdd_cap, cap_cnt = simulate(sigs, d1m, sym, tgt["risk_pct"], init_cash, apply_lot_cap=True)
            st_cap = calc_stats(tr_cap, init_cash)

            if not st_nc or not st_cap:
                print(f"  {label:>12}  | {'データ不足':^36} |")
                continue

            pf_nc  = st_nc["pf"]  if st_nc["pf"]  < 99 else 99.99
            pf_cap = st_cap["pf"] if st_cap["pf"] < 99 else 99.99
            pf_diff = pf_cap - pf_nc
            eq_diff_pct = (eq_cap - eq_nc) / eq_nc * 100 if eq_nc > 0 else 0

            # ロット上限にヒットした割合
            cap_pct = cap_cnt / len(tr_cap) * 100 if len(tr_cap) > 0 else 0

            print(f"  {label:>12}  | "
                  f"{eq_nc:>14,.0f} {pf_nc:>6.2f} {mdd_nc:>6.1f}% {st_nc['sharpe']:>7.2f} | "
                  f"{eq_cap:>14,.0f} {pf_cap:>6.2f} {mdd_cap:>6.1f}% {st_cap['sharpe']:>7.2f} | "
                  f"{cap_cnt:>3}/{len(tr_cap):<3} {pf_diff:>+7.2f} {eq_diff_pct:>+8.1f}%")

            all_results.append({
                "sym": sym, "logic": logic, "init_cash": init_cash, "label": label,
                "nocap_pf": pf_nc, "nocap_sharpe": st_nc["sharpe"], "nocap_mdd": mdd_nc,
                "nocap_eq": eq_nc, "nocap_n": st_nc["n"],
                "cap_pf": pf_cap, "cap_sharpe": st_cap["sharpe"], "cap_mdd": mdd_cap,
                "cap_eq": eq_cap, "cap_n": st_cap["n"],
                "cap_count": cap_cnt, "cap_pct": cap_pct,
                "pf_diff": pf_diff, "eq_diff_pct": eq_diff_pct,
            })

    # ── ロット上限ヒット閾値の特定 ───────────────────────────────
    print("\n" + "="*120)
    print("  ■ ロット上限ヒット分析（銘柄別）")
    print("="*120)

    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        print("  結果なし")
        return

    for sym in df_res["sym"].unique():
        sub = df_res[df_res["sym"] == sym]
        cfg = SYMBOL_CONFIG[sym]
        cs  = cfg.get("contract_size", 100_000)
        max_day   = cs * EXNESS_LOT_DAY
        max_night = cs * EXNESS_LOT_NIGHT

        print(f"\n  {sym} (1ロット={cs:,}単位, 日中上限={max_day:,}, 夜間上限={max_night:,})")

        # ヒットし始める資産水準
        hit_rows = sub[sub["cap_count"] > 0]
        if hit_rows.empty:
            print(f"    → 10億円まで上限ヒットなし ✅")
        else:
            first_hit = hit_rows.iloc[0]
            print(f"    → {first_hit['label']}から上限ヒット開始")
            for _, r in sub.iterrows():
                if r["cap_count"] > 0:
                    print(f"      {r['label']:>12}: "
                          f"ヒット {r['cap_count']:>3}回/{r['cap_n']}回 ({r['cap_pct']:.1f}%)  "
                          f"PF差 {r['pf_diff']:>+.2f}  資産差 {r['eq_diff_pct']:>+.1f}%")

    # ── 夜間上限のインパクトまとめ ────────────────────────────────
    print("\n" + "="*120)
    print("  ■ ロット上限の実務インパクトまとめ")
    print("="*120)

    # 各銘柄 × 初期資産の組み合わせでヒットありの行を抽出
    hit_df = df_res[df_res["cap_count"] > 0].copy()
    if hit_df.empty:
        print("  全銘柄・全資産水準でロット上限ヒットなし。現行のリスク管理で問題なし。")
    else:
        print(f"\n  ロット上限が影響する資産水準:")
        for sym in hit_df["sym"].unique():
            sub = hit_df[hit_df["sym"] == sym]
            first = sub.iloc[0]
            worst = sub.loc[sub["eq_diff_pct"].idxmin()]
            print(f"    {sym:8} 開始: {first['label']:>12}  "
                  f"最大影響: {worst['label']}で資産{worst['eq_diff_pct']:+.1f}%")

    # ── CSV保存 ───────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "backtest_lot_cap.csv")
    df_res.to_csv(out, index=False)
    print(f"\n  結果保存: {out}")

if __name__ == "__main__":
    main()
