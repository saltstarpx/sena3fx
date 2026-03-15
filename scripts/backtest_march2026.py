"""
backtest_march2026.py
=====================
2026年3月単月バックテスト — 初期資金100万円
全データでインジケーターを計算し、3月のトレードのみを評価する。

採用7銘柄 × 本番構成:
  USDJPY:  Logic-C（オーパーツ）      tol=0.30
  EURUSD:  Logic-C（オーパーツ）      tol=0.30
  GBPUSD:  Logic-A（GOLD）           tol=0.30
  USDCAD:  Logic-A（GOLD）           tol=0.30
  NZDUSD:  Logic-A（GOLD）           tol=0.20
  XAUUSD:  Logic-A（GOLD）           tol=0.20
  AUDUSD:  Logic-B（ADX+Streak+4Hボディ≥0.3）tol=0.30
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
RR_RATIO      = 2.5
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

# ── 本番同一: 資産規模連動リスクテーブル ──────────────────────────
EQUITY_RISK_TABLE = [
    (100_000_000, 0.005),  # 1億〜:       0.5%
    ( 70_000_000, 0.010),  # 7000万〜1億: 1.0%
    ( 50_000_000, 0.015),  # 5000万〜7000万: 1.5%
    ( 30_000_000, 0.020),  # 3000万〜5000万: 2.0%
    ( 10_000_000, 0.025),  # 1000万〜3000万: 2.5%
    (          0, 0.030),  # 〜1000万:    3.0%（加速成長期）
]

def equity_base_risk(equity_jpy):
    for threshold, risk in EQUITY_RISK_TABLE:
        if equity_jpy >= threshold:
            return risk
    return 0.030

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E0_WINDOW_MIN   = 2
ADX_MIN         = 20
STREAK_MIN      = 4

MARCH_START = pd.Timestamp("2026-03-01", tz="UTC")
MARCH_END   = pd.Timestamp("2026-04-01", tz="UTC")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 本番採用7銘柄 ────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "tol": 0.30, "label": "USDJPY (Logic-C オーパーツ)"},
    {"sym": "EURUSD",  "logic": "C", "tol": 0.30, "label": "EURUSD (Logic-C オーパーツ)"},
    {"sym": "GBPUSD",  "logic": "A", "tol": 0.30, "label": "GBPUSD (Logic-A GOLD)"},
    {"sym": "USDCAD",  "logic": "A", "tol": 0.30, "label": "USDCAD (Logic-A GOLD)"},
    {"sym": "NZDUSD",  "logic": "A", "tol": 0.20, "label": "NZDUSD (Logic-A GOLD tol=0.20)"},
    {"sym": "XAUUSD",  "logic": "A", "tol": 0.20, "label": "XAUUSD (Logic-A GOLD tol=0.20)"},
    {"sym": "AUDUSD",  "logic": "B", "tol": 0.30, "label": "AUDUSD (Logic-B ADX+Streak)", "h4_body_ratio_min": 0.3},
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

def load_1m(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            df = load_csv(p)
            if len(df) < 10: continue
            return df
    return None

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
def chk_h4_body(b, min_ratio=0.0):
    if min_ratio <= 0: return True
    rng = b["high"] - b["low"]
    if rng <= 0: return False
    return abs(b["close"] - b["open"]) / rng >= min_ratio

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c,
                     tol_factor=0.30, h4_body_ratio_min=0.0):
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
        if not chk_h4_body(h4l, h4_body_ratio_min): continue
        if logic != "C" and not pd.isna(h4l["atr"]) and h4l["atr"] > 0:
            ema_dist = abs(h4l["close"] - h4l["ema20"])
            if ema_dist < A1_EMA_DIST_MIN * h4l["atr"]:
                continue

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

# ── トレード判定 ──────────────────────────────────────────────────
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

# ── シグナル事前生成（全銘柄） ────────────────────────────────────
def load_all_signals():
    """全銘柄の全期間シグナルを生成して返す"""
    sym_signals = {}
    sym_m1_data = {}

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        tol   = tgt["tol"]
        hbr   = tgt.get("h4_body_ratio_min", 0.0)
        print(f"  {tgt['label']} ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データ未発見"); continue

        sym_l = sym.lower()
        d4h = None
        for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
                  os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
            if os.path.exists(p):
                d4h = load_csv(p); break
        if d4h is None:
            d4h = d1m.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open", "close"])

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c,
                                tol_factor=tol, h4_body_ratio_min=hbr)
        for s in sigs:
            s["sym"] = sym
        sym_signals[sym] = sigs
        sym_m1_data[sym] = {"idx": d1m.index, "highs": d1m["high"].values, "lows": d1m["low"].values}
        print(f"{len(sigs)}シグナル ({d1m.index[0].strftime('%Y/%m')}〜{d1m.index[-1].strftime('%Y/%m')})")

    return sym_signals, sym_m1_data


def run_compound_backtest(init_cash, sym_signals, sym_m1_data):
    """
    資産規模連動リスクテーブルによる複利バックテスト。
    全銘柄のシグナルを時系列統合し、ポートフォリオ単位でエクイティを追跡。
    """
    # 全シグナルを時系列統合
    all_sigs = []
    for sym, sigs in sym_signals.items():
        for s in sigs:
            all_sigs.append(s)
    all_sigs.sort(key=lambda x: x["time"])

    equity = init_cash
    peak = init_cash; mdd = 0.0; mdd_yen = 0.0
    trades = []
    milestones = []  # ボーダー到達記録
    THRESHOLDS = [100_000, 500_000, 1_000_000, 3_000_000, 5_000_000,
                  10_000_000, 30_000_000, 50_000_000, 70_000_000, 100_000_000,
                  300_000_000, 500_000_000, 1_000_000_000]
    reached = set()
    for th in THRESHOLDS:
        if init_cash >= th:
            reached.add(th)

    prev_risk_pct = equity_base_risk(init_cash)

    for sig in all_sigs:
        sym = sig["sym"]
        risk_pct = equity_base_risk(equity)
        rm = RiskManager(sym, risk_pct=risk_pct)
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)

        m1d = sym_m1_data[sym]
        sp = m1d["idx"].searchsorted(sig["time"], side="right")
        if sp >= len(m1d["idx"]): continue

        xp, result, half_done = _exit(m1d["highs"][sp:], m1d["lows"][sp:],
                                       sig["ep"], sig["sl"], sig["tp"],
                                       sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            rem = lot * 0.5
        else:
            rem = lot
        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        total_pnl = half_pnl + pnl
        equity += total_pnl

        # リスク%切替チェック
        new_risk_pct = equity_base_risk(equity)
        risk_changed = ""
        if new_risk_pct != prev_risk_pct:
            risk_changed = f" [RISK {prev_risk_pct*100:.1f}%→{new_risk_pct*100:.1f}%]"
            prev_risk_pct = new_risk_pct

        trades.append({
            "time": sig["time"], "sym": sym, "dir": sig["dir"],
            "ep": sig["ep"], "sl": sig["sl"], "tp": sig["tp"],
            "xp": xp, "result": result, "half": half_done,
            "pnl": total_pnl, "equity": equity, "risk_pct": risk_pct,
            "risk_changed": risk_changed,
            "date": sig["time"].strftime("%m/%d"),
            "week": f"W{sig['time'].isocalendar()[1]}"
        })

        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
            mdd_yen = peak - equity

        # マイルストーン到達チェック
        for th in THRESHOLDS:
            if th not in reached and equity >= th:
                reached.add(th)
                milestones.append({
                    "threshold": th,
                    "time": sig["time"],
                    "equity": equity,
                    "trade_no": len(trades)
                })

    return trades, equity, mdd, mdd_yen, milestones


def print_results(init_cash, trades, final_equity, mdd, mdd_yen, milestones):
    """結果表示"""
    if not trades:
        print(f"\n  初期資金¥{init_cash:,.0f}: トレードなし"); return

    total_pnl = final_equity - init_cash
    df = pd.DataFrame(trades)
    df["month"] = df["time"].apply(lambda x: x.strftime("%Y-%m"))
    total_trades = len(df)
    total_wins = len(df[df["pnl"] > 0])
    total_wr = total_wins / total_trades * 100
    total_gw = df[df["pnl"] > 0]["pnl"].sum()
    total_gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    total_pf = total_gw / total_gl if total_gl > 0 else float("inf")

    first_t = df["time"].min().strftime("%Y/%m/%d")
    last_t = df["time"].max().strftime("%Y/%m/%d")

    print(f"\n{'='*110}")
    print(f"  YAGAMI改 全期間バックテスト — 初期資金 ¥{init_cash:,.0f}")
    print(f"  期間: {first_t} 〜 {last_t}  |  リスク: 資産規模連動テーブル（本番同一）  |  複利運用")
    print(f"{'='*110}")

    # ── 銘柄別集計 ──
    logic_names = {"A": "GOLD", "B": "ADX+Stk", "C": "OP"}
    print(f"\n  {'銘柄':<10} {'Logic':<8} {'取引':>5} {'勝':>4} {'負':>4} "
          f"{'勝率':>6} {'PF':>6} {'損益(¥)':>16}")
    print("  " + "-" * 65)
    for tgt in TARGETS:
        sym = tgt["sym"]
        st = df[df["sym"] == sym]
        if len(st) == 0: continue
        w = len(st[st["pnl"] > 0]); l = len(st[st["pnl"] <= 0])
        wr = w / len(st) * 100
        gw = st[st["pnl"] > 0]["pnl"].sum(); gl = abs(st[st["pnl"] < 0]["pnl"].sum())
        pf = gw / gl if gl > 0 else float("inf")
        pf_s = f"{pf:.2f}" if pf < 99 else "INF"
        print(f"  {sym:<10} {logic_names[tgt['logic']]:<8} {len(st):>5} {w:>4} {l:>4} "
              f"{wr:>5.0f}% {pf_s:>6} {st['pnl'].sum():>+16,.0f}")
    pf_s = f"{total_pf:.2f}" if total_pf < 99 else "INF"
    print("  " + "-" * 65)
    print(f"  {'合計':<10} {'':8} {total_trades:>5} {total_wins:>4} {total_trades-total_wins:>4} "
          f"{total_wr:>5.0f}% {pf_s:>6} {total_pnl:>+16,.0f}")

    # ── サマリー ──
    print(f"\n  初期資金:    ¥{init_cash:>16,.0f}")
    print(f"  最終資金:    ¥{final_equity:>16,.0f}")
    print(f"  総損益:      ¥{total_pnl:>+16,.0f} ({total_pnl/init_cash*100:+.1f}%)")
    print(f"  最大DD:       {mdd:.1f}% (¥{mdd_yen:,.0f})")

    # ── マイルストーン到達日時 ──
    if milestones:
        print(f"\n  資産マイルストーン到達:")
        for m in milestones:
            th_label = f"¥{m['threshold']:>16,.0f}"
            t_str = m["time"].strftime("%Y/%m/%d %H:%M")
            print(f"    {th_label} 到達 → {t_str} (#{m['trade_no']}トレード目, 残高¥{m['equity']:,.0f})")

    # ── リスク%切替ポイント ──
    risk_changes = [t for t in trades if t["risk_changed"]]
    if risk_changes:
        print(f"\n  リスク%切替ポイント:")
        for t in risk_changes:
            t_str = t["time"].strftime("%Y/%m/%d %H:%M")
            print(f"    {t_str} | 残高¥{t['equity']:,.0f}{t['risk_changed']}")

    # ── 月別内訳 ──
    monthly = df.groupby("month").agg(
        trades=("pnl", "count"),
        wins=("pnl", lambda x: (x > 0).sum()),
        pnl=("pnl", "sum")
    )
    print(f"\n  月別:")
    print(f"    {'月':>7} {'件数':>5} {'勝率':>6} {'損益(¥)':>16} {'累計損益(¥)':>16}")
    print("    " + "-" * 55)
    cum_pnl = 0
    plus_months = 0
    for m, row in monthly.iterrows():
        wr_m = row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0
        cum_pnl += row["pnl"]
        sign = "+" if row["pnl"] > 0 else ""
        if row["pnl"] > 0: plus_months += 1
        print(f"    {m:>7} {row['trades']:>5.0f} {wr_m:>5.0f}% {row['pnl']:>+16,.0f} {cum_pnl:>+16,.0f}")
    print(f"    月次プラス: {plus_months}/{len(monthly)} ({plus_months/len(monthly)*100:.0f}%)")

    # ── CSV出力 ──
    suffix = f"_{init_cash//10000}万" if init_cash >= 10000 else f"_{init_cash}"
    out_csv = os.path.join(OUT_DIR, f"backtest_fullperiod_compound{suffix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  CSV保存: {out_csv}")


# ── メイン ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n{'='*110}")
    print(f"  シグナル生成（全銘柄・全期間）")
    print(f"{'='*110}")

    sym_signals, sym_m1_data = load_all_signals()

    total_sigs = sum(len(s) for s in sym_signals.values())
    print(f"\n  合計シグナル: {total_sigs}件")

    # ── 100万円バックテスト ──
    trades_100, eq_100, mdd_100, mdd_yen_100, ms_100 = \
        run_compound_backtest(1_000_000, sym_signals, sym_m1_data)
    print_results(1_000_000, trades_100, eq_100, mdd_100, mdd_yen_100, ms_100)

    # ── 10万円バックテスト ──
    trades_10, eq_10, mdd_10, mdd_yen_10, ms_10 = \
        run_compound_backtest(100_000, sym_signals, sym_m1_data)
    print_results(100_000, trades_10, eq_10, mdd_10, mdd_yen_10, ms_10)

    print(f"\n  総実行時間: {time.time()-t0:.1f}秒")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
