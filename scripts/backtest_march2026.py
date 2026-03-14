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
INIT_CASH     = 1_000_000
RR_RATIO      = 2.5
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000
RISK_PCT      = 0.02   # 固定2%

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

# ── メイン ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n{'='*90}")
    print(f"  YAGAMI改 2026年3月 単月バックテスト")
    print(f"  初期資金: ¥{INIT_CASH:,.0f}  |  リスク: 固定2%  |  期間: 2026/3/1〜3/13")
    print(f"{'='*90}")

    all_trades = []
    sym_results = {}

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        tol   = tgt["tol"]
        hbr   = tgt.get("h4_body_ratio_min", 0.0)
        print(f"\n  {tgt['label']} ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データ未発見"); continue

        # 4hデータ: ohlcディレクトリ優先、なければresample
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

        # 全期間でシグナル生成（インジケーターウォームアップ含む）
        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c,
                                tol_factor=tol, h4_body_ratio_min=hbr)

        # 3月のシグナルのみフィルタ
        march_sigs = [s for s in sigs if MARCH_START <= s["time"] < MARCH_END]
        print(f"全{len(sigs)}sig → 3月{len(march_sigs)}sig", end="", flush=True)

        # シミュレーション（3月のシグナルのみ）
        rm = RiskManager(sym, risk_pct=RISK_PCT)
        m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values

        sym_trades = []
        for sig in march_sigs:
            rm.risk_pct = RISK_PCT
            lot = rm.calc_lot(INIT_CASH, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
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
                rem = lot * 0.5
            else:
                rem = lot

            pnl   = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
            total = half_pnl + pnl

            trade = {
                "time": sig["time"], "sym": sym, "dir": sig["dir"],
                "ep": sig["ep"], "sl": sig["sl"], "tp": sig["tp"],
                "xp": xp, "result": result, "half": half_done,
                "pnl": total, "date": sig["time"].strftime("%m/%d"),
                "week": f"W{sig['time'].isocalendar()[1]}"
            }
            sym_trades.append(trade)
            all_trades.append(trade)

        wins = [t for t in sym_trades if t["pnl"] > 0]
        losses = [t for t in sym_trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in sym_trades)
        wr = len(wins) / len(sym_trades) * 100 if sym_trades else 0
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else float("inf")

        sym_results[sym] = {
            "trades": len(sym_trades), "wins": len(wins), "losses": len(losses),
            "wr": wr, "pf": pf, "pnl": total_pnl, "gw": gw, "gl": gl,
            "trade_list": sym_trades
        }
        pf_s = f"{pf:.2f}" if pf < 99 else "INF"
        print(f" → {len(sym_trades)}トレード WR={wr:.0f}% PF={pf_s} PnL=¥{total_pnl:+,.0f}")

    if not all_trades:
        print("\n3月のトレードなし"); return

    # ── 時系列ソート・ポートフォリオ統計 ──────────────────────────
    all_trades.sort(key=lambda x: x["time"])
    df = pd.DataFrame(all_trades)

    # エクイティカーブ
    equity = INIT_CASH
    peak = INIT_CASH; mdd = 0; mdd_yen = 0
    eq_history = []
    for _, row in df.iterrows():
        equity += row["pnl"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
            mdd_yen = peak - equity
        eq_history.append({"time": row["time"], "equity": equity})

    final_equity = equity
    total_pnl = final_equity - INIT_CASH
    total_trades = len(df)
    total_wins = len(df[df["pnl"] > 0])
    total_wr = total_wins / total_trades * 100
    total_gw = df[df["pnl"] > 0]["pnl"].sum()
    total_gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    total_pf = total_gw / total_gl if total_gl > 0 else float("inf")

    # ── 結果表示 ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ポートフォリオ結果（2026年3月）")
    print(f"{'='*90}")

    print(f"\n  {'銘柄':<10} {'Logic':<8} {'tol':>5} {'取引':>4} {'勝':>3} {'負':>3} "
          f"{'勝率':>6} {'PF':>6} {'損益(¥)':>12} {'利益(¥)':>12} {'損失(¥)':>12}")
    print("  " + "-" * 88)

    logic_names = {"A": "GOLD", "B": "ADX+Stk", "C": "OP"}
    for tgt in TARGETS:
        sym = tgt["sym"]
        if sym not in sym_results: continue
        r = sym_results[sym]
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 99 else "INF"
        print(f"  {sym:<10} {logic_names[tgt['logic']]:<8} {tgt['tol']:>5.2f} "
              f"{r['trades']:>4} {r['wins']:>3} {r['losses']:>3} "
              f"{r['wr']:>5.0f}% {pf_s:>6} {r['pnl']:>+12,.0f} "
              f"{r['gw']:>12,.0f} {-r['gl']:>12,.0f}")

    pf_s = f"{total_pf:.2f}" if total_pf < 99 else "INF"
    print("  " + "-" * 88)
    print(f"  {'合計':<10} {'':8} {'':>5} "
          f"{total_trades:>4} {total_wins:>3} {total_trades-total_wins:>3} "
          f"{total_wr:>5.0f}% {pf_s:>6} {total_pnl:>+12,.0f} "
          f"{total_gw:>12,.0f} {-total_gl:>12,.0f}")

    print(f"\n  初期資金:    ¥{INIT_CASH:>12,.0f}")
    print(f"  最終資金:    ¥{final_equity:>12,.0f}")
    print(f"  総損益:      ¥{total_pnl:>+12,.0f} ({total_pnl/INIT_CASH*100:+.1f}%)")
    print(f"  最大DD:       {mdd:.1f}% (¥{mdd_yen:,.0f})")

    # ── 週別内訳 ──────────────────────────────────────────────────
    weekly = df.groupby("week").agg(
        trades=("pnl", "count"),
        wins=("pnl", lambda x: (x > 0).sum()),
        pnl=("pnl", "sum")
    )
    print(f"\n  週別:")
    for wk, row in weekly.iterrows():
        wr_w = row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0
        print(f"    {wk}: {row['trades']}件 WR={wr_w:.0f}% PnL=¥{row['pnl']:+,.0f}")

    # ── 全トレード詳細 ────────────────────────────────────────────
    print(f"\n  全トレード詳細:")
    print(f"  {'日時':<18} {'銘柄':<8} {'方向':>4} {'EP':>12} {'SL':>12} {'TP':>12} "
          f"{'決済':>12} {'結果':>4} {'半利確':>4} {'損益(¥)':>12}")
    print("  " + "-" * 108)
    for t in all_trades:
        dir_s = "L" if t["dir"] == 1 else "S"
        half_s = "Y" if t["half"] else ""
        time_s = t["time"].strftime("%m/%d %H:%M")
        print(f"  {time_s:<18} {t['sym']:<8} {dir_s:>4} {t['ep']:>12.5f} {t['sl']:>12.5f} "
              f"{t['tp']:>12.5f} {t['xp']:>12.5f} {t['result']:>4} {half_s:>4} {t['pnl']:>+12,.0f}")

    # ── CSV出力 ────────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "backtest_march2026.csv")
    df_out = pd.DataFrame(all_trades)
    df_out.to_csv(out_csv, index=False)
    print(f"\n  CSV保存: {out_csv}")
    print(f"  実行時間: {time.time()-t0:.1f}秒")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
