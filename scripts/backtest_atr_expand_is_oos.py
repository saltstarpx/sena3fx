"""
backtest_atr_expand_is_oos.py
==============================
ATR拡張フィルターの IS/OOS 過学習チェック

【目的】
  lean+EMA_dist (new_baseline) vs lean+EMA_dist+ATR_expand
  を IS と OOS の両期間で比較し、一貫性（過学習なし）を確認する

【期間】
  IS:  2025-01-01 ~ 2025-03-02 (1m データが存在する最初の2ヶ月)
  OOS: 2025-03-03 ~ 2026-02-27 (12ヶ月)

【採用基準】
  ① OOS でも IS と同方向の改善（符号一致）
  ② OOS改善幅 ≥ IS改善幅 × 0.0（少なくとも逆転しないこと）
  ③ FX: 2/3銘柄でIS・OOS両方で改善
  ④ XAUUSD: IS・OOS両方でPF改善

【銘柄・設定】
  FX: EURUSD / GBPUSD / AUDUSD（lean+EMA_dist: streak≥4, E1エントリー）
  XAUUSD: lean+EMA_dist+1d_trend（streak=0, E2エントリー）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 定数 ──────────────────────────────────────────────────────────
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
MAX_LOOKAHEAD = 20_000

A1_EMA_DIST_MIN   = 1.0
A4_ATR_PERIOD     = 5
A3_DEFAULT_TOL    = 0.30
E1_MAX_WAIT_MIN   = 5
E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3

IS_START  = "2025-01-01"
IS_END    = "2025-03-02"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

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

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def load_data(sym_upper, sym_lower):
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")
    def _load(tf):
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p): return load_csv(p)
        # フォールバック: data/ ディレクトリの全期間ファイル
        p2 = os.path.join(DATA_DIR, f"{sym_lower}_{tf}.csv")
        if os.path.exists(p2): return load_csv(p2)
        return None
    return _load("1m"), _load("15m"), _load("4h")


# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if need_1d:
        d1 = df.resample("1D").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna(subset=["open","close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None


def build_1h(data_15m):
    df = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    df["atr"]     = calc_atr(df, 14)
    df["ema20"]   = df["close"].ewm(span=20, adjust=False).mean()
    df["atr_avg"] = df["atr"].rolling(A4_ATR_PERIOD).mean()
    return df


# ── フィルター ────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])


def check_klow(bar):
    o, c, l = bar["open"], bar["close"], bar["low"]
    return (min(o, c) - l) / o < KLOW_THR if o > 0 else False


def check_ema_dist(h4_bar):
    dist = abs(h4_bar["close"] - h4_bar["ema20"])
    atr  = h4_bar["atr"]
    if pd.isna(atr) or atr <= 0: return False
    return dist >= atr * A1_EMA_DIST_MIN


def check_atr_expand(data_1h, signal_time):
    h1b = data_1h[data_1h.index < signal_time]
    if len(h1b) < 2: return False
    latest = h1b.iloc[-1]
    atr_now = latest["atr"]; atr_avg = latest["atr_avg"]
    if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg <= 0: return False
    return atr_now > atr_avg


# ── エントリー ────────────────────────────────────────────────────
def pick_entry_1m(signal_time, direction, spread, method, atr_1m, m1_cache):
    m1_idx = m1_cache["idx"]
    start  = m1_idx.searchsorted(signal_time, side="left")
    if method == "E1":
        end_time = signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            o = m1_cache["opens"][i]; c = m1_cache["closes"][i]
            if direction == 1 and c <= o: continue
            if direction == -1 and c >= o: continue
            ni = i + 1
            if ni >= len(m1_idx): return None, None
            return m1_idx[ni], m1_cache["opens"][ni] + (spread if direction == 1 else -spread)
        return None, None
    else:  # E2
        win_min = max(2, E2_ALT_WINDOW_MIN)
        end_time = signal_time + pd.Timedelta(minutes=win_min)
        end = m1_idx.searchsorted(end_time, side="left")
        for i in range(start, min(end, len(m1_idx))):
            bar_time  = m1_idx[i]
            bar_range = m1_cache["highs"][i] - m1_cache["lows"][i]
            if atr_1m is not None:
                atr_val = atr_1m.get(bar_time, np.nan)
                if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                    continue
            return bar_time, m1_cache["opens"][i] + (spread if direction == 1 else -spread)
        return None, None


# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(data_1m, data_15m, data_4h,
                     spread_pips, pip_size, sym_cfg,
                     use_atr_expand, atr_1m=None, m1_cache=None):
    spread    = spread_pips * pip_size
    streak    = sym_cfg["streak"]
    need_1d   = sym_cfg["use_1d"]
    method    = sym_cfg["entry"]

    data_4h, data_1d = build_4h(data_4h, need_1d)
    data_1h = build_1h(data_15m)

    if m1_cache is None:
        m1_cache = {
            "idx":    data_1m.index,
            "opens":  data_1m["open"].values,
            "closes": data_1m["close"].values,
            "highs":  data_1m["high"].values,
            "lows":   data_1m["low"].values,
        }

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()
    min_idx    = max(2, streak if streak > 0 else 2)

    for i in range(min_idx, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak if streak > 0 else 2, 2): continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # Streak
        if streak > 0:
            recent = h4_before["trend"].iloc[-streak:].values
            if not all(t == trend for t in recent): continue

        # 1d EMA（XAUUSDのみ）
        if need_1d and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0: continue
            if d1_before.iloc[-1]["trend1d"] != trend: continue

        # KMID / KLOW
        if not check_kmid(h4_latest, trend): continue
        if not check_klow(h4_latest): continue

        # EMA距離（常時ON: new_baseline）
        if not check_ema_dist(h4_latest): continue

        # ATR拡張（オプション）
        if use_atr_expand and not check_atr_expand(data_1h, h1_ct): continue

        tol = atr_val * A3_DEFAULT_TOL

        for direction in [1, -1]:
            if trend != direction: continue
            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
            if abs(v1 - v2) > tol: continue

            et, ep = pick_entry_1m(h1_ct, direction, spread, method, atr_1m, m1_cache)
            if et is None or et in used_times: continue

            raw = ep - spread if direction == 1 else ep + spread
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({"time": et, "dir": direction,
                                 "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── シミュレーション ──────────────────────────────────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half_price = ep + direction * risk * HALF_R
    limit = min(len(highs), MAX_LOOKAHEAD)
    for i in range(limit):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl: return i, sl, "loss", False, -1
            if h  >= tp: return i, tp, "win",  False, -1
            if h  >= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if lows[j]  <= be_sl: return j, be_sl, "win", True, i
                    if highs[j] >= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
        else:
            if h  >= sl: return i, sl, "loss", False, -1
            if lo <= tp: return i, tp, "win",  False, -1
            if lo <= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if highs[j] >= be_sl: return j, be_sl, "win", True, i
                    if lows[j]  <= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
    return -1, None, None, False, -1


def simulate(signals, data_1m, symbol):
    if not signals: return [], [INIT_CASH]
    rm = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH; trades = []; eq_curve = [INIT_CASH]
    m1_times = data_1m.index
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=USDJPY_RATE)
        start_pos = m1_times.searchsorted(sig["time"], side="right")
        if start_pos >= len(m1_times): continue

        exit_i, exit_price, result, half_done, half_i = _find_exit(
            m1_highs[start_pos:], m1_lows[start_pos:],
            ep, sl, tp, risk, direction
        )
        if result is None: continue

        if half_done and half_i >= 0:
            half_ep = ep + direction * risk * HALF_R
            equity += rm.calc_pnl_jpy(direction, ep, half_ep, lot*0.5, USDJPY_RATE, ep)
            remaining_lot = lot * 0.5
        else:
            remaining_lot = lot

        equity += rm.calc_pnl_jpy(direction, ep, exit_price, remaining_lot, USDJPY_RATE, ep)
        exit_time = m1_times[start_pos + exit_i]
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve


def calc_stats(trades, eq_curve, label):
    if not trades:
        return {"variant": label, "n": 0, "wr": 0.0, "pf": 0.0,
                "mdd_pct": 0.0, "monthly_plus": "0/0"}
    df  = pd.DataFrame(trades)
    n   = len(df)
    wr  = (df["result"] == "win").mean()
    eq  = np.array(eq_curve)
    dlt = np.diff(eq)
    gw  = dlt[dlt > 0].sum()
    gl  = abs(dlt[dlt < 0].sum())
    pf  = gw / gl if gl > 0 else float("inf")
    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq - peak) / peak).min()) * 100
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev    = monthly.shift(1).fillna(INIT_CASH)
    mp      = f"{(monthly > prev).sum()}/{len(monthly)}"
    return {"variant": label, "n": n, "wr": round(wr*100,1),
            "pf": round(pf,2), "mdd_pct": round(mdd,1), "monthly_plus": mp}


# ── メイン ───────────────────────────────────────────────────────
def main():
    all_rows = []

    for sym_cfg in SYMBOLS:
        sym_name  = sym_cfg["name"]
        sym_lower = sym_cfg["lower"]
        category  = sym_cfg["category"]

        print(f"\n{'='*65}")
        print(f"  {sym_name}  [{category}]")
        print(f"{'='*65}")

        d1m, d15m, d4h = load_data(sym_name, sym_lower)
        if d1m is None or d15m is None or d4h is None:
            print(f"  [SKIP] データ不足"); continue

        # IS / OOS に分割
        d1m_is   = slice_period(d1m,  IS_START,  IS_END)
        d15m_is  = slice_period(d15m, IS_START,  IS_END)
        d4h_is   = slice_period(d4h,  IS_START,  IS_END)
        # 4H ISには前後バッファも含めて渡す（トレンド計算用）
        d4h_is_buf  = slice_period(d4h, "2024-12-01", IS_END)

        d1m_oos  = slice_period(d1m,  OOS_START, OOS_END)
        d15m_oos = slice_period(d15m, OOS_START, OOS_END)
        d4h_oos  = slice_period(d4h,  OOS_START, OOS_END)
        # 4H OOSにも前バッファ
        d4h_oos_buf = slice_period(d4h, "2025-01-01", OOS_END)

        if d1m_is is None or len(d1m_is) == 0:
            print(f"  [SKIP] IS 1m なし"); continue
        if d1m_oos is None or len(d1m_oos) == 0:
            print(f"  [SKIP] OOS 1m なし"); continue

        print(f"  IS  1m: {len(d1m_is):,} bars (4h: {len(d4h_is):,})")
        print(f"  OOS 1m: {len(d1m_oos):,} bars (4h: {len(d4h_oos):,})")

        cfg         = SYMBOL_CONFIG.get(sym_name, {})
        spread_pips = cfg.get("spread", 0.0)
        pip_size    = cfg.get("pip", 0.0001)

        for period_label, d1m_p, d15m_p, d4h_p in [
            ("IS",  d1m_is,  d15m_is,  d4h_is_buf),
            ("OOS", d1m_oos, d15m_oos, d4h_oos_buf),
        ]:
            atr_1m = calc_atr(d1m_p, 10).to_dict()
            m1_cache = {
                "idx":    d1m_p.index,
                "opens":  d1m_p["open"].values,
                "closes": d1m_p["close"].values,
                "highs":  d1m_p["high"].values,
                "lows":   d1m_p["low"].values,
            }

            print(f"\n  -- {period_label} --")
            results_period = []
            for use_atr_expand, label in [(False, "new_baseline"), (True, "+ATR_expand")]:
                sigs = generate_signals(
                    d1m_p, d15m_p, d4h_p,
                    spread_pips, pip_size, sym_cfg,
                    use_atr_expand, atr_1m, m1_cache,
                )
                trades, eq_curve = simulate(sigs, d1m_p, sym_name)
                stats = calc_stats(trades, eq_curve, label)
                stats["symbol"]   = sym_name
                stats["period"]   = period_label
                stats["category"] = category
                results_period.append(stats)
                all_rows.append(stats)

                print(f"    [{label:14s}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                      f"PF={stats['pf']:.2f}  MDD={stats['mdd_pct']:.1f}%  月次+={stats['monthly_plus']}")

    # ── IS/OOS 整合性レポート ─────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ATR_expand IS/OOS 整合性チェック（new_baseline比）")
    print("="*70)

    df_all = pd.DataFrame(all_rows)

    print("\n■ 銘柄別 IS/OOS PF変化（new_baseline比）")
    print(f"  {'銘柄':8s} {'IS_base':>8s} {'IS_atr':>8s} {'IS_diff':>8s} | "
          f"{'OOS_base':>9s} {'OOS_atr':>9s} {'OOS_diff':>9s} | {'符号一致':>8s} {'採用判定':>10s}")
    print("  " + "-"*80)

    consistency_fx   = []
    consistency_xau  = None

    for sym_cfg in SYMBOLS:
        sym  = sym_cfg["name"]
        cat  = sym_cfg["category"]
        df_s = df_all[df_all["symbol"] == sym]
        if len(df_s) == 0: continue

        is_bl  = df_s[(df_s["period"] == "IS")  & (df_s["variant"] == "new_baseline")]["pf"].values
        is_atr = df_s[(df_s["period"] == "IS")  & (df_s["variant"] == "+ATR_expand")]["pf"].values
        os_bl  = df_s[(df_s["period"] == "OOS") & (df_s["variant"] == "new_baseline")]["pf"].values
        os_atr = df_s[(df_s["period"] == "OOS") & (df_s["variant"] == "+ATR_expand")]["pf"].values

        if len(is_bl) == 0 or len(os_bl) == 0: continue

        is_bl  = is_bl[0];  is_atr = is_atr[0] if len(is_atr) > 0 else None
        os_bl  = os_bl[0];  os_atr = os_atr[0] if len(os_atr) > 0 else None
        if is_atr is None or os_atr is None: continue

        is_diff  = is_atr  - is_bl
        oos_diff = os_atr  - os_bl
        sign_ok  = (is_diff >= 0 and oos_diff >= 0) or (is_diff < 0 and oos_diff < 0)
        verdict  = "✅ 一致" if sign_ok else "❌ 逆転"

        is_diff_s  = f"{'+' if is_diff >= 0 else ''}{is_diff:.2f}"
        oos_diff_s = f"{'+' if oos_diff >= 0 else ''}{oos_diff:.2f}"

        print(f"  {sym:8s} {is_bl:8.2f} {is_atr:8.2f} {is_diff_s:>8s} | "
              f"{os_bl:9.2f} {os_atr:9.2f} {oos_diff_s:>9s} | {verdict:>8s}  {verdict:>10s}")

        if cat == "FX":
            consistency_fx.append((sym, is_diff, oos_diff, sign_ok, oos_diff >= 0))
        else:
            consistency_xau = (sym, is_diff, oos_diff, sign_ok, oos_diff >= 0)

    # カテゴリ判定
    print("\n■ カテゴリ採用判定")
    if consistency_fx:
        fx_improved  = sum(1 for _, _, _, _, improved in consistency_fx if improved)
        fx_consistent = sum(1 for _, _, _, ok, _ in consistency_fx if ok)
        total_fx = len(consistency_fx)
        fx_verdict = "✅ 採用" if fx_improved >= 2 and fx_consistent >= 2 else "❌ 不採用"
        print(f"  FX: OOS改善 {fx_improved}/{total_fx}銘柄, IS/OOS符号一致 {fx_consistent}/{total_fx} → {fx_verdict}")
        for sym, isd, oosd, ok, imp in consistency_fx:
            tag = "✅" if imp and ok else ("⚠️" if imp else "❌")
            print(f"    {tag} {sym}: IS {'+' if isd>=0 else ''}{isd:.2f} / OOS {'+' if oosd>=0 else ''}{oosd:.2f}")

    if consistency_xau:
        sym, isd, oosd, ok, imp = consistency_xau
        xau_verdict = "✅ 採用" if imp and ok else "❌ 不採用"
        print(f"  XAUUSD: OOS改善 {'✅' if imp else '❌'}, IS/OOS符号一致 {'✅' if ok else '❌'} → {xau_verdict}")
        print(f"    {sym}: IS {'+' if isd>=0 else ''}{isd:.2f} / OOS {'+' if oosd>=0 else ''}{oosd:.2f}")

    # 最終推奨
    print("\n■ 最終推奨（v80向け）")
    if consistency_fx:
        fx_improved  = sum(1 for _, _, _, _, improved in consistency_fx if improved)
        fx_consistent = sum(1 for _, _, _, ok, _ in consistency_fx if ok)
        if fx_improved >= 2 and fx_consistent >= 2:
            print("  FX(EURUSD/GBPUSD/AUDUSD): lean + EMA_dist + ATR_expand を採用")
        else:
            print("  FX(EURUSD/GBPUSD/AUDUSD): lean + EMA_dist のみ（ATR_expand 不採用）")
    if consistency_xau:
        sym, isd, oosd, ok, imp = consistency_xau
        if imp and ok:
            print("  XAUUSD: lean + 1d_trend + EMA_dist + ATR_expand を採用")
        else:
            print("  XAUUSD: lean + 1d_trend + EMA_dist のみ（ATR_expand 不採用）")

    # 保存
    out_csv = os.path.join(OUT_DIR, "atr_expand_is_oos_check.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\nCSV保存: {out_csv}")


if __name__ == "__main__":
    main()
