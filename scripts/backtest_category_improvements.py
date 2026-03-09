"""
backtest_category_improvements.py
===================================
カテゴリ別（FX / 貴金属 / 指数）に v77→v78 改善効果を横断的に検証する。

【目的】
  v78 改善はXAUUSDのOOSデータで発見された → XAUUSD専用に過学習している可能性がある。
  同じ改善がカテゴリ内の複数銘柄で一貫して効果があるかを確認し、汎用性を保証する。

【カテゴリ定義】
  FX      : EURUSD / GBPUSD / AUDUSD（1mデータあり）
  METALS  : XAUUSD（貴金属代表）
  INDICES : US30 / SPX500 / NAS100（米国指数）
  ※ USDJPYは1mデータなし（15mで代替）のため別枠参照

【セッションフィルター（経済的根拠ベース、OOSデータ非依存）】
  FX / METALS : UTC  7-22  （London 8-16 + NY 13-22 → 主要流動性セッション）
  INDICES     : UTC 14-22  （NYSE通常取引時間 14:30-21:00、先物翌朝まで）

【過学習チェック基準】
  カテゴリPASS  : 改善がカテゴリ内銘柄の過半数でOOS PFを改善（FX≥2/3, METALS=1/1, INDICES≥2/3）
  過学習フラグ  : IS改善率とOOS改善率の乖離が大きい場合に警告
  推奨除外基準  : カテゴリ内で改善する銘柄が過半数未満の改善は採用しない

IS : 2024-07-01 〜 2025-02-28  (約8ヶ月、最適化期間)
OOS: 2025-03-03 〜 2026-02-27  (約12ヶ月、未知データ)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 定数 ──────────────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
KLOW_THR  = 0.0015     # v77継承（データ非依存、固定値）
IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# ── カテゴリ定義 ────────────────────────────────────────────────
# utc_start/utc_end は「経済的根拠」で設定（OOSデータを見て調整しない）
# FX/METALS: London+NY主要セッション (UTC7-22)
# INDICES  : NYSE通常取引時間      (UTC14-22)
CATEGORIES = {
    "FX": {
        "symbols": [
            ("EURUSD", "eurusd"),
            ("GBPUSD", "gbpusd"),
            ("AUDUSD", "audusd"),
        ],
        "utc_start": 7,
        "utc_end":   22,
        "session":   "London+NY (UTC7-22)",
        "pass_min":  2,   # カテゴリPASS: 最低N銘柄でOOS PFが改善すること
    },
    "METALS": {
        "symbols": [
            ("XAUUSD", "xauusd"),
        ],
        "utc_start": 7,
        "utc_end":   22,
        "session":   "London+NY (UTC7-22)",
        "pass_min":  1,
    },
    "INDICES": {
        "symbols": [
            ("US30",   "us30"),
            ("SPX500", "spx500"),
            ("NAS100", "nas100"),
        ],
        "utc_start": 14,
        "utc_end":   22,
        "session":   "NYSE通常取引 (UTC14-22)",
        "pass_min":  2,
    },
}

# ── 改善バリアント（累積適用、カテゴリ固定パラメータを使用）──────
# 注意: time_filter は各カテゴリの utc_start/utc_end を使用（銘柄ごとに変えない）
VARIANTS = [
    ("v77",   dict(h1_klow=False, time_filter=False, body_min=False, tight_tol=False)),
    ("v78A",  dict(h1_klow=True,  time_filter=False, body_min=False, tight_tol=False)),
    ("v78B",  dict(h1_klow=True,  time_filter=True,  body_min=False, tight_tol=False)),
    ("v78C",  dict(h1_klow=True,  time_filter=True,  body_min=True,  tight_tol=False)),
    ("v78D",  dict(h1_klow=True,  time_filter=True,  body_min=True,  tight_tol=True)),
]

# ── データロード ────────────────────────────────────────────────
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

def load_1m(sym, is_period=True):
    tag = "is" if is_period else "oos"
    df = load_csv(os.path.join(DATA_DIR, f"{sym}_{tag}_1m.csv"))
    if df is None:
        df = load_csv(os.path.join(DATA_DIR, f"{sym}_1m.csv"))
    if df is None:
        # data/ohlc/ の全期間ファイルを探す
        df = load_csv(os.path.join(DATA_DIR, "ohlc", f"{sym.upper()}_1m.csv"))
    if df is None: return None
    start, end = (IS_START, IS_END) if is_period else (OOS_START, OOS_END)
    return slice_period(df, start, end)

# ── インジケーター ──────────────────────────────────────────────
def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    df["atr"]   = calculate_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ok = (body_bottom - l) / o < KLOW_THR if o > 0 else False
    return kmid_ok and klow_ok

# ── シグナル生成 ────────────────────────────────────────────────
def generate_signals_1h(data_1m, data_15m, data_4h,
                        spread_pips, pip_size,
                        h1_klow, time_filter, body_min, tight_tol,
                        utc_start=7, utc_end=22):
    """
    1Hパターンによるシグナル生成。
    utc_start/utc_end はカテゴリ固定値を受け取る（銘柄個別チューニングなし）。
    """
    spread = spread_pips * pip_size

    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    tol_factor    = 0.2 if tight_tol else 0.3   # 改善D
    body_min_fac  = 0.2                           # 改善C（固定値、データ非依存）

    signals = []; used_times = set()
    h1_times = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        # 改善B: セッションフィルター（カテゴリ固定値、銘柄個別チューニングなし）
        if time_filter and not (utc_start <= h1_ct.hour < utc_end):
            continue

        # 完結済み4H足のみ取得（look-ahead bias 修正済み）
        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) == 0: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tol    = atr_val * tol_factor

        for direction in [1, -1]:
            if trend != direction: continue

            if direction == 1:
                v1, v2  = h1_prev2["low"],  h1_prev1["low"]
                conf_ok = h1_prev1["close"] > h1_prev1["open"]
            else:
                v1, v2  = h1_prev2["high"], h1_prev1["high"]
                conf_ok = h1_prev1["close"] < h1_prev1["open"]

            if abs(v1 - v2) > tol: continue
            if not conf_ok: continue

            # 改善C: 確認足の実体サイズ（固定値 ATR×0.2、データ非依存）
            if body_min:
                body = abs(h1_prev1["close"] - h1_prev1["open"])
                if body < atr_val * body_min_fac:
                    continue

            # 4H文脈足 KMID+KLOW（v77 bug① 修正済み）
            if not check_kmid_klow(h4_latest, direction): continue

            # 改善A: 1H確認足 KLOW
            if h1_klow and not check_kmid_klow(h1_prev1, direction): continue

            m1w = data_1m[
                (data_1m.index >= h1_ct) &
                (data_1m.index <  h1_ct + pd.Timedelta(minutes=2))
            ]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue

            raw = eb["open"]
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                ep   = raw + spread
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                ep   = raw - spread
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ────────────────────────────────────────────
def simulate(signals, data_1m, symbol):
    if not signals:
        return [], [INIT_CASH]
    rm     = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH
    trades = []; eq_curve = [INIT_CASH]

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot    = rm.calc_lot(equity, risk, ep, usdjpy_rate=150.0)
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0: continue

        half_done = False; be_sl = None
        result = None; exit_price = None; exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                cur_sl = be_sl if half_done else sl
                if bar["low"] <= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(1, ep, exit_price, lot*rem, 150.0, ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * HALF_R:
                        equity += rm.calc_pnl_jpy(1, ep, ep+risk*HALF_R, lot*0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(1, ep, tp, lot*rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["high"] >= ep + risk * HALF_R:
                    equity += rm.calc_pnl_jpy(1, ep, ep+risk*HALF_R, lot*0.5, 150.0, ep)
                    half_done = True; be_sl = ep
            else:
                cur_sl = be_sl if half_done else sl
                if bar["high"] >= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(-1, ep, exit_price, lot*rem, 150.0, ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * HALF_R:
                        equity += rm.calc_pnl_jpy(-1, ep, ep-risk*HALF_R, lot*0.5, 150.0, ep)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(-1, ep, tp, lot*rem, 150.0, ep)
                    result = "win"; break
                if not half_done and bar["low"] <= ep - risk * HALF_R:
                    equity += rm.calc_pnl_jpy(-1, ep, ep-risk*HALF_R, lot*0.5, 150.0, ep)
                    half_done = True; be_sl = ep

        if result is None: continue
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve

# ── 統計計算 ────────────────────────────────────────────────────
def calc_stats(trades, eq_curve, pair, variant, period):
    if not trades:
        return {
            "pair": pair, "variant": variant, "period": period,
            "n": 0, "wr": 0.0, "pf": 0.0, "mdd": 0.0,
            "kelly": -1.0, "monthly_plus": "0/0"
        }
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["result"] == "win"]
    wr   = len(wins) / n

    eq     = np.array(eq_curve)
    deltas = np.diff(eq)
    gw     = deltas[deltas > 0].sum()
    gl     = abs(deltas[deltas < 0].sum())
    pf     = gw / gl if gl > 0 else float("inf")

    peak   = np.maximum.accumulate(eq)
    mdd    = abs(((eq - peak) / peak).min()) * 100
    kelly  = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly  = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev_eq  = monthly.shift(1).fillna(INIT_CASH)
    mp       = f"{(monthly > prev_eq).sum()}/{len(monthly)}"

    return {
        "pair":         pair,
        "variant":      variant,
        "period":       period,
        "n":            n,
        "wr":           round(wr * 100, 1),
        "pf":           round(pf,   2),
        "mdd":          round(mdd,  1),
        "kelly":        round(kelly, 3),
        "monthly_plus": mp,
    }

# ── カテゴリ過学習チェック ──────────────────────────────────────
def overfitting_check(cat_results, variant_name, baseline="v77"):
    """
    IS改善率とOOS改善率を比較。
    IS改善がOOS改善を大きく上回る場合は過学習の疑い。
    """
    is_rows  = {r["pair"]: r for r in cat_results if r["period"] == "IS"  and r["variant"] == variant_name}
    oos_rows = {r["pair"]: r for r in cat_results if r["period"] == "OOS" and r["variant"] == variant_name}
    is_base  = {r["pair"]: r for r in cat_results if r["period"] == "IS"  and r["variant"] == baseline}
    oos_base = {r["pair"]: r for r in cat_results if r["period"] == "OOS" and r["variant"] == baseline}

    is_deltas  = []
    oos_deltas = []
    for pair in is_rows:
        if pair in is_base and pair in oos_rows and pair in oos_base:
            is_deltas.append( is_rows[pair]["pf"]  -  is_base[pair]["pf"])
            oos_deltas.append(oos_rows[pair]["pf"] - oos_base[pair]["pf"])

    if not is_deltas:
        return "N/A", False

    avg_is  = sum(is_deltas)  / len(is_deltas)
    avg_oos = sum(oos_deltas) / len(oos_deltas)

    # IS改善がOOS改善の2倍以上 かつ IS>0 → 過学習フラグ
    overfit = (avg_is > 0.1) and (avg_oos < avg_is * 0.5)
    ratio_str = f"IS+{avg_is:+.2f} / OOS{avg_oos:+.2f}"
    return ratio_str, overfit

# ── メイン ──────────────────────────────────────────────────────
print("=" * 100)
print("カテゴリ別改善効果バックテスト（過学習チェック付き）")
print(f"IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
print("【過学習対策】カテゴリ内同一セッション時間 / カテゴリ内過半数一致のみ採用 / IS/OOS乖離チェック")
print("=" * 100)

all_results = []          # 全銘柄・全バリアントの統計
category_recommendation = {}  # カテゴリ別推奨バリアント

for cat_name, cat_cfg in CATEGORIES.items():
    symbols    = cat_cfg["symbols"]
    utc_start  = cat_cfg["utc_start"]
    utc_end    = cat_cfg["utc_end"]
    session    = cat_cfg["session"]
    pass_min   = cat_cfg["pass_min"]
    n_symbols  = len(symbols)

    print(f"\n{'#'*100}")
    print(f"  CATEGORY: {cat_name}  ({n_symbols}銘柄)  セッション: {session}")
    print(f"  銘柄: {', '.join(s[0] for s in symbols)}")
    print(f"  カテゴリPASS基準: {pass_min}/{n_symbols}銘柄以上でOOS PF改善")
    print(f"{'#'*100}")

    cat_results = []  # このカテゴリの全結果

    for pair_name, sym in symbols:
        rm     = RiskManager(pair_name, risk_pct=RISK_PCT)
        spread = rm.spread_pips
        pip    = rm.pip_size

        # データロード
        d1m_is   = load_1m(sym, is_period=True)
        d15m_is  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv")),  IS_START,  IS_END)
        d4h_is   = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv")),   IS_START,  IS_END)
        d1m_oos  = load_1m(sym, is_period=False)
        d15m_oos = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv")), OOS_START, OOS_END)
        d4h_oos  = slice_period(load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv")),  OOS_START, OOS_END)

        # data/ohlc/ フォールバック
        if d15m_is is None:
            raw = load_csv(os.path.join(DATA_DIR, "ohlc", f"{pair_name}_15m.csv"))
            if raw is not None:
                d15m_is  = slice_period(raw, IS_START,  IS_END)
                d15m_oos = slice_period(raw, OOS_START, OOS_END)
        if d4h_is is None:
            raw = load_csv(os.path.join(DATA_DIR, "ohlc", f"{pair_name}_4h.csv"))
            if raw is not None:
                d4h_is  = slice_period(raw, IS_START,  IS_END)
                d4h_oos = slice_period(raw, OOS_START, OOS_END)

        missing = [(n, d) for n, d in [("1m_IS",d1m_is),("15m_IS",d15m_is),("4h_IS",d4h_is),
                                        ("1m_OOS",d1m_oos),("15m_OOS",d15m_oos),("4h_OOS",d4h_oos)]
                   if d is None or len(d) == 0]
        if missing:
            print(f"\n  {pair_name}: [SKIP] データ不足 → {[n for n,_ in missing]}")
            continue

        print(f"\n  ── {pair_name}  spread:{spread}pips  ────────────────────────────")
        print(f"  {'バリアント':<8} {'期間':<4} {'件数':>5} {'勝率':>7} {'PF':>6} "
              f"{'MDD':>7} {'Kelly':>7} {'月次+'}")
        print(f"  {'-'*70}")

        for vname, vflags in VARIANTS:
            for period, d1m, d15m, d4h in [
                ("IS",  d1m_is,  d15m_is,  d4h_is),
                ("OOS", d1m_oos, d15m_oos, d4h_oos)
            ]:
                sigs   = generate_signals_1h(
                    d1m, d15m, d4h, spread, pip,
                    utc_start=utc_start, utc_end=utc_end,
                    **vflags
                )
                trades, eq = simulate(sigs, d1m, pair_name)
                st = calc_stats(trades, eq, pair_name, vname, period)
                cat_results.append(st)
                all_results.append(st)
                print(f"  {vname:<8} {period:<4} {st['n']:>5} "
                      f"{st['wr']:>6.1f}% {st['pf']:>6.2f} "
                      f"{st['mdd']:>6.1f}% {st['kelly']:>7.3f} "
                      f"{st['monthly_plus']:>8}")

    if not cat_results:
        print(f"\n  {cat_name}: データ不足のためスキップ")
        continue

    # ── カテゴリ集計 ──────────────────────────────────────────
    print(f"\n  ── {cat_name} カテゴリ集計（OOS） ─────────────────────────────")
    print(f"  {'バリアント':<8} {'avg_PF':>7} {'改善銘柄':>8} {'過学習チェック':<30} {'カテゴリ判定'}")
    print(f"  {'-'*75}")

    v77_oos_pfs = {}
    for pair_name, _ in symbols:
        rows = [r for r in cat_results if r["pair"] == pair_name and
                r["period"] == "OOS" and r["variant"] == "v77"]
        if rows:
            v77_oos_pfs[pair_name] = rows[0]["pf"]

    best_variant = "v77"
    best_avg_pf  = sum(v77_oos_pfs.values()) / len(v77_oos_pfs) if v77_oos_pfs else 0

    for vname, _ in VARIANTS:
        oos_rows = [r for r in cat_results if r["period"] == "OOS" and r["variant"] == vname]
        if not oos_rows: continue

        pairs_present = [r["pair"] for r in oos_rows]
        avg_pf = sum(r["pf"] for r in oos_rows) / len(oos_rows)

        # 改善銘柄数（v77ベースライン比）
        improved = sum(
            1 for r in oos_rows
            if v77_oos_pfs.get(r["pair"], 0) < r["pf"]
        )

        # IS/OOS乖離チェック
        ratio_str, overfit = overfitting_check(cat_results, vname)
        overfit_mark = " ⚠️過学習疑い" if overfit else ""

        # カテゴリ判定
        cat_pass = (improved >= pass_min)
        mark = "✅ PASS" if cat_pass else "❌ FAIL"

        print(f"  {vname:<8} {avg_pf:>7.2f}  "
              f"{improved:>2}/{len(oos_rows):<5}  "
              f"{ratio_str:<30} {mark}{overfit_mark}")

        if cat_pass and avg_pf > best_avg_pf:
            best_avg_pf  = avg_pf
            best_variant = vname

    # 推奨まとめ
    cat_best_oos = [r for r in cat_results if r["period"] == "OOS" and r["variant"] == best_variant]
    recommendation = {
        "category":     cat_name,
        "session":      session,
        "best_variant": best_variant,
        "avg_pf":       round(best_avg_pf, 2),
        "symbols":      {r["pair"]: r["pf"] for r in cat_best_oos},
    }
    category_recommendation[cat_name] = recommendation

    print(f"\n  ★ {cat_name} 推奨バリアント: {best_variant}  (カテゴリ平均OOS PF: {best_avg_pf:.2f})")
    print(f"  ★ セッションフィルター: {session}（経済的根拠ベース・データ非依存）")

# ── 最終推奨サマリー ────────────────────────────────────────────
print("\n" + "=" * 100)
print("最終推奨サマリー（カテゴリ別）")
print("=" * 100)
print(f"{'カテゴリ':<10} {'推奨バリアント':<12} {'セッション':<25} {'avg PF':>7}  銘柄別OOS PF")
print("-" * 100)
for cat_name, rec in category_recommendation.items():
    sym_pfs = "  ".join(f"{p}:{pf:.2f}" for p, pf in rec["symbols"].items())
    print(f"{cat_name:<10} {rec['best_variant']:<12} {rec['session']:<25} {rec['avg_pf']:>7.2f}  {sym_pfs}")

print("\n【過学習対策の実施内容】")
print("  1. セッションフィルターはOOSデータ非依存（London+NY/NYSE通常取引で経済的根拠あり）")
print("  2. 改善パラメータ（ATR×0.2等）は全カテゴリ固定（銘柄ごとにチューニングしない）")
print("  3. カテゴリPASS = 過半数銘柄で改善（1銘柄だけ改善のケースは採用しない）")
print("  4. IS/OOS乖離をチェック（IS改善>>OOS改善の場合は過学習フラグ）")

# ── CSV保存 ──────────────────────────────────────────────────────
df_all  = pd.DataFrame(all_results)
csv_out = os.path.join(OUT_DIR, "category_improvements.csv")
df_all.to_csv(csv_out, index=False)
print(f"\n結果CSV: {csv_out}")

# ── 可視化 ──────────────────────────────────────────────────────
cat_colors  = {"FX": "#3b82f6", "METALS": "#f59e0b", "INDICES": "#10b981"}
var_markers = {"v77": "o", "v78A": "s", "v78B": "^", "v78C": "D", "v78D": "P"}
variant_names = [v for v, _ in VARIANTS]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
fig.suptitle("カテゴリ別 OOS PF推移（v77→v78D）\n"
             "セッション: FX/METALS=London+NY(UTC7-22) / INDICES=NYSE(UTC14-22)\n"
             "★はカテゴリ推奨バリアント",
             fontsize=11, fontweight="bold")

for ax, (cat_name, cat_cfg) in zip(axes, CATEGORIES.items()):
    symbols = [s[0] for s in cat_cfg["symbols"]]
    color   = cat_colors.get(cat_name, "#6b7280")
    rec     = category_recommendation.get(cat_name, {})
    best_v  = rec.get("best_variant", "v77")

    x = range(len(variant_names))
    for pair_name in symbols:
        pfs = []
        for vname in variant_names:
            rows = df_all[(df_all["pair"] == pair_name) &
                          (df_all["variant"] == vname) &
                          (df_all["period"] == "OOS")]
            pfs.append(rows["pf"].values[0] if len(rows) > 0 else 0)
        style = "-o" if pair_name == symbols[0] else "--s"
        ax.plot(x, pfs, style, label=pair_name, markersize=6)

    # カテゴリ平均
    avg_pfs = []
    for vname in variant_names:
        rows = df_all[(df_all["pair"].isin(symbols)) &
                      (df_all["variant"] == vname) &
                      (df_all["period"] == "OOS")]
        avg_pfs.append(rows["pf"].mean() if len(rows) > 0 else 0)
    ax.plot(x, avg_pfs, "k-", linewidth=2.5, label=f"カテゴリ平均", alpha=0.7)

    # 推奨バリアントをマーク
    if best_v in variant_names:
        bx = variant_names.index(best_v)
        ax.axvline(bx, color="red", linestyle=":", linewidth=1.5, alpha=0.6)
        ax.text(bx, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.5,
                f"★{best_v}", color="red", fontsize=8, ha="center")

    ax.axhline(2.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="PF=2.0")
    ax.set_title(f"{cat_name}\n{cat_cfg['session']}", fontsize=9, fontweight="bold")
    ax.set_xlabel("バリアント", fontsize=8)
    ax.set_ylabel("OOS PF", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
chart_out = os.path.join(OUT_DIR, "category_improvements.png")
plt.savefig(chart_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート: {chart_out}")

print("\n全処理完了。")
