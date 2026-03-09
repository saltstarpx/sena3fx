"""
backtest_v77_bugfix.py
======================
v77バグ修正前後の比較バックテスト

【修正したバグ】
バグ①（4Hループ KMID冗長）:
  - 旧: h4_prev1（確認足・既に陽線/陰線確認済み）にKMIDを適用 → 常にTrue
  - 新: h4_prev3（パターン直前の文脈足）にKMIDを適用 → 実質的なフィルタリングが機能

バグ②（1Hループ 先読みバイアス）:
  - 旧: data_4h[index <= h1_current_time] → 形成中の4H足を含む場合がある
  - 新: data_4h[index < h1_current_time]  → 完結済みの4H足のみ使用

【対象銘柄】
1mデータが存在する銘柄: XAUUSD / AUDUSD / EURUSD / GBPUSD / NAS100 / SPX500 / US30
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

# ── 設定 ──────────────────────────────────────────────────
INIT_CASH  = 1_000_000
RISK_PCT   = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
KLOW_THR   = 0.0015
IS_START   = "2025-01-01"
IS_END     = "2025-02-28"
OOS_START  = "2025-03-03"
OOS_END    = "2026-02-27"

# 1mデータが存在する銘柄のみ
PAIRS = {
    "XAUUSD": {"sym": "xauusd", "1m_file": "xauusd_1m.csv"},  # 全期間1mあり
    "AUDUSD": {"sym": "audusd", "1m_file": "audusd_1m.csv"},
    "EURUSD": {"sym": "eurusd", "1m_file": "eurusd_1m.csv"},
    "GBPUSD": {"sym": "gbpusd", "1m_file": "gbpusd_1m.csv"},
    "NAS100": {"sym": "nas100", "1m_file_is": "nas100_is_1m.csv", "1m_file_oos": "nas100_oos_1m.csv"},
    "SPX500": {"sym": "spx500", "1m_file_is": "spx500_is_1m.csv", "1m_file_oos": "spx500_oos_1m.csv"},
    "US30":   {"sym": "us30",   "1m_file_is": "us30_is_1m.csv",   "1m_file_oos": "us30_oos_1m.csv"},
}

# ── ユーティリティ ─────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"})
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def slice_period(df, start, end):
    if df is None:
        return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    df["atr"]   = calculate_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

# ── KMID+KLOW フィルター ───────────────────────────────────
def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THR
    return kmid_ok and klow_ok

# ── シグナル生成: バグあり（オリジナルv77） ─────────────────
def generate_signals_buggy(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    signals = []; used_times = set()

    # 4Hループ: h4_prev1（確認足）にKMID → 常にTrue（バグあり）
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_ct    = h4_times[i]
        h4_prev1 = data_4h.iloc[i - 1]
        h4_prev2 = data_4h.iloc[i - 2]
        h4_cur   = data_4h.iloc[i]
        atr_val  = h4_cur["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue
        trend = h4_cur["trend"]; tol = atr_val * 0.3

        if trend == 1:
            low1, low2 = h4_prev2["low"], h4_prev1["low"]
            if abs(low1 - low2) <= tol and h4_prev1["close"] > h4_prev1["open"]:
                if not check_kmid_klow(h4_prev1, 1): continue  # ← バグ: 常にTrue
                sl = min(low1, low2) - atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h4_ct) & (data_1m.index < h4_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw + spread; risk = raw - sl
                        if 0 < risk <= atr_val * 3:
                            tp = raw + risk * RR_RATIO
                            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                            used_times.add(et)
        if trend == -1:
            h1, h2 = h4_prev2["high"], h4_prev1["high"]
            if abs(h1 - h2) <= tol and h4_prev1["close"] < h4_prev1["open"]:
                if not check_kmid_klow(h4_prev1, -1): continue  # ← バグ: 常にTrue
                sl = max(h1, h2) + atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h4_ct) & (data_1m.index < h4_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw - spread; risk = sl - raw
                        if 0 < risk <= atr_val * 3:
                            tp = raw - risk * RR_RATIO
                            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                            used_times.add(et)

    # 1Hループ: <= で形成中の4H足を含む（バグあり）
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index <= h1_ct]  # ← バグ: <=
        if len(h4_before) == 0: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue
        trend = h4_latest["trend"]; h4_atr = h4_latest["atr"]; tol = atr_val * 0.3

        if trend == 1:
            low1, low2 = h1_prev2["low"], h1_prev1["low"]
            if abs(low1 - low2) <= tol and h1_prev1["close"] > h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, 1): continue
                sl = min(low1, low2) - atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < h1_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw + spread; risk = raw - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw + risk * RR_RATIO
                            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(et)
        if trend == -1:
            h1, h2 = h1_prev2["high"], h1_prev1["high"]
            if abs(h1 - h2) <= tol and h1_prev1["close"] < h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, -1): continue
                sl = max(h1, h2) + atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < h1_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw - spread; risk = sl - raw
                        if 0 < risk <= h4_atr * 2:
                            tp = raw - risk * RR_RATIO
                            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シグナル生成: バグ修正済み ─────────────────────────────
def generate_signals_fixed(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    signals = []; used_times = set()

    # 4Hループ: h4_prev3（文脈足）にKMID適用、i=3 から開始（バグ修正済み）
    h4_times = data_4h.index.tolist()
    for i in range(3, len(h4_times)):  # ← 3から開始
        h4_ct    = h4_times[i]
        h4_prev1 = data_4h.iloc[i - 1]  # 確認足
        h4_prev2 = data_4h.iloc[i - 2]  # パターン1本目
        h4_prev3 = data_4h.iloc[i - 3]  # 文脈足（KMIDフィルター対象）← 修正
        h4_cur   = data_4h.iloc[i]
        atr_val  = h4_cur["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue
        trend = h4_cur["trend"]; tol = atr_val * 0.3

        if trend == 1:
            low1, low2 = h4_prev2["low"], h4_prev1["low"]
            if abs(low1 - low2) <= tol and h4_prev1["close"] > h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, 1): continue  # ← 修正: h4_prev3
                sl = min(low1, low2) - atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h4_ct) & (data_1m.index < h4_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw + spread; risk = raw - sl
                        if 0 < risk <= atr_val * 3:
                            tp = raw + risk * RR_RATIO
                            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                            used_times.add(et)
        if trend == -1:
            h1, h2 = h4_prev2["high"], h4_prev1["high"]
            if abs(h1 - h2) <= tol and h4_prev1["close"] < h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, -1): continue  # ← 修正: h4_prev3
                sl = max(h1, h2) + atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h4_ct) & (data_1m.index < h4_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw - spread; risk = sl - raw
                        if 0 < risk <= atr_val * 3:
                            tp = raw - risk * RR_RATIO
                            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                            used_times.add(et)

    # 1Hループ: < で完結済み4H足のみ使用（バグ修正済み）
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index < h1_ct]  # ← 修正: <
        if len(h4_before) == 0: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue
        trend = h4_latest["trend"]; h4_atr = h4_latest["atr"]; tol = atr_val * 0.3

        if trend == 1:
            low1, low2 = h1_prev2["low"], h1_prev1["low"]
            if abs(low1 - low2) <= tol and h1_prev1["close"] > h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, 1): continue
                sl = min(low1, low2) - atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < h1_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw + spread; risk = raw - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw + risk * RR_RATIO
                            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(et)
        if trend == -1:
            h1, h2 = h1_prev2["high"], h1_prev1["high"]
            if abs(h1 - h2) <= tol and h1_prev1["close"] < h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, -1): continue
                sl = max(h1, h2) + atr_val * 0.15
                m1w = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < h1_ct + pd.Timedelta(minutes=2))]
                if len(m1w) > 0:
                    eb = m1w.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw = eb["open"]; ep = raw - spread; risk = sl - raw
                        if 0 < risk <= h4_atr * 2:
                            tp = raw - risk * RR_RATIO
                            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ───────────────────────────────────────
def simulate(signals, data_1m, symbol, init_cash=1_000_000, risk_pct=0.02, half_r=1.0):
    if not signals:
        return [], [init_cash]
    rm = RiskManager(symbol, risk_pct=risk_pct)
    equity = init_cash; trades = []; eq_curve = [init_cash]

    for sig in signals:
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=150.0)
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0: continue

        half_done = False; be_sl = None; result = None
        exit_price = None; exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                cur_sl = be_sl if half_done else sl
                if bar["low"] <= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot * rem, usdjpy_rate=150.0, ref_price=ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * half_r:
                        equity += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r, lot * 0.5, usdjpy_rate=150.0, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time; rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(direction, ep, tp, lot * rem, usdjpy_rate=150.0, ref_price=ep)
                    result = "win"; break
                if not half_done and bar["high"] >= ep + risk * half_r:
                    equity += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r, lot * 0.5, usdjpy_rate=150.0, ref_price=ep)
                    half_done = True; be_sl = ep
            else:
                cur_sl = be_sl if half_done else sl
                if bar["high"] >= cur_sl:
                    exit_price = cur_sl; exit_time = bar_time
                    rem = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot * rem, usdjpy_rate=150.0, ref_price=ep)
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * half_r:
                        equity += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r, lot * 0.5, usdjpy_rate=150.0, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time; rem = 0.5 if half_done else 1.0
                    equity += rm.calc_pnl_jpy(direction, ep, tp, lot * rem, usdjpy_rate=150.0, ref_price=ep)
                    result = "win"; break
                if not half_done and bar["low"] <= ep - risk * half_r:
                    equity += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r, lot * 0.5, usdjpy_rate=150.0, ref_price=ep)
                    half_done = True; be_sl = ep

        if result is None: continue
        trades.append({"entry_time": sig["time"], "exit_time": exit_time,
                        "dir": direction, "ep": ep, "sl": sl, "tp": tp,
                        "exit_price": exit_price, "result": result, "equity": equity,
                        "tf": sig.get("tf","?")})
        eq_curve.append(equity)

    return trades, eq_curve

# ── 統計 ──────────────────────────────────────────────────
def calc_stats(trades, eq_curve, label):
    if not trades:
        return {"label": label, "n": 0, "wr": 0, "pf": 0,
                "mdd_pct": 0, "kelly": 0, "monthly_plus": "N/A",
                "n_4h": 0, "n_1h": 0}
    df = pd.DataFrame(trades)
    wins  = df[df["result"] == "win"]
    loses = df[df["result"] == "loss"]
    n  = len(df); wr = len(wins) / n
    # PFはエクイティ変化ベースで計算（ロング/ショート符号問題を回避）
    eq = np.array(eq_curve)
    deltas = np.diff(eq)
    gross_win  = deltas[deltas > 0].sum()
    gross_loss = abs(deltas[deltas < 0].sum())
    pf    = gross_win / gross_loss if gross_loss > 0 else float("inf")
    peak = np.maximum.accumulate(eq)
    mdd  = ((eq - peak) / peak).min()
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    # 月次プラス
    df2 = df.copy()
    df2["exit_time"] = pd.to_datetime(df2["exit_time"], utc=True)
    df2["month"] = df2["exit_time"].dt.to_period("M")
    monthly = df2.groupby("month")["equity"].last()
    monthly_prev = monthly.shift(1).fillna(INIT_CASH)
    mp = f"{(monthly > monthly_prev).sum()}/{len(monthly)}"
    n_4h = (df["tf"] == "4h").sum()
    n_1h = (df["tf"] == "1h").sum()
    return {"label": label, "n": n, "wr": wr * 100, "pf": pf,
            "mdd_pct": abs(mdd) * 100, "kelly": kelly,
            "monthly_plus": mp, "n_4h": n_4h, "n_1h": n_1h}

# ── メイン処理 ────────────────────────────────────────────
print("=" * 80)
print("v77 バグ修正前後 比較バックテスト")
print(f"IS: {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
print(f"対象: 1mデータあり銘柄  初期資金: {INIT_CASH:,}円  RR: {RR_RATIO}  半利確: +{HALF_R}R")
print("=" * 80)

all_results = []

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]
    rm  = RiskManager(pair, risk_pct=RISK_PCT)
    spread = rm.spread_pips
    pip    = rm.pip_size

    print(f"\n{'='*60}")
    print(f"  {pair}  スプレッド: {spread}pips")
    print(f"{'='*60}")

    # IS データ
    d15m_is = load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv"))
    d4h_is  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv"))
    if "1m_file" in cfg:
        d1m_full = load_csv(os.path.join(DATA_DIR, cfg["1m_file"]))
        d1m_is  = slice_period(d1m_full, IS_START, IS_END)
    else:
        d1m_is = load_csv(os.path.join(DATA_DIR, cfg["1m_file_is"]))

    # OOS データ
    d15m_oos = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv"))
    d4h_oos  = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv"))
    if "1m_file" in cfg:
        d1m_oos = slice_period(d1m_full, OOS_START, OOS_END)
    else:
        d1m_oos = load_csv(os.path.join(DATA_DIR, cfg["1m_file_oos"]))

    if any(d is None or len(d) == 0 for d in [d1m_is, d15m_is, d4h_is, d1m_oos, d15m_oos, d4h_oos]):
        print(f"  [SKIP] データ不足")
        continue

    d15m_is  = slice_period(d15m_is,  IS_START,  IS_END)
    d4h_is   = slice_period(d4h_is,   IS_START,  IS_END)
    d15m_oos = slice_period(d15m_oos, OOS_START, OOS_END)
    d4h_oos  = slice_period(d4h_oos,  OOS_START, OOS_END)

    for period, d1m, d15m, d4h in [("IS", d1m_is, d15m_is, d4h_is),
                                     ("OOS", d1m_oos, d15m_oos, d4h_oos)]:
        for version, sig_fn in [("buggy", generate_signals_buggy),
                                  ("fixed", generate_signals_fixed)]:
            sigs   = sig_fn(d1m, d15m, d4h, spread, pip)
            trades, eq = simulate(sigs, d1m, pair, INIT_CASH, RISK_PCT, HALF_R)
            label  = f"{pair}_{period}_{version}"
            stats  = calc_stats(trades, eq, label)
            stats["pair"]    = pair
            stats["period"]  = period
            stats["version"] = version
            all_results.append(stats)
            n_4h_sig = sum(1 for s in sigs if s.get("tf") == "4h")
            n_1h_sig = sum(1 for s in sigs if s.get("tf") == "1h")
            print(f"  [{version}][{period}] "
                  f"シグナル:{len(sigs)}件(4H:{n_4h_sig}/1H:{n_1h_sig}) "
                  f"トレード:{stats['n']}件 | "
                  f"勝率{stats['wr']:.1f}% | PF{stats['pf']:.2f} | "
                  f"MDD{stats['mdd_pct']:.1f}% | "
                  f"Kelly{stats['kelly']:.3f} | "
                  f"月次+{stats['monthly_plus']}")

# ── CSV保存 ──────────────────────────────────────────────
df_all = pd.DataFrame(all_results)
csv_path = os.path.join(OUT_DIR, "v77_bugfix_comparison.csv")
df_all.to_csv(csv_path, index=False)
print(f"\n結果CSV: {csv_path}")

# ── 可視化: OOS PF比較 ────────────────────────────────────
pairs_with_data = df_all["pair"].unique().tolist()
oos_df = df_all[df_all["period"] == "OOS"].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("v77 バグ修正前後 OOS比較\n（修正内容: ①4H-KMID文脈足化 ②1H-4H先読みバイアス除去）",
             fontsize=12, fontweight="bold")

metrics = [
    ("pf",       "OOS PF",      "プロフィットファクター", 3.0),
    ("wr",       "OOS 勝率",    "勝率（%）",             65.0),
    ("mdd_pct",  "OOS MDD",     "最大ドローダウン（%）", 20.0),
    ("kelly",    "OOS Kelly",   "ケリー係数",            0.45),
]

x = np.arange(len(pairs_with_data))
colors = {"buggy": "#ef4444", "fixed": "#22c55e"}

for ax, (metric, title, ylabel, ref) in zip(axes.flatten(), metrics):
    for j, ver in enumerate(["buggy", "fixed"]):
        vals = []
        for p in pairs_with_data:
            row = oos_df[(oos_df["pair"] == p) & (oos_df["version"] == ver)]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        ax.bar(x + j * 0.35, vals, 0.35,
               label=f"{'バグあり' if ver=='buggy' else 'バグ修正済'}",
               color=colors[ver], alpha=0.85)
    ax.axhline(ref, color="gray", linestyle="--", linewidth=1.0, label=f"基準={ref}")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(pairs_with_data, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "v77_bugfix_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート: {chart_path}")

# ── サマリー表 ────────────────────────────────────────────
print("\n" + "=" * 80)
print("OOS サマリー比較（バグあり vs バグ修正済）")
print("=" * 80)
print(f"{'銘柄':<8} {'バグあり PF':>11} {'修正済 PF':>10} {'差分':>7} | "
      f"{'バグあり WR':>11} {'修正済 WR':>10}")
print("-" * 70)

for p in pairs_with_data:
    buggy_row = oos_df[(oos_df["pair"] == p) & (oos_df["version"] == "buggy")]
    fixed_row = oos_df[(oos_df["pair"] == p) & (oos_df["version"] == "fixed")]
    if len(buggy_row) == 0 or len(fixed_row) == 0: continue
    bpf = buggy_row["pf"].values[0]; fpf = fixed_row["pf"].values[0]
    bwr = buggy_row["wr"].values[0]; fwr = fixed_row["wr"].values[0]
    diff = fpf - bpf
    marker = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"{p:<8} {bpf:>11.2f} {fpf:>10.2f} {diff:>+6.2f}{marker} | "
          f"{bwr:>10.1f}% {fwr:>9.1f}%")

print("\n全処理完了。")
