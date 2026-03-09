"""
backtest_v79_mdd_trend.py
==========================
【Phase 1】定量分析 — 特徴量抽出 + 時間帯 / ADX / 日足トレンド / ATR体制 別勝率集計
【Phase 2】計量分析 — カイ二乗検定・相関分析・特徴量重要度スコアリング
【Phase 3】v79 改善バックテスト — MDD対策 + トレンドフォロー強化
          カテゴリ別過学習チェック付き

【v79 改善候補（v77ベース）】
  v79A  : + 日足EMA20 方向一致フィルター（1D trend alignment）
  v79B  : + 4H ADX ≥ 20（トレンド強度 — レンジ排除）
  v79C  : + 4H トレンド一貫性（直近4本の4H足が全て同トレンド方向）
  v79D  : + MDD自動スケールダウン（equity < peak×0.95 → 段階的ロット縮小）
  v79BC : B+C のみ（MDD対策メイン）
  v79ABC: A+B+C（1D+ADX+Streak）

【過学習対策】
  - ADXしきい値・Streak本数は固定値（OOSデータで調整しない）
  - カテゴリ内全銘柄で同一パラメータ
  - カテゴリPASS = 過半数銘柄でOOS PF改善

IS: 2024-07-01〜2025-02-28  OOS: 2025-03-03〜2026-02-27
"""
import os, sys, warnings, math
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

# ── 定数 ──────────────────────────────────────────────────
INIT_CASH  = 1_000_000
RISK_PCT   = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
KLOW_THR   = 0.0015
PAT_TOL    = 0.3     # v77継承
IS_START   = "2024-07-01"; IS_END   = "2025-02-28"
OOS_START  = "2025-03-03"; OOS_END  = "2026-02-27"

# v79 新規フィルター閾値（固定値 / データ非依存）
ADX_THRESHOLD    = 20    # 4H ADX ≥ 20 → 強トレンド確認（一般的な基準値）
STREAK_MIN       = 4     # 直近4本の4H足が全て同方向 → トレンド一貫性
MDD_SCALE_START  = 0.05  # 5% ドローダウンでスケールダウン開始
MDD_SCALE_FLOOR  = 0.50  # 最大スケールダウン 50%まで

# カテゴリ定義（経済的根拠ベースのセッション時間）
CATEGORIES = {
    "FX": {
        "symbols": [("EURUSD","eurusd"), ("GBPUSD","gbpusd"), ("AUDUSD","audusd")],
        "utc_start": 7, "utc_end": 22, "pass_min": 2,
    },
    "METALS": {
        "symbols": [("XAUUSD","xauusd")],
        "utc_start": 0, "utc_end": 24, "pass_min": 1,  # 時間フィルターなし（UTC7-22は逆効果）
    },
    "INDICES": {
        "symbols": [("US30","us30"), ("SPX500","spx500"), ("NAS100","nas100")],
        "utc_start": 14, "utc_end": 22, "pass_min": 2,
    },
}

# 分析対象（1mデータあり）
ANALYSIS_PAIRS = [
    ("XAUUSD","xauusd"), ("EURUSD","eurusd"),
    ("GBPUSD","gbpusd"), ("AUDUSD","audusd"),
]

# ── データロード ────────────────────────────────────────────
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
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def load_1m(sym, is_period=True):
    tag = "is" if is_period else "oos"
    for path in [
        os.path.join(DATA_DIR, f"{sym}_{tag}_1m.csv"),
        os.path.join(DATA_DIR, f"{sym}_1m.csv"),
        os.path.join(DATA_DIR, "ohlc", f"{sym.upper()}_1m.csv"),
    ]:
        df = load_csv(path)
        if df is not None:
            start, end = (IS_START, IS_END) if is_period else (OOS_START, OOS_END)
            return slice_period(df, start, end)
    return None

def load_pair_data(sym, pair_name):
    """IS/OOS 全データを一括ロード。フォールバックあり"""
    def try_load_split(key, tf):
        return load_csv(os.path.join(DATA_DIR, f"{sym}_{key}_{tf}.csv"))
    def try_load_ohlc(tf):
        return load_csv(os.path.join(DATA_DIR, "ohlc", f"{pair_name}_{tf}.csv"))

    result = {}
    for period, start, end in [("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)]:
        result[f"1m_{period}"] = load_1m(sym, is_period=(period=="is"))
        for tf in ["15m", "4h"]:
            df = try_load_split(period, tf)
            if df is None:
                df = try_load_ohlc(tf)
            result[f"{tf}_{period}"] = slice_period(df, start, end) if df is not None else None
    return result

# ── インジケーター ──────────────────────────────────────────
def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    return pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(period).mean()

def calculate_adx(df, period=14):
    """Wilder's ADX（4H足に適用）"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = np.where((up_move > down_move) & (up_move   > 0), up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    atr_s    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_s
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.rename("adx")

def add_indicators_4h(df):
    df = df.copy()
    df["atr"]   = calculate_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calculate_adx(df, 14)
    # ATR比（現在ATR / 20期間ATR移動平均）— ボラティリティ体制
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(20).mean()
    # EMA距離（ATR単位）— トレンド強度
    df["ema_dist_atr"] = (df["close"] - df["ema20"]).abs() / df["atr"].replace(0, np.nan)
    return df

def build_1d_from_4h(data_4h):
    """4H足を1D足にリサンプル（日足トレンド用）"""
    df = data_4h.resample("1D").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    df["ema20"]    = df["close"].ewm(span=20, adjust=False).mean()
    df["trend_1d"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    klow_ok = (min(o,c) - l) / o < KLOW_THR if o > 0 else False
    return kmid_ok and klow_ok

# ── シグナル生成（特徴量付き / フィルター設定可） ─────────────
def generate_signals(data_1m, data_15m, data_4h_raw, spread_pips, pip_size,
                     # 時間フィルター（カテゴリ固定）
                     utc_start=0, utc_end=24,
                     # v79 改善フィルター
                     use_1d_trend=False,   # v79A: 日足EMA20方向一致
                     adx_min=0,            # v79B: 4H ADX ≥ adx_min
                     streak_min=0,         # v79C: 直近N本の4H足が同方向
                     # 特徴量収集モード（Phase1分析用）
                     collect_features=False):

    spread  = spread_pips * pip_size
    data_4h = add_indicators_4h(data_4h_raw)
    data_1d = build_1d_from_4h(data_4h) if use_1d_trend else None

    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, 14)

    signals = []; used_times = set()
    h1_times = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        # カテゴリ固定セッションフィルター
        if not (utc_start <= h1_ct.hour < utc_end): continue

        # 完結済み4H足（look-ahead bias 修正済み）
        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(STREAK_MIN, 4): continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tol    = atr_val * PAT_TOL

        # v79B: 4H ADX フィルター
        adx_val = h4_latest.get("adx", 0)
        if adx_min > 0 and (pd.isna(adx_val) or adx_val < adx_min): continue

        # v79C: 4H トレンド一貫性（直近N本が全て同方向）
        if streak_min > 0:
            recent_trends = h4_before["trend"].iloc[-streak_min:].values
            if not all(t == trend for t in recent_trends): continue

        # v79A: 日足トレンド方向一致
        if use_1d_trend and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0: continue
            d1_latest = d1_before.iloc[-1]
            if d1_latest["trend_1d"] != trend: continue

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

            # v77継承: 4H文脈足 KMID+KLOW
            if not check_kmid_klow(h4_latest, direction): continue

            m1w = data_1m[
                (data_1m.index >= h1_ct) &
                (data_1m.index <  h1_ct + pd.Timedelta(minutes=2))
            ]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue

            raw = eb["open"]
            if direction == 1:
                sl   = min(v1,v2) - atr_val * 0.15
                ep   = raw + spread; risk = raw - sl
            else:
                sl   = max(v1,v2) + atr_val * 0.15
                ep   = raw - spread; risk = sl - raw

            if not (0 < risk <= h4_atr * 2): continue

            tp  = raw + direction * risk * RR_RATIO
            sig = {"time": et, "dir": direction,
                   "ep": ep, "sl": sl, "tp": tp, "risk": risk}

            if collect_features:
                # Phase 1 分析用特徴量
                atr_ratio = h4_latest.get("atr_ratio", np.nan)
                ema_dist  = h4_latest.get("ema_dist_atr", np.nan)
                recent4   = h4_before["trend"].iloc[-4:].values
                streak4   = sum(1 for t in reversed(recent4) if t == direction)
                sig["feat"] = {
                    "hour":      h1_ct.hour,
                    "dow":       h1_ct.dayofweek,  # 0=月
                    "adx_4h":   round(float(adx_val), 1) if not pd.isna(adx_val) else 0,
                    "atr_ratio": round(float(atr_ratio), 2) if not pd.isna(atr_ratio) else 1.0,
                    "ema_dist":  round(float(ema_dist), 2) if not pd.isna(ema_dist) else 0,
                    "streak4":   int(streak4),
                    "direction": direction,
                }
            signals.append(sig)
            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション（MDD自動スケールダウン対応） ──────────────
def simulate(signals, data_1m, symbol, use_mdd_scale=False):
    if not signals:
        return [], [INIT_CASH]
    rm     = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH; peak = INIT_CASH
    trades = []; eq_curve = [INIT_CASH]

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]

        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=150.0)

        # v79D: MDD自動スケールダウン
        if use_mdd_scale and peak > 0:
            drawdown = (equity - peak) / peak  # 0〜-1
            if drawdown < -MDD_SCALE_START:
                # drawdown が深いほど lot を縮小（線形）
                scale = max(MDD_SCALE_FLOOR,
                            1.0 + drawdown / MDD_SCALE_START * (1.0 - MDD_SCALE_FLOOR))
                lot   = lot * scale

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
        peak = max(peak, equity)
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result, "equity": equity,
            **({f"feat_{k}": v for k, v in sig["feat"].items()} if "feat" in sig else {}),
        })
        eq_curve.append(equity)

    return trades, eq_curve

# ── 統計計算 ────────────────────────────────────────────────
def calc_stats(trades, eq_curve, pair, variant, period):
    if not trades:
        return {"pair": pair, "variant": variant, "period": period,
                "n": 0, "wr": 0.0, "pf": 0.0, "mdd": 0.0, "kelly": -1.0, "monthly_plus": "0/0"}
    df   = pd.DataFrame(trades)
    n    = len(df); wins = df[df["result"]=="win"]; wr = len(wins)/n
    eq   = np.array(eq_curve)
    d    = np.diff(eq)
    pf   = d[d>0].sum() / abs(d[d<0].sum()) if (d<0).any() else float("inf")
    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq-peak)/peak).min())*100
    kelly = wr - (1-wr)/(pf if pf>0 else 1e-9)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    mp   = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    mpp  = mp.shift(1).fillna(INIT_CASH)
    return {"pair": pair, "variant": variant, "period": period,
            "n": n, "wr": round(wr*100,1), "pf": round(pf,2),
            "mdd": round(mdd,1), "kelly": round(kelly,3),
            "monthly_plus": f"{(mp>mpp).sum()}/{len(mp)}"}

# ════════════════════════════════════════════════════════════
# PHASE 1: 定量分析 — 特徴量別勝率集計
# ════════════════════════════════════════════════════════════

def chi2_test(n_win_a, n_a, n_win_b, n_b):
    """2×2カイ二乗検定（手動実装）"""
    n_loss_a = n_a - n_win_a; n_loss_b = n_b - n_win_b
    total = n_a + n_b
    if total == 0 or n_a == 0 or n_b == 0: return 0.0, 1.0
    exp_wa = n_a * (n_win_a + n_win_b) / total
    exp_wb = n_b * (n_win_a + n_win_b) / total
    exp_la = n_a * (n_loss_a + n_loss_b) / total
    exp_lb = n_b * (n_loss_a + n_loss_b) / total
    chi2 = 0.0
    for o, e in [(n_win_a, exp_wa),(n_win_b, exp_wb),(n_loss_a, exp_la),(n_loss_b, exp_lb)]:
        if e > 0: chi2 += (o - e)**2 / e
    # p値近似（chi2分布 df=1 の右側確率）
    # 0.5*(1 - erf(sqrt(chi2/2))) の近似
    x = math.sqrt(chi2 / 2)
    erf_approx = math.tanh(x * (1.0 + x * (0.278393 + x * 0.230389)))
    p = max(0.0, 1.0 - erf_approx)
    return round(chi2, 2), round(p, 4)

def analyze_features(trades_df, pair_name):
    """特徴量別勝率の集計・検定"""
    df = trades_df.copy()
    if len(df) < 20: return

    df["win"] = (df["result"] == "win").astype(int)
    base_wr   = df["win"].mean()

    rows = []
    # 1. 時間帯（4時間ブロック）
    df["hour_block"] = (df["feat_hour"] // 4) * 4
    for hb, g in df.groupby("hour_block"):
        wr = g["win"].mean()
        n  = len(g)
        chi2, p = chi2_test(g["win"].sum(), n,
                             df[df["hour_block"] != hb]["win"].sum(),
                             len(df) - n)
        rows.append({"feature": f"hour {hb:02d}-{hb+4:02d}",
                     "n": n, "wr": round(wr*100,1), "base_wr": round(base_wr*100,1),
                     "diff": round((wr-base_wr)*100,1), "chi2": chi2, "p": p})

    # 2. ADX体制（<15 / 15-20 / 20-30 / 30+）
    bins = [0, 15, 20, 30, 999]
    labs = ["ADX<15", "ADX15-20", "ADX20-30", "ADX30+"]
    df["adx_bin"] = pd.cut(df["feat_adx_4h"], bins=bins, labels=labs, right=False)
    for ab, g in df.groupby("adx_bin", observed=True):
        if len(g) < 5: continue
        wr = g["win"].mean()
        n  = len(g)
        chi2, p = chi2_test(g["win"].sum(), n,
                             df[df["adx_bin"] != ab]["win"].sum(), len(df)-n)
        rows.append({"feature": f"{ab}", "n": n, "wr": round(wr*100,1),
                     "base_wr": round(base_wr*100,1),
                     "diff": round((wr-base_wr)*100,1), "chi2": chi2, "p": p})

    # 3. 4H トレンド一貫性（streak4）
    for s, g in df.groupby("feat_streak4"):
        if len(g) < 5: continue
        wr = g["win"].mean(); n = len(g)
        chi2, p = chi2_test(g["win"].sum(), n,
                             df[df["feat_streak4"] != s]["win"].sum(), len(df)-n)
        rows.append({"feature": f"streak4={s}", "n": n, "wr": round(wr*100,1),
                     "base_wr": round(base_wr*100,1),
                     "diff": round((wr-base_wr)*100,1), "chi2": chi2, "p": p})

    # 4. ATR体制（低/中/高ボラ）
    df["atr_bin"] = pd.cut(df["feat_atr_ratio"], bins=[0,0.8,1.2,1.6,9],
                           labels=["低ボラ<0.8","中ボラ0.8-1.2","高ボラ1.2-1.6","極高ボラ1.6+"],
                           right=False)
    for ab, g in df.groupby("atr_bin", observed=True):
        if len(g) < 5: continue
        wr = g["win"].mean(); n = len(g)
        chi2, p = chi2_test(g["win"].sum(), n,
                             df[df["atr_bin"] != ab]["win"].sum(), len(df)-n)
        rows.append({"feature": f"{ab}", "n": n, "wr": round(wr*100,1),
                     "base_wr": round(base_wr*100,1),
                     "diff": round((wr-base_wr)*100,1), "chi2": chi2, "p": p})

    # 5. EMA距離（トレンド強度）
    df["ema_bin"] = pd.cut(df["feat_ema_dist"], bins=[0,1,2,3,99],
                           labels=["EMA近<1ATR","EMA中1-2ATR","EMA遠2-3ATR","EMA極3+ATR"],
                           right=False)
    for ab, g in df.groupby("ema_bin", observed=True):
        if len(g) < 5: continue
        wr = g["win"].mean(); n = len(g)
        chi2, p = chi2_test(g["win"].sum(), n,
                             df[df["ema_bin"] != ab]["win"].sum(), len(df)-n)
        rows.append({"feature": f"{ab}", "n": n, "wr": round(wr*100,1),
                     "base_wr": round(base_wr*100,1),
                     "diff": round((wr-base_wr)*100,1), "chi2": chi2, "p": p})

    print(f"\n  {'特徴量':<22} {'件数':>5} {'WR':>7} {'ベースWR':>8} {'差分':>7} {'χ²':>6} {'p値':>7}  有意性")
    print(f"  {'-'*80}")
    sig_count = 0
    for r in sorted(rows, key=lambda x: abs(x["diff"]), reverse=True):
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
        if r["p"] < 0.10: sig_count += 1
        print(f"  {r['feature']:<22} {r['n']:>5} {r['wr']:>6.1f}% {r['base_wr']:>7.1f}%"
              f" {r['diff']:>+7.1f}% {r['chi2']:>6.2f} {r['p']:>7.4f}  {sig}")
    print(f"\n  全{len(df)}件 基準WR={base_wr*100:.1f}%  統計的有意(p<0.10)={sig_count}項目")
    return rows

# ════════════════════════════════════════════════════════════
# PHASE 2: 計量分析 — 特徴量ランキング
# ════════════════════════════════════════════════════════════
def rank_features(all_feature_rows):
    """全銘柄の統計結果を集計し、特徴量の重要度をランキング"""
    from collections import defaultdict
    feature_scores = defaultdict(list)
    for r in all_feature_rows:
        feature_scores[r["feature"]].append(abs(r["diff"]))
    ranked = sorted(feature_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    return ranked

# ════════════════════════════════════════════════════════════
# PHASE 3: v79 改善バックテスト
# ════════════════════════════════════════════════════════════

# バリアント定義
VARIANTS = [
    ("v77",    dict(use_1d_trend=False, adx_min=0,    streak_min=0,    use_mdd_scale=False)),
    ("v79A",   dict(use_1d_trend=True,  adx_min=0,    streak_min=0,    use_mdd_scale=False)),
    ("v79B",   dict(use_1d_trend=False, adx_min=ADX_THRESHOLD, streak_min=0, use_mdd_scale=False)),
    ("v79C",   dict(use_1d_trend=False, adx_min=0,    streak_min=STREAK_MIN, use_mdd_scale=False)),
    ("v79D",   dict(use_1d_trend=False, adx_min=0,    streak_min=0,    use_mdd_scale=True)),
    ("v79BC",  dict(use_1d_trend=False, adx_min=ADX_THRESHOLD, streak_min=STREAK_MIN, use_mdd_scale=False)),
    ("v79ABC", dict(use_1d_trend=True,  adx_min=ADX_THRESHOLD, streak_min=STREAK_MIN, use_mdd_scale=False)),
]

# ── メイン実行 ────────────────────────────────────────────────
print("=" * 110)
print(f"v79 MDD対策 + トレンドフォロー強化バックテスト")
print(f"IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
print(f"ADX閾値={ADX_THRESHOLD}（固定値、データ非依存）  Streak最小={STREAK_MIN}本  MDD Scale開始={int(MDD_SCALE_START*100)}%")
print("=" * 110)

# ─── Phase 1+2: 定量・計量分析 ────────────────────────────────
print("\n" + "█"*110)
print("  PHASE 1+2: 定量分析・計量分析 — OOS期間の特徴量別勝率 + カイ二乗検定")
print("  分析目的: どの市場条件がエントリー勝率に影響するかを定量化する")
print("█"*110)

all_feature_rows = []
analysis_trades_all = {}

for pair_name, sym in ANALYSIS_PAIRS:
    print(f"\n{'='*80}")
    print(f"  {pair_name} 特徴量分析（OOS）")
    print(f"{'='*80}")

    rm = RiskManager(pair_name, risk_pct=RISK_PCT)
    data = load_pair_data(sym, pair_name)

    d1m_oos  = data.get("1m_oos")
    d15m_oos = data.get("15m_oos")
    d4h_oos  = data.get("4h_oos")

    if any(d is None or len(d) == 0 for d in [d1m_oos, d15m_oos, d4h_oos]):
        print(f"  {pair_name}: データ不足 → スキップ")
        continue

    sigs = generate_signals(
        d1m_oos, d15m_oos, d4h_oos, rm.spread_pips, rm.pip_size,
        utc_start=0, utc_end=24, collect_features=True
    )
    trades, eq = simulate(sigs, d1m_oos, pair_name)
    if not trades:
        print(f"  {pair_name}: トレードなし → スキップ")
        continue

    df_trades = pd.DataFrame(trades)
    analysis_trades_all[pair_name] = df_trades

    rows = analyze_features(df_trades, pair_name)
    if rows:
        all_feature_rows.extend(rows)

# 特徴量重要度ランキング（全銘柄集計）
if all_feature_rows:
    print(f"\n{'='*80}")
    print("  計量分析サマリー: 全銘柄共通特徴量の勝率差分ランキング")
    print("  （|WR差分| の平均が大きい = より予測力が高い特徴量）")
    print(f"{'='*80}")
    ranked = rank_features(all_feature_rows)
    print(f"  {'ランク':<4} {'特徴量':<22} {'平均|WR差分|':>12}")
    print(f"  {'-'*42}")
    for rank, (feat, scores) in enumerate(ranked[:15], 1):
        print(f"  {rank:<4} {feat:<22} {np.mean(scores):>11.1f}%")

# ─── Phase 3: カテゴリ別バックテスト ─────────────────────────
print("\n" + "█"*110)
print(f"  PHASE 3: v79 改善バリアント バックテスト（カテゴリ別過学習チェック付き）")
print(f"  v79A: 日足EMA20方向一致  v79B: ADX≥{ADX_THRESHOLD}  v79C: 4H streak≥{STREAK_MIN}  v79D: MDD scale  v79BC/ABC: 組合せ")
print("█"*110)

all_bt_results = []
cat_recommendations = {}

for cat_name, cat_cfg in CATEGORIES.items():
    symbols   = cat_cfg["symbols"]
    utc_start = cat_cfg["utc_start"]
    utc_end   = cat_cfg["utc_end"]
    pass_min  = cat_cfg["pass_min"]
    n_sym     = len(symbols)

    print(f"\n{'#'*110}")
    print(f"  CATEGORY: {cat_name}  ({n_sym}銘柄)  セッション: UTC{utc_start}-{utc_end}")
    print(f"  カテゴリPASS基準: {pass_min}/{n_sym}銘柄以上でOOS PF改善")
    print(f"{'#'*110}")

    cat_rows = []

    for pair_name, sym in symbols:
        rm = RiskManager(pair_name, risk_pct=RISK_PCT)
        data = load_pair_data(sym, pair_name)

        d1m_is   = data.get("1m_is");   d15m_is  = data.get("15m_is");  d4h_is  = data.get("4h_is")
        d1m_oos  = data.get("1m_oos");  d15m_oos = data.get("15m_oos"); d4h_oos = data.get("4h_oos")

        missing = [n for n,d in [("1m_IS",d1m_is),("15m_IS",d15m_is),("4h_IS",d4h_is),
                                  ("1m_OOS",d1m_oos),("15m_OOS",d15m_oos),("4h_OOS",d4h_oos)]
                   if d is None or len(d)==0]
        if missing:
            print(f"\n  {pair_name}: [SKIP] {missing}")
            continue

        print(f"\n  ── {pair_name}  spread:{rm.spread_pips}pips ─────────")
        print(f"  {'バリアント':<9} {'期間':<4} {'件数':>5} {'WR':>7} {'PF':>6} {'MDD':>7} {'Kelly':>7} {'月次+'}")
        print(f"  {'-'*65}")

        for vname, vflags in VARIANTS:
            mdd_scale = vflags.pop("use_mdd_scale", False)
            for period, d1m, d15m, d4h in [
                ("IS",  d1m_is,  d15m_is,  d4h_is),
                ("OOS", d1m_oos, d15m_oos, d4h_oos),
            ]:
                sigs = generate_signals(
                    d1m, d15m, d4h, rm.spread_pips, rm.pip_size,
                    utc_start=utc_start, utc_end=utc_end,
                    **vflags
                )
                trades, eq = simulate(sigs, d1m, pair_name, use_mdd_scale=mdd_scale)
                st = calc_stats(trades, eq, pair_name, vname, period)
                cat_rows.append(st); all_bt_results.append(st)
                print(f"  {vname:<9} {period:<4} {st['n']:>5} "
                      f"{st['wr']:>6.1f}% {st['pf']:>6.2f} "
                      f"{st['mdd']:>6.1f}% {st['kelly']:>7.3f} "
                      f"{st['monthly_plus']:>8}")
            vflags["use_mdd_scale"] = mdd_scale  # 元に戻す

    if not cat_rows:
        continue

    # カテゴリ集計
    v77_oos = {r["pair"]: r["pf"] for r in cat_rows if r["period"]=="OOS" and r["variant"]=="v77"}
    print(f"\n  ── {cat_name} カテゴリ集計（OOS） ────────────────")
    print(f"  {'バリアント':<9} {'avg_PF':>7} {'改善':>6} {'IS乖離チェック':<28} {'判定'}")
    print(f"  {'-'*65}")

    best_v = "v77"; best_pf = sum(v77_oos.values())/len(v77_oos) if v77_oos else 0

    for vname, _ in VARIANTS:
        oos_v = [r for r in cat_rows if r["period"]=="OOS" and r["variant"]==vname]
        is_v  = [r for r in cat_rows if r["period"]=="IS"  and r["variant"]==vname]
        is_77 = [r for r in cat_rows if r["period"]=="IS"  and r["variant"]=="v77"]
        if not oos_v: continue

        avg_pf   = sum(r["pf"] for r in oos_v) / len(oos_v)
        improved = sum(1 for r in oos_v if v77_oos.get(r["pair"],0) < r["pf"])

        # IS/OOS乖離
        is_pf_v77  = sum(r["pf"] for r in is_77)  / max(len(is_77), 1)
        is_pf_vn   = sum(r["pf"] for r in is_v)   / max(len(is_v),  1)
        oos_pf_vn  = avg_pf
        oos_pf_v77 = sum(v77_oos.values()) / max(len(v77_oos), 1)
        is_delta  = is_pf_vn  - is_pf_v77
        oos_delta = oos_pf_vn - oos_pf_v77
        overfit   = (is_delta > 0.15) and (oos_delta < is_delta * 0.5)
        ratio_str = f"IS{is_delta:+.2f}/OOS{oos_delta:+.2f}"

        cat_pass = (improved >= pass_min)
        mark = "✅PASS" if cat_pass else "❌FAIL"
        of   = " ⚠️過学習疑い" if overfit else ""

        print(f"  {vname:<9} {avg_pf:>7.2f}  {improved:>2}/{len(oos_v):<3}  {ratio_str:<28} {mark}{of}")

        if cat_pass and avg_pf > best_pf and not overfit:
            best_pf = avg_pf; best_v = vname

    cat_best_oos = [r for r in cat_rows if r["period"]=="OOS" and r["variant"]==best_v]
    cat_recommendations[cat_name] = {
        "best_variant": best_v,
        "avg_pf": round(best_pf, 2),
        "symbols": {r["pair"]: r["pf"] for r in cat_best_oos},
    }
    print(f"\n  ★ {cat_name} 推奨: {best_v}  (avg OOS PF: {best_pf:.2f})")

# ── 最終サマリー ─────────────────────────────────────────────
print("\n" + "="*110)
print("最終推奨サマリー（カテゴリ別）")
print("="*110)
print(f"{'カテゴリ':<10} {'推奨バリアント':<12} {'avg OOS PF':>10}  銘柄別 OOS PF")
print("-"*110)
for cat_name, rec in cat_recommendations.items():
    pfs = "  ".join(f"{p}:{pf:.2f}" for p, pf in rec["symbols"].items())
    print(f"{cat_name:<10} {rec['best_variant']:<12} {rec['avg_pf']:>10.2f}  {pfs}")

print("\n【フィルター設計方針（過学習防止）】")
print(f"  - ADX閾値 {ADX_THRESHOLD}: 一般的な「強トレンド」基準値（OOSデータで最適化せず）")
print(f"  - Streak {STREAK_MIN}本: 連続方向確認（短すぎず長すぎない固定値）")
print(f"  - MDD Scale開始 {int(MDD_SCALE_START*100)}%: ポートフォリオ理論の標準的ドローダウン管理")
print(f"  - 全カテゴリ同一パラメータ（銘柄ごとにチューニングしない）")

# ── CSV保存 ──────────────────────────────────────────────────
df_out = pd.DataFrame(all_bt_results)
csv_path = os.path.join(OUT_DIR, "v79_mdd_trend_results.csv")
df_out.to_csv(csv_path, index=False)
print(f"\n結果CSV: {csv_path}")

# ── 可視化 ──────────────────────────────────────────────────
cat_colors = {"FX": "#3b82f6", "METALS": "#f59e0b", "INDICES": "#10b981"}
var_names  = [v for v, _ in VARIANTS]

fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle(f"v79 MDD対策 + トレンドフォロー改善 OOS結果\n"
             f"A:1D_EMA  B:ADX≥{ADX_THRESHOLD}  C:Streak≥{STREAK_MIN}  D:MDD_Scale  BC/ABC:組合せ",
             fontsize=11, fontweight="bold")

metrics = [("pf","OOS PF",2.0),("wr","OOS WR(%)",None),("mdd","OOS MDD(%)",None)]

for row_i, (cat_name, cat_cfg) in enumerate(CATEGORIES.items()):
    for col_i, (metric, mlabel, ref) in enumerate(metrics):
        ax = axes[row_i, col_i]
        syms  = [s[0] for s in cat_cfg["symbols"]]
        color = cat_colors.get(cat_name, "#6b7280")
        best_v = cat_recommendations.get(cat_name, {}).get("best_variant", "v77")

        x = np.arange(len(var_names))
        w = 0.6 / max(len(syms), 1)
        for si, pair_name in enumerate(syms):
            vals = []
            for vname in var_names:
                r = df_out[(df_out["pair"]==pair_name)&(df_out["variant"]==vname)&(df_out["period"]=="OOS")]
                vals.append(r[metric].values[0] if len(r) > 0 else 0)
            ax.bar(x + si*w, vals, w, label=pair_name, alpha=0.85)

        if ref:
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        if best_v in var_names:
            bx = var_names.index(best_v)
            ax.axvline(bx, color="red", linestyle=":", linewidth=1.5, alpha=0.5)

        ax.set_title(f"{cat_name} — {mlabel}", fontsize=9, fontweight="bold")
        ax.set_xticks(x + w*(len(syms)-1)/2)
        ax.set_xticklabels(var_names, rotation=35, ha="right", fontsize=7)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "v79_mdd_trend_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート: {chart_path}")
print("\n全処理完了。")
