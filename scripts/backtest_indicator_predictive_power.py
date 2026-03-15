"""
backtest_indicator_predictive_power.py
======================================
YAGAMI改の各指標が将来リターンに対して本当に予測力を持つか検証。

【検証する指標】
  1. KMID   — 4H実体方向 (binary: エントリー方向と一致 1/0)
  2. KLOW   — 4H下ヒゲ比率 (continuous: 0~1)
  3. EMA距離 — 4H close vs EMA20 の距離/ATR (continuous)
  4. ADX    — 4H ADX値 (continuous)
  5. Streak — 4H連続同方向足数 (discrete)
  6. 1D trend — 日足EMA20方向一致 (binary)
  7. tol距離 — 二番底/天井の2つの安値/高値の差/ATR (continuous)

【分析手法】
  ・将来リターン = エントリー後N本(1h)の方向別PnL / ATR(正規化済み)
  ・散布図（各指標 vs 将来リターン）
  ・回帰係数 + p値 + R²
  ・ピアソン/スピアマン相関
  ・ローリング相関（60トレード窓）で安定性評価
  ・IS期間 vs OOS期間 の相関比較（頑健性）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import SYMBOL_CONFIG

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    {"sym": "GBPUSD",  "logic": "A", "tol": 0.30},
    {"sym": "EURUSD",  "logic": "C", "tol": 0.30},
    {"sym": "USDCAD",  "logic": "A", "tol": 0.30},
    {"sym": "NZDUSD",  "logic": "A", "tol": 0.20},
    {"sym": "XAUUSD",  "logic": "A", "tol": 0.20},
    {"sym": "AUDUSD",  "logic": "B", "tol": 0.30},
    {"sym": "USDJPY",  "logic": "C", "tol": 0.30},
]

KLOW_THR        = 0.0015
EMA_DIST_MIN    = 1.0
ADX_MIN         = 20
STREAK_MIN      = 4
LOOKAHEAD_1H    = 10  # 将来リターン計算用（エントリー後10本の1H足）
ROLLING_WINDOW  = 60  # ローリング相関の窓サイズ

# ── データロード ──────────────────────────────────────────────
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
    d1m = None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            d1m = load_csv(p); break
    if d1m is None: return None, None
    d4h = None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
        if os.path.exists(p):
            d4h = load_csv(p); break
    if d4h is None:
        p_is  = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
        p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
        if os.path.exists(p_is) and os.path.exists(p_oos):
            d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
            d4h = d4h[~d4h.index.duplicated(keep="first")].sort_index()
        else:
            d4h = d1m.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open", "close"])
    return d1m, d4h

def split_is_oos(d1m):
    n = int(len(d1m) * 0.4)
    ts = d1m.index[n]
    return d1m[d1m.index < ts].copy(), d1m[d1m.index >= ts].copy(), ts

# ── インジケーター ────────────────────────────────────────────
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

def build_4h(df4h):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calc_adx(df, 14)
    # streak計算
    df["streak"] = 0
    s = 0
    prev = 0
    streaks = []
    for t in df["trend"].values:
        if t == prev:
            s += 1
        else:
            s = 1
            prev = t
        streaks.append(s)
    df["streak"] = streaks
    return df

def build_1h(df):
    r = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    r["atr"]   = calc_atr(r, 14)
    r["ema20"] = r["close"].ewm(span=20, adjust=False).mean()
    return r

def build_1d(d4h):
    d1 = d4h.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return d1

# ══════════════════════════════════════════════════════════════
# シグナル候補 + 指標値を全て記録（フィルターは適用しない）
# ══════════════════════════════════════════════════════════════

def collect_signal_features(d1m, d4h_full, spread, tol=0.30):
    """
    二番底/天井パターンが成立した全候補に対して:
    - 各指標の値を記録
    - 将来リターン（エントリー後の方向別損益/ATR）を計算
    フィルターは適用しない（指標の予測力を個別に評価するため）
    """
    d4h = build_4h(d4h_full)
    d1h = build_1h(d1m)
    d1d = build_1d(d4h_full)

    records = []

    for i in range(2, len(d1h) - LOOKAHEAD_1H):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        cur = d1h.iloc[i]
        atr1h = cur["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue

        # 4H情報
        h4b = d4h[d4h.index < hct]
        if len(h4b) < STREAK_MIN: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)) or h4l["atr"] <= 0: continue
        trend = h4l["trend"]

        # 二番底/天井チェック（緩い閾値 0.50 で広くキャプチャ）
        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        tol_dist = abs(v1 - v2) / atr1h
        if tol_dist > 0.50: continue  # 最低限のパターンフィルターのみ

        # ── 指標値を全て計算 ──
        # 1. KMID
        kmid = 1 if ((d == 1 and h4l["close"] > h4l["open"]) or
                      (d == -1 and h4l["close"] < h4l["open"])) else 0

        # 2. KLOW
        klow = (min(h4l["open"], h4l["close"]) - h4l["low"]) / h4l["open"] if h4l["open"] > 0 else 0

        # 3. EMA距離（ATR正規化）
        ema_dist = abs(h4l["close"] - h4l["ema20"]) / h4l["atr"] if h4l["atr"] > 0 else 0

        # 4. ADX
        adx_val = h4l.get("adx", 0)

        # 5. Streak
        streak_val = h4l.get("streak", 0)

        # 6. 1D trend一致
        d1b = d1d[d1d.index.normalize() < hct.normalize()]
        d1_trend_match = 0
        if len(d1b) > 0:
            d1_trend_match = 1 if d1b.iloc[-1]["trend1d"] == d else 0

        # 7. tol距離（正規化済み）
        # already computed as tol_dist

        # 8. 確認足方向一致
        confirm = 0
        if d == 1 and p1["close"] > p1["open"]: confirm = 1
        if d == -1 and p1["close"] < p1["open"]: confirm = 1

        # 9. 4Hボディ比率
        h4_range = h4l["high"] - h4l["low"]
        h4_body_ratio = abs(h4l["close"] - h4l["open"]) / h4_range if h4_range > 0 else 0

        # ── 将来リターン ──
        # エントリー価格 = 現在1H足のclose + spread
        ep = cur["close"] + (spread if d == 1 else -spread)
        # 将来N本の1Hデータ
        future = d1h.iloc[i+1:i+1+LOOKAHEAD_1H]
        if len(future) < 3: continue

        # 方向別リターン（ATR正規化）
        if d == 1:
            max_fav = (future["high"].max() - ep) / atr1h
            max_adv = (ep - future["low"].min()) / atr1h
            close_ret = (future.iloc[-1]["close"] - ep) / atr1h
        else:
            max_fav = (ep - future["low"].min()) / atr1h
            max_adv = (future["high"].max() - ep) / atr1h
            close_ret = (ep - future.iloc[-1]["close"]) / atr1h

        # SL/TP想定の勝敗判定（RR=2.5, 半利確なし）
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if risk <= 0: continue
        tp = raw + d * risk * 2.5
        # 簡易勝敗
        won = 0
        for j in range(i+1, min(i+1+LOOKAHEAD_1H*4, len(d1h))):
            if d == 1:
                if d1h.iloc[j]["low"] <= sl: won = 0; break
                if d1h.iloc[j]["high"] >= tp: won = 1; break
            else:
                if d1h.iloc[j]["high"] >= sl: won = 0; break
                if d1h.iloc[j]["low"] <= tp: won = 1; break

        records.append({
            "time": hct, "dir": d,
            "kmid": kmid, "klow": klow, "ema_dist": ema_dist,
            "adx": adx_val, "streak": streak_val,
            "d1_trend": d1_trend_match, "tol_dist": tol_dist,
            "confirm": confirm, "h4_body_ratio": h4_body_ratio,
            "close_ret": close_ret, "max_fav": max_fav, "max_adv": max_adv,
            "won": won, "risk_atr": risk / atr1h,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# 分析関数
# ══════════════════════════════════════════════════════════════

def analyze_indicator(df, ind_col, target_col="won", label=None):
    """単一指標の予測力を分析"""
    x = df[ind_col].values.astype(float)
    y = df[target_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 30: return None

    # 相関
    pearson_r, pearson_p = sp_stats.pearsonr(x, y)
    spearman_r, spearman_p = sp_stats.spearmanr(x, y)

    # 単回帰
    slope, intercept, r_val, p_val, se = sp_stats.linregress(x, y)

    # ローリング相関の安定性
    s = pd.Series(x); t_s = pd.Series(y)
    roll_corr = s.rolling(ROLLING_WINDOW).corr(t_s).dropna()
    roll_mean = roll_corr.mean()
    roll_std  = roll_corr.std()
    roll_sign_pct = (roll_corr > 0).mean() if len(roll_corr) > 0 else 0  # 符号一貫率

    # バイナリ指標の場合 WR比較
    wr_1 = wr_0 = None
    unique = np.unique(x)
    if len(unique) == 2:
        wr_1 = y[x == 1].mean() if (x == 1).sum() > 0 else None
        wr_0 = y[x == 0].mean() if (x == 0).sum() > 0 else None

    return {
        "indicator": label or ind_col,
        "n": len(x),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "slope": slope, "r2": r_val**2, "reg_p": p_val, "se": se,
        "roll_mean": roll_mean, "roll_std": roll_std, "roll_sign_pct": roll_sign_pct,
        "wr_1": wr_1, "wr_0": wr_0,
    }


def print_analysis(results, sym, period_label):
    """分析結果を整形出力"""
    print(f"\n  {'指標':18} | {'n':>5} {'Pearson r':>10} {'p値':>10} {'Spearman':>10} {'回帰β':>10} "
          f"{'R²':>8} {'p値':>10} | {'Roll平均':>8} {'Roll安定':>9} {'符号一貫':>8} | {'WR(1)':>6} {'WR(0)':>6}")
    print("  " + "-"*155)
    for r in results:
        if r is None: continue
        sig = ""
        if r["pearson_p"] < 0.001: sig = "***"
        elif r["pearson_p"] < 0.01: sig = "**"
        elif r["pearson_p"] < 0.05: sig = "*"
        elif r["pearson_p"] < 0.10: sig = "†"

        wr1 = f"{r['wr_1']*100:.1f}%" if r["wr_1"] is not None else "  -  "
        wr0 = f"{r['wr_0']*100:.1f}%" if r["wr_0"] is not None else "  -  "

        print(f"  {r['indicator']:18} | {r['n']:>5} {r['pearson_r']:>+10.4f} {r['pearson_p']:>10.4f}{sig:3} "
              f"{r['spearman_r']:>+10.4f} {r['slope']:>+10.5f} "
              f"{r['r2']:>8.5f} {r['reg_p']:>10.4f} | "
              f"{r['roll_mean']:>+8.4f} {r['roll_std']:>9.4f} {r['roll_sign_pct']*100:>7.1f}% | "
              f"{wr1:>6} {wr0:>6}")


# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

INDICATORS = [
    ("kmid",          "KMID(実体方向)"),
    ("klow",          "KLOW(下ヒゲ)"),
    ("ema_dist",      "EMA距離/ATR"),
    ("adx",           "ADX"),
    ("streak",        "Streak(連続)"),
    ("d1_trend",      "日足EMA方向"),
    ("tol_dist",      "tol距離/ATR"),
    ("confirm",       "確認足方向"),
    ("h4_body_ratio", "4Hボディ比率"),
]

def main():
    print("\n" + "="*160)
    print("  YAGAMI改 指標の予測力検証")
    print("  目的: 各指標が将来の勝敗(won)を本当に予測できているか")
    print("  有意水準: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10")
    print("="*160)

    all_data_is  = []
    all_data_oos = []
    sym_results  = {}

    for tgt in TARGETS:
        sym = tgt["sym"]
        print(f"\n  ◆ {sym} (logic={tgt['logic']}, tol={tgt['tol']}) ...", end=" ", flush=True)

        d1m, d4h = load_all(sym)
        if d1m is None:
            print("データなし"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m)
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]

        # IS/OOS それぞれで特徴量収集
        df_is  = collect_signal_features(is_d, d4h, spread, tol=0.50)
        df_oos = collect_signal_features(oos_d, d4h, spread, tol=0.50)
        print(f"IS={len(df_is)}件, OOS={len(df_oos)}件")

        if len(df_is) < 30 or len(df_oos) < 30: continue

        df_is["sym"]  = sym; df_oos["sym"] = sym
        all_data_is.append(df_is)
        all_data_oos.append(df_oos)

        # 銘柄別分析
        results_is = []
        results_oos = []
        for col, lbl in INDICATORS:
            r_is  = analyze_indicator(df_is,  col, target_col="won", label=lbl)
            r_oos = analyze_indicator(df_oos, col, target_col="won", label=lbl)
            results_is.append(r_is)
            results_oos.append(r_oos)
        sym_results[sym] = {"is": results_is, "oos": results_oos}

        print(f"\n  [{sym}] IS期間:")
        print_analysis(results_is, sym, "IS")
        print(f"\n  [{sym}] OOS期間:")
        print_analysis(results_oos, sym, "OOS")

    # ══ 全銘柄統合分析 ══════════════════════════════════════════
    if all_data_oos:
        print("\n" + "="*160)
        print("  ■ 全銘柄統合（OOS）")
        print("="*160)
        df_all_oos = pd.concat(all_data_oos, ignore_index=True)
        results_all = []
        for col, lbl in INDICATORS:
            r = analyze_indicator(df_all_oos, col, target_col="won", label=lbl)
            results_all.append(r)
        print_analysis(results_all, "ALL", "OOS統合")

        # ── IS vs OOS の相関比較 ──
        df_all_is = pd.concat(all_data_is, ignore_index=True)
        print("\n" + "="*100)
        print("  ■ IS vs OOS 相関比較（頑健性チェック）")
        print(f"  {'指標':18} | {'IS Pearson':>11} {'OOS Pearson':>12} | {'IS Spearman':>12} {'OOS Spearman':>13} | {'符号一致':>8} {'方向安定':>8}")
        print("  " + "-"*100)
        results_is_all = []
        for col, lbl in INDICATORS:
            r_is  = analyze_indicator(df_all_is,  col, target_col="won", label=lbl)
            r_oos = analyze_indicator(df_all_oos, col, target_col="won", label=lbl)
            results_is_all.append(r_is)
            if r_is and r_oos:
                sign_match = "✅" if (r_is["pearson_r"] > 0) == (r_oos["pearson_r"] > 0) else "❌"
                # 方向安定: 相関の符号がISでもOOSでも同じで有意
                stable = "✅" if (sign_match == "✅" and
                                 r_is["pearson_p"] < 0.1 and r_oos["pearson_p"] < 0.1) else "⚠️"
                print(f"  {lbl:18} | {r_is['pearson_r']:>+11.4f} {r_oos['pearson_r']:>+12.4f} | "
                      f"{r_is['spearman_r']:>+12.4f} {r_oos['spearman_r']:>+13.4f} | "
                      f"{sign_match:>8} {stable:>8}")

    # ── サマリー ──────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 判定サマリー（OOS統合）")
    print("="*100)
    for r in results_all:
        if r is None: continue
        verdict = "✅ 有意" if r["pearson_p"] < 0.05 else ("⚠️ 弱い" if r["pearson_p"] < 0.10 else "❌ 無意味")
        stability = "安定" if r["roll_sign_pct"] > 0.65 else "不安定"
        print(f"  {r['indicator']:18} → {verdict:10} (p={r['pearson_p']:.4f}, r={r['pearson_r']:+.4f}) "
              f"ローリング符号一貫率={r['roll_sign_pct']*100:.0f}%({stability})")

    # ── CSV保存 ──────────────────────────────────────────────
    csv_rows = []
    for r in results_all:
        if r is None: continue
        csv_rows.append(r)
    out_path = os.path.join(OUT_DIR, "indicator_predictive_power.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"\n  結果保存: {out_path}")

    # 全データ保存（散布図用）
    if all_data_oos:
        df_all_oos.to_csv(os.path.join(OUT_DIR, "indicator_features_oos.csv"), index=False)
        print(f"  特徴量データ保存: {os.path.join(OUT_DIR, 'indicator_features_oos.csv')}")


if __name__ == "__main__":
    main()
