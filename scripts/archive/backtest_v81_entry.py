"""
v81 エントリー条件見直しバックテスト
=====================================
v80（ローソク足フィルター）に続く第2フェーズ。
戦略の「何を二番底と定義するか」自体を見直す。

【テスト対象バリアント】
  v77    : ベースライン（tol=0.3, フィルターなし）
  v81A_tight: tolerance=ATR×0.1（最厳格）
  v81A_mid  : tolerance=ATR×0.2（中間）
  v81B      : neckline_break=True（ネックライン突破必須）
  v81C      : ema_dist_min=1.0（EMA距離≥ATR×1.0）
  v81AB     : A(0.2)+B
  v81BC     : B+C
  v81ABC    : A(0.2)+B+C

【ネックライン突破の定義】
  ロング: 確認足終値 ≥ 直前バー（第1谷）の高値
  ショート: 確認足終値 ≤ 直前バー（第1天井）の安値
  → 二番底パターンの中間高値（ネックライン）を終値で抜けた確認

【設計方針（過学習防止）】
  - tolerance値は 0.1/0.2（いずれもデータ非依存の固定値）
  - neckline_break はやがみメソッドの古典TA原則
  - ema_dist_min=1.0 はATRベース（データ非依存）

IS: 2024-07-01〜2025-02-28  OOS: 2025-03-01〜2026-02-28
銘柄: EURUSD / GBPUSD / AUDUSD / XAUUSD
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG
from strategies.current.yagami_mtf_v79 import generate_signals

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

INIT_CASH = 1_000_000
RR_RATIO  = 2.5
HALF_R    = 1.0
IS_START  = "2024-07-01"; IS_END  = "2025-02-28"
OOS_START = "2025-03-01"; OOS_END = "2026-02-28"
SYMBOLS   = ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"]

VARIANTS = {
    "v77":         dict(tol_factor=0.3, neckline_break=False, ema_dist_min=0.0),
    "v81A_tight":  dict(tol_factor=0.1, neckline_break=False, ema_dist_min=0.0),
    "v81A_mid":    dict(tol_factor=0.2, neckline_break=False, ema_dist_min=0.0),
    "v81B":        dict(tol_factor=0.3, neckline_break=True,  ema_dist_min=0.0),
    "v81C":        dict(tol_factor=0.3, neckline_break=False, ema_dist_min=1.0),
    "v81AB":       dict(tol_factor=0.2, neckline_break=True,  ema_dist_min=0.0),
    "v81BC":       dict(tol_factor=0.3, neckline_break=True,  ema_dist_min=1.0),
    "v81ABC":      dict(tol_factor=0.2, neckline_break=True,  ema_dist_min=1.0),
}


def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def load_data(sym_upper, tf, period):
    sym_lower = sym_upper.lower()
    p = os.path.join(DATA_DIR, f"{sym_lower}_{period}_{tf}.csv")
    if os.path.exists(p):
        return load_csv(p)
    p2 = os.path.join(DATA_DIR, "ohlc", f"{sym_upper}_{tf}.csv")
    if os.path.exists(p2):
        df = load_csv(p2)
        if df is not None:
            s = IS_START if period == "is" else OOS_START
            e = IS_END   if period == "is" else OOS_END
            return df[(df.index >= s) & (df.index <= e)].copy()
    return None


def load_symbol_data(sym, period):
    d4h  = load_data(sym, "4h", period)
    d15m = load_data(sym, "15m", period)
    d1m  = load_data(sym, "1m", period)
    if d1m is None or len(d1m) == 0:
        d1m = d15m
    return d1m, d15m, d4h


def simulate_trades(signals, data_intra, uj_intra, rm, period_start):
    if not signals:
        return [], [(pd.Timestamp(period_start, tz="UTC"), INIT_CASH)]
    uj_init = 150.0
    if uj_intra is not None and len(uj_intra) > 0:
        uj_init = float(uj_intra.iloc[0]["close"])
    m1t = data_intra.index.values; m1h = data_intra["high"].values; m1l = data_intra["low"].values
    ujt = uj_intra.index.values   if uj_intra is not None else None
    ujc = uj_intra["close"].values if uj_intra is not None else None
    equity = INIT_CASH
    eq_tl  = [(pd.Timestamp(period_start, tz="UTC"), equity)]
    trades = []
    for sig in signals:
        ep, sl, tp = sig["ep"], sig["sl"], sig["tp"]
        risk, direction = sig["risk"], sig["dir"]
        si = np.searchsorted(m1t, np.datetime64(sig["time"]), side="right")
        if si >= len(m1t): continue
        lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=uj_init)
        if lot <= 0: continue
        half_tp = (ep + (tp - ep) * (HALF_R / RR_RATIO)) if direction == 1 \
             else (ep - (ep - tp) * (HALF_R / RR_RATIO))
        half_done = False; sl_cur = sl; result = None; exit_i = None
        for i in range(si, len(m1t)):
            h = m1h[i]; lo = m1l[i]
            if direction == 1:
                if lo <= sl_cur:   result = "SL"; exit_i = i; break
                if not half_done and h >= half_tp: half_done = True; sl_cur = ep
                if h >= tp:        result = "TP"; exit_i = i; break
            else:
                if h >= sl_cur:    result = "SL"; exit_i = i; break
                if not half_done and lo <= half_tp: half_done = True; sl_cur = ep
                if lo <= tp:       result = "TP"; exit_i = i; break
        if result is None:
            result = "BE" if half_done else "OPEN"; exit_i = len(m1t) - 1
        if result == "OPEN": continue
        et = pd.Timestamp(m1t[exit_i]); et = et.tz_localize("UTC") if et.tzinfo is None else et
        uj_ex = uj_init
        if ujt is not None:
            idx = np.searchsorted(ujt, m1t[exit_i], side="right") - 1
            if idx >= 0: uj_ex = float(ujc[idx])
        if result == "TP":
            pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=uj_init,  ref_price=ep) +
                   rm.calc_pnl_jpy(direction, ep, tp,      lot*0.5, usdjpy_rate=uj_ex,   ref_price=ep)) \
                  if half_done else \
                   rm.calc_pnl_jpy(direction, ep, tp, lot, usdjpy_rate=uj_ex, ref_price=ep)
        elif result == "SL":
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=uj_init, ref_price=ep) \
                  if half_done else \
                  rm.calc_pnl_jpy(direction, ep, sl_cur, lot, usdjpy_rate=uj_ex, ref_price=ep)
        else:
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=uj_init, ref_price=ep)
        equity += pnl; eq_tl.append((et, equity))
        trades.append({"exit_time": et, "result": result, "pnl": pnl, "equity": equity, "dir": direction})
    return trades, eq_tl


def calc_metrics(trades, eq_tl, label):
    if not trades:
        return {"period": label, "n": 0, "wr": 0.0, "pf": 0.0, "mdd": 0.0, "monthly_pos": 0.0}
    df = pd.DataFrame(trades)
    n = len(df); wr = (df["result"] == "TP").mean() * 100
    gp = df[df["pnl"] > 0]["pnl"].sum(); gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    eq_df = pd.DataFrame(eq_tl, columns=["time", "equity"]).sort_values("time")
    peak  = np.maximum.accumulate(eq_df["equity"].values)
    mdd   = (peak - eq_df["equity"].values).max()
    df["mo"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    mp = df.groupby("mo")["pnl"].sum()
    monthly_pos = (mp > 0).sum() / len(mp) * 100 if len(mp) > 0 else 0.0
    return {"period": label, "n": n, "wr": wr, "pf": pf, "mdd": mdd, "monthly_pos": monthly_pos}


print("=" * 76)
print("v81 エントリー条件見直しバックテスト — tolerance / neckline / EMA距離")
print("=" * 76)
print(f"IS: {IS_START}〜{IS_END}   OOS: {OOS_START}〜{OOS_END}")
print()

uj_is  = load_data("USDJPY", "15m", "is")
uj_oos = load_data("USDJPY", "15m", "oos")
all_res = {sym: {} for sym in SYMBOLS}

for sym in SYMBOLS:
    cfg = SYMBOL_CONFIG.get(sym)
    if cfg is None: continue
    rm = RiskManager(sym)
    spread_pips = cfg["spread"]; pip_size = cfg["pip"]

    print(f"\n{'━'*76}")
    print(f"▶ {sym}  spread={spread_pips}pips  pip={pip_size}")
    print(f"{'━'*76}")
    print(f"  {'variant':12s}  {'IS n':>5s}  {'IS WR':>6s}  {'IS PF':>6s}  "
          f"{'OOS n':>6s}  {'OOS WR':>6s}  {'OOS PF':>7s}  {'月次+':>5s}  判定")
    print(f"  {'─'*74}")

    for v_name, v_params in VARIANTS.items():
        all_res[sym][v_name] = {}
        is_m = oos_m = None
        for period, p_start, uj in [("is", IS_START, uj_is), ("oos", OOS_START, uj_oos)]:
            d1m, d15m, d4h = load_symbol_data(sym, period)
            if d4h is None or d15m is None or d1m is None: continue
            try:
                sigs = generate_signals(d1m, d15m, d4h,
                                        spread_pips=spread_pips, rr_ratio=RR_RATIO,
                                        pip_size=pip_size, **v_params)
            except Exception as e:
                print(f"  {v_name} {period}: {e}"); continue
            trades, eq_tl = simulate_trades(sigs, d1m, uj, rm, p_start)
            m = calc_metrics(trades, eq_tl, period.upper())
            all_res[sym][v_name][period.upper()] = m
            if period == "is": is_m = m
            else:              oos_m = m

        if is_m is None or oos_m is None: continue
        v77_oos_pf = all_res[sym].get("v77", {}).get("OOS", {}).get("pf")
        if v77_oos_pf is not None and v_name != "v77":
            delta = oos_m["pf"] - v77_oos_pf
            ok = "✅" if delta > 0 else "❌"
        else:
            delta = 0.0; ok = "  "
        print(f"  {v_name:12s}  {is_m['n']:5d}  {is_m['wr']:5.1f}%  {is_m['pf']:6.2f}  "
              f"{oos_m['n']:6d}  {oos_m['wr']:5.1f}%  {oos_m['pf']:7.2f}  "
              f"{oos_m['monthly_pos']:4.0f}%  {ok} ({delta:+.2f})")


# ── OOS PF サマリー ─────────────────────────────────────────
print("\n\n" + "=" * 76)
print("【OOS PF サマリー】")
print("=" * 76)
hdr = f"{'variant':12s} | " + " | ".join(f"{s:8s}" for s in SYMBOLS) + " | avg改善"
print(hdr); print("─" * len(hdr))

for v_name in VARIANTS:
    pfs = []; v77s = []
    cells = []; n_imp = 0; n_tot = 0
    for sym in SYMBOLS:
        op = all_res[sym].get(v_name, {}).get("OOS", {}).get("pf")
        vp = all_res[sym].get("v77",  {}).get("OOS", {}).get("pf")
        if op is None:
            cells.append(f"{'N/A':8s}")
        else:
            pfs.append(op)
            star = ""
            if v_name != "v77" and vp is not None:
                v77s.append(vp); n_tot += 1
                if op > vp: n_imp += 1; star = "✓"
                else: star = " "
            cells.append(f"{op:6.2f}{star} ")
    avg = ""
    if v_name != "v77" and pfs and v77s:
        d = np.mean(pfs) - np.mean(v77s)
        avg = f" {'+' if d>=0 else ''}{d:.2f} ({n_imp}/{n_tot}改善)"
    print(f"{v_name:12s} | " + " | ".join(cells) + f"|{avg}")


# ── IS/OOS 過学習チェック ─────────────────────────────────────
print("\n\n" + "=" * 76)
print("【過学習チェック】IS改善 vs OOS改善")
print("=" * 76)
print(f"{'variant':12s}  {'sym':6s}  IS変化   OOS変化  乖離   判定")
print("─" * 60)
for v_name in VARIANTS:
    if v_name == "v77": continue
    for sym in SYMBOLS:
        ri_77  = all_res[sym].get("v77",   {}).get("IS",  {})
        ro_77  = all_res[sym].get("v77",   {}).get("OOS", {})
        ri_v   = all_res[sym].get(v_name,  {}).get("IS",  {})
        ro_v   = all_res[sym].get(v_name,  {}).get("OOS", {})
        if not all([ri_77, ro_77, ri_v, ro_v]): continue
        is_d  = ri_v.get("pf", 0) - ri_77.get("pf", 0)
        oos_d = ro_v.get("pf", 0) - ro_77.get("pf", 0)
        gap   = is_d - oos_d
        if abs(is_d) < 0.05 and abs(oos_d) < 0.05: verdict = "→ 効果なし"
        elif is_d > 0.2 and is_d > oos_d * 2:       verdict = "⚠ 過学習疑い"
        elif oos_d > 0 and oos_d >= is_d * 0.5:     verdict = "✅ 有効・過学習なし"
        elif oos_d > 0:                              verdict = "✅ OOS改善"
        else:                                        verdict = "❌ OOS悪化"
        print(f"{v_name:12s}  {sym:6s}  IS:{is_d:+.2f}  OOS:{oos_d:+.2f}  "
              f"gap:{gap:+.2f}  {verdict}")


# ── CSV出力 ───────────────────────────────────────────────────
rows = []
for sym in SYMBOLS:
    for v_name, v_params in VARIANTS.items():
        for period in ["IS", "OOS"]:
            m = all_res[sym].get(v_name, {}).get(period)
            if m is None: continue
            rows.append({"symbol": sym, "variant": v_name, "period": period,
                          "n_trades": m["n"], "win_rate": round(m["wr"], 2),
                          "pf": round(m["pf"], 3), "mdd_jpy": round(m["mdd"]),
                          "monthly_pos_pct": round(m["monthly_pos"], 1),
                          **v_params})
csv_path = os.path.join(OUT_DIR, "backtest_v81_entry.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n→ {csv_path}")
print("✅ v81バックテスト完了")
