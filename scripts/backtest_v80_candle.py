"""
v80 ローソク足フィルター バックテスト
=====================================
ベースライン: v77相当（セッションフィルターなし）
比較対象:
  v80A: body_ratio_min=0.5  確認足の実体比率 ≥ 50%
  v80B: ascending_only=True  上昇型二番底/下降型二番天井のみ
  v80C: wick_limit=True      確認足の逆ヒゲ ≤ 実体
  v80AB: A+B
  v80BC: B+C
  v80ABC: A+B+C（完全組み合わせ）

設計方針（過学習防止）:
  - 全パラメータは固定値（データ非依存の古典TA基準）
  - body_ratio_min=0.5 は「強い実体」の業界標準
  - ascending_only は古典TA原則（上昇型二番底=ブルダイバージェンス）
  - wick_limit は方向一致でない抵抗を示すヒゲの排除

IS期間:  2024-07-01 〜 2025-02-28
OOS期間: 2025-03-01 〜 2026-02-28
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

# ── パラメータ ─────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0

IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-01"
OOS_END   = "2026-02-28"

# ── テスト対象銘柄 ─────────────────────────────────────────
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"]

# ── バリアント定義 ─────────────────────────────────────────
VARIANTS = {
    "v77":    dict(body_ratio_min=0.0, ascending_only=False, wick_limit=False),
    "v80A":   dict(body_ratio_min=0.5, ascending_only=False, wick_limit=False),
    "v80B":   dict(body_ratio_min=0.0, ascending_only=True,  wick_limit=False),
    "v80C":   dict(body_ratio_min=0.0, ascending_only=False, wick_limit=True),
    "v80AB":  dict(body_ratio_min=0.5, ascending_only=True,  wick_limit=False),
    "v80BC":  dict(body_ratio_min=0.0, ascending_only=True,  wick_limit=True),
    "v80ABC": dict(body_ratio_min=0.5, ascending_only=True,  wick_limit=True),
}


# ── データロード ──────────────────────────────────────────
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
    split_path = os.path.join(DATA_DIR, f"{sym_lower}_{period}_{tf}.csv")
    if os.path.exists(split_path):
        return load_csv(split_path)
    ohlc_path = os.path.join(DATA_DIR, "ohlc", f"{sym_upper}_{tf}.csv")
    if os.path.exists(ohlc_path):
        df = load_csv(ohlc_path)
        if df is not None:
            start = IS_START if period == "is" else OOS_START
            end   = IS_END   if period == "is" else OOS_END
            return df[(df.index >= start) & (df.index <= end)].copy()
    return None


def load_symbol_data(sym, period):
    data_4h  = load_data(sym, "4h", period)
    data_15m = load_data(sym, "15m", period)
    data_1m  = load_data(sym, "1m", period)
    if data_1m is None or len(data_1m) == 0:
        data_1m = data_15m
    return data_1m, data_15m, data_4h


# ── シミュレーション ──────────────────────────────────────
def simulate_trades(signals, data_intra, usdjpy_intra, rm, period_start):
    if not signals:
        return [], [(pd.Timestamp(period_start, tz="UTC"), INIT_CASH)]

    usdjpy_init = 150.0
    if usdjpy_intra is not None and len(usdjpy_intra) > 0:
        usdjpy_init = float(usdjpy_intra.iloc[0]["close"])

    m1_times  = data_intra.index.values
    m1_highs  = data_intra["high"].values
    m1_lows   = data_intra["low"].values
    uj_times  = usdjpy_intra.index.values   if usdjpy_intra is not None else None
    uj_closes = usdjpy_intra["close"].values if usdjpy_intra is not None else None

    equity      = INIT_CASH
    eq_timeline = [(pd.Timestamp(period_start, tz="UTC"), equity)]
    trades      = []

    for sig in signals:
        ep, sl, tp = sig["ep"], sig["sl"], sig["tp"]
        risk, direction = sig["risk"], sig["dir"]
        entry_time = sig["time"]

        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times):
            continue
        lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0:
            continue

        half_tp = (ep + (tp - ep) * (HALF_R / RR_RATIO)) if direction == 1 \
             else (ep - (ep - tp) * (HALF_R / RR_RATIO))
        half_done = False; sl_current = sl; result = None; exit_idx = None

        for i in range(start_idx, len(m1_times)):
            h = m1_highs[i]; lo = m1_lows[i]
            if direction == 1:
                if lo <= sl_current:
                    result = "SL"; exit_idx = i; break
                if not half_done and h >= half_tp:
                    half_done = True; sl_current = ep
                if h >= tp:
                    result = "TP"; exit_idx = i; break
            else:
                if h >= sl_current:
                    result = "SL"; exit_idx = i; break
                if not half_done and lo <= half_tp:
                    half_done = True; sl_current = ep
                if lo <= tp:
                    result = "TP"; exit_idx = i; break

        if result is None:
            result = "BE" if half_done else "OPEN"
            exit_idx = len(m1_times) - 1
        if result == "OPEN":
            continue

        exit_time = pd.Timestamp(m1_times[exit_idx])
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")

        usdjpy_at_exit = usdjpy_init
        if uj_times is not None:
            uj_idx = np.searchsorted(uj_times, m1_times[exit_idx], side="right") - 1
            if uj_idx >= 0:
                usdjpy_at_exit = float(uj_closes[uj_idx])

        if result == "TP":
            pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5, usdjpy_rate=usdjpy_init, ref_price=ep) +
                   rm.calc_pnl_jpy(direction, ep, tp, lot * 0.5, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
                   ) if half_done else \
                  rm.calc_pnl_jpy(direction, ep, tp, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5, usdjpy_rate=usdjpy_init, ref_price=ep) \
                  if half_done else \
                  rm.calc_pnl_jpy(direction, ep, sl_current, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:  # BE
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5, usdjpy_rate=usdjpy_init, ref_price=ep)

        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                        "result": result, "pnl": pnl, "equity": equity, "dir": direction})

    return trades, eq_timeline


def calc_metrics(trades, eq_timeline, period_label):
    if not trades:
        return {"period": period_label, "n": 0, "wr": 0.0, "pf": 0.0,
                "mdd": 0.0, "monthly_pos": 0.0, "monthly_pnl": pd.Series(dtype=float)}
    df = pd.DataFrame(trades)
    n  = len(df); wr = (df["result"] == "TP").mean() * 100
    gp = df[df["pnl"] > 0]["pnl"].sum(); gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    eq_df  = pd.DataFrame(eq_timeline, columns=["time", "equity"]).sort_values("time")
    peak   = np.maximum.accumulate(eq_df["equity"].values)
    mdd    = (peak - eq_df["equity"].values).max()
    df["exit_month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    monthly_pnl = df.groupby("exit_month")["pnl"].sum()
    monthly_pos = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100 if len(monthly_pnl) > 0 else 0.0
    return {"period": period_label, "n": n, "wr": wr, "pf": pf,
            "mdd": mdd, "monthly_pos": monthly_pos, "monthly_pnl": monthly_pnl}


# ── メイン実行 ────────────────────────────────────────────
print("=" * 72)
print("v80 ローソク足フィルター バックテスト — IS/OOS・v77比較")
print("=" * 72)
print(f"IS:  {IS_START}〜{IS_END}   OOS: {OOS_START}〜{OOS_END}")
print(f"バリアント: {', '.join(VARIANTS.keys())}")
print()

usdjpy_is  = load_data("USDJPY", "15m", "is")
usdjpy_oos = load_data("USDJPY", "15m", "oos")

# {sym: {variant: {period: metrics}}}
all_results = {sym: {} for sym in SYMBOLS}

for sym in SYMBOLS:
    sym_cfg = SYMBOL_CONFIG.get(sym)
    if sym_cfg is None:
        continue
    rm = RiskManager(sym)
    spread_pips = sym_cfg["spread"]
    pip_size    = sym_cfg["pip"]

    print(f"\n{'━'*72}")
    print(f"▶ {sym}  (spread={spread_pips}pips, pip={pip_size})")
    print(f"{'━'*72}")
    print(f"  {'variant':8s}  {'IS n':>5s}  {'IS WR':>7s}  {'IS PF':>7s}  "
          f"{'OOS n':>6s}  {'OOS WR':>7s}  {'OOS PF':>7s}  {'月次+':>5s}  判定")
    print(f"  {'─'*70}")

    for v_name, v_params in VARIANTS.items():
        all_results[sym][v_name] = {}
        is_m = oos_m = None

        for period, p_start, uj_data in [
            ("is",  IS_START,  usdjpy_is),
            ("oos", OOS_START, usdjpy_oos),
        ]:
            d1m, d15m, d4h = load_symbol_data(sym, period)
            if d4h is None or d15m is None or d1m is None:
                continue
            try:
                sigs = generate_signals(
                    d1m, d15m, d4h,
                    spread_pips=spread_pips, rr_ratio=RR_RATIO, pip_size=pip_size,
                    **v_params
                )
            except Exception as e:
                print(f"  {v_name} {period.upper()}: エラー {e}")
                continue

            trades, eq_tl = simulate_trades(sigs, d1m, uj_data, rm, p_start)
            m = calc_metrics(trades, eq_tl, period.upper())
            all_results[sym][v_name][period.upper()] = m
            if period == "is":
                is_m = m
            else:
                oos_m = m

        if is_m is None or oos_m is None:
            continue

        # v77比較（OOS PF変化）
        v77_oos_pf = all_results[sym].get("v77", {}).get("OOS", {}).get("pf", None)
        if v77_oos_pf is not None and v_name != "v77":
            delta = oos_m["pf"] - v77_oos_pf
            ok = "✅" if delta > 0 else "❌"
        else:
            delta = 0.0; ok = "  "

        print(f"  {v_name:8s}  {is_m['n']:5d}  {is_m['wr']:6.1f}%  {is_m['pf']:7.2f}  "
              f"{oos_m['n']:6d}  {oos_m['wr']:6.1f}%  {oos_m['pf']:7.2f}  "
              f"{oos_m['monthly_pos']:4.0f}%  {ok} ({delta:+.2f})")


# ── サマリー: OOS PF比較テーブル ──────────────────────────
print("\n\n" + "=" * 72)
print("【OOS PF サマリー】各バリアント vs v77")
print("=" * 72)

header = f"{'variant':8s} | " + " | ".join(f"{s:8s}" for s in SYMBOLS) + " | avg改善"
print(header)
print("─" * len(header))

for v_name in VARIANTS:
    pfs = []
    v77s = []
    for sym in SYMBOLS:
        oos_pf  = all_results[sym].get(v_name, {}).get("OOS", {}).get("pf")
        v77_pf  = all_results[sym].get("v77", {}).get("OOS", {}).get("pf")
        if oos_pf is not None:
            pfs.append(oos_pf)
            if v77_pf is not None:
                v77s.append(v77_pf)

    cells = []
    n_improved = 0; n_total = 0
    for sym in SYMBOLS:
        oos_pf = all_results[sym].get(v_name, {}).get("OOS", {}).get("pf")
        v77_pf = all_results[sym].get("v77", {}).get("OOS", {}).get("pf")
        if oos_pf is None:
            cells.append(f"{'N/A':8s}")
        else:
            star = ""
            if v_name != "v77" and v77_pf is not None:
                n_total += 1
                if oos_pf > v77_pf:
                    n_improved += 1; star = "✓"
                else:
                    star = " "
            cells.append(f"{oos_pf:6.2f}{star} ")

    avg_delta = ""
    if v_name != "v77" and pfs and v77s:
        delta = np.mean(pfs) - np.mean(v77s)
        sign  = "+" if delta >= 0 else ""
        avg_delta = f" {sign}{delta:.2f} ({n_improved}/{n_total}改善)"

    print(f"{v_name:8s} | " + " | ".join(cells) + f"|{avg_delta}")

# ── IS/OOS 過学習チェック（OOS変化 vs IS変化） ────────────
print("\n\n" + "=" * 72)
print("【過学習チェック】IS改善 vs OOS改善（v77との差）")
print("=" * 72)
print(f"{'variant':8s}  {'銘柄':6s}  IS PF変化  OOS PF変化  乖離   判定")
print("─" * 55)

for v_name in VARIANTS:
    if v_name == "v77":
        continue
    for sym in SYMBOLS:
        r_v77_is  = all_results[sym].get("v77",  {}).get("IS",  {})
        r_v77_oos = all_results[sym].get("v77",  {}).get("OOS", {})
        r_vN_is   = all_results[sym].get(v_name, {}).get("IS",  {})
        r_vN_oos  = all_results[sym].get(v_name, {}).get("OOS", {})
        if not all([r_v77_is, r_v77_oos, r_vN_is, r_vN_oos]):
            continue

        is_imp  = r_vN_is.get("pf", 0)  - r_v77_is.get("pf", 0)
        oos_imp = r_vN_oos.get("pf", 0) - r_v77_oos.get("pf", 0)
        gap = is_imp - oos_imp

        if abs(is_imp) < 0.05 and abs(oos_imp) < 0.05:
            verdict = "→ 効果なし"
        elif is_imp > 0.2 and is_imp > oos_imp * 2:
            verdict = "⚠ 過学習疑い"
        elif oos_imp > 0 and oos_imp >= is_imp * 0.5:
            verdict = "✅ 有効・過学習なし"
        elif oos_imp > 0:
            verdict = "✅ OOS改善"
        else:
            verdict = "❌ OOS悪化"

        print(f"{v_name:8s}  {sym:6s}  IS:{is_imp:+.2f}  OOS:{oos_imp:+.2f}  "
              f"gap:{gap:+.2f}  {verdict}")

# ── CSV出力 ───────────────────────────────────────────────
rows = []
for sym in SYMBOLS:
    for v_name in VARIANTS:
        for period in ["IS", "OOS"]:
            m = all_results[sym].get(v_name, {}).get(period)
            if m is None:
                continue
            rows.append({
                "symbol": sym, "variant": v_name, "period": period,
                "n_trades": m["n"], "win_rate": round(m["wr"], 2),
                "pf": round(m["pf"], 3), "mdd_jpy": round(m["mdd"]),
                "monthly_positive_pct": round(m["monthly_pos"], 1),
                **{k: v for k, v in VARIANTS[v_name].items()},
            })
csv_path = os.path.join(OUT_DIR, "backtest_v80_candle.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n→ {csv_path}")
print("\n✅ v80バックテスト完了")
