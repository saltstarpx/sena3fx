"""
YAGAMI改 (v79) バックテスト — IS/OOS分割・v77比較
==================================================
戦略: yagami_mtf_v79.py のgenerate_signalsを使用

カテゴリ別フィルター:
  FX     (EURUSD/GBPUSD/AUDUSD):    v79BC — ADX≥20 + Streak≥4
  METALS (XAUUSD):                   v79A  — 日足EMA20方向一致
  INDICES (US30/SPX500/NAS100):      両バリアント検証（採用不可想定）

IS期間:  2024-07-01 〜 2025-02-28
OOS期間: 2025-03-01 〜 2026-02-28

ロット: 固定2%リスク  RR: 2.5  半利確: +1R到達で50%決済・SLをBEへ
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG
from strategies.current.yagami_mtf_v79 import generate_signals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── フォント設定 ──────────────────────────────────────────
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

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

# ── 銘柄カテゴリ設定 ──────────────────────────────────────
CATEGORIES = {
    "FX": {
        "symbols": ["EURUSD", "GBPUSD", "AUDUSD"],
        # v77: セッションフィルターのみ（カテゴリ比較の baseline）
        # v79BC: v77_session + ADX≥20 + Streak≥4
        # UTC 7-22 = London+NY主要セッション（経済的根拠ベース）
        "v77_params":  dict(adx_min=0,  streak_min=0, use_1d_trend=False, utc_start=7, utc_end=22),
        "v79_params":  dict(adx_min=20, streak_min=4, use_1d_trend=False, utc_start=7, utc_end=22),
        "v79_label":   "v79BC (ADX≥20+Streak≥4, UTC7-22)",
    },
    "METALS": {
        "symbols": ["XAUUSD"],
        # v77: セッションフィルターなし（XAUUSDはUTC7-22が逆効果のためv77そのまま）
        # v79A: 日足EMA20のみ追加（セッションフィルターなし）
        "v77_params":  dict(adx_min=0, streak_min=0, use_1d_trend=False),
        "v79_params":  dict(adx_min=0, streak_min=0, use_1d_trend=True),
        "v79_label":   "v79A (日足EMA20)",
    },
    "INDICES": {
        "symbols": ["US30", "SPX500", "NAS100"],
        # 指数: 採用不可想定（v77ベースのみ、過去検証でPF<1.5）
        "v77_params":  dict(adx_min=0, streak_min=0, use_1d_trend=False),
        "v79_params":  dict(adx_min=0, streak_min=0, use_1d_trend=False),
        "v79_label":   "v77同等（改善なし）",
    },
}

COLORS = {
    "EURUSD": "#3b82f6", "GBPUSD": "#22c55e", "AUDUSD": "#f97316",
    "XAUUSD": "#eab308",
    "US30":   "#8b5cf6", "SPX500": "#ec4899", "NAS100": "#06b6d4",
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
    """
    IS/OOS分割ファイルを優先。なければohlcフォルダの全期間ファイルをスライス。
    sym_upper: "EURUSD" 等（大文字）
    tf: "1m" / "15m" / "4h"
    period: "is" / "oos"
    """
    sym_lower = sym_upper.lower()

    # IS/OOS分割ファイル（data/ 小文字）
    split_path = os.path.join(DATA_DIR, f"{sym_lower}_{period}_{tf}.csv")
    if os.path.exists(split_path):
        return load_csv(split_path)

    # フォールバック: ohlcフォルダ（大文字）をスライス
    ohlc_path = os.path.join(DATA_DIR, "ohlc", f"{sym_upper}_{tf}.csv")
    if os.path.exists(ohlc_path):
        df = load_csv(ohlc_path)
        if df is not None:
            start = IS_START if period == "is" else OOS_START
            end   = IS_END   if period == "is" else OOS_END
            return df[(df.index >= start) & (df.index <= end)].copy()

    return None


def load_symbol_data(sym, period):
    """
    1m / 15m / 4h をまとめてロード。
    1m がない場合は 15m で代用。
    Returns: (data_1m, data_15m, data_4h, entry_tf)
    """
    data_4h = load_data(sym, "4h", period)

    # 15m（1H足リサンプル元）
    data_15m = load_data(sym, "15m", period)

    # エントリー用: 1m 優先、なければ 15m
    data_1m = load_data(sym, "1m", period)
    entry_tf = "1m"
    if data_1m is None or len(data_1m) == 0:
        data_1m = data_15m
        entry_tf = "15m"

    return data_1m, data_15m, data_4h, entry_tf


# ── トレードシミュレーション ──────────────────────────────
def simulate_trades(signals, data_intra, usdjpy_intra, rm, period_start):
    """
    半利確あり（+1R到達で50%決済・SLをBEへ）シミュレーション
    signals: generate_signals() の戻り値リスト
    """
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
        ep         = sig["ep"]
        sl         = sig["sl"]
        tp         = sig["tp"]
        risk       = sig["risk"]
        direction  = sig["dir"]
        entry_time = sig["time"]
        tf         = sig["tf"]

        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times):
            continue

        lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0:
            continue

        # 半利確ターゲット（+1R = RR_RATIO の何割）
        half_tp = (ep + (tp - ep) * (HALF_R / RR_RATIO)) if direction == 1 \
             else (ep - (ep - tp) * (HALF_R / RR_RATIO))

        half_done  = False
        sl_current = sl
        result     = None
        exit_idx   = None

        for i in range(start_idx, len(m1_times)):
            h = m1_highs[i]
            lo = m1_lows[i]
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
            result   = "BE" if half_done else "OPEN"
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
            if half_done:
                pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                        usdjpy_rate=usdjpy_init, ref_price=ep) +
                       rm.calc_pnl_jpy(direction, ep, tp, lot * 0.5,
                                        usdjpy_rate=usdjpy_at_exit, ref_price=ep))
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, tp, lot,
                                       usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            if half_done:
                pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                       usdjpy_rate=usdjpy_init, ref_price=ep)
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, sl_current, lot,
                                       usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:  # BE
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                    usdjpy_rate=usdjpy_init, ref_price=ep)

        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({
            "entry_time": entry_time,
            "exit_time":  exit_time,
            "result":     result,
            "pnl":        pnl,
            "equity":     equity,
            "tf":         tf,
            "dir":        direction,
        })

    return trades, eq_timeline


# ── 統計計算 ──────────────────────────────────────────────
def calc_metrics(trades, eq_timeline, period_label):
    if not trades:
        return {
            "period": period_label, "n": 0, "wr": 0.0, "pf": 0.0,
            "mdd": 0.0, "monthly_pos": 0.0,
            "total_profit": 0, "final_equity": INIT_CASH,
            "monthly_pnl": pd.Series(dtype=float),
        }

    df = pd.DataFrame(trades)
    n  = len(df)
    wr = (df["result"] == "TP").mean() * 100

    gp = df[df["pnl"] > 0]["pnl"].sum()
    gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")

    # MDD
    eq_df   = pd.DataFrame(eq_timeline, columns=["time", "equity"]).sort_values("time")
    eq_vals = eq_df["equity"].values
    peak    = np.maximum.accumulate(eq_vals)
    dd      = peak - eq_vals
    mdd     = dd.max()

    # 月次黒字率
    df["exit_month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    monthly_pnl = df.groupby("exit_month")["pnl"].sum()
    monthly_pos = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100

    return {
        "period":       period_label,
        "n":            n,
        "wr":           wr,
        "pf":           pf,
        "mdd":          mdd,
        "monthly_pos":  monthly_pos,
        "total_profit": eq_df["equity"].iloc[-1] - INIT_CASH,
        "final_equity": eq_df["equity"].iloc[-1],
        "monthly_pnl":  monthly_pnl,
    }


# ── メイン実行 ────────────────────────────────────────────
print("=" * 70)
print("YAGAMI改 (v79) バックテスト — カテゴリ別フィルター・v77比較")
print("=" * 70)
print(f"IS期間:  {IS_START} 〜 {IS_END}")
print(f"OOS期間: {OOS_START} 〜 {OOS_END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%/trade  RR: {RR_RATIO}")
print()

# USDJPY（円換算用）
usdjpy_is  = load_data("USDJPY", "15m", "is")
usdjpy_oos = load_data("USDJPY", "15m", "oos")

# 結果格納
all_results = {}  # {sym: {variant: {period: metrics}}}

for cat_name, cat_cfg in CATEGORIES.items():
    print(f"\n{'━'*70}")
    print(f"【カテゴリ: {cat_name}】  {cat_cfg['v79_label']}")
    print(f"{'━'*70}")

    for sym in cat_cfg["symbols"]:
        print(f"\n  ▶ {sym}")
        all_results[sym] = {}

        # RiskManager 取得
        sym_cfg = SYMBOL_CONFIG.get(sym)
        if sym_cfg is None:
            print(f"    ⚠ SYMBOL_CONFIG に {sym} が未登録。スキップ。")
            continue
        rm = RiskManager(sym)

        spread_pips = sym_cfg["spread"]
        pip_size    = sym_cfg["pip"]

        for variant, v_params, v_label in [
            ("v77", cat_cfg["v77_params"], "v77（ベースライン）"),
            ("v79", cat_cfg["v79_params"], cat_cfg["v79_label"]),
        ]:
            all_results[sym][variant] = {}
            print(f"\n    [{variant}] {v_label}")

            for period, p_start, p_end, uj_data in [
                ("IS",  IS_START,  IS_END,  usdjpy_is),
                ("OOS", OOS_START, OOS_END, usdjpy_oos),
            ]:
                data_1m, data_15m, data_4h, entry_tf = load_symbol_data(sym, period.lower())

                if data_4h is None or len(data_4h) == 0:
                    print(f"      {period}: データなし（4H）")
                    continue
                if data_15m is None or len(data_15m) == 0:
                    print(f"      {period}: データなし（15m）")
                    continue
                if data_1m is None or len(data_1m) == 0:
                    print(f"      {period}: データなし（1m/15m）")
                    continue

                # シグナル生成
                try:
                    sigs = generate_signals(
                        data_1m, data_15m, data_4h,
                        spread_pips=spread_pips,
                        rr_ratio=RR_RATIO,
                        pip_size=pip_size,
                        **v_params
                    )
                except Exception as e:
                    print(f"      {period}: generate_signals エラー: {e}")
                    continue

                # シミュレーション
                trades, eq_tl = simulate_trades(sigs, data_1m, uj_data, rm, p_start)

                # 指標計算
                metrics = calc_metrics(trades, eq_tl, period)
                all_results[sym][variant][period] = {
                    "trades": trades, "eq_tl": eq_tl, "metrics": metrics
                }

                m = metrics
                print(f"      {period}: n={m['n']:3d}  WR={m['wr']:.1f}%  "
                      f"PF={m['pf']:.2f}  MDD={m['mdd']/1000:.1f}k円  "
                      f"月次+:{m['monthly_pos']:.0f}%  "
                      f"総損益:{m['total_profit']/1000:+.0f}k円")

# ── サマリーテーブル ────────────────────────────────────
print("\n\n" + "=" * 70)
print("【サマリー比較】 v77 vs v79 — OOS期間")
print("=" * 70)

header = f"{'銘柄':8s}  {'カテゴリ':8s}  {'v77 PF':>8s}  {'v79 PF':>8s}  {'変化':>8s}  {'v77 WR':>7s}  {'v79 WR':>7s}  判定"
print(header)
print("─" * len(header))

cat_pf_v77 = {}  # {cat: [pf]}
cat_pf_v79 = {}

for cat_name, cat_cfg in CATEGORIES.items():
    cat_pf_v77[cat_name] = []
    cat_pf_v79[cat_name] = []

    for sym in cat_cfg["symbols"]:
        if sym not in all_results:
            continue
        r77 = all_results[sym].get("v77", {}).get("OOS", {}).get("metrics")
        r79 = all_results[sym].get("v79", {}).get("OOS", {}).get("metrics")
        if r77 is None or r79 is None:
            continue

        pf77 = r77["pf"]
        pf79 = r79["pf"]
        wr77 = r77["wr"]
        wr79 = r79["wr"]
        change = pf79 - pf77
        sign   = "+" if change >= 0 else ""
        ok     = "✅" if pf79 >= pf77 else "❌"

        print(f"{sym:8s}  {cat_name:8s}  {pf77:8.2f}  {pf79:8.2f}  "
              f"{sign}{change:7.2f}  {wr77:6.1f}%  {wr79:6.1f}%  {ok}")

        cat_pf_v77[cat_name].append(pf77)
        cat_pf_v79[cat_name].append(pf79)

print("─" * len(header))
for cat_name in CATEGORIES:
    if not cat_pf_v77[cat_name]:
        continue
    avg77 = np.mean(cat_pf_v77[cat_name])
    avg79 = np.mean(cat_pf_v79[cat_name])
    change = avg79 - avg77
    sign   = "+" if change >= 0 else ""
    n_improved = sum(1 for a, b in zip(cat_pf_v77[cat_name], cat_pf_v79[cat_name]) if b >= a)
    n_total    = len(cat_pf_v77[cat_name])
    ok = "✅" if n_improved > n_total / 2 else "❌"
    print(f"{'avg':8s}  {cat_name:8s}  {avg77:8.2f}  {avg79:8.2f}  "
          f"{sign}{change:7.2f}  {'':7s}  {'':7s}  {ok} ({n_improved}/{n_total}改善)")

# ── IS/OOS過学習チェック ──────────────────────────────────
print("\n\n" + "=" * 70)
print("【過学習チェック】 IS改善 vs OOS改善（v77→v79）")
print("=" * 70)
print(f"{'銘柄':8s}  {'IS改善':>9s}  {'OOS改善':>9s}  {'乖離':>9s}  判定")
print("─" * 50)
for cat_name, cat_cfg in CATEGORIES.items():
    for sym in cat_cfg["symbols"]:
        if sym not in all_results:
            continue
        r77_is  = all_results[sym].get("v77", {}).get("IS",  {}).get("metrics")
        r79_is  = all_results[sym].get("v79", {}).get("IS",  {}).get("metrics")
        r77_oos = all_results[sym].get("v77", {}).get("OOS", {}).get("metrics")
        r79_oos = all_results[sym].get("v79", {}).get("OOS", {}).get("metrics")
        if not all([r77_is, r79_is, r77_oos, r79_oos]):
            continue

        is_imp  = r79_is["pf"]  - r77_is["pf"]
        oos_imp = r79_oos["pf"] - r77_oos["pf"]
        gap     = is_imp - oos_imp

        # 過学習判定: IS改善がOOS改善の2倍以上かつIS改善が有意（>0.2）
        if is_imp > 0.2 and is_imp > oos_imp * 2:
            verdict = "⚠ 過学習疑い"
        elif oos_imp >= is_imp * 0.7 or oos_imp >= 0:
            verdict = "✅ 過学習なし"
        else:
            verdict = "⚡ 要確認"

        s77_is  = f"{r77_is['pf']:.2f}"
        s79_is  = f"{r79_is['pf']:.2f}"
        sign_is = "+" if is_imp >= 0 else ""
        sign_oo = "+" if oos_imp >= 0 else ""
        print(f"{sym:8s}  IS:{sign_is}{is_imp:+.2f}({s77_is}→{s79_is})  "
              f"OOS:{sign_oo}{oos_imp:+.2f}  gap:{gap:+.2f}  {verdict}")

# ── エクイティカーブ描画 ──────────────────────────────────
print("\n\nエクイティカーブを生成中...")

def plot_equity_curves():
    # FX + METALS のみ描画（指数は採用不可のため省略）
    plot_syms = ["EURUSD", "GBPUSD", "AUDUSD", "XAUUSD"]
    plot_syms = [s for s in plot_syms if s in all_results]

    n_sym = len(plot_syms)
    if n_sym == 0:
        return

    fig, axes = plt.subplots(n_sym, 2, figsize=(16, 4 * n_sym),
                              squeeze=False, facecolor="#1a1a2e")
    fig.suptitle("YAGAMI改 (v79) バックテスト — v77 vs v79 エクイティカーブ",
                 fontsize=15, color="white", y=1.01)

    for row, sym in enumerate(plot_syms):
        cat_name = next(c for c, cfg in CATEGORIES.items() if sym in cfg["symbols"])
        cfg = CATEGORIES[cat_name]
        color = COLORS.get(sym, "#ffffff")

        for col, (period, p_label) in enumerate([("IS", "IS"), ("OOS", "OOS")]):
            ax = axes[row][col]
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

            plotted = False
            for variant, v_label, lw, ls in [
                ("v77", "v77 baseline", 1.5, "--"),
                ("v79", cfg["v79_label"], 2.0, "-"),
            ]:
                r = all_results.get(sym, {}).get(variant, {}).get(period)
                if r is None:
                    continue
                eq_tl = r["eq_tl"]
                if not eq_tl:
                    continue
                times  = [t for t, _ in eq_tl]
                equities = [e for _, e in eq_tl]
                label_str = f"{v_label}  PF={r['metrics']['pf']:.2f}"
                alpha = 0.5 if variant == "v77" else 1.0
                ax.plot(times, [e / 1e4 for e in equities],
                        label=label_str, linewidth=lw, linestyle=ls,
                        color=color if variant == "v79" else "#888888",
                        alpha=alpha)
                plotted = True

            ax.axhline(INIT_CASH / 1e4, color="#555", linestyle=":", linewidth=0.8)
            ax.set_title(f"{sym} [{p_label}]", color="white", fontsize=11)
            ax.set_ylabel("資産 (万円)", color="white", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")
            if plotted:
                ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                          framealpha=0.8, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "backtest_yagami_kai.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  → {out_path}")


plot_equity_curves()

# ── CSV出力 ───────────────────────────────────────────────
rows = []
for cat_name, cat_cfg in CATEGORIES.items():
    for sym in cat_cfg["symbols"]:
        if sym not in all_results:
            continue
        for variant in ["v77", "v79"]:
            for period in ["IS", "OOS"]:
                r = all_results[sym].get(variant, {}).get(period)
                if r is None:
                    continue
                m = r["metrics"]
                rows.append({
                    "symbol":    sym,
                    "category":  cat_name,
                    "variant":   variant,
                    "period":    period,
                    "n_trades":  m["n"],
                    "win_rate":  round(m["wr"], 2),
                    "pf":        round(m["pf"], 3),
                    "mdd_jpy":   round(m["mdd"]),
                    "monthly_positive_pct": round(m["monthly_pos"], 1),
                    "total_profit_jpy": round(m["total_profit"]),
                })

csv_path = os.path.join(OUT_DIR, "backtest_yagami_kai.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  → {csv_path}")

print("\n✅ バックテスト完了")
