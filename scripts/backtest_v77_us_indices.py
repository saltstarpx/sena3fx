"""
v77ロジック × US30・US500 バックテスト
3モード比較: 4Hベース / 1Hベース / ハイブリッド（4H+1H）

設定:
  初期資金: 1,000,000円
  損切リスク: 総資産の2%
  スプレッド: US30=3.0pt / US500=0.5pt（実測値）
  半利確: +1Rで50%決済 + SLをBEへ移動
  RR比: 2.5
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# ── パス設定 ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 日本語フォント設定
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────
INIT_CASH  = 1_000_000   # 初期資金 100万円
RISK_PCT   = 0.02        # 損切リスク 2%
RR_RATIO   = 2.5         # リスクリワード比
HALF_R     = 1.0         # 半利確トリガー（1R）

IS_START   = "2025-01-01"
IS_END     = "2025-02-28"
OOS_START  = "2025-03-03"
OOS_END    = "2026-02-27"

PAIRS = {
    "US30": {
        "spread_pt": 3.0,    # 実測平均スプレッド（ポイント）
        "pip":       1.0,    # 1ポイント = 1pip相当
        "is_1m":    os.path.join(DATA_DIR, "us30_is_1m.csv"),
        "is_15m":   os.path.join(DATA_DIR, "us30_is_15m.csv"),
        "is_4h":    os.path.join(DATA_DIR, "us30_is_4h.csv"),
        "oos_1m":   os.path.join(DATA_DIR, "us30_oos_1m.csv"),
        "oos_15m":  os.path.join(DATA_DIR, "us30_oos_15m.csv"),
        "oos_4h":   os.path.join(DATA_DIR, "us30_oos_4h.csv"),
        "color":    "#f59e0b",
    },
    "US500": {
        "spread_pt": 0.5,
        "pip":       0.1,
        "is_1m":    os.path.join(DATA_DIR, "spx500_is_1m.csv"),
        "is_15m":   os.path.join(DATA_DIR, "spx500_is_15m.csv"),
        "is_4h":    os.path.join(DATA_DIR, "spx500_is_4h.csv"),
        "oos_1m":   os.path.join(DATA_DIR, "spx500_oos_1m.csv"),
        "oos_15m":  os.path.join(DATA_DIR, "spx500_oos_15m.csv"),
        "oos_4h":   os.path.join(DATA_DIR, "spx500_oos_4h.csv"),
        "color":    "#3b82f6",
    },
}

MODES = ["4h", "1h", "hybrid"]
MODE_LABELS = {"4h": "4Hベース", "1h": "1Hベース", "hybrid": "ハイブリッド"}


# ── データ読み込み ─────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


# ── ATR計算 ───────────────────────────────────────────
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── KMID+KLOWフィルター ────────────────────────────────
def check_kmid_klow(bar, direction: int) -> bool:
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    if direction == 1:
        if c <= o:
            return False
        lower_wick_ratio = (min(o, c) - l) / (o + 1e-10)
        if lower_wick_ratio >= 0.0015:
            return False
    else:
        if c >= o:
            return False
        upper_wick_ratio = (h - max(o, c)) / (o + 1e-10)
        if upper_wick_ratio >= 0.0015:
            return False
    return True


# ── シグナル生成（モード別） ──────────────────────────
def generate_signals_mode(data_1m, data_15m, data_4h, spread_pt, mode="hybrid", rr_ratio=2.5):
    """
    mode:
      "4h"     : 4時間足パターンのみ
      "1h"     : 1時間足パターンのみ（4Hトレンドフィルターは維持）
      "hybrid" : 4H + 1H 両方（既存v77と同じ）
    """
    spread = spread_pt  # インデックス銘柄はポイント単位

    # 4時間足: ATR・EMA・トレンド
    d4h = data_4h.copy()
    d4h["atr"]   = calc_atr(d4h)
    d4h["ema20"] = d4h["close"].ewm(span=20, adjust=False).mean()
    d4h["trend"] = np.where(d4h["close"] > d4h["ema20"], 1, -1)

    # 1時間足: 15分足からリサンプリング
    d1h = pd.DataFrame({
        "open":   data_15m["open"].resample("1h").first(),
        "high":   data_15m["high"].resample("1h").max(),
        "low":    data_15m["low"].resample("1h").min(),
        "close":  data_15m["close"].resample("1h").last(),
        "volume": data_15m["volume"].resample("1h").sum(),
    }).dropna()
    d1h["atr"] = calc_atr(d1h)

    signals = []
    used_times = set()

    # ── 4Hベースのシグナル ──
    if mode in ("4h", "hybrid"):
        h4_times = d4h.index.tolist()
        for i in range(2, len(h4_times)):
            h4_time    = h4_times[i]
            h4_prev1   = d4h.iloc[i - 1]
            h4_prev2   = d4h.iloc[i - 2]
            h4_current = d4h.iloc[i]
            atr_val = h4_current["atr"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue
            trend     = h4_current["trend"]
            tolerance = atr_val * 0.3

            # ロング: 二番底
            if trend == 1:
                low1, low2 = h4_prev2["low"], h4_prev1["low"]
                if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                    if not check_kmid_klow(h4_prev1, 1):
                        continue
                    sl = min(low1, low2) - atr_val * 0.15
                    m1_win = data_1m[
                        (data_1m.index >= h4_time) &
                        (data_1m.index <  h4_time + pd.Timedelta(minutes=2))
                    ]
                    if len(m1_win) > 0:
                        bar = m1_win.iloc[0]
                        et  = bar.name
                        if et not in used_times:
                            raw_ep = bar["open"]
                            ep     = raw_ep + spread
                            risk   = raw_ep - sl
                            if 0 < risk <= atr_val * 3:
                                tp = raw_ep + risk * rr_ratio
                                signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl,
                                                "tp": tp, "risk": risk, "tf": "4h"})
                                used_times.add(et)

            # ショート: 二番天井
            if trend == -1:
                high1, high2 = h4_prev2["high"], h4_prev1["high"]
                if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                    if not check_kmid_klow(h4_prev1, -1):
                        continue
                    sl = max(high1, high2) + atr_val * 0.15
                    m1_win = data_1m[
                        (data_1m.index >= h4_time) &
                        (data_1m.index <  h4_time + pd.Timedelta(minutes=2))
                    ]
                    if len(m1_win) > 0:
                        bar = m1_win.iloc[0]
                        et  = bar.name
                        if et not in used_times:
                            raw_ep = bar["open"]
                            ep     = raw_ep - spread
                            risk   = sl - raw_ep
                            if 0 < risk <= atr_val * 3:
                                tp = raw_ep - risk * rr_ratio
                                signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl,
                                                "tp": tp, "risk": risk, "tf": "4h"})
                                used_times.add(et)

    # ── 1Hベースのシグナル ──
    if mode in ("1h", "hybrid"):
        h1_times = d1h.index.tolist()
        for i in range(2, len(h1_times)):
            h1_time    = h1_times[i]
            h1_prev1   = d1h.iloc[i - 1]
            h1_prev2   = d1h.iloc[i - 2]
            atr_val    = d1h.iloc[i]["atr"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue
            h4_before = d4h[d4h.index <= h1_time]
            if len(h4_before) == 0:
                continue
            h4_latest = h4_before.iloc[-1]
            trend     = h4_latest["trend"]
            h4_atr    = h4_latest["atr"]
            tolerance = atr_val * 0.3

            # ロング: 二番底
            if trend == 1:
                low1, low2 = h1_prev2["low"], h1_prev1["low"]
                if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                    if not check_kmid_klow(h4_latest, 1):
                        continue
                    sl = min(low1, low2) - atr_val * 0.15
                    m1_win = data_1m[
                        (data_1m.index >= h1_time) &
                        (data_1m.index <  h1_time + pd.Timedelta(minutes=2))
                    ]
                    if len(m1_win) > 0:
                        bar = m1_win.iloc[0]
                        et  = bar.name
                        if et not in used_times:
                            raw_ep = bar["open"]
                            ep     = raw_ep + spread
                            risk   = raw_ep - sl
                            if 0 < risk <= h4_atr * 2:
                                tp = raw_ep + risk * rr_ratio
                                signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl,
                                                "tp": tp, "risk": risk, "tf": "1h"})
                                used_times.add(et)

            # ショート: 二番天井
            if trend == -1:
                high1, high2 = h1_prev2["high"], h1_prev1["high"]
                if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                    if not check_kmid_klow(h4_latest, -1):
                        continue
                    sl = max(high1, high2) + atr_val * 0.15
                    m1_win = data_1m[
                        (data_1m.index >= h1_time) &
                        (data_1m.index <  h1_time + pd.Timedelta(minutes=2))
                    ]
                    if len(m1_win) > 0:
                        bar = m1_win.iloc[0]
                        et  = bar.name
                        if et not in used_times:
                            raw_ep = bar["open"]
                            ep     = raw_ep - spread
                            risk   = sl - raw_ep
                            if 0 < risk <= h4_atr * 2:
                                tp = raw_ep - risk * rr_ratio
                                signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl,
                                                "tp": tp, "risk": risk, "tf": "1h"})
                                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── バックテストシミュレーター ─────────────────────────
def simulate(signals, data_1m, init_cash=1_000_000, risk_pct=0.02, half_r=1.0):
    """
    シグナルリストを元に1分足でSL/TP/半利確を判定してトレードをシミュレートする。
    半利確: +1Rで50%決済 + SLをBEへ移動
    """
    trades = []
    cash   = init_cash
    equity_curve = []

    for sig in signals:
        entry_time = sig["time"]
        ep   = sig["ep"]
        sl   = sig["sl"]
        tp   = sig["tp"]
        risk = sig["risk"]
        d    = sig["dir"]

        # ポジションサイズ: 初期資金ベース固定（複利爆発防止）
        # リスク額は常に初期資金の2%固定とする
        risk_amt  = init_cash * risk_pct
        pos_size  = risk_amt / risk if risk > 0 else 0
        if pos_size <= 0:
            continue

        # 半利確ライン
        half_tp_price = ep + d * risk * half_r

        # 1分足でSL/TP判定
        future_bars = data_1m[data_1m.index > entry_time]
        if len(future_bars) == 0:
            continue

        half_done  = False
        be_sl      = sl          # BEに移動後のSL
        exit_price = None
        exit_time  = None
        exit_reason = "timeout"
        half_pnl   = 0.0

        for _, bar in future_bars.iterrows():
            h, l = bar["high"], bar["low"]

            # 半利確チェック（未実施の場合）
            if not half_done:
                triggered = (d == 1 and h >= half_tp_price) or (d == -1 and l <= half_tp_price)
                if triggered:
                    half_done  = True
                    half_pnl   = (half_tp_price - ep) * d * pos_size * 0.5
                    pos_size  *= 0.5   # 残り50%
                    be_sl      = ep    # SLをBEへ移動

            # SL判定（BEに移動済みの場合はBE-SLで判定）
            current_sl = be_sl if half_done else sl
            sl_hit = (d == 1 and l <= current_sl) or (d == -1 and h >= current_sl)
            if sl_hit:
                exit_price  = current_sl
                exit_time   = bar.name
                exit_reason = "sl"
                break

            # TP判定
            tp_hit = (d == 1 and h >= tp) or (d == -1 and l <= tp)
            if tp_hit:
                exit_price  = tp
                exit_time   = bar.name
                exit_reason = "tp"
                break

        if exit_price is None:
            # 期間末に強制決済
            last_bar   = future_bars.iloc[-1]
            exit_price = last_bar["close"]
            exit_time  = last_bar.name
            exit_reason = "timeout"

        # 残りポジションの損益
        remain_pnl = (exit_price - ep) * d * pos_size
        total_pnl  = half_pnl + remain_pnl

        cash += total_pnl
        equity_curve.append(cash)

        trades.append({
            "entry_time":   entry_time,
            "exit_time":    exit_time,
            "dir":          d,
            "ep":           ep,
            "sl":           sl,
            "tp":           tp,
            "exit_price":   exit_price,
            "exit_reason":  exit_reason,
            "pnl":          total_pnl,
            "half_done":    half_done,
            "tf":           sig.get("tf", "?"),
            "cash":         cash,
        })

    return trades, equity_curve


# ── 統計計算 ──────────────────────────────────────────
def calc_stats(trades, init_cash):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    wins  = df[df["pnl"] > 0]
    loses = df[df["pnl"] <= 0]
    wr    = len(wins) / len(df) if len(df) > 0 else 0
    avg_w = wins["pnl"].mean()  if len(wins)  > 0 else 0
    avg_l = loses["pnl"].mean() if len(loses) > 0 else 0
    pf    = (wins["pnl"].sum() / abs(loses["pnl"].sum())) if loses["pnl"].sum() != 0 else float("inf")

    # MDD
    equity = [init_cash] + list(df["cash"])
    peak   = init_cash
    mdd    = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > mdd:
            mdd = dd

    # ケリー基準
    if avg_l != 0 and wr < 1:
        rr_actual = abs(avg_w / avg_l) if avg_l != 0 else 0
        kelly = wr - (1 - wr) / rr_actual if rr_actual > 0 else 0
    else:
        kelly = 0

    final_cash = df["cash"].iloc[-1]
    total_ret  = (final_cash - init_cash) / init_cash * 100

    # 月次勝率
    df["ym"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly_pnl = df.groupby("ym")["pnl"].sum()
    plus_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)

    return {
        "trades":       len(df),
        "win_rate":     wr,
        "pf":           pf,
        "avg_win":      avg_w,
        "avg_loss":     avg_l,
        "total_pnl":    df["pnl"].sum(),
        "total_ret":    total_ret,
        "final_cash":   final_cash,
        "mdd":          mdd,
        "kelly":        kelly,
        "plus_months":  plus_months,
        "total_months": total_months,
        "monthly_pnl":  monthly_pnl,
    }


# ── メイン ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("v77 バックテスト: US30 / US500 × 3モード（IS/OOS）")
    print(f"初期資金: {INIT_CASH:,}円 | リスク: {RISK_PCT*100:.0f}% | RR: {RR_RATIO}")
    print("=" * 70)

    all_results = {}  # {pair: {mode: {is: stats, oos: stats}}}

    for pair, cfg in PAIRS.items():
        all_results[pair] = {}
        spread = cfg["spread_pt"]
        print(f"\n{'='*60}")
        print(f"銘柄: {pair}  スプレッド: {spread}pt")
        print(f"{'='*60}")

        for period_label, start, end, key_1m, key_15m, key_4h in [
            ("IS",  IS_START,  IS_END,  "is_1m",  "is_15m",  "is_4h"),
            ("OOS", OOS_START, OOS_END, "oos_1m", "oos_15m", "oos_4h"),
        ]:
            # データ読み込み
            d1m  = load_csv(cfg[key_1m])
            d15m = load_csv(cfg[key_15m])
            d4h  = load_csv(cfg[key_4h])

            # 期間フィルター
            d1m  = d1m[start:end]
            d15m = d15m[start:end]
            d4h  = d4h[start:end]

            print(f"\n  [{period_label}] {start} 〜 {end}")

            for mode in MODES:
                print(f"    [{mode}] シグナル生成中...", end=" ", flush=True)
                sigs = generate_signals_mode(d1m, d15m, d4h, spread, mode=mode, rr_ratio=RR_RATIO)
                print(f"{len(sigs)}件", end=" → ", flush=True)

                trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH, risk_pct=RISK_PCT, half_r=HALF_R)
                stats = calc_stats(trades, INIT_CASH)

                if pair not in all_results:
                    all_results[pair] = {}
                if mode not in all_results[pair]:
                    all_results[pair][mode] = {}
                all_results[pair][mode][period_label] = {
                    "stats": stats,
                    "trades": trades,
                    "equity": eq,
                }

                if stats:
                    print(f"勝率={stats['win_rate']:.1%} PF={stats['pf']:.2f} "
                          f"リターン={stats['total_ret']:+.1f}% MDD={stats['mdd']:.1%}")
                else:
                    print("データなし")

    # ── 結果テーブル出力 ──
    print("\n" + "=" * 70)
    print("バックテスト結果サマリー")
    print("=" * 70)
    header = f"{'銘柄':<8} {'モード':<10} {'期間':<5} {'件数':>5} {'勝率':>7} {'PF':>6} {'リターン':>9} {'MDD':>7} {'ケリー':>7} {'月+':>5}"
    print(header)
    print("-" * 70)
    for pair in PAIRS:
        for mode in MODES:
            for period in ["IS", "OOS"]:
                s = all_results[pair][mode][period]["stats"]
                if not s:
                    print(f"{pair:<8} {MODE_LABELS[mode]:<10} {period:<5} {'N/A':>5}")
                    continue
                print(f"{pair:<8} {MODE_LABELS[mode]:<10} {period:<5} "
                      f"{s['trades']:>5} {s['win_rate']:>7.1%} {s['pf']:>6.2f} "
                      f"{s['total_ret']:>+8.1f}% {s['mdd']:>6.1%} "
                      f"{s['kelly']:>7.3f} {s['plus_months']}/{s['total_months']}")

    # ── 可視化 ──
    _plot_results(all_results)

    return all_results


def _plot_results(all_results):
    pairs  = list(PAIRS.keys())
    modes  = MODES
    n_pairs = len(pairs)
    n_modes = len(modes)

    mode_colors = {"4h": "#f59e0b", "1h": "#3b82f6", "hybrid": "#10b981"}
    period_ls   = {"IS": "--", "OOS": "-"}

    fig, axes = plt.subplots(n_pairs, n_modes + 1, figsize=(20, 5 * n_pairs))
    fig.patch.set_facecolor("#0f172a")
    plt.suptitle("v77 バックテスト: US30 / US500 × 3モード（IS vs OOS）",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    for pi, pair in enumerate(pairs):
        # 左3列: 各モードのエクイティカーブ
        for mi, mode in enumerate(modes):
            ax = axes[pi][mi]
            ax.set_facecolor("#1e293b")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")

            for period in ["IS", "OOS"]:
                eq = all_results[pair][mode][period]["equity"]
                if not eq:
                    continue
                xs = range(len(eq))
                ax.plot(xs, eq, color=mode_colors[mode],
                        linestyle=period_ls[period], linewidth=1.5,
                        label=f"{period}", alpha=0.9)
                ax.axhline(INIT_CASH, color="#475569", linewidth=0.8, linestyle=":")

            s_is  = all_results[pair][mode]["IS"]["stats"]
            s_oos = all_results[pair][mode]["OOS"]["stats"]
            title = f"{pair} {MODE_LABELS[mode]}"
            if s_is and s_oos:
                title += f"\nIS: WR={s_is['win_rate']:.0%} PF={s_is['pf']:.1f} | OOS: WR={s_oos['win_rate']:.0%} PF={s_oos['pf']:.1f}"
            ax.set_title(title, color="white", fontsize=9, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/10000:.0f}万"))
            ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)

        # 右端列: モード別PF比較棒グラフ
        ax = axes[pi][n_modes]
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        x     = np.arange(n_modes)
        width = 0.35
        for pi2, period in enumerate(["IS", "OOS"]):
            pfs = []
            for mode in modes:
                s = all_results[pair][mode][period]["stats"]
                pfs.append(s["pf"] if s else 0)
            bars = ax.bar(x + pi2 * width, pfs, width,
                          label=period, alpha=0.85,
                          color=["#60a5fa", "#34d399"][pi2],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, pfs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=8)

        ax.axhline(1.0, color="#ef4444", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([MODE_LABELS[m] for m in modes], color="white", fontsize=8)
        ax.set_title(f"{pair} プロフィットファクター比較", color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "v77_us_indices_backtest.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nチャート保存: {out_png}")


if __name__ == "__main__":
    main()
