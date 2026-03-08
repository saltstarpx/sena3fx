"""
v77ロジック × 全12銘柄 バックテスト（1Hベースのみ）
OANDAの公表スプレッドを使用

設定:
  初期資金: 1,000,000円
  損切リスク: 総資産の2%
  モード: 1Hベース（本番採用）
  半利確: +1Rで50%決済 + SLをBEへ移動
  RR比: 2.5

OANDAの公表スプレッド（pips）:
  USDJPY=1.4, EURUSD=1.4, GBPUSD=2.0, AUDUSD=1.4
  USDCAD=2.2, USDCHF=1.8, NZDUSD=1.7
  EURJPY=1.8, GBPJPY=3.1, EURGBP=1.7
  US30=3.0pt, SPX500=0.5pt
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0

IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# OANDAの公表スプレッド（pips）
# FX銘柄: pips単位、指数: ポイント単位
PAIRS = {
    "USDJPY": {"spread": 0.8, "pip": 0.01, "sym": "usdjpy", "color": "#ef4444"},
    "EURUSD": {"spread": 0.8, "pip": 0.0001, "sym": "eurusd", "color": "#f97316"},
    "GBPUSD": {"spread": 1.3, "pip": 0.0001, "sym": "gbpusd", "color": "#eab308"},
    "AUDUSD": {"spread": 1.4, "pip": 0.0001, "sym": "audusd", "color": "#22c55e"},
    "USDCAD": {"spread": 2.2, "pip": 0.0001, "sym": "usdcad", "color": "#14b8a6"},
    "USDCHF": {"spread": 1.8, "pip": 0.0001, "sym": "usdchf", "color": "#3b82f6"},
    "NZDUSD": {"spread": 1.7, "pip": 0.0001, "sym": "nzdusd", "color": "#8b5cf6"},
    "EURJPY": {"spread": 2.0, "pip": 0.01,   "sym": "eurjpy", "color": "#ec4899"},
    "GBPJPY": {"spread": 3.0, "pip": 0.01,   "sym": "gbpjpy", "color": "#f43f5e"},
    "EURGBP": {"spread": 1.7, "pip": 0.0001, "sym": "eurgbp", "color": "#a855f7"},
    "US30":   {"spread": 3.0, "pip": 1.0,    "sym": "us30",   "color": "#f59e0b"},
    "SPX500": {"spread": 0.5, "pip": 0.1,    "sym": "spx500", "color": "#06b6d4"},
}

# ── データ読み込み ─────────────────────────────────────
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
    return df[(df.index >= start) & (df.index <= end)].copy()

# ── シグナル生成（1Hベース） ─────────────────────────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    """
    1Hベース: 15分足をリサンプリングして1H足を生成し、
    4H足でトレンド判定 + 1H足で二番底/二番天井パターン検出
    """
    signals = []

    # 1H足を15分足から生成
    data_1h = pd.DataFrame({
        "open":   data_15m["open"].resample("1h").first(),
        "high":   data_15m["high"].resample("1h").max(),
        "low":    data_15m["low"].resample("1h").min(),
        "close":  data_15m["close"].resample("1h").last(),
        "volume": data_15m["volume"].resample("1h").sum() if "volume" in data_15m.columns else 0,
    }).dropna(subset=["open","close"])

    # 4H足のEMA20
    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()

    # ATR（4H足）
    data_4h["tr"] = np.maximum(
        data_4h["high"] - data_4h["low"],
        np.maximum(
            abs(data_4h["high"] - data_4h["close"].shift(1)),
            abs(data_4h["low"]  - data_4h["close"].shift(1))
        )
    )
    data_4h["atr"] = data_4h["tr"].rolling(14).mean()

    spread = spread_pips * pip_size

    # 1H足の各バーでパターン検出
    h1_bars = data_1h.index
    for i in range(5, len(h1_bars)):
        bar_time = h1_bars[i]

        # 4H足のトレンド判定（直近の確定4H足）
        past_4h = data_4h[data_4h.index < bar_time]
        if len(past_4h) < 21:
            continue
        last_4h = past_4h.iloc[-1]
        trend_up   = last_4h["close"] > last_4h["ema20"]
        trend_down = last_4h["close"] < last_4h["ema20"]

        # ATR
        atr_4h = last_4h["atr"]
        if pd.isna(atr_4h) or atr_4h <= 0:
            continue

        # KMID: 直前4H足の実体方向
        if len(past_4h) >= 2:
            prev_4h = past_4h.iloc[-1]
            kmid_bull = prev_4h["close"] > prev_4h["open"]
            kmid_bear = prev_4h["close"] < prev_4h["open"]
        else:
            continue

        # KLOW: 直前4H足の下ヒゲ比率 < 0.15%
        if prev_4h["open"] > 0:
            lower_wick_ratio = (min(prev_4h["open"], prev_4h["close"]) - prev_4h["low"]) / prev_4h["open"]
        else:
            continue
        klow_ok = lower_wick_ratio < 0.0015

        # 直近1H足5本でパターン検出
        recent = data_1h.iloc[i-5:i]
        if len(recent) < 4:
            continue

        # ロング: 二番底パターン
        if trend_up and kmid_bull and klow_ok:
            lows = recent["low"].values
            min_idx = np.argmin(lows)
            if min_idx > 0 and min_idx < len(lows) - 1:
                second_low_idx = None
                for j in range(min_idx + 1, len(lows)):
                    if abs(lows[j] - lows[min_idx]) <= atr_4h * 0.3:
                        second_low_idx = j
                        break
                if second_low_idx is not None:
                    confirm_bar = data_1h.iloc[i-1]
                    if confirm_bar["close"] > confirm_bar["open"]:  # 陽線確認
                        sl_raw = min(lows[min_idx], lows[second_low_idx]) - atr_4h * 0.15
                        # 1分足でエントリー（次の1分足始値）
                        next_1m = data_1m[data_1m.index > bar_time]
                        if len(next_1m) == 0:
                            continue
                        entry_bar = next_1m.iloc[0]
                        ep = entry_bar["open"] + spread
                        risk = ep - sl_raw
                        if risk <= 0 or risk > atr_4h * 3:
                            continue
                        tp = ep + risk * rr_ratio
                        signals.append({
                            "time": entry_bar.name,
                            "direction": "long",
                            "ep": ep, "sl": sl_raw, "tp": tp,
                            "risk": risk,
                        })

        # ショート: 二番天井パターン
        if trend_down and kmid_bear and klow_ok:
            highs = recent["high"].values
            max_idx = np.argmax(highs)
            if max_idx > 0 and max_idx < len(highs) - 1:
                second_high_idx = None
                for j in range(max_idx + 1, len(highs)):
                    if abs(highs[j] - highs[max_idx]) <= atr_4h * 0.3:
                        second_high_idx = j
                        break
                if second_high_idx is not None:
                    confirm_bar = data_1h.iloc[i-1]
                    if confirm_bar["close"] < confirm_bar["open"]:  # 陰線確認
                        sl_raw = max(highs[max_idx], highs[second_high_idx]) + atr_4h * 0.15
                        next_1m = data_1m[data_1m.index > bar_time]
                        if len(next_1m) == 0:
                            continue
                        entry_bar = next_1m.iloc[0]
                        ep = entry_bar["open"] - spread
                        risk = sl_raw - ep
                        if risk <= 0 or risk > atr_4h * 3:
                            continue
                        tp = ep - risk * rr_ratio
                        signals.append({
                            "time": entry_bar.name,
                            "direction": "short",
                            "ep": ep, "sl": sl_raw, "tp": tp,
                            "risk": risk,
                        })

    return pd.DataFrame(signals)

# ── シミュレーション ─────────────────────────────────────
def simulate(signals, data_1m, init_cash=1_000_000, risk_pct=0.02, half_r=1.0):
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")

    trades = []
    equity = init_cash

    for _, sig in signals.iterrows():
        direction = sig["direction"]
        ep = sig["ep"]
        sl = sig["sl"]
        tp = sig["tp"]
        risk = sig["risk"]

        risk_amt  = init_cash * risk_pct  # 固定ポジションサイズ（初期資金ベース）
        lot_size  = risk_amt / risk if risk > 0 else 0

        # 1分足でSL/TP/半利確を判定
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0:
            continue

        half_done = False
        be_sl = None
        result = None
        exit_price = None
        exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == "long":
                # 半利確チェック
                if not half_done and bar["high"] >= ep + risk * half_r:
                    half_done = True
                    be_sl = ep
                    # 半利確: 50%決済
                    pnl_half = (ep + risk * half_r - ep) * lot_size * 0.5
                    equity += pnl_half

                # SL判定
                current_sl = be_sl if half_done else sl
                if bar["low"] <= current_sl:
                    exit_price = current_sl
                    exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (exit_price - ep) * lot_size * remaining
                    equity += pnl
                    result = "win" if pnl > 0 else "loss"
                    break

                # TP判定
                if bar["high"] >= tp:
                    exit_price = tp
                    exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (exit_price - ep) * lot_size * remaining
                    equity += pnl
                    result = "win"
                    break

            else:  # short
                if not half_done and bar["low"] <= ep - risk * half_r:
                    half_done = True
                    be_sl = ep
                    pnl_half = (ep - (ep - risk * half_r)) * lot_size * 0.5
                    equity += pnl_half

                current_sl = be_sl if half_done else sl
                if bar["high"] >= current_sl:
                    exit_price = current_sl
                    exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (ep - exit_price) * lot_size * remaining
                    equity += pnl
                    result = "win" if pnl > 0 else "loss"
                    break

                if bar["low"] <= tp:
                    exit_price = tp
                    exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (ep - exit_price) * lot_size * remaining
                    equity += pnl
                    result = "win"
                    break

        if result is None:
            continue

        total_pnl = equity - init_cash - sum(t.get("pnl", 0) for t in trades)
        trades.append({
            "entry_time": sig["time"],
            "exit_time": exit_time,
            "direction": direction,
            "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price,
            "result": result,
            "equity": equity,
        })

    if not trades:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")

    df_trades = pd.DataFrame(trades)
    eq_series = pd.Series([init_cash] + df_trades["equity"].tolist(), name="equity")
    return df_trades, eq_series

# ── 統計計算 ─────────────────────────────────────────────
def calc_stats(trades, eq_series, label):
    if len(trades) == 0:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0,
                "return_pct": 0, "mdd_pct": 0, "kelly": 0, "monthly_plus": "N/A"}

    wins  = trades[trades["result"] == "win"]
    loses = trades[trades["result"] == "loss"]
    n = len(trades)
    wr = len(wins) / n if n > 0 else 0

    gross_win  = (wins["exit_price"] - wins["ep"]).abs().sum() if len(wins) > 0 else 0
    gross_loss = (loses["exit_price"] - loses["ep"]).abs().sum() if len(loses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    eq = eq_series.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = dd.min()

    ret = (eq[-1] - eq[0]) / eq[0]

    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    # 月次プラス率
    if "exit_time" in trades.columns and len(trades) > 0:
        trades2 = trades.copy()
        trades2["exit_time"] = pd.to_datetime(trades2["exit_time"], utc=True)
        trades2["month"] = trades2["exit_time"].dt.to_period("M")
        monthly = trades2.groupby("month")["equity"].last()
        monthly_pnl = monthly.diff().fillna(monthly.iloc[0] - INIT_CASH)
        n_plus = (monthly_pnl > 0).sum()
        n_total = len(monthly_pnl)
        monthly_str = f"{n_plus}/{n_total}"
    else:
        monthly_str = "N/A"

    return {
        "label": label,
        "n": n,
        "winrate": wr * 100,
        "pf": pf,
        "return_pct": ret * 100,
        "return_abs": (eq[-1] - eq[0]),
        "mdd_pct": abs(mdd) * 100,
        "kelly": kelly,
        "monthly_plus": monthly_str,
    }

# ── メイン処理 ─────────────────────────────────────────
print("=" * 70)
print("v77（1Hベース）全12銘柄バックテスト")
print(f"IS: {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%  RR: {RR_RATIO}")
print("=" * 70)

all_results = []
eq_curves = {}  # {pair: {"is": eq_series, "oos": eq_series}}

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]
    spread = cfg["spread"]
    pip = cfg["pip"]
    spread_price = spread * pip  # 価格単位に変換

    print(f"\n--- {pair} (スプレッド: {spread}pips) ---")

    # データ読み込み
    d1m_is  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_1m.csv"))
    d15m_is = load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv"))
    d4h_is  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv"))
    d1m_oos  = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_1m.csv"))
    d15m_oos = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv"))
    d4h_oos  = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv"))

    if any(d is None for d in [d1m_is, d15m_is, d4h_is, d1m_oos, d15m_oos, d4h_oos]):
        print(f"  [SKIP] データ不足")
        continue

    # 期間スライス
    d1m_is   = slice_period(d1m_is,   IS_START,  IS_END)
    d15m_is  = slice_period(d15m_is,  IS_START,  IS_END)
    d4h_is   = slice_period(d4h_is,   IS_START,  IS_END)
    d1m_oos  = slice_period(d1m_oos,  OOS_START, OOS_END)
    d15m_oos = slice_period(d15m_oos, OOS_START, OOS_END)
    d4h_oos  = slice_period(d4h_oos,  OOS_START, OOS_END)

    eq_curves[pair] = {}

    for period, d1m, d15m, d4h in [
        ("IS",  d1m_is,  d15m_is,  d4h_is),
        ("OOS", d1m_oos, d15m_oos, d4h_oos),
    ]:
        sigs = generate_signals_1h(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
        trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH, risk_pct=RISK_PCT, half_r=HALF_R)
        stats = calc_stats(trades, eq, f"{pair}_{period}")
        stats["pair"] = pair
        stats["period"] = period
        stats["spread"] = spread
        all_results.append(stats)
        eq_curves[pair][period] = eq

        print(f"  [{period}] {stats['n']}件 | 勝率{stats['winrate']:.1f}% | PF{stats['pf']:.2f} | "
              f"リターン+{stats['return_pct']:.1f}% | MDD{stats['mdd_pct']:.1f}% | "
              f"ケリー{stats['kelly']:.3f} | 月次+{stats['monthly_plus']}")

# ── 結果保存 ─────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(OUT_DIR, "v77_all_pairs_results.csv"), index=False)

# ── 可視化 ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle("v77（1Hベース）全12銘柄バックテスト結果", fontsize=16, fontweight="bold", y=0.98)

pair_list = list(eq_curves.keys())
for idx, pair in enumerate(pair_list):
    ax = axes[idx // 4][idx % 4]
    cfg = PAIRS[pair]
    color = cfg["color"]

    if "IS" in eq_curves[pair]:
        eq_is = eq_curves[pair]["IS"].values
        ax.plot(eq_is / INIT_CASH * 100 - 100, color=color, alpha=0.5, lw=1.5, label="IS")
    if "OOS" in eq_curves[pair]:
        eq_oos = eq_curves[pair]["OOS"].values
        ax.plot(eq_oos / INIT_CASH * 100 - 100, color=color, lw=2, label="OOS")

    # 統計テキスト
    is_stats  = next((r for r in all_results if r["pair"] == pair and r["period"] == "IS"),  None)
    oos_stats = next((r for r in all_results if r["pair"] == pair and r["period"] == "OOS"), None)

    txt = ""
    if is_stats:
        txt += f"IS: {is_stats['n']}件 勝率{is_stats['winrate']:.0f}% PF{is_stats['pf']:.2f}\n"
    if oos_stats:
        txt += f"OOS: {oos_stats['n']}件 勝率{oos_stats['winrate']:.0f}% PF{oos_stats['pf']:.2f}\n"
        txt += f"OOS MDD:{oos_stats['mdd_pct']:.1f}% ケリー:{oos_stats['kelly']:.3f}"

    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_title(f"{pair}  スプレッド:{cfg['spread']}pips", fontsize=10, fontweight="bold")
    ax.set_ylabel("リターン (%)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3)

# 空のサブプロットを非表示
for idx in range(len(pair_list), 12):
    axes[idx // 4][idx % 4].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "v77_all_pairs_backtest.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n\nチャート保存: {out_path}")

# ── サマリーテーブル表示 ─────────────────────────────────
print("\n" + "=" * 90)
print(f"{'銘柄':<8} {'スプレッド':>6} {'期間':>5} {'件数':>5} {'勝率':>7} {'PF':>6} {'リターン':>10} {'MDD':>7} {'ケリー':>7} {'月次+':>7}")
print("-" * 90)
for r in all_results:
    print(f"{r['pair']:<8} {r['spread']:>5.1f}pips {r['period']:>5} {r['n']:>5} "
          f"{r['winrate']:>6.1f}% {r['pf']:>6.2f} {r['return_pct']:>9.1f}% "
          f"{r['mdd_pct']:>6.1f}% {r['kelly']:>7.3f} {r['monthly_plus']:>7}")

print("\n完了。")
