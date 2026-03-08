"""
backtest_nzdusd_1y.py
=====================
NZDUSD 1年間バックテスト（OOS期間: 2025-03-03 〜 2026-02-27）
- 初期資金: 100万円
- リスク: 総資産の2%（risk_manager.py による全通貨ペア対応）
- モード: 1H（本番推奨）
- エントリー: 足確定後2分以内の最初の1分足始値
- 半利確: +1R で半分決済、残りはBEストップ → +2.5R TP
- 同一バー順序: SL優先（保守的アプローチ）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager

# ── 設定 ─────────────────────────────────────────────────
SYMBOL     = "NZDUSD"
PERIOD_START = "2025-03-03"
PERIOD_END   = "2026-02-27"
INIT_CASH  = 1_000_000   # 100万円
RISK_PCT   = 0.02         # 2%
RR_RATIO   = 2.5
HALF_R     = 1.0
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 日本語フォント設定
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break

# ── ユーティリティ ────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def slice_period(df, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)]

def calculate_atr(df, period=14):
    hl  = df["high"] - df["low"]
    hc  = abs(df["high"] - df["close"].shift())
    lc  = abs(df["low"]  - df["close"].shift())
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

KLOW_THRESHOLD = 0.0015

def check_kmid_klow(prev_4h_bar, direction):
    o, h, l, c = prev_4h_bar["open"], prev_4h_bar["high"], prev_4h_bar["low"], prev_4h_bar["close"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THRESHOLD
    return kmid_ok and klow_ok

# ── シグナル生成（1Hモード: v77本体と同一ロジック） ────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread = spread_pips * pip_size

    data_4h = data_4h.copy()
    data_4h["atr"]   = calculate_atr(data_4h, 14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_15m.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    data_1h["atr"] = calculate_atr(data_1h, 14)

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1   = data_1h.iloc[i - 1]
        h1_prev2   = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]
        atr_val    = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest["atr"]) or pd.isna(h4_latest["ema20"]):
            continue
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tolerance = atr_val * 0.3

        # ロング: 二番底
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, direction=1):
                    continue
                sl = min(low1, low2) - atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({"time": entry_time, "dir": 1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, direction=-1):
                    continue
                sl = max(high1, high2) + atr_val * 0.15
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({"time": entry_time, "dir": -1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()

# ── シミュレーション（SL優先・保守的アプローチ） ──────────
def simulate(signals, data_1m, init_cash=1_000_000, risk_pct=0.02, half_r=1.0,
             symbol="NZDUSD", usdjpy_1m=None):
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")
    rm     = RiskManager(symbol, risk_pct=risk_pct)
    trades = []
    equity = init_cash

    for _, sig in signals.iterrows():
        direction = sig["dir"]
        ep   = sig["ep"]
        sl   = sig["sl"]
        tp   = sig["tp"]
        risk = sig["risk"]

        usdjpy_rate = rm.get_usdjpy_rate(usdjpy_1m, sig["time"]) if usdjpy_1m is not None else 150.0
        lot_size    = rm.calc_lot(equity, risk, ep, usdjpy_rate=usdjpy_rate)

        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0:
            continue

        half_done = False; be_sl = None; result = None
        exit_price = None; exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == 1:  # ロング
                current_sl = be_sl if half_done else sl

                # ① SL優先
                if bar["low"] <= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                          usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl
                    result  = "win" if pnl > 0 else "loss"
                    break

                # ② TP
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * half_r:
                        equity   += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r,
                                                    lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    equity    += rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                 usdjpy_rate=usdjpy_rate, ref_price=ep)
                    result = "win"
                    break

                # ③ 半利確
                if not half_done and bar["high"] >= ep + risk * half_r:
                    half_done = True; be_sl = ep
                    equity   += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r,
                                                lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)

            else:  # ショート
                current_sl = be_sl if half_done else sl

                # ① SL優先
                if bar["high"] >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                          usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl
                    result  = "win" if pnl > 0 else "loss"
                    break

                # ② TP
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * half_r:
                        equity   += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r,
                                                    lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    equity    += rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                 usdjpy_rate=usdjpy_rate, ref_price=ep)
                    result = "win"
                    break

                # ③ 半利確
                if not half_done and bar["low"] <= ep - risk * half_r:
                    half_done = True; be_sl = ep
                    equity   += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r,
                                                lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)

        if result is not None and exit_price is not None:
            trades.append({
                "entry_time": sig["time"],
                "exit_time":  exit_time,
                "dir":        direction,
                "ep":         ep,
                "exit_price": exit_price,
                "result":     result,
                "equity":     equity,
                "tf":         sig.get("tf", "1h"),
            })

    trades_df = pd.DataFrame(trades)
    eq_series = pd.Series(
        [init_cash] + (trades_df["equity"].tolist() if len(trades_df) > 0 else []),
        name="equity"
    )
    return trades_df, eq_series

# ── 統計計算 ─────────────────────────────────────────────
def calc_stats(trades, eq):
    if len(trades) == 0:
        return {}
    n  = len(trades)
    wr = (trades["result"] == "win").sum() / n
    wins   = trades[trades["result"] == "win"]
    losses = trades[trades["result"] == "loss"]
    avg_win  = wins["equity"].diff().fillna(wins["equity"] - INIT_CASH).mean() if len(wins) > 0 else 0
    avg_loss = losses["equity"].diff().fillna(losses["equity"] - INIT_CASH).mean() if len(losses) > 0 else 0
    pf = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if abs(avg_loss * len(losses)) > 0 else float("inf")

    eq_arr = np.array(eq.tolist())
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak
    mdd    = dd.min()
    ret    = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]
    kelly  = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    # 月次プラス率
    trades2 = trades.copy()
    trades2["exit_time"] = pd.to_datetime(trades2["exit_time"], utc=True)
    trades2["month"]     = trades2["exit_time"].dt.to_period("M")
    monthly         = trades2.groupby("month")["equity"].last()
    monthly_shifted = monthly.shift(1).fillna(INIT_CASH)
    monthly_plus    = (monthly > monthly_shifted).sum()
    monthly_total   = len(monthly)

    return {
        "n":            n,
        "winrate":      wr * 100,
        "pf":           pf,
        "return_pct":   ret * 100,
        "return_abs":   eq_arr[-1] - eq_arr[0],
        "final_equity": eq_arr[-1],
        "mdd_pct":      abs(mdd) * 100,
        "kelly":        kelly,
        "monthly_plus": f"{monthly_plus}/{monthly_total}",
        "avg_win_jpy":  avg_win,
        "avg_loss_jpy": avg_loss,
    }

# ── メイン処理 ────────────────────────────────────────────
print("=" * 70)
print(f"NZDUSD 1年間バックテスト（{PERIOD_START} 〜 {PERIOD_END}）")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%  RR: {RR_RATIO}  半利確: +{HALF_R}R")
print("=" * 70)

rm = RiskManager(SYMBOL, risk_pct=RISK_PCT)
print(f"スプレッド: {rm.spread_pips}pips  タイプ: {rm.quote_type}  pip_size: {rm.pip_size}")

# データ読み込み
d1m  = load_csv(os.path.join(DATA_DIR, "nzdusd_oos_1m.csv"))
d15m = load_csv(os.path.join(DATA_DIR, "nzdusd_oos_15m.csv"))
d4h  = load_csv(os.path.join(DATA_DIR, "nzdusd_oos_4h.csv"))
usdjpy_1m = load_csv(os.path.join(DATA_DIR, "usdjpy_oos_1m.csv"))

for name, df in [("1m", d1m), ("15m", d15m), ("4h", d4h), ("usdjpy_1m", usdjpy_1m)]:
    if df is None:
        print(f"[ERROR] {name} データが見つかりません")
        sys.exit(1)

# 期間スライス
d1m       = slice_period(d1m,       PERIOD_START, PERIOD_END)
d15m      = slice_period(d15m,      PERIOD_START, PERIOD_END)
d4h       = slice_period(d4h,       PERIOD_START, PERIOD_END)
usdjpy_1m = slice_period(usdjpy_1m, PERIOD_START, PERIOD_END)

print(f"1分足: {len(d1m)}行  15分足: {len(d15m)}行  4時間足: {len(d4h)}行")

# シグナル生成
print("\nシグナル生成中...")
sigs = generate_signals_1h(d1m, d15m, d4h, rm.spread_pips, rm.pip_size, rr_ratio=RR_RATIO)
print(f"シグナル数: {len(sigs)}件")

# シミュレーション
print("シミュレーション実行中...")
trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH, risk_pct=RISK_PCT, half_r=HALF_R,
                      symbol=SYMBOL, usdjpy_1m=usdjpy_1m)

# 統計
stats = calc_stats(trades, eq)
print("\n" + "=" * 70)
print("【バックテスト結果】")
print("=" * 70)
print(f"トレード数    : {stats['n']}件")
print(f"勝率          : {stats['winrate']:.1f}%")
print(f"プロフィットF : {stats['pf']:.2f}")
print(f"最終資産      : {stats['final_equity']:,.0f}円")
print(f"純利益        : +{stats['return_abs']:,.0f}円  ({stats['return_pct']:+.1f}%)")
print(f"最大DD        : {stats['mdd_pct']:.1f}%")
print(f"ケリー基準    : {stats['kelly']:.3f}")
print(f"月次プラス率  : {stats['monthly_plus']}")
print("=" * 70)

# ── 可視化 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"NZDUSD 1年間バックテスト（{PERIOD_START} 〜 {PERIOD_END}）\n"
             f"初期資金100万円 / リスク2% / 1Hモード / スプレッド{rm.spread_pips}pips",
             fontsize=13, fontweight="bold")

# 1. 資産曲線
ax1 = axes[0, 0]
eq_arr = np.array(eq.tolist())
ax1.plot(eq_arr, color="#22c55e", linewidth=1.5)
ax1.axhline(INIT_CASH, color="gray", linestyle="--", alpha=0.5, label="初期資金")
ax1.fill_between(range(len(eq_arr)), INIT_CASH, eq_arr,
                 where=eq_arr >= INIT_CASH, alpha=0.15, color="#22c55e")
ax1.fill_between(range(len(eq_arr)), INIT_CASH, eq_arr,
                 where=eq_arr < INIT_CASH, alpha=0.15, color="#ef4444")
ax1.set_title("資産曲線（円）", fontweight="bold")
ax1.set_ylabel("資産（円）")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/10000:.0f}万"))
ax1.legend()
ax1.grid(alpha=0.3)

# 2. ドローダウン
ax2 = axes[0, 1]
peak = np.maximum.accumulate(eq_arr)
dd   = (eq_arr - peak) / peak * 100
ax2.fill_between(range(len(dd)), dd, 0, color="#ef4444", alpha=0.6)
ax2.set_title("ドローダウン（%）", fontweight="bold")
ax2.set_ylabel("DD（%）")
ax2.grid(alpha=0.3)

# 3. 月次損益
ax3 = axes[1, 0]
if len(trades) > 0:
    trades2 = trades.copy()
    trades2["exit_time"] = pd.to_datetime(trades2["exit_time"], utc=True)
    trades2["month"]     = trades2["exit_time"].dt.to_period("M")
    monthly_eq = trades2.groupby("month")["equity"].last()
    monthly_shifted = monthly_eq.shift(1).fillna(INIT_CASH)
    monthly_pnl = monthly_eq - monthly_shifted
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly_pnl.values]
    ax3.bar(range(len(monthly_pnl)), monthly_pnl.values / 10000, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(monthly_pnl)))
    ax3.set_xticklabels([str(p) for p in monthly_pnl.index], rotation=45, fontsize=8)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title("月次損益（万円）", fontweight="bold")
    ax3.set_ylabel("損益（万円）")
    ax3.grid(alpha=0.3, axis="y")

# 4. サマリーテキスト
ax4 = axes[1, 1]
ax4.axis("off")
summary = (
    f"【最終結果サマリー】\n\n"
    f"  初期資金:    ¥{INIT_CASH:>12,.0f}\n"
    f"  最終資産:    ¥{stats['final_equity']:>12,.0f}\n"
    f"  純利益:      ¥{stats['return_abs']:>+12,.0f}\n"
    f"  リターン:    {stats['return_pct']:>+10.1f}%\n\n"
    f"  トレード数:  {stats['n']:>10}件\n"
    f"  勝率:        {stats['winrate']:>10.1f}%\n"
    f"  PF:          {stats['pf']:>10.2f}\n"
    f"  最大DD:      {stats['mdd_pct']:>10.1f}%\n"
    f"  ケリー基準:  {stats['kelly']:>10.3f}\n"
    f"  月次プラス:  {stats['monthly_plus']:>10}\n\n"
    f"  スプレッド:  {rm.spread_pips}pips\n"
    f"  リスク/取引: {RISK_PCT*100:.0f}%（総資産連動）\n"
    f"  半利確:      +{HALF_R}R で半分決済\n"
    f"  SL:          ATR×0.15（過学習なし）"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0fdf4", alpha=0.8))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "nzdusd_1y_backtest.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nチャート保存: {out_path}")

# トレード一覧CSV
if len(trades) > 0:
    csv_path = os.path.join(OUT_DIR, "nzdusd_1y_trades.csv")
    trades.to_csv(csv_path, index=False)
    print(f"トレード一覧: {csv_path}")

print("\n完了")
