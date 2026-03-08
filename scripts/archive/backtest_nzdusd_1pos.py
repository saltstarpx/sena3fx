"""
backtest_nzdusd_1pos.py
=======================
Step1修正版: 同時ポジション制限（1ポジションのみ）を追加
- 前のトレードが終了するまで次のエントリーを禁止
- 勝率集計をTP/SL/BEで正確に分類
- 修正前（862件）vs 修正後（実態）を比較

修正内容:
  ① 同時ポジション制限: open_positionフラグで前のトレード終了まで待機
  ② 勝率集計: win(TP) / loss(SL) / be(BEストップ) の3分類
  ③ 損益計算: BEストップは半利確分+0.5Rのみ計上（損益ゼロ）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.risk_manager import RiskManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────
SYMBOL     = "NZDUSD"
SYM        = "nzdusd"
INIT_CASH  = 1_000_000
RISK_PCT   = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
OOS_START  = "2025-03-03"
OOS_END    = "2026-02-27"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── データ読み込み ─────────────────────────────────────────
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

def calculate_atr(df, period=14):
    hl   = df["high"] - df["low"]
    hc   = abs(df["high"] - df["close"].shift())
    lc   = abs(df["low"]  - df["close"].shift())
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    df["atr"]   = calculate_atr(df)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

KLOW_THR = 0.0015
def check_kmid_klow(bar, direction):
    o = bar["open"]; c = bar["close"]; l = bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THR
    return kmid_ok and klow_ok

# ── シグナル生成（1Hモード） ──────────────────────────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h)

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        t         = h1_times[i]
        h1_prev2  = data_1h.iloc[i - 2]
        h1_prev1  = data_1h.iloc[i - 1]
        h1_curr   = data_1h.iloc[i]
        atr_val   = h1_curr["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = data_4h[data_4h.index <= t]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest["atr"]) or pd.isna(h4_latest["ema20"]):
            continue
        trend     = h4_latest["trend"]
        h4_atr    = h4_latest["atr"]
        tolerance = atr_val * 0.3

        # ロング: 二番底
        if trend == 1:
            low1 = h1_prev2["low"]; low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, 1):
                    continue
                sl = min(low1, low2) - atr_val * 0.15
                m1_win = data_1m[(data_1m.index >= t) &
                                  (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1_win) > 0:
                    eb = m1_win.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw_ep = eb["open"]; ep = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * RR_RATIO
                            signals.append({"time": et, "dir": 1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                            used_times.add(et)

        # ショート: 二番天井
        if trend == -1:
            high1 = h1_prev2["high"]; high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                if not check_kmid_klow(h4_latest, -1):
                    continue
                sl = max(high1, high2) + atr_val * 0.15
                m1_win = data_1m[(data_1m.index >= t) &
                                  (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1_win) > 0:
                    eb = m1_win.iloc[0]; et = eb.name
                    if et not in used_times:
                        raw_ep = eb["open"]; ep = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * RR_RATIO
                            signals.append({"time": et, "dir": -1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

# ── シミュレーション（1ポジション制限 + TP/SL/BE分類） ────
def simulate_1pos(signals, data_1m, init_cash=1_000_000, risk_pct=0.02):
    """
    修正①: 同時ポジション制限
      - open_position_exit_time を記録
      - 次のシグナルのentry_timeがopen_position_exit_time以降でなければスキップ
    修正②: 勝率集計をTP/SL/BEで正確に分類
      - result = "TP" / "SL" / "BE"
    """
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash])

    rm     = RiskManager(SYMBOL, risk_pct=risk_pct)
    trades = []
    equity = init_cash
    open_position_exit_time = None  # 現在保有中のポジションの終了時刻

    for _, sig in signals.iterrows():
        entry_time = sig["time"]

        # ── 修正①: 前のポジションが終了していなければスキップ ──
        if open_position_exit_time is not None:
            if entry_time < open_position_exit_time:
                continue  # まだ保有中 → スキップ

        direction = sig["dir"]
        ep   = sig["ep"]
        sl   = sig["sl"]
        tp   = sig["tp"]
        risk = sig["risk"]

        # ロットサイズ計算（Type B: XXX/USD → USDJPY換算）
        # NZDUSDはType B: lot * price_diff * usdjpy_rate
        usdjpy_rate = 150.0  # 簡易固定値（実際は動的取得）
        # 損切額上限: min(総資産×2%, 50万円)
        risk_amt = min(equity * risk_pct, 500_000)
        lot_size = risk_amt / (risk * usdjpy_rate)

        future = data_1m[data_1m.index > entry_time]
        if len(future) == 0:
            continue

        half_done  = False
        be_sl      = None
        result     = None
        exit_price = None
        exit_time  = None
        half_pnl   = 0.0

        for bar_time, bar in future.iterrows():
            if direction == 1:  # ロング
                current_sl = be_sl if half_done else sl

                # ① SL優先
                if bar["low"] <= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price,
                                           lot_size * remaining,
                                           usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl
                    # BEストップ: half_done=True かつ exit_price≈ep
                    if half_done and abs(exit_price - ep) < 1e-6:
                        result = "BE"
                    else:
                        result = "TP" if pnl > 0 else "SL"
                    break

                # ② TP
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * HALF_R:
                        hp = rm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                              lot_size * 0.5,
                                              usdjpy_rate=usdjpy_rate, ref_price=ep)
                        equity += hp; half_pnl = hp; half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price,
                                           lot_size * remaining,
                                           usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl; result = "TP"
                    break

                # ③ 半利確
                if not half_done and bar["high"] >= ep + risk * HALF_R:
                    hp = rm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                          lot_size * 0.5,
                                          usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += hp; half_pnl = hp
                    half_done = True; be_sl = ep

            else:  # ショート
                current_sl = be_sl if half_done else sl

                # ① SL優先
                if bar["high"] >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price,
                                           lot_size * remaining,
                                           usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl
                    if half_done and abs(exit_price - ep) < 1e-6:
                        result = "BE"
                    else:
                        result = "TP" if pnl > 0 else "SL"
                    break

                # ② TP
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * HALF_R:
                        hp = rm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                              lot_size * 0.5,
                                              usdjpy_rate=usdjpy_rate, ref_price=ep)
                        equity += hp; half_pnl = hp; half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = rm.calc_pnl_jpy(direction, ep, exit_price,
                                           lot_size * remaining,
                                           usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += pnl; result = "TP"
                    break

                # ③ 半利確
                if not half_done and bar["low"] <= ep - risk * HALF_R:
                    hp = rm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                          lot_size * 0.5,
                                          usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity += hp; half_pnl = hp
                    half_done = True; be_sl = ep

        if result is None or exit_time is None:
            continue

        # 次のエントリー可能時刻を更新
        open_position_exit_time = exit_time

        trades.append({
            "entry_time": entry_time,
            "exit_time":  exit_time,
            "dir":        direction,
            "ep":         ep,
            "sl":         sl,
            "tp":         tp,
            "exit_price": exit_price,
            "result":     result,
            "half_done":  half_done,
            "half_pnl":   half_pnl,
            "equity":     equity,
        })

    if not trades:
        return pd.DataFrame(), pd.Series([init_cash])
    df = pd.DataFrame(trades)
    eq = pd.Series([init_cash] + df["equity"].tolist())
    return df, eq

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"NZDUSD 1年間バックテスト [Step1修正版: 1ポジション制限]")
print(f"期間: {OOS_START} 〜 {OOS_END}  初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%")
print("=" * 70)

d1m  = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_1m.csv"))
d15m = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_15m.csv"))
d4h  = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_4h.csv"))

d1m  = slice_period(d1m,  OOS_START, OOS_END)
d15m = slice_period(d15m, OOS_START, OOS_END)
d4h  = slice_period(d4h,  OOS_START, OOS_END)

rm = RiskManager(SYMBOL, risk_pct=RISK_PCT)
print(f"\nシグナル生成中（1Hモード）...")
signals = generate_signals_1h(d1m, d15m, d4h, rm.spread_pips, rm.pip_size)
print(f"生成シグナル数: {len(signals)}件")

print(f"\nシミュレーション実行中（1ポジション制限あり）...")
trades, eq = simulate_1pos(signals, d1m, init_cash=INIT_CASH, risk_pct=RISK_PCT)
print(f"実行トレード数: {len(trades)}件")

# ── 統計 ──────────────────────────────────────────────────
n_tp = (trades["result"] == "TP").sum()
n_sl = (trades["result"] == "SL").sum()
n_be = (trades["result"] == "BE").sum()
n    = len(trades)

wr_strict   = n_tp / (n_tp + n_sl + n_be) if n > 0 else 0  # BE含む
wr_excl_be  = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0  # BE除く

# PF計算（円換算ベース）
eq_arr = eq.values
pnl_arr = np.diff(eq_arr)
gross_win  = pnl_arr[pnl_arr > 0].sum()
gross_loss = abs(pnl_arr[pnl_arr < 0].sum())
pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

peak = np.maximum.accumulate(eq_arr)
dd   = (eq_arr - peak) / peak
mdd  = abs(dd.min()) * 100

ret_pct = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] * 100
ret_abs = eq_arr[-1] - eq_arr[0]

kelly = wr_excl_be - (1 - wr_excl_be) / (pf if pf > 0 else 1e-9)

# 月次集計
trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
trades["month"] = trades["exit_time"].dt.to_period("M")
monthly = trades.groupby("month")["equity"].last()
monthly_shifted = monthly.shift(1).fillna(INIT_CASH)
monthly_plus = (monthly > monthly_shifted).sum()
monthly_total = len(monthly)

print(f"\n{'='*70}")
print(f"【結果サマリー】")
print(f"{'='*70}")
print(f"  総トレード数:        {n}件")
print(f"  TP（利確）:          {n_tp}件")
print(f"  SL（損切り）:        {n_sl}件")
print(f"  BE（BEストップ）:    {n_be}件（実質±0、半利確分は計上済み）")
print(f"")
print(f"  勝率（BE含む）:      {wr_strict*100:.1f}%")
print(f"  勝率（BE除く）:      {wr_excl_be*100:.1f}%  ← 実態に近い")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:    {mdd:.1f}%")
print(f"  純利益:              {ret_abs:+,.0f}円")
print(f"  リターン:            {ret_pct:+.1f}%")
print(f"  ケリー係数:          {kelly:.3f}")
print(f"  月次プラス:          {monthly_plus}/{monthly_total}")
print(f"  最終資産:            {eq_arr[-1]:,.0f}円")
print(f"{'='*70}")

# ── 可視化 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"NZDUSD 1年間バックテスト [Step1修正版: 1ポジション制限]\n"
    f"初期資金100万円 / リスク2% / 1Hモード / スプレッド{rm.spread_pips}pips",
    fontsize=12, fontweight="bold"
)

# 資産曲線
ax = axes[0, 0]
ax.plot(eq_arr / 1e6, color="#3b82f6", linewidth=1.5)
ax.axhline(INIT_CASH / 1e6, color="gray", linestyle="--", alpha=0.5)
ax.set_title("資産曲線", fontweight="bold")
ax.set_ylabel("資産（百万円）")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}M"))
ax.grid(alpha=0.3)
ax.text(0.05, 0.92,
        f"最終: {eq_arr[-1]/1e6:.2f}百万円\n+{ret_pct:.1f}%",
        transform=ax.transAxes, fontsize=10, color="#3b82f6",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ドローダウン
ax = axes[1, 0]
ax.fill_between(range(len(dd)), dd * 100, 0, color="#ef4444", alpha=0.6)
ax.set_title("ドローダウン", fontweight="bold")
ax.set_ylabel("DD（%）")
ax.grid(alpha=0.3)
ax.text(0.05, 0.1, f"最大DD: {mdd:.1f}%",
        transform=ax.transAxes, fontsize=10, color="#ef4444",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 月次損益
ax = axes[0, 1]
monthly_pnl = monthly - monthly_shifted
colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in monthly_pnl.values]
ax.bar(range(len(monthly_pnl)), monthly_pnl.values / 1e4, color=colors, alpha=0.8)
ax.set_xticks(range(len(monthly_pnl)))
ax.set_xticklabels([str(p) for p in monthly_pnl.index], rotation=45, fontsize=8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("月次損益（万円）", fontweight="bold")
ax.set_ylabel("損益（万円）")
ax.grid(alpha=0.3, axis="y")

# TP/SL/BE内訳パイチャート
ax = axes[1, 1]
ax.axis("off")
summary = (
    f"{'':=<38}\n"
    f"  {'項目':<14} {'修正前':>8}  {'修正後':>8}\n"
    f"{'':=<38}\n"
    f"  {'総トレード数':<14} {'862件':>8}  {f'{n}件':>8}\n"
    f"  {'TP（利確）':<14} {'311件':>8}  {f'{n_tp}件':>8}\n"
    f"  {'SL（損切り）':<14} {'237件':>8}  {f'{n_sl}件':>8}\n"
    f"  {'BE（BEスト）':<14} {'314件':>8}  {f'{n_be}件':>8}\n"
    f"{'':=<38}\n"
    f"  {'勝率(BE含む)':<14} {'36.1%':>8}  {f'{wr_strict*100:.1f}%':>8}\n"
    f"  {'勝率(BE除く)':<14} {'56.8%':>8}  {f'{wr_excl_be*100:.1f}%':>8}\n"
    f"  {'PF':<14} {'2.75':>8}  {f'{pf:.2f}':>8}\n"
    f"  {'最大DD':<14} {'8.9%':>8}  {f'{mdd:.1f}%':>8}\n"
    f"  {'純利益':<14} {'1.71億':>8}  {f'{ret_abs/1e4:.0f}万':>8}\n"
    f"  {'月次+':<14} {'12/12':>8}  {f'{monthly_plus}/{monthly_total}':>8}\n"
    f"{'':=<38}\n"
    f"  ※修正前は同時9ポジションを\n"
    f"    保有していた（バグ）\n"
    f"  ※修正後が実態に近い成績"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f9ff", alpha=0.9))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "nzdusd_1pos_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nチャート保存: {out_path}")

# CSVも保存
trades.to_csv(os.path.join(OUT_DIR, "nzdusd_1pos_trades.csv"), index=False)
print("完了。")
