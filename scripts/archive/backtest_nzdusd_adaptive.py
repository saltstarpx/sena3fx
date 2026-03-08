"""
backtest_nzdusd_adaptive.py
===========================
AdaptiveRiskManager版: 資産規模 × DD 連動型リスク逓減
- 資産規模テーブルと DD テーブルの両方を評価し、低い方を採用
- 1ポジション制限（同時保有禁止）
- TP/SL/BE 正確分類
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
from utils.risk_manager import AdaptiveRiskManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────
SYMBOL     = "NZDUSD"
SYM        = "nzdusd"
INIT_CASH  = 1_000_000
BASE_RISK  = 0.02       # 基本リスク率（2%）
RR_RATIO   = 2.5
HALF_R     = 1.0
USDJPY_RATE = 150.0     # 簡易固定値（実際は動的取得）
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
    time_col = "time" if "time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    for c in ["open","high","low","close"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df

def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)]

# # ── シグナル生成（backtest_v77_correct.pyの正しい実装を使用） ──────
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from backtest_v77_correct import generate_signals_1h as _gen_1h, calculate_atr

# ── シミュレーション（AdaptiveRiskManager版） ──────────────
def simulate_adaptive(signals, data_1m, init_cash=1_000_000, base_risk=0.02):
    """
    AdaptiveRiskManager を使ったシミュレーション。
    - 資産規模 × DD 連動でリスク%を自動調整
    - 1ポジション制限
    - TP/SL/BE 正確分類
    """
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash])

    arm    = AdaptiveRiskManager(SYMBOL, base_risk_pct=base_risk)
    trades = []
    equity = init_cash
    arm.update_peak(equity)  # 初期ピーク設定
    open_position_exit_time = None

    for sig in signals:
        entry_time = sig["time"]

        # ── 1ポジション制限 ──
        if open_position_exit_time is not None:
            if entry_time < open_position_exit_time:
                continue

        direction = sig["dir"]
        ep   = sig["ep"]
        sl   = sig["sl"]
        tp   = sig["tp"]
        risk = sig["risk"]

        # ── AdaptiveRiskManager でロットサイズ計算 ──
        lot_size, eff_risk, reason = arm.calc_lot_adaptive(
            equity=equity,
            sl_distance=risk,
            ref_price=ep,
            usdjpy_rate=USDJPY_RATE,
        )

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

                # ① SL優先（保守的アプローチ）
                if bar["low"] <= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot_size * remaining,
                                            usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += pnl
                    if half_done and abs(exit_price - ep) < 1e-6:
                        result = "BE"
                    else:
                        result = "TP" if pnl > 0 else "SL"
                    break

                # ② TP
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                               lot_size * 0.5,
                                               usdjpy_rate=USDJPY_RATE, ref_price=ep)
                        equity += hp; half_pnl = hp; half_done = True
                        arm.update_peak(equity)  # 半利確後もピーク更新
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot_size * remaining,
                                            usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += pnl; result = "TP"
                    break

                # ③ 半利確
                if not half_done and bar["high"] >= ep + risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                           lot_size * 0.5,
                                           usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += hp; half_pnl = hp
                    half_done = True; be_sl = ep
                    arm.update_peak(equity)  # 半利確後もピーク更新

            else:  # ショート
                current_sl = be_sl if half_done else sl

                # ① SL優先
                if bar["high"] >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot_size * remaining,
                                            usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += pnl
                    if half_done and abs(exit_price - ep) < 1e-6:
                        result = "BE"
                    else:
                        result = "TP" if pnl > 0 else "SL"
                    break

                # ② TP
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                               lot_size * 0.5,
                                               usdjpy_rate=USDJPY_RATE, ref_price=ep)
                        equity += hp; half_pnl = hp; half_done = True
                        arm.update_peak(equity)
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot_size * remaining,
                                            usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += pnl; result = "TP"
                    break

                # ③ 半利確
                if not half_done and bar["low"] <= ep - risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                           lot_size * 0.5,
                                           usdjpy_rate=USDJPY_RATE, ref_price=ep)
                    equity += hp; half_pnl = hp
                    half_done = True; be_sl = ep
                    arm.update_peak(equity)

        if result is None or exit_time is None:
            continue

        # 決済後にピーク更新
        arm.update_peak(equity)
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
            "eff_risk":   eff_risk,
            "lot_size":   lot_size,
            "equity":     equity,
            "peak":       arm.peak_equity,
            "dd":         arm.current_dd_rate(equity),
            "reason":     reason,
        })

    if not trades:
        return pd.DataFrame(), pd.Series([init_cash])
    df = pd.DataFrame(trades)
    eq = pd.Series([init_cash] + df["equity"].tolist())
    return df, eq

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"NZDUSD 1年間バックテスト [AdaptiveRiskManager版]")
print(f"期間: {OOS_START} 〜 {OOS_END}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print(f"資産規模 × DD 連動型リスク逓減（低い方を採用）")
print("=" * 70)

d1m  = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_1m.csv"))
d1h  = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_1h.csv"))
d4h  = load_csv(os.path.join(DATA_DIR, f"{SYM}_oos_4h.csv"))

d1m  = slice_period(d1m,  OOS_START, OOS_END)
d1h  = slice_period(d1h,  OOS_START, OOS_END)
d4h  = slice_period(d4h,  OOS_START, OOS_END)

arm_ref = AdaptiveRiskManager(SYMBOL, base_risk_pct=BASE_RISK)
print(f"\nシグナル生成中（1Hモード）...")
signals = _gen_1h(d1m, d1h, d4h, arm_ref.spread_pips, arm_ref.pip_size)
print(f"生成シグナル数: {len(signals)}件")

print(f"\nシミュレーション実行中（AdaptiveRiskManager + 1ポジション制限）...")
trades, eq = simulate_adaptive(signals, d1m, init_cash=INIT_CASH, base_risk=BASE_RISK)
print(f"実行トレード数: {len(trades)}件")

# ── 統計 ──────────────────────────────────────────────────
n_tp = (trades["result"] == "TP").sum()
n_sl = (trades["result"] == "SL").sum()
n_be = (trades["result"] == "BE").sum()
n    = len(trades)

wr_strict  = n_tp / (n_tp + n_sl + n_be) if n > 0 else 0
wr_excl_be = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0

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
kelly   = wr_excl_be - (1 - wr_excl_be) / (pf if pf > 0 else 1e-9)

# 月次集計
trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
trades["month"] = trades["exit_time"].dt.to_period("M")
monthly = trades.groupby("month")["equity"].last()
monthly_shifted = monthly.shift(1).fillna(INIT_CASH)
monthly_plus  = (monthly > monthly_shifted).sum()
monthly_total = len(monthly)

# リスク%の分布
risk_dist = trades["eff_risk"].value_counts().sort_index()

print(f"\n{'='*70}")
print(f"【結果サマリー】")
print(f"{'='*70}")
print(f"  総トレード数:          {n}件")
print(f"  TP（利確）:            {n_tp}件")
print(f"  SL（損切り）:          {n_sl}件")
print(f"  BE（BEストップ）:      {n_be}件")
print(f"")
print(f"  勝率（BE含む）:        {wr_strict*100:.1f}%")
print(f"  勝率（BE除く）:        {wr_excl_be*100:.1f}%  ← 実態に近い")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:      {mdd:.1f}%")
print(f"  純利益:                {ret_abs:+,.0f}円")
print(f"  リターン:              {ret_pct:+.1f}%")
print(f"  ケリー係数:            {kelly:.3f}")
print(f"  月次プラス:            {monthly_plus}/{monthly_total}")
print(f"  最終資産:              {eq_arr[-1]:,.0f}円")
print(f"")
print(f"【リスク%の適用分布】")
for risk_val, cnt in risk_dist.items():
    print(f"  {risk_val*100:.2f}%: {cnt}件 ({cnt/n*100:.1f}%)")
print(f"{'='*70}")

# ── 可視化 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"NZDUSD 1年間バックテスト [AdaptiveRiskManager版]\n"
    f"初期資金100万円 / 基本リスク2% / 資産規模×DD連動逓減 / 1Hモード",
    fontsize=12, fontweight="bold"
)

# 資産曲線
ax = axes[0, 0]
ax.plot(eq_arr / 1e6, color="#3b82f6", linewidth=1.5, label="AdaptiveRM")
ax.axhline(INIT_CASH / 1e6, color="gray", linestyle="--", alpha=0.5)
ax.set_title("資産曲線", fontweight="bold")
ax.set_ylabel("資産（百万円）")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
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
monthly_pnl = monthly.diff().fillna(monthly - INIT_CASH)
colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly_pnl]
ax.bar(range(len(monthly_pnl)), monthly_pnl.values / 1e4, color=colors, alpha=0.8)
ax.set_title("月次損益（万円）", fontweight="bold")
ax.set_ylabel("損益（万円）")
ax.set_xticks(range(len(monthly_pnl)))
ax.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45, fontsize=7)
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(alpha=0.3, axis="y")
ax.text(0.05, 0.92, f"月次プラス: {monthly_plus}/{monthly_total}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# リスク%の推移
ax = axes[1, 1]
ax.plot(trades["eff_risk"].values * 100, color="#f97316", linewidth=0.8, alpha=0.8)
ax.set_title("実効リスク%の推移", fontweight="bold")
ax.set_ylabel("リスク%")
ax.set_ylim(0, BASE_RISK * 100 * 1.1)
ax.axhline(BASE_RISK * 100, color="gray", linestyle="--", alpha=0.5, label=f"基本{BASE_RISK*100:.0f}%")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.text(0.05, 0.92,
        f"2.0%: {(trades['eff_risk']==0.02).sum()}件\n"
        f"1.5%: {(trades['eff_risk']==0.015).sum()}件\n"
        f"1.0%: {(trades['eff_risk']==0.01).sum()}件\n"
        f"0.75%: {(trades['eff_risk']==0.0075).sum()}件",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "nzdusd_adaptive_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nチャート保存: {out_path}")
print("完了。")
