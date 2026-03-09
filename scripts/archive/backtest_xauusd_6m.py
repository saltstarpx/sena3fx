"""
backtest_xauusd_6m.py
=====================
XAUUSD 半年間バックテスト（2025-01-01 〜 2025-06-30）

【設定】
  - ロジック: backtest_multi_fast.py と同一（AND条件 4H+1H KMID/KLOW）
  - スプレッド: 0.90pips（XAUUSD ゼロ口座）
  - pip_size: 0.01（XAUUSD: 1pip = $0.01）
  - quote_type: B（USD建て → USDJPY換算）
  - KLOW_THR: 0.0015（他銘柄と統一）
  - 手数料: ゼロ口座 片道0.2ドル/ロット（資産から直接控除）
  - USDJPY固定レート: 150円（バックテスト簡易設定）

【データ】
  1分足: xauusd_1m.csv（2025-01-01〜2025-06-30）
  15分足: xauusd_15m.csv（1H足生成用）
  4H足: xauusd_4h.csv（トレンド判定用）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.risk_manager     import AdaptiveRiskManager
from utils.position_manager import PositionManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────
SYMBOL     = "XAUUSD"
INIT_CASH  = 1_000_000
BASE_RISK  = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
START_DATE = "2025-01-01"
END_DATE   = "2025-06-30"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

KLOW_THR    = 0.0015
USDJPY_RATE = 150.0   # 固定レート（簡易）

# ── データ読み込み ─────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THR
    return kmid_ok and klow_ok

# ── シグナル生成（4H AND 1H KMID/KLOW） ──────────────────
def generate_signals(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = data_4h[data_4h.index <= h1_ct]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)) or pd.isna(h4_latest.get("ema20", np.nan)):
            continue

        trend     = h4_latest["trend"]
        h4_atr    = h4_latest["atr"]
        tolerance = atr_val * 0.3

        for direction, low_or_high in [(1, "low"), (-1, "high")]:
            if trend != direction:
                continue
            v1 = h1_prev2[low_or_high]
            v2 = h1_prev1[low_or_high]
            if abs(v1 - v2) > tolerance:
                continue
            # 反転確認足（4H足 AND 1H足 の両方でKMID/KLOWを満たす場合のみ通過）
            if direction == 1 and h1_prev1["close"] <= h1_prev1["open"]:
                continue
            if direction == -1 and h1_prev1["close"] >= h1_prev1["open"]:
                continue
            if not check_kmid_klow(h4_latest, direction):   # 4H足フィルター（品質）
                continue
            if not check_kmid_klow(h1_prev1, direction):   # 1H足フィルター（タイミング）
                continue

            entry_window_end = h1_ct + pd.Timedelta(minutes=2)
            m1_window = data_1m[
                (data_1m.index >= h1_ct) & (data_1m.index < entry_window_end)
            ]
            if len(m1_window) == 0:
                continue
            entry_bar  = m1_window.iloc[0]
            entry_time = entry_bar.name
            if entry_time in used_times:
                continue

            raw_ep = entry_bar["open"]
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                ep   = raw_ep + spread
                risk = raw_ep - sl
                tp   = raw_ep + risk * RR_RATIO
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                ep   = raw_ep - spread
                risk = sl - raw_ep
                tp   = raw_ep - risk * RR_RATIO

            if 0 < risk <= h4_atr * 2:
                signals.append({
                    "time": entry_time, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ─────────────────────────────────────
def simulate(signals, data_1m, init_cash, base_risk):
    arm    = AdaptiveRiskManager(SYMBOL, base_risk_pct=base_risk)
    equity = init_cash
    trades = []

    times  = data_1m.index.values
    highs  = data_1m["high"].values
    lows   = data_1m["low"].values

    for sig in signals:
        entry_time = sig["time"]
        direction  = sig["dir"]
        ep         = sig["ep"]
        sl         = sig["sl"]
        tp         = sig["tp"]
        risk       = sig["risk"]

        idx_arr = np.searchsorted(times, np.datetime64(entry_time))
        if idx_arr >= len(times):
            continue

        lot, eff_risk, reason = arm.calc_lot_adaptive(
            equity=equity, sl_distance=risk,
            ref_price=ep, usdjpy_rate=USDJPY_RATE,
        )
        if lot <= 0:
            continue

        # エントリー手数料を資産から控除
        entry_commission = arm.calc_commission_jpy(lot, USDJPY_RATE)
        equity -= entry_commission

        half_done  = False
        be_sl      = None
        half_pnl   = 0.0
        result     = None
        exit_time  = None
        exit_price = None

        for j in range(idx_arr + 1, len(times)):
            bar_high = highs[j]
            bar_low  = lows[j]
            bar_time = times[j]
            current_sl = be_sl if half_done else sl

            if direction == 1:  # ロング
                if bar_low <= current_sl:
                    exit_price = current_sl
                    exit_time  = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * remaining, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else ("TP" if pnl > 0 else "SL")
                    break
                if bar_high >= tp:
                    if not half_done and bar_high >= ep + risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                               lot * 0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp
                        equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                        half_done = True
                        be_sl     = ep
                    exit_price = tp
                    exit_time  = bar_time
                    pnl = arm.calc_pnl_jpy(direction, ep, tp,
                                            lot * 0.5, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    result = "TP"
                    break
                if not half_done and bar_high >= ep + risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                           lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    half_done = True
                    be_sl     = ep

            else:  # ショート
                if bar_high >= current_sl:
                    exit_price = current_sl
                    exit_time  = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * remaining, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else ("TP" if pnl > 0 else "SL")
                    break
                if bar_low <= tp:
                    if not half_done and bar_low <= ep - risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                               lot * 0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp
                        equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                        half_done = True
                        be_sl     = ep
                    exit_price = tp
                    exit_time  = bar_time
                    pnl = arm.calc_pnl_jpy(direction, ep, tp,
                                            lot * 0.5, USDJPY_RATE, ep)
                    equity += pnl
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    result = "TP"
                    break
                if not half_done and bar_low <= ep - risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                           lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    equity -= arm.calc_commission_jpy(lot * 0.5, USDJPY_RATE)
                    half_done = True
                    be_sl     = ep

        if result is None:
            continue

        pnl_delta = equity - (trades[-1]["equity_after"] if trades else init_cash)
        trades.append({
            "symbol":       SYMBOL,
            "entry_time":   entry_time,
            "exit_time":    exit_time,
            "dir":          direction,
            "ep":           ep,
            "sl":           sl,
            "tp":           tp,
            "exit_price":   exit_price,
            "result":       result,
            "lot":          lot,
            "eff_risk":     eff_risk,
            "half_pnl":     half_pnl,
            "equity_after": equity,
            "pnl_delta":    pnl_delta,
        })

    return trades

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"XAUUSD 半年間バックテスト [AND条件: 4H+1H KMID/KLOW]")
print(f"期間: {START_DATE} 〜 {END_DATE}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print("=" * 70)

# データ読み込み
data_1m_all  = load_csv(os.path.join(DATA_DIR, "xauusd_1m.csv"))
data_15m_all = load_csv(os.path.join(DATA_DIR, "xauusd_15m.csv"))
data_4h_all  = load_csv(os.path.join(DATA_DIR, "xauusd_4h.csv"))

if data_1m_all is None or data_15m_all is None or data_4h_all is None:
    print("エラー: データファイルが見つかりません")
    sys.exit(1)

# 期間スライス（4H足はEMA計算のためウォームアップ期間を含める）
data_1m  = slice_period(data_1m_all,  START_DATE, END_DATE)
data_15m = slice_period(data_15m_all, START_DATE, END_DATE)
data_4h  = slice_period(data_4h_all,  START_DATE, END_DATE)

cfg      = __import__("utils.risk_manager", fromlist=["SYMBOL_CONFIG"]).SYMBOL_CONFIG[SYMBOL]
spread_p = cfg["spread"]
pip_size = cfg["pip"]

print(f"データ行数: 1m={len(data_1m)}, 15m={len(data_15m)}, 4h={len(data_4h)}")
print(f"スプレッド: {spread_p}pips  pip_size: {pip_size}")

# シグナル生成
print("\nシグナル生成中...")
signals = generate_signals(data_1m, data_15m, data_4h, spread_p, pip_size)
print(f"シグナル数: {len(signals)}件")

# シミュレーション
print("シミュレーション実行中...")
trades = simulate(signals, data_1m, INIT_CASH, BASE_RISK)
print(f"完了: {len(trades)}件")

if not trades:
    print("トレードなし")
    sys.exit(0)

# ── 集計 ──────────────────────────────────────────────────
df_trades = pd.DataFrame(trades)
df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"], utc=True)
df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"], utc=True)
df_trades["month"] = df_trades["exit_time"].dt.to_period("M")

n_total = len(df_trades)
n_tp    = (df_trades["result"] == "TP").sum()
n_sl    = (df_trades["result"] == "SL").sum()
n_be    = (df_trades["result"] == "BE").sum()
win_rate = n_tp / (n_tp + n_sl) * 100 if (n_tp + n_sl) > 0 else 0

# PF計算
tp_trades = df_trades[df_trades["result"] == "TP"]
sl_trades = df_trades[df_trades["result"] == "SL"]
gross_profit = tp_trades["pnl_delta"].sum() if len(tp_trades) > 0 else 0
gross_loss   = abs(sl_trades["pnl_delta"].sum()) if len(sl_trades) > 0 else 1
pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

# 資産曲線
eq_curve = [INIT_CASH] + df_trades["equity_after"].tolist()
eq_arr   = np.array(eq_curve)
peak     = np.maximum.accumulate(eq_arr)
dd_arr   = (eq_arr - peak) / peak * 100
max_dd   = dd_arr.min()

final_equity = df_trades["equity_after"].iloc[-1]
ret_pct      = (final_equity - INIT_CASH) / INIT_CASH * 100

# 月次損益
monthly_pnl = df_trades.groupby("month")["pnl_delta"].sum()
n_pos_months = (monthly_pnl > 0).sum()

# ケリー係数
if (n_tp + n_sl) > 0:
    wr_dec = n_tp / (n_tp + n_sl)
    avg_win  = tp_trades["pnl_delta"].mean() if len(tp_trades) > 0 else 0
    avg_loss = abs(sl_trades["pnl_delta"].mean()) if len(sl_trades) > 0 else 1
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = wr_dec - (1 - wr_dec) / rr_ratio if rr_ratio > 0 else 0
else:
    kelly = 0

print("\n" + "=" * 70)
print(f"【XAUUSD 半年間バックテスト 結果サマリー（{START_DATE} 〜 {END_DATE}）】")
print("=" * 70)
print(f"  トレード数:            {n_total}件（TP:{n_tp} SL:{n_sl} BE:{n_be}）")
print(f"  勝率（BE除く）:        {win_rate:.1f}%")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:      {max_dd:.1f}%")
print(f"  純利益:                {final_equity - INIT_CASH:+,.0f}円")
print(f"  リターン（6ヶ月）:     {ret_pct:+.1f}%")
print(f"  ケリー係数:            {kelly:.3f}")
print(f"  月次プラス:            {n_pos_months}/{len(monthly_pnl)}")
print(f"  最終資産:              {final_equity:,.0f}円")
print("=" * 70)

# 月次詳細
print("\n【月次損益】")
for m, pnl in monthly_pnl.items():
    sign = "+" if pnl >= 0 else ""
    print(f"  {m}: {sign}{pnl/1e4:.1f}万円")

# ── チャート描画 ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
fig.suptitle(
    f"XAUUSD 半年間バックテスト [4H AND 1H KMID/KLOW]\n"
    f"期間: {START_DATE}〜{END_DATE}  初期資金: {INIT_CASH:,}円",
    fontsize=13, fontweight="bold"
)

# (1) 資産曲線
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(range(len(eq_arr)), np.array(eq_arr) / 1e6, color="#d97706", linewidth=1.5)
ax1.axhline(INIT_CASH / 1e6, color="gray", linestyle="--", alpha=0.5)
ax1.text(0.04, 0.93,
         f"最終: {final_equity/1e6:.2f}M円\n{ret_pct:+.1f}%",
         transform=ax1.transAxes, fontsize=10, color="#d97706",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax1.set_title("資産曲線", fontweight="bold")
ax1.set_ylabel("資産（百万円）")
ax1.set_xlabel("トレード番号")
ax1.grid(alpha=0.3)

# (2) ドローダウン
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(range(len(dd_arr)), dd_arr, 0, color="#ef4444", alpha=0.5)
ax2.plot(range(len(dd_arr)), dd_arr, color="#ef4444", linewidth=0.8)
mdd_i = int(np.argmin(dd_arr))
ax2.annotate(
    f"最大DD\n{dd_arr[mdd_i]:.2f}%",
    xy=(mdd_i, dd_arr[mdd_i]),
    xytext=(mdd_i + max(3, len(dd_arr)//10), dd_arr[mdd_i] + abs(dd_arr[mdd_i]) * 0.25),
    fontsize=9, color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85)
)
ax2.set_title("ドローダウン", fontweight="bold")
ax2.set_ylabel("DD（%）")
ax2.set_xlabel("トレード番号")
ax2.grid(alpha=0.3)

# (3) 月次損益
ax3 = fig.add_subplot(gs[0, 1])
months_str = [str(m) for m in monthly_pnl.index]
colors     = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly_pnl.values]
bars = ax3.bar(range(len(months_str)), monthly_pnl.values / 1e4, color=colors, alpha=0.85)
for i, (bar, val) in enumerate(zip(bars, monthly_pnl.values)):
    sign = "+" if val >= 0 else ""
    ax3.text(i, val / 1e4 + (0.5 if val >= 0 else -1.5),
             f"{sign}{val/1e4:.0f}", ha="center", fontsize=9, fontweight="bold")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(range(len(months_str)))
ax3.set_xticklabels(months_str, rotation=30, fontsize=9)
ax3.set_ylabel("損益（万円）")
ax3.set_title(f"月次損益（月次プラス: {n_pos_months}/{len(monthly_pnl)}）", fontweight="bold")
ax3.grid(alpha=0.3, axis="y")

# (4) 結果分布（TP/SL/BE）
ax4 = fig.add_subplot(gs[1, 1])
labels = ["TP", "SL", "BE"]
counts = [n_tp, n_sl, n_be]
colors4 = ["#22c55e", "#ef4444", "#94a3b8"]
wedges, texts, autotexts = ax4.pie(
    counts, labels=labels, colors=colors4, autopct="%1.1f%%",
    startangle=90, textprops={"fontsize": 11}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax4.set_title(
    f"結果分布（計{n_total}件）\n"
    f"勝率(BE除く): {win_rate:.1f}%  PF: {pf:.2f}  最大DD: {max_dd:.1f}%",
    fontweight="bold", fontsize=10
)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "xauusd_6m_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nチャート保存: {out_path}")

# CSV保存
csv_path = os.path.join(OUT_DIR, "xauusd_6m_trades.csv")
df_trades.to_csv(csv_path, index=False)
print(f"取引履歴CSV保存: {csv_path}")
