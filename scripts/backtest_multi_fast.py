"""
backtest_multi_fast.py
======================
複数銘柄同時バックテスト（高速版）

【高速化のポイント】
- 各銘柄を独立してシミュレーション（単体バックテスト）
- 決済チェックをバー単位ではなく「次の決済バー」を事前に検索（O(n)化）
- PositionManagerのルールを事後フィルタリングで適用

【選定銘柄（相関<0.5・流動性上位）】
  USDJPY, AUDUSD, EURJPY, EURGBP, US30

【期間】OOS: 2025-03-03 〜 2025-06-30（3ヶ月）
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
from utils.risk_manager     import AdaptiveRiskManager
from utils.position_manager import PositionManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────
INIT_CASH  = 1_000_000
BASE_RISK  = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
OOS_START  = "2025-03-03"
OOS_END    = "2025-06-30"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 選定銘柄（相関<0.5・流動性上位）
SYMBOLS = ["USDJPY", "AUDUSD", "EURJPY", "EURGBP", "US30"]
KLOW_THR = 0.0015

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

# ── シグナル生成（1Hモード） ───────────────────────────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size):
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
            # 反転確認足
            if direction == 1 and h1_prev1["close"] <= h1_prev1["open"]:
                continue
            if direction == -1 and h1_prev1["close"] >= h1_prev1["open"]:
                continue
            if not check_kmid_klow(h4_latest, direction):
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

# ── 単体シミュレーション（高速版） ────────────────────────
def simulate_single(symbol, signals, data_1m, init_cash, base_risk):
    """
    1銘柄を独立してシミュレーション。
    各シグナルの決済バーを事前に検索してO(n)で処理。
    戻り値: trades のリスト（entry_time, exit_time, result, pnl_jpy, equity_after）
    """
    arm    = AdaptiveRiskManager(symbol, base_risk_pct=base_risk)
    equity = init_cash
    trades = []

    # 1分足をnumpy配列に変換して高速化
    times  = data_1m.index.values
    highs  = data_1m["high"].values
    lows   = data_1m["low"].values
    opens  = data_1m["open"].values

    # USDJPY換算レートは簡易固定値（150円）
    USDJPY_RATE = 150.0

    i_start = 0  # 検索開始インデックス（前回エントリーより後ろから検索）

    for sig in signals:
        entry_time = sig["time"]
        direction  = sig["dir"]
        ep         = sig["ep"]
        sl         = sig["sl"]
        tp         = sig["tp"]
        risk       = sig["risk"]

        # エントリーバーのインデックスを検索
        idx_arr = np.searchsorted(times, np.datetime64(entry_time))
        if idx_arr >= len(times):
            continue

        # ロットサイズ計算
        lot, eff_risk, reason = arm.calc_lot_adaptive(
            equity=equity, sl_distance=risk,
            ref_price=ep, usdjpy_rate=USDJPY_RATE,
        )

        # 決済チェック（エントリーバーの次から）
        half_done = False
        be_sl     = None
        half_pnl  = 0.0
        result    = None
        exit_time = None
        exit_price = None

        for j in range(idx_arr + 1, len(times)):
            bar_high = highs[j]
            bar_low  = lows[j]
            bar_time = times[j]

            current_sl = be_sl if half_done else sl

            if direction == 1:  # ロング
                # SL優先（保守的）
                if bar_low <= current_sl:
                    exit_price = current_sl
                    exit_time  = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else ("TP" if pnl > 0 else "SL")
                    break
                # TP
                if bar_high >= tp:
                    if not half_done and bar_high >= ep + risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                               lot * 0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp
                        half_done = True; be_sl = ep
                        arm.update_peak(equity)
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl; result = "TP"; break
                # 半利確のみ
                if not half_done and bar_high >= ep + risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep + risk * HALF_R,
                                           lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    half_done = True; be_sl = ep
                    arm.update_peak(equity)

            else:  # ショート
                if bar_high >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl
                    result = "BE" if (half_done and abs(exit_price - ep) < risk * 0.01) else ("TP" if pnl > 0 else "SL")
                    break
                if bar_low <= tp:
                    if not half_done and bar_low <= ep - risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                               lot * 0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp
                        half_done = True; be_sl = ep
                        arm.update_peak(equity)
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price,
                                            lot * remaining, USDJPY_RATE, ep)
                    equity += pnl; result = "TP"; break
                if not half_done and bar_low <= ep - risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep - risk * HALF_R,
                                           lot * 0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp
                    half_done = True; be_sl = ep
                    arm.update_peak(equity)

        if result is None:
            continue  # 未決済（期末超え）はスキップ

        arm.update_peak(equity)
        trades.append({
            "symbol":     symbol,
            "entry_time": entry_time,
            "exit_time":  exit_time,
            "dir":        direction,
            "ep":         ep,
            "sl":         sl,
            "tp":         tp,
            "exit_price": exit_price,
            "result":     result,
            "lot":        lot,
            "eff_risk":   eff_risk,
            "half_pnl":   half_pnl,
            "equity_after": equity,
        })

    return trades, equity

# ── PositionManagerによる事後フィルタリング ───────────────
def apply_position_filter(all_trades_df, init_cash, base_risk):
    """
    全銘柄の取引履歴を時系列順にマージし、
    PositionManagerのルールで採用/不採用を決定する。
    採用されたトレードのみで最終的な資産曲線を計算する。
    """
    pm     = PositionManager()
    equity = init_cash
    equity_curve = [init_cash]
    accepted_trades = []

    # 全トレードを entry_time でソート
    df = all_trades_df.sort_values("entry_time").reset_index(drop=True)

    # 現在保有中のポジション情報 {symbol: {exit_time, pnl_sequence}}
    open_pos = {}  # symbol -> {exit_time, pnl_delta}

    for _, row in df.iterrows():
        sym        = row["symbol"]
        entry_time = row["entry_time"]
        exit_time  = row["exit_time"]

        # エントリー前に決済済みポジションを処理
        closed = [s for s, p in open_pos.items()
                  if p["exit_time"] <= entry_time]
        for s in closed:
            pos = open_pos.pop(s)
            equity += pos["pnl_delta"]
            equity_curve.append(equity)
            pm.close_position(s)

        # 既に同銘柄を保有中ならスキップ
        if sym in open_pos:
            continue

        # PositionManagerチェック
        eff_risk = row["eff_risk"]
        ok, reason = pm.can_enter(sym, eff_risk)
        if not ok:
            continue

        # 採用
        pm.open_position(sym, eff_risk, entry_time=pd.Timestamp(entry_time))
        # 損益 = equity_after - (前のequity)
        # 単体シミュレーションのequity_afterは累積値なので差分を取る
        pnl_delta = row["pnl_delta"]
        open_pos[sym] = {
            "exit_time": exit_time,
            "pnl_delta": pnl_delta,
        }
        accepted_trades.append(row.to_dict())

    # 残りのオープンポジションを決済
    for s, pos in open_pos.items():
        equity += pos["pnl_delta"]
        equity_curve.append(equity)

    return pd.DataFrame(accepted_trades), pd.Series(equity_curve)

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"複数銘柄同時バックテスト（高速版）[AdaptiveRiskManager + PositionManager]")
print(f"期間: {OOS_START} 〜 {OOS_END}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print(f"選定銘柄: {SYMBOLS}")
print("=" * 70)

# ── Step1: 各銘柄を独立してシミュレーション ──────────────
all_trades = []
sym_equity = {}  # symbol -> 最終equity

for sym in SYMBOLS:
    sym_lower = sym.lower()
    d1m  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_1m.csv"))
    d15m = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_15m.csv"))
    d4h  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv"))

    if d1m is None or d15m is None or d4h is None:
        print(f"  {sym}: データなし → スキップ")
        continue

    d1m  = slice_period(d1m,  OOS_START, OOS_END)
    d15m = slice_period(d15m, OOS_START, OOS_END)
    d4h  = slice_period(d4h,  OOS_START, OOS_END)

    if d1m is None or len(d1m) == 0:
        print(f"  {sym}: OOS期間のデータなし → スキップ")
        continue

    arm_ref = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    print(f"  {sym}: シグナル生成中... (spread={arm_ref.spread_pips}pips)")
    sigs = generate_signals_1h(d1m, d15m, d4h, arm_ref.spread_pips, arm_ref.pip_size)
    print(f"    → シグナル{len(sigs)}件 / シミュレーション実行中...")

    trades, final_eq = simulate_single(sym, sigs, d1m, INIT_CASH, BASE_RISK)
    sym_equity[sym] = final_eq

    # pnl_delta（1トレードごとの損益）を計算
    prev_eq = INIT_CASH
    for t in trades:
        t["pnl_delta"] = t["equity_after"] - prev_eq
        prev_eq = t["equity_after"]

    n_tp = sum(1 for t in trades if t["result"] == "TP")
    n_sl = sum(1 for t in trades if t["result"] == "SL")
    n_be = sum(1 for t in trades if t["result"] == "BE")
    wr   = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0
    ret  = (final_eq - INIT_CASH) / INIT_CASH * 100
    print(f"    → 完了: {len(trades)}件 TP:{n_tp} SL:{n_sl} BE:{n_be} "
          f"勝率:{wr*100:.0f}% リターン:{ret:+.1f}%")

    all_trades.extend(trades)

if not all_trades:
    print("トレードなし。終了。")
    sys.exit(0)

all_trades_df = pd.DataFrame(all_trades)
all_trades_df["entry_time"] = pd.to_datetime(all_trades_df["entry_time"], utc=True)
all_trades_df["exit_time"]  = pd.to_datetime(all_trades_df["exit_time"],  utc=True)

# ── Step2: PositionManagerフィルタリング ─────────────────
print(f"\n全シグナル合計: {len(all_trades_df)}件")
print("PositionManagerフィルタリング中...")

accepted, eq_series = apply_position_filter(all_trades_df, INIT_CASH, BASE_RISK)
print(f"採用トレード: {len(accepted)}件 / 全{len(all_trades_df)}件")

# ── Step3: 統計 ──────────────────────────────────────────
eq_arr = eq_series.values
n_tp = (accepted["result"] == "TP").sum() if len(accepted) > 0 else 0
n_sl = (accepted["result"] == "SL").sum() if len(accepted) > 0 else 0
n_be = (accepted["result"] == "BE").sum() if len(accepted) > 0 else 0
n    = len(accepted)

wr_strict  = n_tp / (n_tp + n_sl + n_be) if (n_tp + n_sl + n_be) > 0 else 0
wr_excl_be = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0

pnl_arr    = np.diff(eq_arr)
gross_win  = pnl_arr[pnl_arr > 0].sum()
gross_loss = abs(pnl_arr[pnl_arr < 0].sum())
pf  = gross_win / gross_loss if gross_loss > 0 else float("inf")

peak = np.maximum.accumulate(eq_arr)
dd   = (eq_arr - peak) / peak
mdd  = abs(dd.min()) * 100

ret_pct = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] * 100
ret_abs = eq_arr[-1] - eq_arr[0]
kelly   = wr_excl_be - (1 - wr_excl_be) / (pf if pf > 0 else 1e-9)

# 月次集計
if len(accepted) > 0:
    accepted["month"] = accepted["exit_time"].dt.to_period("M")
    monthly_eq = accepted.groupby("month")["equity_after"].last()
    monthly_shifted = monthly_eq.shift(1).fillna(INIT_CASH)
    monthly_plus  = (monthly_eq > monthly_shifted).sum()
    monthly_total = len(monthly_eq)
else:
    monthly_plus = monthly_total = 0

print(f"\n{'='*70}")
print(f"【複数銘柄同時バックテスト 結果サマリー】")
print(f"{'='*70}")
print(f"  採用トレード数:        {n}件（TP:{n_tp} SL:{n_sl} BE:{n_be}）")
print(f"  勝率（BE除く）:        {wr_excl_be*100:.1f}%")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:      {mdd:.1f}%")
print(f"  純利益:                {ret_abs:+,.0f}円")
print(f"  リターン（3ヶ月）:     {ret_pct:+.1f}%")
print(f"  ケリー係数:            {kelly:.3f}")
print(f"  月次プラス:            {monthly_plus}/{monthly_total}")
print(f"  最終資産:              {eq_arr[-1]:,.0f}円")

if len(accepted) > 0:
    print(f"\n【銘柄別採用トレード数】")
    for sym in SYMBOLS:
        sub = accepted[accepted["symbol"] == sym]
        if len(sub) == 0:
            print(f"  {sym:8s}: 0件")
            continue
        n_tp_ = (sub["result"] == "TP").sum()
        n_sl_ = (sub["result"] == "SL").sum()
        n_be_ = (sub["result"] == "BE").sum()
        wr_   = n_tp_ / (n_tp_ + n_sl_) if (n_tp_ + n_sl_) > 0 else 0
        print(f"  {sym:8s}: {len(sub):3d}件 TP:{n_tp_:3d} SL:{n_sl_:3d} BE:{n_be_:3d} 勝率:{wr_*100:.0f}%")
print(f"{'='*70}")

# ── Step4: 可視化 ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"複数銘柄同時バックテスト（高速版）\n"
    f"銘柄: {', '.join(SYMBOLS)}  期間: {OOS_START}〜{OOS_END}  初期資金: {INIT_CASH:,}円",
    fontsize=11, fontweight="bold"
)

# 資産曲線
ax = axes[0, 0]
ax.plot(eq_arr / 1e6, color="#3b82f6", linewidth=1.5)
ax.axhline(INIT_CASH / 1e6, color="gray", linestyle="--", alpha=0.5)
ax.set_title("資産曲線", fontweight="bold")
ax.set_ylabel("資産（百万円）")
ax.grid(alpha=0.3)
ax.text(0.05, 0.92,
        f"最終: {eq_arr[-1]/1e6:.3f}百万円\n{ret_pct:+.1f}%（3ヶ月）",
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
if len(accepted) > 0:
    monthly_pnl = monthly_eq.diff().fillna(monthly_eq.iloc[0] - INIT_CASH)
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly_pnl]
    ax.bar(range(len(monthly_pnl)), monthly_pnl.values / 1e4, color=colors, alpha=0.8)
    ax.set_xticks(range(len(monthly_pnl)))
    ax.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45, fontsize=8)
    ax.text(0.05, 0.92, f"月次プラス: {monthly_plus}/{monthly_total}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.set_title("月次損益（万円）", fontweight="bold")
ax.set_ylabel("損益（万円）")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(alpha=0.3, axis="y")

# 銘柄別トレード数と勝率
ax = axes[1, 1]
if len(accepted) > 0:
    sym_counts = []
    sym_wrs    = []
    for sym in SYMBOLS:
        sub = accepted[accepted["symbol"] == sym]
        n_tp_ = (sub["result"] == "TP").sum()
        n_sl_ = (sub["result"] == "SL").sum()
        wr_   = n_tp_ / (n_tp_ + n_sl_) if (n_tp_ + n_sl_) > 0 else 0
        sym_counts.append(len(sub))
        sym_wrs.append(wr_ * 100)

    x = range(len(SYMBOLS))
    bars = ax.bar(x, sym_counts, color="#6366f1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SYMBOLS, fontsize=9)
    ax.set_ylabel("採用トレード数")
    ax2 = ax.twinx()
    ax2.plot(x, sym_wrs, "o-", color="#f59e0b", linewidth=2, markersize=8)
    ax2.set_ylabel("勝率（%）", color="#f59e0b")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#f59e0b")
ax.set_title("銘柄別採用トレード数・勝率", fontweight="bold")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "multi_fast_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nチャート保存: {out_path}")

if len(accepted) > 0:
    accepted.to_csv(os.path.join(OUT_DIR, "multi_fast_trades.csv"), index=False)
    print("取引履歴CSV保存完了。")
print("完了。")
