"""
backtest_multi_adaptive.py
==========================
複数銘柄同時バックテスト（PositionManager組み込み版）

【設計】
1. 全銘柄のシグナルを生成して時系列でマージ
2. 時刻順に1件ずつ処理（リアルタイム運用と同じ順序）
3. PositionManagerで全体・グループ・サブグループ制約を適用
4. AdaptiveRiskManagerで資産規模×DD連動リスク逓減
5. 1ポジション/銘柄（同銘柄の重複エントリー禁止）

【対象銘柄】（OOSデータが存在する全15銘柄）
  FX    : USDJPY, EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF,
           NZDUSD, EURJPY, GBPJPY, EURGBP
  貴金属 : XAUUSD, XAGUSD
  指数   : US30, SPX500, NAS100
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
from utils.risk_manager    import AdaptiveRiskManager, SYMBOL_CONFIG
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
OOS_END    = "2026-02-27"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 対象銘柄（OOSデータが存在するもの）
SYMBOLS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF",
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "XAUUSD", "XAGUSD",
    "US30", "SPX500", "NAS100",
]

# ── シグナル生成関数（backtest_v77_correct.pyから直接コピー） ──
KLOW_THR = 0.0015

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
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def calculate_atr(df, period=14):
    hl  = df["high"] - df["low"]
    hc  = abs(df["high"] - df["close"].shift())
    lc  = abs(df["low"]  - df["close"].shift())
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(prev_4h_bar, direction):
    o = prev_4h_bar["open"]
    c = prev_4h_bar["close"]
    l = prev_4h_bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THR
    return kmid_ok and klow_ok

def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    """1Hモード: 1H足でパターン検出 → 1分足でエントリー"""
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    # 15分足から1H足を集約
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1   = data_1h.iloc[i - 1]
        h1_prev2   = data_1h.iloc[i - 2]
        atr_val    = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)) or pd.isna(h4_latest.get("ema20", np.nan)):
            continue

        trend     = h4_latest["trend"]
        h4_atr    = h4_latest["atr"]
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
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                            })
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
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                            })
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── メインシミュレーション ─────────────────────────────────
def simulate_multi(all_signals_df, data_1m_dict, init_cash=1_000_000, base_risk=0.02):
    """
    全銘柄のシグナルを時系列順に処理するシミュレーション。

    Parameters
    ----------
    all_signals_df : pd.DataFrame
        全銘柄のシグナルをマージして時刻順にソートしたもの
        columns: [time, symbol, dir, ep, sl, tp, risk]
    data_1m_dict   : dict[symbol -> pd.DataFrame]
        銘柄ごとの1分足データ
    """
    # 銘柄ごとのAdaptiveRiskManager
    arm_dict = {sym: AdaptiveRiskManager(sym, base_risk_pct=base_risk) for sym in SYMBOLS}
    pm       = PositionManager()

    equity = init_cash
    peak   = init_cash

    # 保有中ポジションの詳細情報
    open_pos_detail = {}  # symbol -> {entry_time, ep, sl, tp, risk, lot, half_done, be_sl, half_pnl, usdjpy_rate}

    trades = []
    equity_curve = [init_cash]

    # 全シグナルを時刻順に処理
    for _, sig in all_signals_df.iterrows():
        entry_time = sig["time"]
        symbol     = sig["symbol"]
        direction  = sig["dir"]
        ep         = sig["ep"]
        sl         = sig["sl"]
        tp         = sig["tp"]
        risk       = sig["risk"]

        arm = arm_dict[symbol]
        data_1m = data_1m_dict[symbol]

        # ── 保有中ポジションの決済チェック（エントリー前に処理） ──
        closed_syms = []
        for sym, detail in open_pos_detail.items():
            d1m = data_1m_dict[sym]
            future = d1m[d1m.index > detail["entry_time"]]
            # entry_time以降のバーで決済チェック（簡易: 現在のentry_timeまで）
            future_until = future[future.index <= entry_time]

            for bar_time, bar in future_until.iterrows():
                if bar_time <= detail.get("last_checked", detail["entry_time"]):
                    continue

                dir_  = detail["dir"]
                ep_   = detail["ep"]
                sl_   = detail["sl"]
                tp_   = detail["tp"]
                risk_ = detail["risk"]
                lot_  = detail["lot"]
                arm_  = arm_dict[sym]
                ur    = detail["usdjpy_rate"]

                half_done = detail["half_done"]
                be_sl_    = detail.get("be_sl")
                current_sl = be_sl_ if half_done else sl_

                result     = None
                exit_price = None

                if dir_ == 1:  # ロング
                    if bar["low"] <= current_sl:
                        exit_price = current_sl
                        remaining  = 0.5 if half_done else 1.0
                        pnl = arm_.calc_pnl_jpy(dir_, ep_, exit_price, lot_ * remaining,
                                                 usdjpy_rate=ur, ref_price=ep_)
                        equity += pnl
                        result = "BE" if (half_done and abs(exit_price - ep_) < 1e-6) else ("TP" if pnl > 0 else "SL")
                    elif bar["high"] >= tp_:
                        if not half_done and bar["high"] >= ep_ + risk_ * HALF_R:
                            hp = arm_.calc_pnl_jpy(dir_, ep_, ep_ + risk_ * HALF_R,
                                                    lot_ * 0.5, usdjpy_rate=ur, ref_price=ep_)
                            equity += hp
                            detail["half_pnl"] += hp
                            detail["half_done"] = True
                            detail["be_sl"]     = ep_
                            arm_.update_peak(equity)
                        exit_price = tp_
                        remaining  = 0.5 if detail["half_done"] else 1.0
                        pnl = arm_.calc_pnl_jpy(dir_, ep_, exit_price, lot_ * remaining,
                                                 usdjpy_rate=ur, ref_price=ep_)
                        equity += pnl; result = "TP"
                    elif not detail["half_done"] and bar["high"] >= ep_ + risk_ * HALF_R:
                        hp = arm_.calc_pnl_jpy(dir_, ep_, ep_ + risk_ * HALF_R,
                                                lot_ * 0.5, usdjpy_rate=ur, ref_price=ep_)
                        equity += hp
                        detail["half_pnl"] += hp
                        detail["half_done"] = True
                        detail["be_sl"]     = ep_
                        arm_.update_peak(equity)

                else:  # ショート
                    if bar["high"] >= current_sl:
                        exit_price = current_sl
                        remaining  = 0.5 if half_done else 1.0
                        pnl = arm_.calc_pnl_jpy(dir_, ep_, exit_price, lot_ * remaining,
                                                 usdjpy_rate=ur, ref_price=ep_)
                        equity += pnl
                        result = "BE" if (half_done and abs(exit_price - ep_) < 1e-6) else ("TP" if pnl > 0 else "SL")
                    elif bar["low"] <= tp_:
                        if not half_done and bar["low"] <= ep_ - risk_ * HALF_R:
                            hp = arm_.calc_pnl_jpy(dir_, ep_, ep_ - risk_ * HALF_R,
                                                    lot_ * 0.5, usdjpy_rate=ur, ref_price=ep_)
                            equity += hp
                            detail["half_pnl"] += hp
                            detail["half_done"] = True
                            detail["be_sl"]     = ep_
                            arm_.update_peak(equity)
                        exit_price = tp_
                        remaining  = 0.5 if detail["half_done"] else 1.0
                        pnl = arm_.calc_pnl_jpy(dir_, ep_, exit_price, lot_ * remaining,
                                                 usdjpy_rate=ur, ref_price=ep_)
                        equity += pnl; result = "TP"
                    elif not detail["half_done"] and bar["low"] <= ep_ - risk_ * HALF_R:
                        hp = arm_.calc_pnl_jpy(dir_, ep_, ep_ - risk_ * HALF_R,
                                                lot_ * 0.5, usdjpy_rate=ur, ref_price=ep_)
                        equity += hp
                        detail["half_pnl"] += hp
                        detail["half_done"] = True
                        detail["be_sl"]     = ep_
                        arm_.update_peak(equity)

                detail["last_checked"] = bar_time

                if result is not None and exit_price is not None:
                    arm_.update_peak(equity)
                    peak = max(peak, equity)
                    trades.append({
                        "symbol":     sym,
                        "entry_time": detail["entry_time"],
                        "exit_time":  bar_time,
                        "dir":        dir_,
                        "ep":         ep_,
                        "exit_price": exit_price,
                        "result":     result,
                        "half_pnl":   detail["half_pnl"],
                        "equity":     equity,
                    })
                    equity_curve.append(equity)
                    closed_syms.append(sym)
                    break

        # 決済済みポジションを削除
        for sym in closed_syms:
            del open_pos_detail[sym]
            pm.close_position(sym)

        # ── 新規エントリー判定 ──
        if symbol in open_pos_detail:
            continue  # 同銘柄を既に保有中

        # PositionManagerでチェック
        eff_risk, reason = arm.effective_risk_pct(equity)
        ok, pm_reason = pm.can_enter(symbol, eff_risk)
        if not ok:
            continue

        # ロットサイズ計算（USDJPY換算レートは簡易固定値）
        usdjpy_rate = 150.0
        lot, eff_risk, risk_reason = arm.calc_lot_adaptive(
            equity=equity,
            sl_distance=risk,
            ref_price=ep,
            usdjpy_rate=usdjpy_rate,
        )

        # ポジション登録
        pm.open_position(symbol, eff_risk, entry_time=entry_time)
        open_pos_detail[symbol] = {
            "entry_time": entry_time,
            "dir":        direction,
            "ep":         ep,
            "sl":         sl,
            "tp":         tp,
            "risk":       risk,
            "lot":        lot,
            "half_done":  False,
            "be_sl":      None,
            "half_pnl":   0.0,
            "usdjpy_rate": usdjpy_rate,
            "last_checked": entry_time,
        }

    # ── 期末に残っているポジションを強制決済（終値） ──
    for sym, detail in open_pos_detail.items():
        d1m = data_1m_dict[sym]
        if len(d1m) > 0:
            last_bar   = d1m.iloc[-1]
            exit_price = last_bar["close"]
            dir_       = detail["dir"]
            ep_        = detail["ep"]
            lot_       = detail["lot"]
            ur         = detail["usdjpy_rate"]
            arm_       = arm_dict[sym]
            remaining  = 0.5 if detail["half_done"] else 1.0
            pnl = arm_.calc_pnl_jpy(dir_, ep_, exit_price, lot_ * remaining,
                                     usdjpy_rate=ur, ref_price=ep_)
            equity += pnl
            trades.append({
                "symbol":     sym,
                "entry_time": detail["entry_time"],
                "exit_time":  last_bar.name,
                "dir":        dir_,
                "ep":         ep_,
                "exit_price": exit_price,
                "result":     "FORCED",
                "half_pnl":   detail["half_pnl"],
                "equity":     equity,
            })
            equity_curve.append(equity)

    df = pd.DataFrame(trades)
    eq = pd.Series(equity_curve)
    return df, eq

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"複数銘柄同時バックテスト [AdaptiveRiskManager + PositionManager]")
print(f"期間: {OOS_START} 〜 {OOS_END}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print(f"全体6ポジ / グループ2ポジ / サブグループ1ポジ / リスク上限8%")
print("=" * 70)

# ── データ読み込み & シグナル生成 ──────────────────────────
all_signals = []
data_1m_dict = {}

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
    print(f"    → {len(sigs)}件")

    for s in sigs:
        s["symbol"] = sym
    all_signals.extend(sigs)
    data_1m_dict[sym] = d1m

# 時系列順にソート
all_signals_df = pd.DataFrame(all_signals).sort_values("time").reset_index(drop=True)
print(f"\n全シグナル合計: {len(all_signals_df)}件")

# ── シミュレーション実行 ───────────────────────────────────
print(f"\nシミュレーション実行中...")
trades, eq = simulate_multi(all_signals_df, data_1m_dict, init_cash=INIT_CASH, base_risk=BASE_RISK)
print(f"実行トレード数: {len(trades)}件")

if len(trades) == 0:
    print("トレードなし。終了。")
    sys.exit(0)

# ── 統計 ──────────────────────────────────────────────────
n_tp = (trades["result"] == "TP").sum()
n_sl = (trades["result"] == "SL").sum()
n_be = (trades["result"] == "BE").sum()
n_fo = (trades["result"] == "FORCED").sum()
n    = len(trades)

wr_strict  = n_tp / (n_tp + n_sl + n_be) if (n_tp + n_sl + n_be) > 0 else 0
wr_excl_be = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0

eq_arr  = eq.values
pnl_arr = np.diff(eq_arr)
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
trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
trades["month"] = trades["exit_time"].dt.to_period("M")
monthly_eq = trades.groupby("month")["equity"].last()
monthly_shifted = monthly_eq.shift(1).fillna(INIT_CASH)
monthly_plus  = (monthly_eq > monthly_shifted).sum()
monthly_total = len(monthly_eq)

# 銘柄別集計
sym_stats = trades.groupby("symbol").agg(
    trades_n=("result", "count"),
    tp=("result", lambda x: (x == "TP").sum()),
    sl=("result", lambda x: (x == "SL").sum()),
    be=("result", lambda x: (x == "BE").sum()),
).reset_index()

print(f"\n{'='*70}")
print(f"【結果サマリー】")
print(f"{'='*70}")
print(f"  総トレード数:          {n}件（TP:{n_tp} SL:{n_sl} BE:{n_be} 強制:{n_fo}）")
print(f"  勝率（BE含む）:        {wr_strict*100:.1f}%")
print(f"  勝率（BE除く）:        {wr_excl_be*100:.1f}%")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:      {mdd:.1f}%")
print(f"  純利益:                {ret_abs:+,.0f}円")
print(f"  リターン:              {ret_pct:+.1f}%")
print(f"  ケリー係数:            {kelly:.3f}")
print(f"  月次プラス:            {monthly_plus}/{monthly_total}")
print(f"  最終資産:              {eq_arr[-1]:,.0f}円")
print(f"\n【銘柄別トレード数】")
for _, row in sym_stats.iterrows():
    wr = row["tp"] / (row["tp"] + row["sl"]) if (row["tp"] + row["sl"]) > 0 else 0
    print(f"  {row['symbol']:8s}: {row['trades_n']:3d}件 TP:{row['tp']:3d} SL:{row['sl']:3d} BE:{row['be']:3d} 勝率:{wr*100:.0f}%")
print(f"{'='*70}")

# ── 可視化 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"複数銘柄同時バックテスト [AdaptiveRiskManager + PositionManager]\n"
    f"初期資金100万円 / 基本リスク2% / 全体6ポジ・グループ2ポジ・サブグループ1ポジ / リスク上限8%",
    fontsize=11, fontweight="bold"
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
monthly_pnl = monthly_eq.diff().fillna(monthly_eq.iloc[0] - INIT_CASH)
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

# 銘柄別トレード数
ax = axes[1, 1]
sym_stats_sorted = sym_stats.sort_values("trades_n", ascending=True)
bars = ax.barh(sym_stats_sorted["symbol"], sym_stats_sorted["trades_n"],
               color="#6366f1", alpha=0.8)
ax.set_title("銘柄別トレード数", fontweight="bold")
ax.set_xlabel("トレード数")
ax.grid(alpha=0.3, axis="x")
for bar, (_, row) in zip(bars, sym_stats_sorted.iterrows()):
    wr = row["tp"] / (row["tp"] + row["sl"]) if (row["tp"] + row["sl"]) > 0 else 0
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"勝率{wr*100:.0f}%", va="center", fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "multi_adaptive_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nチャート保存: {out_path}")

# CSV保存
trades.to_csv(os.path.join(OUT_DIR, "multi_adaptive_trades.csv"), index=False)
print("取引履歴CSV保存完了。")
print("完了。")
