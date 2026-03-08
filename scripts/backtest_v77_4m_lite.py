"""
backtest_v77_4m_lite.py  v2
===========================
backtest_v77_correct.py の軽量版（シグナル生成ロジック修正版）

修正内容（v2）:
- 二番底/天井パターンを3点構造（底1→山→底2）で厳密に検出
- 反転確認足の実体サイズ条件追加（ATR×0.3以上）
- クールダウン: 同銘柄の直前シグナルから4H以上経過
- 二番底が一番底より高い位置（上昇二番底）の確認
- 山の高値が底1終値よりATR×0.5以上高い確認

期間: 2025-11-01 〜 2026-02-27（OOS期間のみ）
モード: Hybridのみ（4H足 + 1H足）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── パラメータ ────────────────────────────────────────────
INIT_CASH    = 1_000_000
RISK_PCT     = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
KLOW_THR     = 0.0015      # 下ヒゲ比率上限
BODY_MIN_ATR = 0.3         # 反転確認足の最小実体（ATR倍率）
MOUNTAIN_MIN = 0.5         # 山の高さ最小（ATR倍率）: 底1終値からの距離
COOLDOWN_H   = 4           # 同銘柄クールダウン時間（時間）
START        = "2025-11-01"
END          = "2026-02-27"

PAIRS = {
    "USDJPY": {"sym": "usdjpy"},
    "EURUSD": {"sym": "eurusd"},
    "GBPUSD": {"sym": "gbpusd"},
    "AUDUSD": {"sym": "audusd"},
    "USDCAD": {"sym": "usdcad"},
    "USDCHF": {"sym": "usdchf"},
    "NZDUSD": {"sym": "nzdusd"},
    "EURJPY": {"sym": "eurjpy"},
    "GBPJPY": {"sym": "gbpjpy"},
    "EURGBP": {"sym": "eurgbp"},
    "US30":   {"sym": "us30"},
    "SPX500": {"sym": "spx500"},
}

# ── データ読み込み ────────────────────────────────────────
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

def try_load(sym, tf):
    for name in [f"{sym}_oos_{tf}.csv", f"{sym}_{tf}.csv"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return load_csv(p)
    return None

def slice_period(df, start, end):
    return df[(df.index >= start) & (df.index <= end)].copy()

# ── テクニカル指標 ─────────────────────────────────────────
def calculate_atr(df, period=14):
    high_low   = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

# ── KMIDフィルター（方向性確認） ──────────────────────────
def check_kmid(bar, direction):
    """確認足が方向に沿った実体を持つか（KMID）"""
    o = bar["open"]; c = bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)

# ── KLOWフィルター（下ヒゲ比率） ─────────────────────────
def check_klow(bar, direction):
    """下ヒゲが小さいか（KLOW）"""
    o = bar["open"]; c = bar["close"]; l = bar["low"]; h = bar["high"]
    if direction == 1:
        body_bottom = min(o, c)
        ratio = (body_bottom - l) / o if o > 0 else 0
    else:
        body_top = max(o, c)
        ratio = (h - body_top) / o if o > 0 else 0
    return ratio < KLOW_THR

# ── 二番底/天井パターン検出（3点構造） ───────────────────
def detect_double_bottom(bars, atr_val, direction):
    """
    3点構造の二番底/天井パターンを検出する。
    
    上昇（direction=1）: 底1 → 山 → 底2（上昇二番底）
    下降（direction=-1）: 天井1 → 谷 → 天井2（下降二番天井）
    
    bars: 直近N本のDataFrame（古い順）
    戻り値: (detected, sl_price, confirm_bar_idx)
    """
    n = len(bars)
    if n < 5:
        return False, None, None

    tolerance = atr_val * 0.3
    body_min  = atr_val * BODY_MIN_ATR
    mountain_min = atr_val * MOUNTAIN_MIN

    if direction == 1:
        # 上昇二番底: 底1 → 山 → 底2 → 反転確認足
        # bars[-4]: 底1候補, bars[-3]: 山候補, bars[-2]: 底2候補, bars[-1]: 反転確認足
        for i in range(n - 4, max(n - 8, 0), -1):
            b1  = bars.iloc[i]      # 底1
            mid = bars.iloc[i+1]    # 山
            b2  = bars.iloc[i+2]    # 底2
            conf= bars.iloc[i+3]    # 反転確認足

            low1  = b1["low"]
            peak  = mid["high"]
            low2  = b2["low"]
            conf_body = abs(conf["close"] - conf["open"])

            # 条件1: 底1と底2の安値が近い（tolerance以内）
            if abs(low1 - low2) > tolerance:
                continue
            # 条件2: 底2は底1より高い（上昇二番底）または同水準
            if low2 < low1 - tolerance:
                continue
            # 条件3: 山の高値が底1の終値よりmountain_min以上高い
            if peak < b1["close"] + mountain_min:
                continue
            # 条件4: 反転確認足が陽線かつ実体がbody_min以上
            if conf["close"] <= conf["open"]:
                continue
            if conf_body < body_min:
                continue
            # 条件5: KMIDフィルター（確認足）
            if not check_kmid(conf, 1):
                continue
            # 条件6: KLOWフィルター（確認足）
            if not check_klow(conf, 1):
                continue

            sl = min(low1, low2) - atr_val * 0.15
            return True, sl, i + 3  # 確認足のインデックス

    else:
        # 下降二番天井: 天井1 → 谷 → 天井2 → 反転確認足
        for i in range(n - 4, max(n - 8, 0), -1):
            t1  = bars.iloc[i]      # 天井1
            mid = bars.iloc[i+1]    # 谷
            t2  = bars.iloc[i+2]    # 天井2
            conf= bars.iloc[i+3]    # 反転確認足

            high1 = t1["high"]
            trough = mid["low"]
            high2 = t2["high"]
            conf_body = abs(conf["close"] - conf["open"])

            # 条件1: 天井1と天井2の高値が近い
            if abs(high1 - high2) > tolerance:
                continue
            # 条件2: 天井2は天井1より低い（下降二番天井）または同水準
            if high2 > high1 + tolerance:
                continue
            # 条件3: 谷の安値が天井1の終値よりmountain_min以上低い
            if trough > t1["close"] - mountain_min:
                continue
            # 条件4: 反転確認足が陰線かつ実体がbody_min以上
            if conf["close"] >= conf["open"]:
                continue
            if conf_body < body_min:
                continue
            # 条件5: KMIDフィルター
            if not check_kmid(conf, -1):
                continue
            # 条件6: KLOWフィルター
            if not check_klow(conf, -1):
                continue

            sl = max(high1, high2) + atr_val * 0.15
            return True, sl, i + 3

    return False, None, None

# ── シグナル生成（Hybridモード） ──────────────────────────
def generate_signals_hybrid(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)

    signals    = []
    used_times = set()
    last_signal_time = pd.Timestamp("2000-01-01", tz="UTC")  # クールダウン管理

    # ── 4Hシグナル ──
    h4_times = data_4h.index.tolist()
    for i in range(8, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_current = data_4h.iloc[i]
        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # クールダウンチェック
        if (h4_current_time - last_signal_time).total_seconds() / 3600 < COOLDOWN_H:
            continue

        trend = h4_current["trend"]
        bars = data_4h.iloc[i-7:i]  # 直近7本（確認足を含む）

        detected, sl, _ = detect_double_bottom(bars, atr_val, trend)
        if not detected:
            continue

        # エントリー: 4H足確定後2分以内の1分足始値
        m1w = data_1m[(data_1m.index >= h4_current_time) &
                      (data_1m.index < h4_current_time + pd.Timedelta(minutes=2))]
        if len(m1w) == 0:
            continue
        eb = m1w.iloc[0]; et = eb.name
        if et in used_times:
            continue

        raw_ep = eb["open"]
        if trend == 1:
            ep   = raw_ep + spread
            risk = raw_ep - sl
            tp   = raw_ep + risk * rr_ratio
        else:
            ep   = raw_ep - spread
            risk = sl - raw_ep
            tp   = raw_ep - risk * rr_ratio

        if risk <= 0 or risk > atr_val * 3:
            continue

        signals.append({"time": et, "dir": trend, "ep": ep, "sl": sl,
                        "tp": tp, "risk": risk, "tf": "4h"})
        used_times.add(et)
        last_signal_time = et

    # クールダウンをリセット（4Hと1Hは独立して管理）
    last_signal_time_1h = pd.Timestamp("2000-01-01", tz="UTC")

    # ── 1Hシグナル ──
    h1_times = data_1h.index.tolist()
    for i in range(8, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_current = data_1h.iloc[i]
        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # クールダウンチェック（1H独立）
        if (h1_current_time - last_signal_time_1h).total_seconds() / 3600 < COOLDOWN_H:
            continue

        # 4H足のトレンド確認
        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest["atr"]) or pd.isna(h4_latest["ema20"]):
            continue
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        bars = data_1h.iloc[i-7:i]  # 直近7本
        detected, sl, _ = detect_double_bottom(bars, atr_val, trend)
        if not detected:
            continue

        # エントリー
        m1w = data_1m[(data_1m.index >= h1_current_time) &
                      (data_1m.index < h1_current_time + pd.Timedelta(minutes=2))]
        if len(m1w) == 0:
            continue
        eb = m1w.iloc[0]; et = eb.name
        if et in used_times:
            continue

        raw_ep = eb["open"]
        if trend == 1:
            ep   = raw_ep + spread
            risk = raw_ep - sl
            tp   = raw_ep + risk * rr_ratio
        else:
            ep   = raw_ep - spread
            risk = sl - raw_ep
            tp   = raw_ep - risk * rr_ratio

        if risk <= 0 or risk > h4_atr * 2:
            continue

        signals.append({"time": et, "dir": trend, "ep": ep, "sl": sl,
                        "tp": tp, "risk": risk, "tf": "1h"})
        used_times.add(et)
        last_signal_time_1h = et

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

# ── シミュレーション ──────────────────────────────────────
def simulate(signals, data_1m, init_cash, risk_pct, half_r, symbol, usdjpy_1m=None):
    rm = RiskManager(symbol, risk_pct=risk_pct)
    if signals is None or len(signals) == 0:
        return [], [init_cash]
    equity    = init_cash
    eq_series = [equity]
    trades    = []

    for _, sig in signals.iterrows():
        ep         = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk       = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]
        future     = data_1m[data_1m.index > entry_time]
        if len(future) == 0:
            continue

        # USDJPY レート取得
        usdjpy_at_entry = 150.0
        if usdjpy_1m is not None:
            uj = usdjpy_1m[usdjpy_1m.index <= entry_time]
            if len(uj) > 0:
                usdjpy_at_entry = uj.iloc[-1]["close"]

        lot = rm.calc_lot(equity, risk, ref_price=ep, usdjpy_rate=usdjpy_at_entry)
        if lot <= 0:
            continue

        # 半利確TP（1.0R到達）
        half_tp = (ep + (tp - ep) * (half_r / RR_RATIO) if direction == 1
                   else ep - (ep - tp) * (half_r / RR_RATIO))
        half_done = False
        sl_current = sl  # SLはBE移動後に更新
        result = None; exit_time = None; exit_price = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                # SL判定（SL/TP同時到達はSL優先）
                if bar["low"] <= sl_current:
                    result = "SL"; exit_price = sl_current; exit_time = bar_time; break
                # 半利確
                if not half_done and bar["high"] >= half_tp:
                    half_done = True
                    sl_current = ep  # BEに移動
                # TP判定
                if bar["high"] >= tp:
                    result = "TP"; exit_price = tp; exit_time = bar_time; break
            else:
                if bar["high"] >= sl_current:
                    result = "SL"; exit_price = sl_current; exit_time = bar_time; break
                if not half_done and bar["low"] <= half_tp:
                    half_done = True
                    sl_current = ep
                if bar["low"] <= tp:
                    result = "TP"; exit_price = tp; exit_time = bar_time; break

        if result is None:
            result     = "BE" if half_done else "OPEN"
            exit_price = sl_current
            exit_time  = future.index[-1]
        if result == "OPEN":
            continue

        # USDJPY レート（決済時）
        usdjpy_at_exit = 150.0
        if usdjpy_1m is not None:
            uj = usdjpy_1m[usdjpy_1m.index <= exit_time]
            if len(uj) > 0:
                usdjpy_at_exit = uj.iloc[-1]["close"]

        # PnL計算
        if result == "TP":
            if half_done:
                half_pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                            usdjpy_rate=usdjpy_at_entry, ref_price=ep)
                full_pnl = rm.calc_pnl_jpy(direction, ep, tp, lot * 0.5,
                                            usdjpy_rate=usdjpy_at_exit, ref_price=ep)
                pnl = half_pnl + full_pnl
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, tp, lot,
                                      usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            if half_done:
                half_pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                            usdjpy_rate=usdjpy_at_entry, ref_price=ep)
                # BE-SLなので残り半分は±0
                pnl = half_pnl
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot,
                                      usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:  # BE（期間終了時にBE-SL）
            half_pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                        usdjpy_rate=usdjpy_at_entry, ref_price=ep)
            pnl = half_pnl

        equity += pnl
        eq_series.append(equity)
        trades.append({
            "entry_time": entry_time, "exit_time": exit_time,
            "symbol": symbol, "direction": direction,
            "ep": ep, "sl": sl, "tp": tp,
            "result": result, "pnl": pnl, "equity": equity, "tf": sig["tf"]
        })

    return trades, eq_series

# ── 統計計算 ──────────────────────────────────────────────
def calc_stats(trades, eq, label):
    if not trades:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0, "return_pct": 0,
                "return_abs": 0, "mdd_pct": 0, "kelly": 0, "monthly_plus": ""}
    df   = pd.DataFrame(trades)
    wins = df[df["result"] == "TP"]
    losses = df[df["result"] == "SL"]
    n  = len(df); wr = len(wins) / n if n > 0 else 0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss   = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf  = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    eq_arr = np.array(eq)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak
    mdd    = dd.min()
    ret    = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]
    avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1
    kelly    = wr - (1 - wr) / (avg_win / avg_loss) if avg_loss > 0 else 0
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    monthly_plus = f"{(monthly > 0).sum()}/{len(monthly)}"
    return {"label": label, "n": n, "winrate": wr * 100, "pf": pf,
            "return_pct": ret * 100, "return_abs": eq[-1] - eq[0],
            "mdd_pct": abs(mdd) * 100, "kelly": kelly, "monthly_plus": monthly_plus}

# ── メイン処理 ────────────────────────────────────────────
print("=" * 70)
print(f"v77 Hybrid 12銘柄 バックテスト v2  {START} 〜 {END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%  RR: {RR_RATIO}")
print(f"クールダウン: {COOLDOWN_H}H  反転足実体: ATR×{BODY_MIN_ATR}  山高さ: ATR×{MOUNTAIN_MIN}")
print("=" * 70)

all_results = []
all_trades  = []
eq_curves   = {}

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]
    rm  = RiskManager(pair, risk_pct=RISK_PCT)
    spread = rm.spread_pips; pip = rm.pip_size

    d1m  = try_load(sym, "1m")
    d15m = try_load(sym, "15m")
    d4h  = try_load(sym, "4h")

    if any(d is None for d in [d1m, d15m, d4h]):
        print(f"  [{pair}] SKIP: データ不足")
        continue

    d1m  = slice_period(d1m,  START, END)
    d15m = slice_period(d15m, START, END)
    d4h  = slice_period(d4h,  START, END)

    if len(d1m) == 0:
        print(f"  [{pair}] SKIP: 期間内データなし")
        continue

    usdjpy_1m = None
    if rm.quote_type != "A":
        uj = try_load("usdjpy", "1m")
        if uj is not None:
            usdjpy_1m = slice_period(uj, START, END)

    sigs   = generate_signals_hybrid(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
    trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH, risk_pct=RISK_PCT,
                          half_r=HALF_R, symbol=pair, usdjpy_1m=usdjpy_1m)
    stats  = calc_stats(trades, eq, pair)
    stats["pair"] = pair; stats["spread"] = spread
    all_results.append(stats)
    for t in trades:
        all_trades.append(t)
    eq_curves[pair] = eq

    tf_info = ""
    if trades:
        df_t = pd.DataFrame(trades)
        tf_info = f" 4H:{(df_t['tf']=='4h').sum()} 1H:{(df_t['tf']=='1h').sum()}"

    print(f"  [{pair}] {stats['n']}件{tf_info} | 勝率{stats['winrate']:.1f}% | "
          f"PF{stats['pf']:.2f} | リターン{stats['return_pct']:+.1f}% | "
          f"MDD{stats['mdd_pct']:.1f}% | 月次+{stats['monthly_plus']}")

# ── 結果保存 ──────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
df_trades  = pd.DataFrame(all_trades)
df_results.to_csv(os.path.join(OUT_DIR, "v77_4m_lite_results.csv"), index=False)
df_trades.to_csv(os.path.join(OUT_DIR, "v77_4m_lite_trades.csv"), index=False)

# ── チャート ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"v77 Hybrid 12銘柄バックテスト v2\n{START} 〜 {END}", fontsize=13, fontweight="bold")

ax1 = axes[0]
colors = plt.cm.tab20(np.linspace(0, 1, len(eq_curves)))
for (pair, eq), color in zip(eq_curves.items(), colors):
    if len(eq) > 1:
        ax1.plot(range(len(eq)), [e / INIT_CASH * 100 for e in eq],
                 label=pair, color=color, linewidth=1.2)
ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylabel("資産（初期=100）")
ax1.set_title("銘柄別 資産曲線")
ax1.legend(loc="upper left", fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
if len(df_results) > 0:
    pairs_sorted = df_results.sort_values("pf", ascending=True)
    colors_bar = ["#22c55e" if pf >= 1.5 else "#f97316" if pf >= 1.0 else "#ef4444"
                  for pf in pairs_sorted["pf"]]
    bars = ax2.barh(pairs_sorted["pair"], pairs_sorted["pf"], color=colors_bar)
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax2.axvline(1.5, color="orange", linestyle="--", linewidth=1)
    ax2.set_xlabel("プロフィットファクター")
    ax2.set_title("銘柄別 PF（緑≥1.5 / 橙≥1.0 / 赤<1.0）")
    for bar, (_, row) in zip(bars, pairs_sorted.iterrows()):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"勝率{row['winrate']:.0f}% | {row['n']}件 | MDD{row['mdd_pct']:.1f}%",
                 va="center", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v77_4m_lite_result.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n結果保存: results/v77_4m_lite_result.png")

# ── 全銘柄合算サマリー ────────────────────────────────────
if len(df_results) > 0:
    print("\n" + "=" * 70)
    print("全銘柄サマリー")
    print("=" * 70)
    print(df_results[["pair","n","winrate","pf","return_pct","mdd_pct","monthly_plus","spread"]].to_string(index=False))
    total_pnl = df_results["return_abs"].sum()
    print(f"\n合計PnL: {total_pnl:,.0f}円 ({total_pnl/INIT_CASH*100:.1f}%)")
