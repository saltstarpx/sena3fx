"""
v77ロジック × 全12銘柄 × 3モード（1H/4H/ハイブリッド）バックテスト
最低スプレッド（Exness ゼロ口座 生スプレッド）を使用
設定:
  初期資金: 1,000,000円
  損切リスク: 総資産の2%（固定ポジションサイズ）
  半利確: +1Rで50%決済 + SLをBEへ移動
  RR比: 2.5
最低スプレッド（pips）:
  USDJPY=0.42, EURUSD=0.04, GBPUSD=0.13, AUDUSD=0.01
  USDCAD=0.10, USDCHF=0.10, NZDUSD=0.10
  EURJPY=0.21, GBPJPY=0.35, EURGBP=0.50
  US30=3.0pt（ティック実測値）, SPX500=0.5pt（ティック実測値）
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

# ── 設定 ──────────────────────────────────────────────────
INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
IS_START  = "2024-07-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

MODES = ["1H", "4H", "Hybrid"]

# 最低スプレッド（Exness ゼロ口座 生スプレッド）
PAIRS = {
    "USDJPY": {"spread": 0.42, "pip": 0.01,   "sym": "usdjpy", "color": "#ef4444"},
    "EURUSD": {"spread": 0.04, "pip": 0.0001, "sym": "eurusd", "color": "#f97316"},
    "GBPUSD": {"spread": 0.13, "pip": 0.0001, "sym": "gbpusd", "color": "#eab308"},
    "AUDUSD": {"spread": 0.01, "pip": 0.0001, "sym": "audusd", "color": "#22c55e"},
    "USDCAD": {"spread": 0.10, "pip": 0.0001, "sym": "usdcad", "color": "#14b8a6"},
    "USDCHF": {"spread": 0.10, "pip": 0.0001, "sym": "usdchf", "color": "#3b82f6"},
    "NZDUSD": {"spread": 0.10, "pip": 0.0001, "sym": "nzdusd", "color": "#8b5cf6"},
    "EURJPY": {"spread": 0.21, "pip": 0.01,   "sym": "eurjpy", "color": "#ec4899"},
    "GBPJPY": {"spread": 0.35, "pip": 0.01,   "sym": "gbpjpy", "color": "#f43f5e"},
    "EURGBP": {"spread": 0.50, "pip": 0.0001, "sym": "eurgbp", "color": "#a855f7"},
    "US30":   {"spread": 3.0,  "pip": 1.0,    "sym": "us30",   "color": "#f59e0b"},
    "SPX500": {"spread": 0.5,  "pip": 0.1,    "sym": "spx500", "color": "#06b6d4"},
}

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

def make_4h_from_1h(data_1h):
    """1H足から4H足を生成"""
    return pd.DataFrame({
        "open":  data_1h["open"].resample("4h").first(),
        "high":  data_1h["high"].resample("4h").max(),
        "low":   data_1h["low"].resample("4h").min(),
        "close": data_1h["close"].resample("4h").last(),
    }).dropna(subset=["open","close"])

# ── 共通ユーティリティ ─────────────────────────────────────
def add_ema_atr(df, span=20, atr_period=14):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1))
        )
    )
    df["atr"] = tr.rolling(atr_period).mean()
    return df

def detect_double_bottom(bars, i, atr_tol=0.3):
    """直近5本の中で二番底パターンを検出。ロングシグナルを返す"""
    if i < 4:
        return None
    recent = bars.iloc[i-4:i+1]
    lows = recent["low"].values
    min_idx = np.argmin(lows)
    if min_idx == 0 or min_idx == 4:
        return None
    # 二番底: 最安値の前後に高値があり、最安値付近に2つの底がある
    first_low  = lows[:min_idx].min()
    second_low = lows[min_idx+1:].min() if min_idx < 4 else None
    if second_low is None:
        return None
    atr = recent["atr"].iloc[-1] if "atr" in recent.columns else None
    if atr is None or atr <= 0:
        return None
    if abs(first_low - second_low) <= atr * atr_tol:
        return min(first_low, second_low)
    return None

def detect_double_top(bars, i, atr_tol=0.3):
    """直近5本の中で二番天井パターンを検出。ショートシグナルを返す"""
    if i < 4:
        return None
    recent = bars.iloc[i-4:i+1]
    highs = recent["high"].values
    max_idx = np.argmax(highs)
    if max_idx == 0 or max_idx == 4:
        return None
    first_high  = highs[:max_idx].max()
    second_high = highs[max_idx+1:].max() if max_idx < 4 else None
    if second_high is None:
        return None
    atr = recent["atr"].iloc[-1] if "atr" in recent.columns else None
    if atr is None or atr <= 0:
        return None
    if abs(first_high - second_high) <= atr * atr_tol:
        return max(first_high, second_high)
    return None

# ── シグナル生成: 1Hベース ────────────────────────────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    signals = []
    # 1H足を15分足から生成
    data_1h = pd.DataFrame({
        "open":  data_15m["open"].resample("1h").first(),
        "high":  data_15m["high"].resample("1h").max(),
        "low":   data_15m["low"].resample("1h").min(),
        "close": data_15m["close"].resample("1h").last(),
    }).dropna(subset=["open","close"])
    data_4h = add_ema_atr(data_4h)
    data_1h = add_ema_atr(data_1h)
    spread = spread_pips * pip_size
    h1_bars = data_1h.reset_index()
    for i in range(5, len(h1_bars)):
        bar = h1_bars.iloc[i]
        bar_time = bar["timestamp"]
        # 4H足のトレンド判定
        prev_4h = data_4h[data_4h.index < bar_time]
        if len(prev_4h) < 2:
            continue
        last_4h = prev_4h.iloc[-1]
        if pd.isna(last_4h["ema20"]) or pd.isna(last_4h["atr"]):
            continue
        trend_up   = last_4h["close"] > last_4h["ema20"]
        trend_down = last_4h["close"] < last_4h["ema20"]
        atr_4h = last_4h["atr"]
        # 1H足ATR
        atr_1h = bar["atr"] if not pd.isna(bar.get("atr", np.nan)) else atr_4h
        # KMID: 直前4H足の実体方向
        prev_4h2 = prev_4h.iloc[-1]
        kmid_bull = prev_4h2["close"] >= prev_4h2["open"]
        kmid_bear = prev_4h2["close"] < prev_4h2["open"]
        # KLOW: 直前4H足の下ヒゲ比率 < 0.15%
        body_low = min(prev_4h2["open"], prev_4h2["close"])
        klow_ok = (body_low - prev_4h2["low"]) / prev_4h2["open"] < 0.0015 if prev_4h2["open"] > 0 else True
        # 1H足でパターン検出
        recent_1h = h1_bars.iloc[max(0,i-4):i+1].copy()
        recent_1h = recent_1h.merge(
            data_1h[["atr"]].reset_index(), on="timestamp", how="left"
        ) if "atr" not in recent_1h.columns else recent_1h
        # ロング: 二番底
        if trend_up and kmid_bull and klow_ok:
            bottom = detect_double_bottom(recent_1h.set_index("timestamp"), len(recent_1h)-1, atr_tol=0.3)
            if bottom is not None:
                confirm = bar["close"] > bar["open"]  # 確認足が陽線
                if confirm:
                    sl_raw = bottom - atr_4h * 0.15
                    risk = bar["close"] + spread - sl_raw
                    if 0 < risk <= atr_4h * 3:
                        ep = bar["close"] + spread
                        tp = ep + risk * rr_ratio
                        signals.append({"time": bar_time, "direction": "long",
                                        "ep": ep, "sl": sl_raw, "tp": tp, "risk": risk})
        # ショート: 二番天井
        if trend_down and kmid_bear:
            top = detect_double_top(recent_1h.set_index("timestamp"), len(recent_1h)-1, atr_tol=0.3)
            if top is not None:
                confirm = bar["close"] < bar["open"]  # 確認足が陰線
                if confirm:
                    sl_raw = top + atr_4h * 0.15
                    risk = sl_raw - (bar["close"] - spread)
                    if 0 < risk <= atr_4h * 3:
                        ep = bar["close"] - spread
                        tp = ep - risk * rr_ratio
                        signals.append({"time": bar_time, "direction": "short",
                                        "ep": ep, "sl": sl_raw, "tp": tp, "risk": risk})
    return pd.DataFrame(signals)

# ── シグナル生成: 4Hベース ────────────────────────────────
def generate_signals_4h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    signals = []
    data_4h = add_ema_atr(data_4h)
    spread = spread_pips * pip_size
    bars_4h = data_4h.reset_index()
    for i in range(5, len(bars_4h)):
        bar = bars_4h.iloc[i]
        bar_time = bar["timestamp"]
        if pd.isna(bar["ema20"]) or pd.isna(bar["atr"]):
            continue
        trend_up   = bar["close"] > bar["ema20"]
        trend_down = bar["close"] < bar["ema20"]
        atr_4h = bar["atr"]
        # KMID / KLOW（直前4H足）
        prev_bar = bars_4h.iloc[i-1]
        kmid_bull = prev_bar["close"] >= prev_bar["open"]
        kmid_bear = prev_bar["close"] < prev_bar["open"]
        body_low = min(prev_bar["open"], prev_bar["close"])
        klow_ok = (body_low - prev_bar["low"]) / prev_bar["open"] < 0.0015 if prev_bar["open"] > 0 else True
        recent_4h = bars_4h.iloc[max(0,i-4):i+1].set_index("timestamp")
        # ロング
        if trend_up and kmid_bull and klow_ok:
            bottom = detect_double_bottom(recent_4h, len(recent_4h)-1, atr_tol=0.3)
            if bottom is not None and bar["close"] > bar["open"]:
                sl_raw = bottom - atr_4h * 0.15
                risk = bar["close"] + spread - sl_raw
                if 0 < risk <= atr_4h * 3:
                    ep = bar["close"] + spread
                    tp = ep + risk * rr_ratio
                    signals.append({"time": bar_time, "direction": "long",
                                    "ep": ep, "sl": sl_raw, "tp": tp, "risk": risk})
        # ショート
        if trend_down and kmid_bear:
            top = detect_double_top(recent_4h, len(recent_4h)-1, atr_tol=0.3)
            if top is not None and bar["close"] < bar["open"]:
                sl_raw = top + atr_4h * 0.15
                risk = sl_raw - (bar["close"] - spread)
                if 0 < risk <= atr_4h * 3:
                    ep = bar["close"] - spread
                    tp = ep - risk * rr_ratio
                    signals.append({"time": bar_time, "direction": "short",
                                    "ep": ep, "sl": sl_raw, "tp": tp, "risk": risk})
    return pd.DataFrame(signals)

# ── シグナル生成: ハイブリッド（1H + 4H 両方） ─────────────
def generate_signals_hybrid(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    sig_1h = generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio)
    sig_4h = generate_signals_4h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio)
    if len(sig_1h) == 0 and len(sig_4h) == 0:
        return pd.DataFrame()
    combined = pd.concat([sig_1h, sig_4h], ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time","direction"]).reset_index(drop=True)
    return combined

# ── シミュレーション（保守的アプローチ: 同一バー内はSL優先） ──────────
# 同一バーで半利確ラインとSL/TPが同時に発生した場合の処理順序:
#   ① まず現在のSL（またはBE-SL）に触れているか判定 → 触れていたら即損切
#   ② SLに触れていない場合のみTPを判定
#   ③ SL/TPどちらにも触れていない場合のみ半利確ラインを判定
# これにより「半利確→同バーでSL」という楽観的シナリオを排除し、
# 実運用より悪い結果は出ない保守的な評価が得られる。
def simulate(signals, data_1m, init_cash=1_000_000, risk_pct=0.02, half_r=1.0):
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")
    trades = []
    equity = init_cash
    for _, sig in signals.iterrows():
        direction = sig["direction"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        risk_amt = init_cash * risk_pct
        lot_size = risk_amt / risk if risk > 0 else 0
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0:
            continue
        half_done = False; be_sl = None; result = None
        exit_price = None; exit_time = None
        for bar_time, bar in future.iterrows():
            if direction == "long":
                current_sl = be_sl if half_done else sl

                # ① SL判定を最優先（保守的: 同一バーでSLと半利確が重なる場合はSLを優先）
                if bar["low"] <= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (exit_price - ep) * lot_size * remaining
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break

                # ② TP判定
                if bar["high"] >= tp:
                    # 同一バーで半利確ラインも超えていた場合は半利確→TP完結として処理
                    if not half_done and bar["high"] >= ep + risk * half_r:
                        equity += risk * half_r * lot_size * 0.5  # 半利確分 +0.5R
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    equity += (exit_price - ep) * lot_size * remaining
                    result = "win"; break

                # ③ 半利確チェック（SL/TPどちらにも触れなかった場合のみ）
                if not half_done and bar["high"] >= ep + risk * half_r:
                    half_done = True; be_sl = ep
                    equity += risk * half_r * lot_size * 0.5  # +0.5R

            else:  # short
                current_sl = be_sl if half_done else sl

                # ① SL判定を最優先
                if bar["high"] >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    pnl = (ep - exit_price) * lot_size * remaining
                    equity += pnl; result = "win" if pnl > 0 else "loss"; break

                # ② TP判定
                if bar["low"] <= tp:
                    # 同一バーで半利確ラインも下抜けていた場合は半利確→TP完結として処理
                    if not half_done and bar["low"] <= ep - risk * half_r:
                        equity += risk * half_r * lot_size * 0.5  # 半利確分 +0.5R
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining = 0.5 if half_done else 1.0
                    equity += (ep - exit_price) * lot_size * remaining
                    result = "win"; break

                # ③ 半利確チェック（SL/TPどちらにも触れなかった場合のみ）
                if not half_done and bar["low"] <= ep - risk * half_r:
                    half_done = True; be_sl = ep
                    equity += risk * half_r * lot_size * 0.5  # +0.5R

        if result is None:
            continue
        trades.append({"entry_time": sig["time"], "exit_time": exit_time,
                        "direction": direction, "ep": ep, "sl": sl, "tp": tp,
                        "exit_price": exit_price, "result": result, "equity": equity})
    if not trades:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")
    df_trades = pd.DataFrame(trades)
    eq_series = pd.Series([init_cash] + df_trades["equity"].tolist(), name="equity")
    return df_trades, eq_series

# ── 統計計算 ──────────────────────────────────────────────
def calc_stats(trades, eq_series, label):
    if len(trades) == 0:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0,
                "return_pct": 0, "mdd_pct": 0, "kelly": 0, "monthly_plus": "N/A"}
    wins  = trades[trades["result"] == "win"]
    loses = trades[trades["result"] == "loss"]
    n = len(trades)
    wr = len(wins) / n
    gross_win  = (wins["exit_price"]  - wins["ep"]).abs().sum()  if len(wins)  > 0 else 0
    gross_loss = (loses["exit_price"] - loses["ep"]).abs().sum() if len(loses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    eq = eq_series.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = dd.min()
    ret = (eq[-1] - eq[0]) / eq[0]
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    # 月次プラス率
    if len(trades) > 0 and "exit_time" in trades.columns:
        trades2 = trades.copy()
        trades2["exit_time"] = pd.to_datetime(trades2["exit_time"], utc=True)
        trades2["month"] = trades2["exit_time"].dt.to_period("M")
        monthly = trades2.groupby("month")["equity"].last()
        monthly_shifted = monthly.shift(1).fillna(INIT_CASH)
        monthly_plus = (monthly > monthly_shifted).sum()
        monthly_total = len(monthly)
        monthly_str = f"{monthly_plus}/{monthly_total}"
    else:
        monthly_str = "N/A"
    return {
        "label": label, "n": n,
        "winrate": wr * 100,
        "pf": pf,
        "return_pct": ret * 100,
        "return_abs": eq[-1] - eq[0],
        "mdd_pct": abs(mdd) * 100,
        "kelly": kelly,
        "monthly_plus": monthly_str,
    }

# ── メイン処理 ────────────────────────────────────────────
print("=" * 80)
print("v77 全12銘柄 × 3モード（1H/4H/Hybrid）バックテスト")
print(f"IS: {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%  RR: {RR_RATIO}  スプレッド: 最低値")
print("=" * 80)

all_results = []
eq_curves = {}  # {pair: {mode: {"IS": eq, "OOS": eq}}}

sig_funcs = {
    "1H":     generate_signals_1h,
    "4H":     generate_signals_4h,
    "Hybrid": generate_signals_hybrid,
}

for pair, cfg in PAIRS.items():
    sym    = cfg["sym"]
    spread = cfg["spread"]
    pip    = cfg["pip"]
    print(f"\n{'='*60}")
    print(f"  {pair}  スプレッド: {spread}pips")
    print(f"{'='*60}")

    # データ読み込み
    d1m_is   = load_csv(os.path.join(DATA_DIR, f"{sym}_is_1m.csv"))
    d15m_is  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv"))
    d4h_is   = load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv"))
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

    for mode, sig_fn in sig_funcs.items():
        eq_curves[pair][mode] = {}
        for period, d1m, d15m, d4h in [
            ("IS",  d1m_is,  d15m_is,  d4h_is),
            ("OOS", d1m_oos, d15m_oos, d4h_oos),
        ]:
            sigs = sig_fn(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
            trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH,
                                  risk_pct=RISK_PCT, half_r=HALF_R)
            stats = calc_stats(trades, eq, f"{pair}_{mode}_{period}")
            stats.update({"pair": pair, "mode": mode, "period": period, "spread": spread})
            all_results.append(stats)
            eq_curves[pair][mode][period] = eq
            print(f"  [{mode}][{period}] {stats['n']}件 | 勝率{stats['winrate']:.1f}% | "
                  f"PF{stats['pf']:.2f} | リターン+{stats['return_pct']:.1f}% | "
                  f"MDD{stats['mdd_pct']:.1f}% | ケリー{stats['kelly']:.3f} | "
                  f"月次+{stats['monthly_plus']}")

# ── 結果CSV保存 ───────────────────────────────────────────
df_results = pd.DataFrame(all_results)
csv_path = os.path.join(OUT_DIR, "v77_3modes_results_v2.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n結果CSV保存: {csv_path}")

# ── 可視化1: モード別OOS PF比較チャート ───────────────────
pairs_list = list(PAIRS.keys())
n_pairs = len(pairs_list)
mode_colors = {"1H": "#3b82f6", "4H": "#f97316", "Hybrid": "#22c55e"}

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("v77 全12銘柄 × 3モード バックテスト比較\n（最低スプレッド / 初期資金100万円 / リスク2% / RR2.5）",
             fontsize=13, fontweight="bold", y=0.98)

x = np.arange(n_pairs)
w = 0.25

for ax_idx, (metric, ylabel, title, hline) in enumerate([
    ("pf",          "プロフィットファクター", "OOS PF比較（モード別）",       1.5),
    ("mdd_pct",     "最大ドローダウン（%）",  "OOS MDD比較（モード別）",      20.0),
    ("winrate",     "勝率（%）",              "OOS 勝率比較（モード別）",     50.0),
    ("kelly",       "ケリー係数",             "OOS ケリー係数比較（モード別）", 0.3),
]):
    ax = axes[ax_idx // 2][ax_idx % 2]
    for m_idx, mode in enumerate(MODES):
        vals = []
        for pair in pairs_list:
            row = next((r for r in all_results
                        if r["pair"] == pair and r["mode"] == mode and r["period"] == "OOS"), None)
            vals.append(row[metric] if row else 0)
        ax.bar(x + (m_idx - 1) * w, vals, w,
               label=mode, color=mode_colors[mode], alpha=0.85)
    ax.axhline(hline, color="red", linestyle="--", lw=1, alpha=0.6,
               label=f"基準={hline}")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
chart1_path = os.path.join(OUT_DIR, "v77_3modes_oos_comparison_v2.png")
plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート1保存: {chart1_path}")

# ── 可視化2: スコアカード（OOS PF順） ─────────────────────
fig2, ax2 = plt.subplots(figsize=(22, 14))
ax2.axis("off")

col_labels = ["銘柄", "モード", "スプレッド",
              "IS勝率", "IS PF", "IS MDD", "IS月次+",
              "OOS勝率", "OOS PF", "OOS MDD", "OOS月次+", "OOSケリー", "評価"]

def grade(oos_pf, oos_mdd, oos_kelly):
    if oos_pf >= 3.0 and oos_mdd <= 15 and oos_kelly >= 0.5:
        return "S"
    elif oos_pf >= 2.0 and oos_mdd <= 25 and oos_kelly >= 0.3:
        return "A"
    elif oos_pf >= 1.5 and oos_mdd <= 35:
        return "B"
    elif oos_pf >= 1.0:
        return "C"
    else:
        return "D"

grade_colors = {"S": "#2ecc71", "A": "#3498db", "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c"}

table_data = []
for r_oos in all_results:
    if r_oos["period"] != "OOS":
        continue
    r_is = next((r for r in all_results
                 if r["pair"] == r_oos["pair"] and r["mode"] == r_oos["mode"] and r["period"] == "IS"), None)
    g = grade(r_oos["pf"], r_oos["mdd_pct"], r_oos["kelly"])
    table_data.append([
        r_oos["pair"],
        r_oos["mode"],
        f'{r_oos["spread"]}pips',
        f'{r_is["winrate"]:.1f}%'  if r_is else "—",
        f'{r_is["pf"]:.2f}'        if r_is else "—",
        f'{r_is["mdd_pct"]:.1f}%'  if r_is else "—",
        r_is["monthly_plus"]       if r_is else "—",
        f'{r_oos["winrate"]:.1f}%',
        f'{r_oos["pf"]:.2f}',
        f'{r_oos["mdd_pct"]:.1f}%',
        r_oos["monthly_plus"],
        f'{r_oos["kelly"]:.3f}',
        g,
    ])

# OOS PFでソート
table_data.sort(key=lambda r: float(r[8]), reverse=True)

tbl = ax2.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.5)

for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

for i, row in enumerate(table_data):
    g = row[-1]
    tbl[(i+1, len(col_labels)-1)].set_facecolor(grade_colors.get(g, "white"))
    tbl[(i+1, len(col_labels)-1)].set_text_props(color="white", fontweight="bold")
    bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
    for j in range(len(col_labels)-1):
        tbl[(i+1, j)].set_facecolor(bg)
    # モード列に色付け
    mode_val = row[1]
    tbl[(i+1, 1)].set_facecolor(mode_colors.get(mode_val, "#ffffff"))
    tbl[(i+1, 1)].set_text_props(color="white", fontweight="bold")

ax2.set_title("v77 全12銘柄 × 3モード スコアカード（OOS PF順）\n最低スプレッド / 初期資金100万円 / リスク2% / RR2.5",
              fontsize=12, fontweight="bold", pad=20)

chart2_path = os.path.join(OUT_DIR, "v77_3modes_scorecard_v2.png")
plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"スコアカード保存: {chart2_path}")
print("\n全処理完了。")
