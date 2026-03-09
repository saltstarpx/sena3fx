"""
backtest_v77_correct.py
=======================
v77本体（yagami_mtf_v77.py）のgenerate_signalsロジックを正確に移植した
全12銘柄 × 3モード（1H/4H/Hybrid）バックテスト

【前回スクリプトとの主な修正点】
1. エントリータイミング: 足終値→ 足確定後2分以内の最初の1分足始値（成行）
2. パターン検出: 直近5本の複雑な検出→ 直前2本（i-2, i-1）のシンプルな二番底/天井
3. リスク幅計算: ep（終値+スプレッド）-sl → raw_ep（1分足始値）-sl（チャートレベル）
4. TP計算: ep+risk*RR → raw_ep+risk*RR（チャートレベル）
5. 1Hリスク上限: atr_4h*3 → h4_atr*2（より厳格）
6. used_times重複防止: なし → あり（同時刻の二重エントリー防止）
7. スプレッド変換: spread_pips*pip_size（全通貨ペア対応）
8. simulate: 同一バーSL優先（保守的アプローチ）
9. シグナルキー: direction → dir（v77本体に合わせる）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# risk_manager をインポート（銘柄名を渡すだけで自動設定）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG
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
INIT_CASH  = 1_000_000
RISK_PCT   = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0          # 半利確ライン: +1R
KLOW_THR   = 0.0015       # KLOWフィルター閾値
IS_START   = "2024-07-01"
IS_END     = "2025-02-28"
OOS_START  = "2025-03-03"
OOS_END    = "2026-02-27"

# 銘柄設定は risk_manager.py の SYMBOL_CONFIG から自動取得
# 銘柄名を渡すだけで spread / pip_size / 円換算タイプが決まる
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
    "NAS100": {"sym": "nas100"},
    "XAUUSD": {"sym": "xauusd"},
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

# ── ATR / EMA計算 ─────────────────────────────────────────
def calculate_atr(df, period=14):
    high_low    = df["high"] - df["low"]
    high_close  = abs(df["high"] - df["close"].shift())
    low_close   = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

# ── KMID + KLOW フィルター（v77本体と同一） ───────────────
def check_kmid_klow(prev_4h_bar, direction):
    o = prev_4h_bar["open"]
    c = prev_4h_bar["close"]
    l = prev_4h_bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    klow_ok     = klow_ratio < KLOW_THR
    return kmid_ok and klow_ok

# ── シグナル生成: 4Hベース（v77本体と同一ロジック） ─────────
def generate_signals_4h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    signals    = []
    used_times = set()
    h4_times   = data_4h.index.tolist()

    # [BUG①修正] i=3 から開始。h4_prev3（パターン直前の文脈足）にKMIDを適用。
    # h4_prev1は確認足（既に陽線/陰線確認済み）のため、そこへのKMIDは常にTrue。
    for i in range(3, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1    = data_4h.iloc[i - 1]   # 確認足（陽線/陰線チェック対象）
        h4_prev2    = data_4h.iloc[i - 2]   # パターン1本目
        h4_prev3    = data_4h.iloc[i - 3]   # 文脈足（KMIDフィルター対象）← 修正
        h4_current  = data_4h.iloc[i]
        atr_val     = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        trend     = h4_current["trend"]
        tolerance = atr_val * 0.3

        # ロング: 二番底
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, direction=1):  # ← 修正: h4_prev3
                    continue
                sl = min(low1, low2) - atr_val * 0.15
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({"time": entry_time, "dir": 1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                            "tf": "4h"})
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                if not check_kmid_klow(h4_prev3, direction=-1):  # ← 修正: h4_prev3
                    continue
                sl = max(high1, high2) + atr_val * 0.15
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({"time": entry_time, "dir": -1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                            "tf": "4h"})
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

# ── シグナル生成: 1Hベース（v77本体と同一ロジック） ─────────
def generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread = spread_pips * pip_size
    data_4h = add_indicators(data_4h)

    # 1H足を15分足から集約
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
        h1_prev1    = data_1h.iloc[i - 1]
        h1_prev2    = data_1h.iloc[i - 2]
        h1_current  = data_1h.iloc[i]
        atr_val     = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # [BUG②修正] 完結済み4H足のみ取得（< で形成中の足を除外）
        h4_before = data_4h[data_4h.index < h1_current_time]
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
                        if 0 < risk <= h4_atr * 2:      # 1H: リスク上限 h4_atr*2
                            tp = raw_ep + risk * rr_ratio
                            signals.append({"time": entry_time, "dir": 1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                            "tf": "1h"})
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
                        if 0 < risk <= h4_atr * 2:      # 1H: リスク上限 h4_atr*2
                            tp = raw_ep - risk * rr_ratio
                            signals.append({"time": entry_time, "dir": -1,
                                            "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                            "tf": "1h"})
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

# ── シグナル生成: ハイブリッド（1H + 4H 統合） ──────────────
def generate_signals_hybrid(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    sig_1h = generate_signals_1h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio)
    sig_4h = generate_signals_4h(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio)
    if len(sig_1h) == 0 and len(sig_4h) == 0:
        return pd.DataFrame()
    combined = pd.concat([sig_1h, sig_4h], ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time","dir"]).reset_index(drop=True)
    return combined

# ── シミュレーション（保守的アプローチ: 同一バー内はSL優先） ──
# 同一バーで半利確ラインとSL/TPが同時発生した場合の処理順序:
#   ① まず現在のSL（またはBE-SL）に触れているか判定 → 触れていたら即損切
#   ② SLに触れていない場合のみTPを判定
#   ③ SL/TPどちらにも触れていない場合のみ半利確ラインを判定
def simulate(signals, data_1m, init_cash=1_000_000, risk_pct=0.02, half_r=1.0,
             symbol="USDJPY", usdjpy_1m=None):
    """
    Parameters
    ----------
    symbol : str
        銘柄名。RiskManagerが自動的に円換算ロジックを選択する。
    usdjpy_1m : pd.DataFrame or None
        USDJPY の1分足データ。Type B/C/D 銘柄のロットサイズ計算に使用。
        None の場合はデフォルトレート（150.0）を使用。
    """
    if signals is None or len(signals) == 0:
        return pd.DataFrame(), pd.Series([init_cash], name="equity")
    rm     = RiskManager(symbol, risk_pct=risk_pct)
    trades = []
    equity = init_cash
    for _, sig in signals.iterrows():
        direction = sig["dir"]   # v77本体は "dir" キー（1 or -1）
        ep   = sig["ep"]
        sl   = sig["sl"]
        tp   = sig["tp"]
        risk = sig["risk"]
        # エントリー時点の USDJPY レートを取得（Type B/C/D 用）
        usdjpy_rate = rm.get_usdjpy_rate(usdjpy_1m, sig["time"]) if usdjpy_1m is not None else 150.0
        # ロットサイズ: 総資産 × risk_pct ÷ (SL距離 × 1通貨あたり円価値)
        lot_size = rm.calc_lot(equity, risk, ep, usdjpy_rate=usdjpy_rate)
        future = data_1m[data_1m.index > sig["time"]]
        if len(future) == 0:
            continue
        half_done = False; be_sl = None; result = None
        exit_price = None; exit_time = None

        for bar_time, bar in future.iterrows():
            if direction == 1:  # ロング
                current_sl = be_sl if half_done else sl

                # ① SL判定を最優先
                if bar["low"] <= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl        = rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                   usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity    += pnl
                    result     = "win" if pnl > 0 else "loss"
                    break

                # ② TP判定
                if bar["high"] >= tp:
                    if not half_done and bar["high"] >= ep + risk * half_r:
                        equity   += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r,
                                                    lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    equity    += rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                 usdjpy_rate=usdjpy_rate, ref_price=ep)
                    result     = "win"
                    break

                # ③ 半利確チェック
                if not half_done and bar["high"] >= ep + risk * half_r:
                    half_done = True; be_sl = ep
                    equity   += rm.calc_pnl_jpy(direction, ep, ep + risk * half_r,
                                                lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)

            else:  # ショート
                current_sl = be_sl if half_done else sl

                # ① SL判定を最優先
                if bar["high"] >= current_sl:
                    exit_price = current_sl; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    pnl        = rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                 usdjpy_rate=usdjpy_rate, ref_price=ep)
                    equity    += pnl
                    result     = "win" if pnl > 0 else "loss"
                    break

                # ② TP判定
                if bar["low"] <= tp:
                    if not half_done and bar["low"] <= ep - risk * half_r:
                        equity   += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r,
                                                    lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)
                        half_done = True
                    exit_price = tp; exit_time = bar_time
                    remaining  = 0.5 if half_done else 1.0
                    equity    += rm.calc_pnl_jpy(direction, ep, exit_price, lot_size * remaining,
                                                 usdjpy_rate=usdjpy_rate, ref_price=ep)
                    result     = "win"
                    break

                # ③ 半利確チェック
                if not half_done and bar["low"] <= ep - risk * half_r:
                    half_done = True; be_sl = ep
                    equity   += rm.calc_pnl_jpy(direction, ep, ep - risk * half_r,
                                                lot_size * 0.5, usdjpy_rate=usdjpy_rate, ref_price=ep)

        if result is None:
            continue
        trades.append({"entry_time": sig["time"], "exit_time": exit_time,
                        "dir": direction, "ep": ep, "sl": sl, "tp": tp,
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
                "return_pct": 0, "return_abs": 0, "mdd_pct": 0,
                "kelly": 0, "monthly_plus": "N/A"}
    wins  = trades[trades["result"] == "win"]
    loses = trades[trades["result"] == "loss"]
    n  = len(trades)
    wr = len(wins) / n
    # PFはequityデルタベースで計算（ロング/ショート符号問題を回避）
    eq    = eq_series.values
    deltas = np.diff(eq)
    gross_win  = deltas[deltas > 0].sum()
    gross_loss = abs(deltas[deltas < 0].sum())
    pf    = gross_win / gross_loss if gross_loss > 0 else float("inf")
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    ret   = (eq[-1] - eq[0]) / eq[0]
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    # 月次プラス率
    if len(trades) > 0 and "exit_time" in trades.columns:
        trades2 = trades.copy()
        trades2["exit_time"] = pd.to_datetime(trades2["exit_time"], utc=True)
        trades2["month"] = trades2["exit_time"].dt.to_period("M")
        monthly = trades2.groupby("month")["equity"].last()
        monthly_shifted = monthly.shift(1).fillna(INIT_CASH)
        monthly_plus  = (monthly > monthly_shifted).sum()
        monthly_total = len(monthly)
        monthly_str   = f"{monthly_plus}/{monthly_total}"
    else:
        monthly_str = "N/A"
    return {
        "label": label, "n": n,
        "winrate":    wr * 100,
        "pf":         pf,
        "return_pct": ret * 100,
        "return_abs": eq[-1] - eq[0],
        "mdd_pct":    abs(mdd) * 100,
        "kelly":      kelly,
        "monthly_plus": monthly_str,
    }

# ── メイン処理 ────────────────────────────────────────────
print("=" * 80)
print("v77 全12銘柄 × 3モード（1H/4H/Hybrid）バックテスト [正式版]")
print(f"IS: {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%  RR: {RR_RATIO}  スプレッド: 最低値")
print(f"エントリー: 足確定後2分以内の最初の1分足始値  半利確: +{HALF_R}R")
print("=" * 80)

all_results = []
eq_curves   = {}

sig_funcs = {
    "1H":     generate_signals_1h,
    "4H":     generate_signals_4h,
    "Hybrid": generate_signals_hybrid,
}

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]
    rm  = RiskManager(pair, risk_pct=RISK_PCT)  # 銘柄名を渡すだけで自動設定
    spread = rm.spread_pips
    pip    = rm.pip_size
    print(f"\n{'='*60}")
    print(f"  {pair}  スプレッド: {spread}pips  タイプ: {rm.quote_type}")
    print(f"{'='*60}")

    # 1m: IS/OOSスプリット版を優先、なければ全期間1ファイルから期間スライス
    def load_1m(sym_lc, period_start, period_end):
        split_path = os.path.join(DATA_DIR, f"{sym_lc}_{'is' if period_start == IS_START else 'oos'}_1m.csv")
        full_path  = os.path.join(DATA_DIR, f"{sym_lc}_1m.csv")
        df = load_csv(split_path)
        if df is None:
            df = load_csv(full_path)
        return slice_period(df, period_start, period_end) if df is not None else None

    d1m_is   = load_1m(sym, IS_START,  IS_END)
    d15m_is  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv"))
    d4h_is   = load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv"))
    d1m_oos  = load_1m(sym, OOS_START, OOS_END)
    d15m_oos = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv"))
    d4h_oos  = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv"))

    if any(d is None or len(d) == 0 for d in [d1m_is, d15m_is, d4h_is, d1m_oos, d15m_oos, d4h_oos]):
        print(f"  [SKIP] データ不足")
        continue

    d15m_is  = slice_period(d15m_is,  IS_START,  IS_END)
    d4h_is   = slice_period(d4h_is,   IS_START,  IS_END)
    d15m_oos = slice_period(d15m_oos, OOS_START, OOS_END)
    d4h_oos  = slice_period(d4h_oos,  OOS_START, OOS_END)

    eq_curves[pair] = {}

    for mode, sig_fn in sig_funcs.items():
        eq_curves[pair][mode] = {}
        for period, d1m, d15m, d4h in [
            ("IS",  d1m_is,  d15m_is,  d4h_is),
            ("OOS", d1m_oos, d15m_oos, d4h_oos),
        ]:
            sigs   = sig_fn(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
            # USDJPY以外の銘柄はUSDJPYレートが必要なため1分足データを渡す
            usdjpy_1m = load_csv(os.path.join(DATA_DIR, "usdjpy_is_1m.csv")) \
                        if rm.quote_type != "A" and period == "IS" else \
                        load_csv(os.path.join(DATA_DIR, "usdjpy_oos_1m.csv")) \
                        if rm.quote_type != "A" else None
            if usdjpy_1m is not None:
                usdjpy_1m = slice_period(usdjpy_1m,
                                         IS_START if period == "IS" else OOS_START,
                                         IS_END   if period == "IS" else OOS_END)
            trades, eq = simulate(sigs, d1m, init_cash=INIT_CASH,
                                  risk_pct=RISK_PCT, half_r=HALF_R,
                                  symbol=pair, usdjpy_1m=usdjpy_1m)
            label  = f"{pair}_{mode}_{period}"
            stats  = calc_stats(trades, eq, label)
            stats["pair"]   = pair
            stats["mode"]   = mode
            stats["period"] = period
            stats["spread"] = spread
            all_results.append(stats)
            eq_curves[pair][mode][period] = eq
            print(f"  [{mode}][{period}] {stats['n']}件 | "
                  f"勝率{stats['winrate']:.1f}% | PF{stats['pf']:.2f} | "
                  f"リターン+{stats['return_pct']:.1f}% | "
                  f"MDD{stats['mdd_pct']:.1f}% | "
                  f"ケリー{stats['kelly']:.3f} | "
                  f"月次+{stats['monthly_plus']}")

# ── 結果CSV保存 ───────────────────────────────────────────
df_results = pd.DataFrame(all_results)
csv_path   = os.path.join(OUT_DIR, "v77_correct_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n結果CSV保存: {csv_path}")

# ── 可視化1: モード別OOS PF比較チャート ───────────────────
oos_df = df_results[df_results["period"] == "OOS"].copy()
pairs  = list(PAIRS.keys())
modes  = ["1H", "4H", "Hybrid"]
colors = {"1H": "#3b82f6", "4H": "#f97316", "Hybrid": "#22c55e"}
x      = np.arange(len(pairs))
width  = 0.25

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("v77 全12銘柄 × 3モード バックテスト比較 [正式版]\n"
             "（最低スプレッド / 初期資金100万円 / リスク2% / RR2.5 / 半利確+1R）",
             fontsize=13, fontweight="bold")

metrics = [
    ("pf",         "OOS PF比較（モード別）",     "プロフィットファクター", 1.5),
    ("mdd_pct",    "OOS MDD比較（モード別）",     "最大ドローダウン（%）",  20.0),
    ("winrate",    "OOS 勝率比較（モード別）",    "勝率（%）",              50.0),
    ("kelly",      "OOS ケリー係数比較（モード別）", "ケリー係数",           0.3),
]

for ax, (metric, title, ylabel, ref) in zip(axes.flatten(), metrics):
    for j, mode in enumerate(modes):
        vals = []
        for p in pairs:
            row = oos_df[(oos_df["pair"] == p) & (oos_df["mode"] == mode)]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        ax.bar(x + j * width, vals, width, label=mode, color=colors[mode], alpha=0.8)
    ax.axhline(ref, color="red", linestyle="--", linewidth=0.8, label=f"基準={ref}")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
chart1_path = os.path.join(OUT_DIR, "v77_correct_oos_comparison.png")
plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート1保存: {chart1_path}")

# ── 可視化2: スコアカード ─────────────────────────────────
is_df  = df_results[df_results["period"] == "IS"].copy()
oos_df = df_results[df_results["period"] == "OOS"].copy()

merged = []
for _, row in oos_df.iterrows():
    is_row = is_df[(is_df["pair"] == row["pair"]) & (is_df["mode"] == row["mode"])]
    if len(is_row) == 0:
        continue
    is_row = is_row.iloc[0]
    merged.append({
        "pair":         row["pair"],
        "mode":         row["mode"],
        "spread":       f"{row['spread']}pips",
        "is_wr":        f"{is_row['winrate']:.1f}%",
        "is_pf":        f"{is_row['pf']:.2f}",
        "is_mdd":       f"{is_row['mdd_pct']:.1f}%",
        "is_monthly":   is_row["monthly_plus"],
        "oos_wr":       f"{row['winrate']:.1f}%",
        "oos_pf":       f"{row['pf']:.2f}",
        "oos_mdd":      f"{row['mdd_pct']:.1f}%",
        "oos_monthly":  row["monthly_plus"],
        "oos_kelly":    f"{row['kelly']:.3f}",
        "oos_pf_val":   row["pf"],
        "oos_kelly_val": row["kelly"],
    })

merged_df = pd.DataFrame(merged).sort_values("oos_pf_val", ascending=False).reset_index(drop=True)

def grade(pf, kelly):
    if pf >= 2.0 and kelly >= 0.3:  return "A"
    if pf >= 1.5:                    return "B"
    if pf >= 1.0:                    return "C"
    return "D"

merged_df["grade"] = merged_df.apply(
    lambda r: grade(r["oos_pf_val"], r["oos_kelly_val"]), axis=1)

grade_colors = {"A": "#16a34a", "B": "#2563eb", "C": "#d97706", "D": "#dc2626"}
mode_colors  = {"1H": "#3b82f6", "4H": "#f97316", "Hybrid": "#22c55e"}

fig2, ax2 = plt.subplots(figsize=(20, len(merged_df) * 0.45 + 2.5))
ax2.axis("off")
cols = ["銘柄","モード","スプレッド",
        "IS勝率","IS PF","IS MDD","IS月次+",
        "OOS勝率","OOS PF","OOS MDD","OOS月次+","OOSケリー","評価"]
col_widths = [0.07, 0.06, 0.07,
              0.06, 0.05, 0.06, 0.06,
              0.06, 0.05, 0.06, 0.06, 0.07, 0.05]
table_data = []
for _, r in merged_df.iterrows():
    table_data.append([
        r["pair"], r["mode"], r["spread"],
        r["is_wr"], r["is_pf"], r["is_mdd"], r["is_monthly"],
        r["oos_wr"], r["oos_pf"], r["oos_mdd"], r["oos_monthly"],
        r["oos_kelly"], r["grade"],
    ])

tbl = ax2.table(
    cellText=table_data,
    colLabels=cols,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.4)

for j in range(len(cols)):
    tbl[(0, j)].set_facecolor("#1e293b")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

for i, (_, r) in enumerate(merged_df.iterrows(), start=1):
    tbl[(i, 1)].set_facecolor(mode_colors.get(r["mode"], "#ffffff"))
    tbl[(i, 1)].set_text_props(color="white", fontweight="bold")
    gc = grade_colors.get(r["grade"], "#ffffff")
    tbl[(i, 12)].set_facecolor(gc)
    tbl[(i, 12)].set_text_props(color="white", fontweight="bold")
    bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
    for j in range(len(cols)):
        if j not in [1, 12]:
            tbl[(i, j)].set_facecolor(bg)

ax2.set_title("v77 全12銘柄 × 3モード スコアカード（OOS PF順）[正式版]\n"
              "最低スプレッド / 初期資金100万円 / リスク2% / RR2.5 / 半利確+1R",
              fontsize=12, fontweight="bold", pad=20)

chart2_path = os.path.join(OUT_DIR, "v77_correct_scorecard.png")
plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"スコアカード保存: {chart2_path}")
print("\n全処理完了。")
