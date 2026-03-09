"""
backtest_1m_entry_accuracy.py
==============================
1分足エントリー精度向上の比較検証（YAGAMI改 v77ベース）

【検証内容】
HTFシグナル確定後の1m エントリー方法を4通り比較する。

■ E0（baseline / 現行）
    HTF足確定直後の最初の1m始値で無条件エントリー（v77/v79現行）
    ウィンドウ: HTF足確定 + 2分以内

■ E1（1m方向確認待ち）
    HTF信号後、エントリー方向の1m陽/陰線（close > open）が出たら
    次の1m足始値でエントリー。最大5分待つ。
    ロング: 最初の1m陽線の次足始値
    ショート: 最初の1m陰線の次足始値

■ E2（1mスパイクフィルター）
    E0と同じ最初の1m足でエントリーを試みるが、
    その1m足のレンジが直近10本の1m ATR × 2 を超える場合はスキップして
    次の1m足（計3分以内）を使う。
    急騰・急落・ギャップ起動後の悪いエントリーを回避。

■ E3（E1 + E2 複合）
    E1の方向確認 + E2のスパイクフィルターを組み合わせる。
    最大5分以内に条件を満たす1m足が出なければスキップ。

【対象銘柄】
    USDJPY (1mなし → 15m代用でシグナル生成のみ、エントリー比較は別銘柄で実施)
    EURUSD / GBPUSD / AUDUSD / XAUUSD（1mあり）

【期間】
    OOS: 2025-03-03 〜 2026-02-27

【設定】
    USDJPY: v77パラメータ（ADX/Streak不使用）
    FX:     v79BC（adx_min=20, streak_min=4）
    XAUUSD: v79A（use_1d_trend=True）

【過学習対策】
    エントリー方法はデータ非依存の古典TA手法のみ（パラメータなし）
    E1の待機上限5分は「HTF足が確定してから次の重要バーまでの一般的な時間」
    E2のATR×2は「通常レンジの2倍 = 明らかな異常」の業界標準的基準
"""
import os
import sys
import warnings
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

# ── 定数 ──────────────────────────────────────────────────────────
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
TOL_FACTOR  = 0.3
USDJPY_RATE = 150.0

OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

# E1: 方向確認待ち最大ウィンドウ（分）
E1_MAX_WAIT_MIN = 5
# E2: スパイク判定閾値（1m ATR × 倍率）
E2_SPIKE_ATR_MULT = 2.0
# E2: スパイク時の代替探索ウィンドウ（分）
E2_ALT_WINDOW_MIN = 3

# ── 銘柄設定 ──────────────────────────────────────────────────────
SYMBOLS = [
    {
        "name":        "EURUSD",
        "lower":       "eurusd",
        "has_1m":      True,
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "GBPUSD",
        "lower":       "gbpusd",
        "has_1m":      True,
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "AUDUSD",
        "lower":       "audusd",
        "has_1m":      True,
        "adx_min":     20,
        "streak_min":  4,
        "use_1d":      False,
        "utc_start":   7,
        "utc_end":     22,
    },
    {
        "name":        "XAUUSD",
        "lower":       "xauusd",
        "has_1m":      True,
        "adx_min":     0,
        "streak_min":  0,
        "use_1d":      True,
        "utc_start":   0,
        "utc_end":     24,
    },
]

ENTRY_METHODS = ["E0", "E1", "E2", "E3"]

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def slice_period(df, start, end):
    if df is None:
        return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def load_data(sym_lower, sym_upper):
    """IS+OOS統合データを data/ohlc/ から読み込む（OOS期間でスライス）"""
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")

    def _load(tf):
        # data/ohlc/{UPPER}_{tf}.csv を優先
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p):
            return load_csv(p)
        # フォールバック: data/{lower}_oos_{tf}.csv
        p2 = os.path.join(DATA_DIR, f"{sym_lower}_oos_{tf}.csv")
        return load_csv(p2)

    d1m  = _load("1m")
    d15m = _load("15m")
    d4h  = _load("4h")

    return d1m, d15m, d4h


# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def calc_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high - high.shift(1)
    down = low.shift(1) - low
    pdm  = np.where((up > down) & (up > 0),   up,   0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    a    = 1.0 / period
    atr_ = pd.Series(tr.values,  index=df.index).ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    mdi  = 100 * pd.Series(mdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atr_
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.ewm(alpha=a, adjust=False).mean()


def build_indicators(df4h, adx_min, use_1d):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    if adx_min > 0:
        df["adx"] = calc_adx(df, 14)
    if use_1d:
        d1 = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]  = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
        return df, d1
    return df, None


def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bot = min(o, c)
    klow_ok  = (body_bot - l) / o < KLOW_THR if o > 0 else False
    return kmid_ok and klow_ok


# ── 1m ATR（スパイクフィルター用）─────────────────────────────────
def build_1m_atr(data_1m, period=10):
    """1m足の ATR を事前計算して Series で返す"""
    return calc_atr(data_1m, period)


# ── エントリー価格決定（メソッド別）────────────────────────────────
def pick_entry_1m(data_1m, signal_time, direction, spread,
                  method, atr_1m=None):
    """
    signal_time: HTF足が確定した時刻（この時刻以降の1m足を探す）
    Returns: (entry_time, entry_price) または (None, None)
    """
    if method == "E0":
        # 現行: signal_time から2分以内の最初の1m始値
        window = data_1m[
            (data_1m.index >= signal_time) &
            (data_1m.index <  signal_time + pd.Timedelta(minutes=2))
        ]
        if len(window) == 0:
            return None, None
        bar = window.iloc[0]
        return bar.name, bar["open"] + (spread if direction == 1 else -spread)

    elif method == "E1":
        # 方向確認待ち: 最大5分、方向一致の1m陽/陰線が出たら次足始値
        window = data_1m[
            (data_1m.index >= signal_time) &
            (data_1m.index <  signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN))
        ]
        for idx, (bar_time, bar) in enumerate(window.iterrows()):
            # 方向一致の確認足を発見
            if direction == 1 and bar["close"] > bar["open"]:
                pass
            elif direction == -1 and bar["close"] < bar["open"]:
                pass
            else:
                continue
            # 次の1m足を取得
            next_bars = data_1m[data_1m.index > bar_time]
            if len(next_bars) == 0:
                return None, None
            nb = next_bars.iloc[0]
            return nb.name, nb["open"] + (spread if direction == 1 else -spread)
        return None, None

    elif method == "E2":
        # スパイクフィルター: 最初の1m足が過大ならスキップして次を試みる（3分以内）
        window = data_1m[
            (data_1m.index >= signal_time) &
            (data_1m.index <  signal_time + pd.Timedelta(minutes=max(2, E2_ALT_WINDOW_MIN)))
        ]
        for bar_time, bar in window.iterrows():
            bar_range = bar["high"] - bar["low"]
            # スパイク判定
            if atr_1m is not None and bar_time in atr_1m.index:
                atr_val = atr_1m.loc[bar_time]
                if not pd.isna(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                    continue  # スパイクのためスキップ
            return bar.name, bar["open"] + (spread if direction == 1 else -spread)
        return None, None

    elif method == "E3":
        # E1 + E2 複合: 方向確認 + スパイクフィルター
        window = data_1m[
            (data_1m.index >= signal_time) &
            (data_1m.index <  signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN))
        ]
        for bar_time, bar in window.iterrows():
            # スパイクチェック
            bar_range = bar["high"] - bar["low"]
            if atr_1m is not None and bar_time in atr_1m.index:
                atr_val = atr_1m.loc[bar_time]
                if not pd.isna(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                    continue
            # 方向確認
            if direction == 1 and bar["close"] > bar["open"]:
                pass
            elif direction == -1 and bar["close"] < bar["open"]:
                pass
            else:
                continue
            # 次の1m足でエントリー
            next_bars = data_1m[data_1m.index > bar_time]
            if len(next_bars) == 0:
                return None, None
            nb = next_bars.iloc[0]
            return nb.name, nb["open"] + (spread if direction == 1 else -spread)
        return None, None

    return None, None


# ── シグナル生成（エントリーメソッド別）──────────────────────────
def generate_signals(data_1m, data_15m, data_4h,
                     spread_pips, pip_size,
                     adx_min, streak_min, use_1d,
                     utc_start, utc_end,
                     method, atr_1m=None):
    spread = spread_pips * pip_size

    data_4h, data_1d = build_indicators(data_4h, adx_min, use_1d)

    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    data_1h["atr"] = calc_atr(data_1h, 14)

    signals   = []
    used_times = set()
    h1_times  = data_1h.index.tolist()
    min_idx   = max(2, streak_min if streak_min > 0 else 0)

    for i in range(min_idx, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if not (utc_start <= h1_ct.hour < utc_end):
            continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < max(streak_min, 2):
            continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)):
            continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        if adx_min > 0:
            adx_val = h4_latest.get("adx", 0)
            if pd.isna(adx_val) or adx_val < adx_min:
                continue

        if streak_min > 0:
            recent = h4_before["trend"].iloc[-streak_min:].values
            if not all(t == trend for t in recent):
                continue

        if use_1d and data_1d is not None:
            d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0:
                continue
            if d1_before.iloc[-1]["trend1d"] != trend:
                continue

        tol = atr_val * TOL_FACTOR

        for direction in [1, -1]:
            if trend != direction:
                continue

            if direction == 1:
                v1 = h1_prev2["low"];  v2 = h1_prev1["low"]
                conf_ok = h1_prev1["close"] > h1_prev1["open"]
            else:
                v1 = h1_prev2["high"]; v2 = h1_prev1["high"]
                conf_ok = h1_prev1["close"] < h1_prev1["open"]

            if abs(v1 - v2) > tol:
                continue
            if not conf_ok:
                continue
            if not check_kmid_klow(h4_latest, direction):
                continue

            # エントリー価格をメソッド別に決定
            et, ep = pick_entry_1m(
                data_1m, h1_ct, direction, spread, method, atr_1m
            )
            if et is None or et in used_times:
                continue

            raw = ep - spread if direction == 1 else ep + spread  # raw open
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── シミュレーション（ベクトル演算版）────────────────────────────
def _find_exit_vectorized(highs, lows, times, ep, sl, tp, risk, direction):
    """
    numpy 配列で SL/TP/半利確 の最初のヒットを検出。
    Returns: (exit_idx, exit_price, result, half_done, half_idx)
    """
    half_price = ep + direction * risk * HALF_R

    for i in range(len(highs)):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            sl_hit   = lo <= sl
            tp_hit   = h  >= tp
            half_hit = h  >= half_price
        else:
            sl_hit   = h  >= sl
            tp_hit   = lo <= tp
            half_hit = lo <= half_price

        if sl_hit:
            return i, sl, "loss", False, -1
        if tp_hit:
            return i, tp, "win", False, -1
        if half_hit:
            # 半利確発動 → SL を BE（エントリー価格）に移動して再探索
            be_sl = ep
            for j in range(i + 1, len(highs)):
                h2 = highs[j]; lo2 = lows[j]
                if direction == 1:
                    if lo2 <= be_sl:
                        return j, be_sl, "win", True, i
                    if h2 >= tp:
                        return j, tp, "win", True, i
                else:
                    if h2 >= be_sl:
                        return j, be_sl, "win", True, i
                    if lo2 <= tp:
                        return j, tp, "win", True, i
            return -1, None, None, True, i  # データ期間内に未決済

    return -1, None, None, False, -1  # 未決済


def simulate(signals, data_1m, symbol):
    if not signals:
        return [], [INIT_CASH]
    rm     = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH
    trades = []; eq_curve = [INIT_CASH]

    # 1m データをキャッシュ（バイナリサーチで高速アクセス）
    m1_times = data_1m.index
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=USDJPY_RATE)

        start_pos = m1_times.searchsorted(sig["time"], side="right")
        if start_pos >= len(m1_times):
            continue

        highs = m1_highs[start_pos:]
        lows  = m1_lows[start_pos:]
        times = m1_times[start_pos:]

        exit_i, exit_price, result, half_done, half_i = _find_exit_vectorized(
            highs, lows, times, ep, sl, tp, risk, direction
        )

        if result is None:
            continue

        # 半利確分の損益
        if half_done and half_i >= 0:
            half_ep = ep + direction * risk * HALF_R
            equity += rm.calc_pnl_jpy(direction, ep, half_ep, lot * 0.5, USDJPY_RATE, ep)
            remaining_lot = lot * 0.5
        else:
            remaining_lot = lot

        equity += rm.calc_pnl_jpy(direction, ep, exit_price, remaining_lot, USDJPY_RATE, ep)

        exit_time = times[exit_i]
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "ep": ep, "sl": sl, "tp": tp,
            "exit_price": exit_price, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve


# ── 統計計算 ──────────────────────────────────────────────────────
def calc_stats(trades, eq_curve, symbol, method):
    if not trades:
        return {
            "symbol": symbol, "method": method,
            "n": 0, "wr": 0.0, "pf": 0.0,
            "mdd_pct": 0.0, "kelly": -1.0, "monthly_plus": "0/0"
        }
    df   = pd.DataFrame(trades)
    n    = len(df)
    wr   = (df["result"] == "win").mean()

    eq  = np.array(eq_curve)
    dlt = np.diff(eq)
    gw  = dlt[dlt > 0].sum()
    gl  = abs(dlt[dlt < 0].sum())
    pf  = gw / gl if gl > 0 else float("inf")

    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq - peak) / peak).min()) * 100
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)

    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    monthly  = df.groupby(df["exit_time"].dt.to_period("M"))["equity"].last()
    prev_eq  = monthly.shift(1).fillna(INIT_CASH)
    mp       = f"{(monthly > prev_eq).sum()}/{len(monthly)}"

    return {
        "symbol":       symbol,
        "method":       method,
        "n":            n,
        "wr":           round(wr * 100, 1),
        "pf":           round(pf, 2),
        "mdd_pct":      round(mdd, 1),
        "kelly":        round(kelly, 3),
        "monthly_plus": mp,
    }


# ── メイン ───────────────────────────────────────────────────────
def main():
    all_results = []

    for sym_cfg in SYMBOLS:
        sym_name  = sym_cfg["name"]
        sym_lower = sym_cfg["lower"]
        print(f"\n{'='*60}")
        print(f"  {sym_name}")
        print(f"{'='*60}")

        d1m, d15m, d4h = load_data(sym_lower, sym_name)

        if d15m is None or d4h is None:
            print(f"  [SKIP] {sym_name}: 15m または 4h データが見つかりません")
            continue

        if d1m is None:
            print(f"  [SKIP] {sym_name}: 1m データが見つかりません")
            continue

        # OOS期間でスライス
        d1m_oos  = slice_period(d1m,  OOS_START, OOS_END)
        d15m_oos = slice_period(d15m, OOS_START, OOS_END)
        d4h_oos  = slice_period(d4h,  OOS_START, OOS_END)

        if d1m_oos is None or len(d1m_oos) == 0:
            print(f"  [SKIP] {sym_name}: OOS 1m データなし")
            continue

        print(f"  OOS 1m: {len(d1m_oos):,} bars | 4h: {len(d4h_oos):,} bars")

        # スパイクフィルター用1m ATR
        atr_1m = build_1m_atr(d1m_oos, period=10)

        cfg = SYMBOL_CONFIG.get(sym_name, {})
        spread_pips = cfg.get("spread", 0.0)
        pip_size    = cfg.get("pip", 0.0001)

        for method in ENTRY_METHODS:
            sigs = generate_signals(
                d1m_oos, d15m_oos, d4h_oos,
                spread_pips=spread_pips,
                pip_size=pip_size,
                adx_min=sym_cfg["adx_min"],
                streak_min=sym_cfg["streak_min"],
                use_1d=sym_cfg["use_1d"],
                utc_start=sym_cfg["utc_start"],
                utc_end=sym_cfg["utc_end"],
                method=method,
                atr_1m=atr_1m,
            )
            trades, eq_curve = simulate(sigs, d1m_oos, sym_name)
            stats = calc_stats(trades, eq_curve, sym_name, method)
            all_results.append(stats)

            entry_rate = f"{len(sigs)}/{len(sigs)}" if method == "E0" else f"{len(sigs)}"
            print(f"  [{method}] n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                  f"PF={stats['pf']:.2f}  MDD={stats['mdd_pct']:.1f}%  "
                  f"Kelly={stats['kelly']:.3f}  月次+={stats['monthly_plus']}")

    # ── サマリーテーブル ────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  1m エントリーメソッド比較サマリー（OOS）")
    print("="*70)

    df_res = pd.DataFrame(all_results)
    df_pivot = df_res.pivot_table(
        index="symbol", columns="method",
        values=["n", "wr", "pf", "mdd_pct", "kelly"],
        aggfunc="first"
    )

    # PF比較テーブル
    print("\n■ プロフィットファクター比較")
    pf_table = df_res.pivot_table(
        index="symbol", columns="method", values="pf", aggfunc="first"
    )
    print(pf_table.to_string())

    print("\n■ 勝率比較（%）")
    wr_table = df_res.pivot_table(
        index="symbol", columns="method", values="wr", aggfunc="first"
    )
    print(wr_table.to_string())

    print("\n■ トレード数比較")
    n_table = df_res.pivot_table(
        index="symbol", columns="method", values="n", aggfunc="first"
    )
    print(n_table.to_string())

    print("\n■ MDD比較（%）")
    mdd_table = df_res.pivot_table(
        index="symbol", columns="method", values="mdd_pct", aggfunc="first"
    )
    print(mdd_table.to_string())

    # ── 推奨メソッドまとめ ────────────────────────────────────────
    print("\n\n■ 推奨メソッド（PF最大）")
    for sym in df_res["symbol"].unique():
        sym_df = df_res[df_res["symbol"] == sym]
        best = sym_df.loc[sym_df["pf"].idxmax()]
        baseline = sym_df[sym_df["method"] == "E0"]["pf"].values[0]
        change = best["pf"] - baseline
        sign   = "+" if change >= 0 else ""
        print(f"  {sym}: 推奨={best['method']} "
              f"PF={best['pf']:.2f} (E0比: {sign}{change:.2f})")

    # ── CSV保存 ──────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "backtest_1m_entry_accuracy.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n結果保存: {out_csv}")

    # ── グラフ出力 ────────────────────────────────────────────────
    symbols_with_data = df_res["symbol"].unique()
    n_sym = len(symbols_with_data)
    if n_sym == 0:
        return

    fig, axes = plt.subplots(1, n_sym, figsize=(5 * n_sym, 5), squeeze=False)

    colors = {"E0": "gray", "E1": "steelblue", "E2": "darkorange", "E3": "green"}

    for ax_idx, sym in enumerate(symbols_with_data):
        ax = axes[0][ax_idx]
        sym_df = df_res[df_res["symbol"] == sym].set_index("method")
        for method in ENTRY_METHODS:
            if method in sym_df.index:
                pf_val = sym_df.loc[method, "pf"]
                ax.bar(method, pf_val, color=colors[method], alpha=0.8, label=method)
        ax.set_title(sym)
        ax.set_ylabel("PF (OOS)")
        ax.axhline(2.0, color="red", linestyle="--", linewidth=0.8, label="PF=2.0")
        ax.set_ylim(0, max(df_res[df_res["symbol"] == sym]["pf"].max() * 1.2, 2.5))

    plt.suptitle("1m エントリーメソッド比較 OOS PF\n"
                 "E0=現行 | E1=方向確認待ち | E2=スパイクフィルター | E3=E1+E2",
                 fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "backtest_1m_entry_accuracy.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"グラフ保存: {out_png}")


if __name__ == "__main__":
    main()
