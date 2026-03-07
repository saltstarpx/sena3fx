"""
候補ファクターの個別バックテスト検証
====================================
スクリーニングで発見した有望ファクターを1つずつv76に追加し、
トレード数を大幅に減らさずPF・勝率を改善できるものを探す。

発見した有望ファクター:
1. KMID (ロングIC=+0.40★, ショートIC=-0.40★) → 方向別に陽線/陰線の実体比率
2. KSFT (全体IC=-0.12★, ロングIC=+0.28★, ショートIC=-0.40★) → 足の重心位置
3. RSV5 (ロングIC=+0.25★, ショートIC=-0.25★) → 5本ストキャスティクス
4. KLOW (全体IC=-0.15★) → 下ヒゲ比率
5. RSQR20 (全体IC=-0.12, p=0.057) → トレンド線形性（ボーダーライン）
6. ADX14 (ショートIC=-0.18) → トレンド強度

弱点分析の発見:
- アジア時間ロングが弱い (WR=50%, avg_pnl=5.77)
- 月曜日が弱い (WR=52%)
- 30分以内の決済 → ほぼ負け (WR=35%)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SPREAD = 0.4
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31", tz="UTC")

def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# ── データ ──
print("データ読み込み中...")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h = load(f"{DATA}/usdjpy_is_4h.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h = load(f"{DATA}/usdjpy_oos_4h.csv")
data_15m = pd.concat([is_15m, oos_15m]).sort_index()
data_15m = data_15m[~data_15m.index.duplicated(keep="first")]
data_4h = pd.concat([is_4h, oos_4h]).sort_index()
data_4h = data_4h[~data_4h.index.duplicated(keep="first")]

# ── v76シグナル生成 ──
from strategies.current.yagami_mtf_v76 import generate_signals
print("v76シグナル生成中...")
signals = generate_signals(data_15m, data_15m, data_4h, spread_pips=SPREAD)
sig_map = {s["time"]: s for s in signals}

# ── 4Hファクター事前計算 ──
print("4Hファクター計算中...")
c = data_4h["close"]
h = data_4h["high"]
l = data_4h["low"]
o = data_4h["open"]
v = data_4h["volume"]

factors_4h = pd.DataFrame(index=data_4h.index)

# KMID: 実体比率
factors_4h["KMID"] = (c - o) / (o + 1e-12)
# KSFT: 足の重心
factors_4h["KSFT"] = (2*c - h - l) / (o + 1e-12)
# KLOW: 下ヒゲ比率
factors_4h["KLOW"] = (np.minimum(o, c) - l) / (o + 1e-12)
# KUP: 上ヒゲ比率
factors_4h["KUP"] = (h - np.maximum(o, c)) / (o + 1e-12)
# RSV5: ストキャスティクス
min_l5 = l.rolling(5).min()
max_h5 = h.rolling(5).max()
factors_4h["RSV5"] = (c - min_l5) / (max_h5 - min_l5 + 1e-12)
# RSV10
min_l10 = l.rolling(10).min()
max_h10 = h.rolling(10).max()
factors_4h["RSV10"] = (c - min_l10) / (max_h10 - min_l10 + 1e-12)
# RSQR20: トレンド線形性
def rsquare(x):
    if len(x) < 3: return np.nan
    y = np.arange(len(x))
    _, _, r, _, _ = stats.linregress(y, x)
    return r ** 2
factors_4h["RSQR20"] = c.rolling(20).apply(rsquare, raw=True)
# ADX14
plus_dm = h.diff()
minus_dm = -l.diff()
plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
atr14 = tr.rolling(14).mean()
plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-12)
minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-12)
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
factors_4h["ADX14"] = dx.rolling(14).mean()
# SUMP14 (RSI的)
diff = c - c.shift(1)
gain = diff.clip(lower=0)
loss_s = (-diff).clip(lower=0)
factors_4h["SUMP14"] = gain.rolling(14).sum() / (gain.rolling(14).sum() + loss_s.rolling(14).sum() + 1e-12)
# EMA乖離率
ema20 = c.ewm(span=20, adjust=False).mean()
factors_4h["EMA_DEV20"] = (c - ema20) / (ema20 + 1e-12)
# BB位置
ma20 = c.rolling(20).mean()
std20 = c.rolling(20).std()
factors_4h["BB_POS20"] = (c - (ma20 - 2*std20)) / (4*std20 + 1e-12)
# ROC5
factors_4h["ROC5"] = c / c.shift(5) - 1


def get_factor_at_time(t):
    """エントリー時点の4Hファクター値を取得"""
    h4_before = factors_4h[factors_4h.index <= t]
    if len(h4_before) == 0:
        return None
    return h4_before.iloc[-1]


def run_backtest_with_filter(filter_func, label=""):
    """
    v76シグナルにフィルターを適用してバックテスト。
    filter_func(signal, factor_values) → True=エントリー / False=スキップ
    """
    trades = []
    pos = None
    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name
        if pos is not None:
            d = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + sl_pnl
                trades.append({"pnl": total, "dir": pos["dir"], "time": pos["entry_time"]})
                pos = None; continue
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + tp_pnl
                trades.append({"pnl": total, "dir": pos["dir"], "time": pos["entry_time"]})
                pos = None; continue
        if pos is None and t in sig_map:
            if START <= t <= END:
                sig = sig_map[t]
                fv = get_factor_at_time(t)
                if fv is not None and filter_func(sig, fv):
                    pos = {**sig, "entry_time": t, "half_closed": False}
    return pd.DataFrame(trades)


def calc_stats(df_trades):
    """統計計算"""
    if df_trades.empty:
        return {"trades": 0, "wr": 0, "pf": 0, "total_pnl": 0, "avg_pnl": 0, "p_value": 1}
    wins = df_trades[df_trades["pnl"] > 0]
    losses = df_trades[df_trades["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
    wr = len(wins) / len(df_trades) * 100
    t_stat, p_val = stats.ttest_1samp(df_trades["pnl"], 0) if len(df_trades) >= 2 else (0, 1)
    monthly = df_trades.set_index("time").resample("ME")["pnl"].sum()
    plus_m = (monthly > 0).sum()
    total_m = len(monthly[monthly != 0])
    cumsum = df_trades["pnl"].cumsum()
    mdd = (cumsum.cummax() - cumsum).max()
    return {
        "trades": len(df_trades),
        "wr": wr,
        "pf": pf,
        "total_pnl": df_trades["pnl"].sum(),
        "avg_pnl": df_trades["pnl"].mean(),
        "p_value": p_val,
        "plus_months": f"{plus_m}/{total_m}",
        "mdd": mdd,
    }


# ══════════════════════════════════════════════════════════════════
# フィルター定義
# ══════════════════════════════════════════════════════════════════

filters = {}

# ベースライン: フィルターなし
filters["v76_baseline"] = lambda sig, fv: True

# ── ファクターA: KMID方向一致フィルター ──
# ロング時は陽線（KMID>0）、ショート時は陰線（KMID<0）を要求
filters["A_KMID_align"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["KMID"] > 0) or
    (sig["dir"] == -1 and fv["KMID"] < 0)
)

# ── ファクターB: KSFT方向一致 ──
# ロング: 足の重心が上（KSFT>0）、ショート: 重心が下
filters["B_KSFT_align"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["KSFT"] > 0) or
    (sig["dir"] == -1 and fv["KSFT"] < 0)
)

# ── ファクターC: RSV5 方向一致 ──
# ロング: RSV5 > 0.3 (上昇中)、ショート: RSV5 < 0.7 (下降中)
filters["C_RSV5_dir"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["RSV5"] > 0.3) or
    (sig["dir"] == -1 and fv["RSV5"] < 0.7)
)

# ── ファクターD: RSV5 逆張り回避 ──
# ロング: RSV5 > 0.2 (売られすぎからの反発OK、ただし底なし除外)
# ショート: RSV5 < 0.8
filters["D_RSV5_extreme"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["RSV5"] > 0.15) or
    (sig["dir"] == -1 and fv["RSV5"] < 0.85)
)

# ── ファクターE: KLOW低い方が勝ち → 下ヒゲ小さい時のみ ──
filters["E_KLOW_small"] = lambda sig, fv: fv["KLOW"] < 0.0015

# ── ファクターF: ADX弱トレンド回避 ──
# ADXが低すぎる = レンジ相場 → エントリー見送り
filters["F_ADX_min"] = lambda sig, fv: not pd.isna(fv["ADX14"]) and fv["ADX14"] > 20

# ── ファクターG: RSQR20低い方が勝ち → トレンドが直線的すぎる時は回避 ──
filters["G_RSQR20_low"] = lambda sig, fv: not pd.isna(fv["RSQR20"]) and fv["RSQR20"] < 0.6

# ── ファクターH: SUMP14 (RSI的) 方向一致 ──
filters["H_SUMP14_align"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["SUMP14"] > 0.45) or
    (sig["dir"] == -1 and fv["SUMP14"] < 0.55)
)

# ── ファクターI: BB位置 方向一致 ──
filters["I_BB_POS_align"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["BB_POS20"] > 0.3) or
    (sig["dir"] == -1 and fv["BB_POS20"] < 0.7)
) if not pd.isna(fv["BB_POS20"]) else True

# ── ファクターJ: 複合 KMID + RSV5 ──
filters["J_KMID_RSV5"] = lambda sig, fv: (
    ((sig["dir"] == 1 and fv["KMID"] > 0) or (sig["dir"] == -1 and fv["KMID"] < 0)) and
    ((sig["dir"] == 1 and fv["RSV5"] > 0.2) or (sig["dir"] == -1 and fv["RSV5"] < 0.8))
)

# ── ファクターK: 複合 KMID + KSFT ──
filters["K_KMID_KSFT"] = lambda sig, fv: (
    ((sig["dir"] == 1 and fv["KMID"] > 0) or (sig["dir"] == -1 and fv["KMID"] < 0)) and
    ((sig["dir"] == 1 and fv["KSFT"] > 0) or (sig["dir"] == -1 and fv["KSFT"] < 0))
)

# ── ファクターL: 複合 KMID + KLOW ──
filters["L_KMID_KLOW"] = lambda sig, fv: (
    ((sig["dir"] == 1 and fv["KMID"] > 0) or (sig["dir"] == -1 and fv["KMID"] < 0)) and
    fv["KLOW"] < 0.0015
)

# ── ファクターM: KMID緩め（中立含む） ──
filters["M_KMID_loose"] = lambda sig, fv: (
    (sig["dir"] == 1 and fv["KMID"] > -0.001) or
    (sig["dir"] == -1 and fv["KMID"] < 0.001)
)


# ══════════════════════════════════════════════════════════════════
# 全フィルターでバックテスト実行
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  候補ファクター個別バックテスト結果")
print("=" * 80)
print(f"  {'フィルター':<20s} {'トレ数':>6s} {'勝率':>6s} {'PF':>7s} {'総損益':>10s} {'平均損益':>8s} {'MDD':>8s} {'p値':>8s} {'月+':>6s}")
print(f"  {'─'*76}")

all_results = {}
for name, filt in filters.items():
    df = run_backtest_with_filter(filt, name)
    s = calc_stats(df)
    all_results[name] = s
    sig = "★" if s["p_value"] < 0.05 else " "
    print(f"  {name:<20s} {s['trades']:>6d} {s['wr']:>5.1f}% {s['pf']:>7.3f} {s['total_pnl']:>+10.1f} {s['avg_pnl']:>+8.1f} {s['mdd']:>8.1f} {s['p_value']:>7.4f}{sig} {s['plus_months']:>6s}")

# ── v76対比サマリー ──
base = all_results["v76_baseline"]
print(f"\n{'='*80}")
print(f"  v76ベースライン対比")
print(f"{'='*80}")
print(f"  {'フィルター':<20s} {'Δトレ':>6s} {'ΔWR':>7s} {'ΔPF':>8s} {'Δ総損益':>10s} {'Δ平均損益':>10s} {'採用':>6s}")
print(f"  {'─'*68}")

for name, s in all_results.items():
    if name == "v76_baseline":
        continue
    d_trades = s["trades"] - base["trades"]
    d_wr = s["wr"] - base["wr"]
    d_pf = s["pf"] - base["pf"]
    d_total = s["total_pnl"] - base["total_pnl"]
    d_avg = s["avg_pnl"] - base["avg_pnl"]

    # 採用基準: PF改善 & トレード数が50%以上維持 & 平均損益改善
    adopt = "○" if (d_pf > 0 and s["trades"] >= base["trades"] * 0.5 and d_avg > 0) else "×"
    if d_pf > 0.2 and s["trades"] >= base["trades"] * 0.4 and d_avg > 5:
        adopt = "◎"

    print(f"  {name:<20s} {d_trades:>+6d} {d_wr:>+6.1f}% {d_pf:>+8.3f} {d_total:>+10.1f} {d_avg:>+10.1f} {adopt:>6s}")

# ── 最終推奨 ──
print(f"\n{'='*80}")
print(f"  最終推奨")
print(f"{'='*80}")

# PF改善かつトレード数50%以上の中で、PF×平均損益が最大のものを推奨
candidates = []
for name, s in all_results.items():
    if name == "v76_baseline":
        continue
    if s["pf"] > base["pf"] and s["trades"] >= base["trades"] * 0.4:
        score = s["pf"] * s["avg_pnl"]
        candidates.append((name, s, score))

candidates.sort(key=lambda x: x[2], reverse=True)
if candidates:
    for rank, (name, s, score) in enumerate(candidates[:5], 1):
        print(f"  {rank}. {name}")
        print(f"     {s['trades']}回, 勝率{s['wr']:.1f}%, PF {s['pf']:.3f}, 総損益{s['total_pnl']:+.1f}pips, 平均{s['avg_pnl']:+.1f}pips")
else:
    print("  v76を超える候補なし")

# CSV保存
rows = [{"filter": name, **s} for name, s in all_results.items()]
pd.DataFrame(rows).to_csv(f"{RESULTS}/qlib_factor_backtest.csv", index=False)
print(f"\n  結果保存: {RESULTS}/qlib_factor_backtest.csv")
