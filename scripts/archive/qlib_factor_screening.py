"""
qlibファクタースクリーニング: v76シグナルの勝敗を予測できるファクターを探す
============================================================================
v76の373トレードに対して、エントリー時点の各種ファクター値を計算し、
勝ちトレードと負けトレードを分離するファクターを特定する。

方針: v76の味噌汁に「お揚げか豆腐か」を見つける
- v76が既に使っているもの（EMA20, ATR14）は除外
- シンプルで解釈可能なファクターのみ
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ── データ読み込み ──
def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

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
signals = generate_signals(data_15m, data_15m, data_4h, spread_pips=0.4)
print(f"  シグナル数: {len(signals)}")

# ── バックテスト（トレード結果取得） ──
print("バックテスト実行中...")
trades = []
pos = None
sig_map = {s["time"]: s for s in signals}
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31", tz="UTC")

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
            trades.append({**pos["signal"], "pnl": total, "result": "win" if total > 0 else "loss"})
            pos = None; continue
        if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
            tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
            total = pos.get("half_pnl", 0) + tp_pnl
            trades.append({**pos["signal"], "pnl": total, "result": "win" if total > 0 else "loss"})
            pos = None; continue
    if pos is None and t in sig_map:
        if START <= t <= END:
            sig = sig_map[t]
            pos = {**sig, "signal": sig, "half_closed": False}

df_trades = pd.DataFrame(trades)
print(f"  トレード数: {len(df_trades)}, 勝ち: {(df_trades['result']=='win').sum()}, 負け: {(df_trades['result']=='loss').sum()}")


# ══════════════════════════════════════════════════════════════════
# ファクター計算（qlib Alpha158 + FX向け追加ファクター）
# ══════════════════════════════════════════════════════════════════

def compute_factors_4h(df):
    """4時間足でファクター計算"""
    f = pd.DataFrame(index=df.index)

    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ── KBAR（ローソク足形状） v76はclose>openだけ使用 ──
    f["KMID"] = (c - o) / (o + 1e-12)             # 実体比率
    f["KLEN"] = (h - l) / (o + 1e-12)             # レンジ比率
    f["KSFT"] = (2*c - h - l) / (o + 1e-12)       # 価格位置（上か下か）
    f["KUP"] = (h - np.maximum(o, c)) / (o + 1e-12)  # 上ヒゲ比率
    f["KLOW"] = (np.minimum(o, c) - l) / (o + 1e-12)  # 下ヒゲ比率

    # ── ROC（モメンタム） ──
    for w in [3, 5, 10, 20]:
        f[f"ROC{w}"] = c / c.shift(w) - 1

    # ── RSV（ストキャスティクス的） ──
    for w in [5, 10, 20]:
        min_l = l.rolling(w).min()
        max_h = h.rolling(w).max()
        f[f"RSV{w}"] = (c - min_l) / (max_h - min_l + 1e-12)

    # ── SUMP/SUMN (RSI的) ──
    for w in [5, 10, 14, 20]:
        diff = c - c.shift(1)
        gain = diff.clip(lower=0)
        loss = (-diff).clip(lower=0)
        sum_gain = gain.rolling(w).sum()
        sum_loss = loss.rolling(w).sum()
        f[f"SUMP{w}"] = sum_gain / (sum_gain + sum_loss + 1e-12)  # = RSI/100

    # ── CNTP/CNTN/CNTD（連騰/連落比率） ──
    for w in [5, 10, 20]:
        up = (c > c.shift(1)).astype(float)
        dn = (c < c.shift(1)).astype(float)
        f[f"CNTP{w}"] = up.rolling(w).mean()
        f[f"CNTN{w}"] = dn.rolling(w).mean()
        f[f"CNTD{w}"] = f[f"CNTP{w}"] - f[f"CNTN{w}"]

    # ── IMAX/IMIN (Aroon的) ──
    for w in [5, 10, 20]:
        f[f"IMAX{w}"] = h.rolling(w).apply(lambda x: np.argmax(x) / w, raw=True)
        f[f"IMIN{w}"] = l.rolling(w).apply(lambda x: np.argmin(x) / w, raw=True)
        f[f"IMXD{w}"] = f[f"IMAX{w}"] - f[f"IMIN{w}"]

    # ── STD（ボラティリティ） v76はATRを使用、STDは別情報 ──
    for w in [5, 10, 20]:
        f[f"STD{w}"] = c.rolling(w).std() / (c + 1e-12)

    # ── RSQR（トレンドの線形性） ──
    for w in [5, 10, 20]:
        def rsquare(x):
            if len(x) < 3:
                return np.nan
            y = np.arange(len(x))
            slope, intercept, r, p, se = stats.linregress(y, x)
            return r ** 2
        f[f"RSQR{w}"] = c.rolling(w).apply(rsquare, raw=True)

    # ── BETA（トレンド傾き） ──
    for w in [5, 10, 20]:
        def slope(x):
            if len(x) < 3:
                return np.nan
            y = np.arange(len(x))
            s, _, _, _, _ = stats.linregress(y, x)
            return s
        f[f"BETA{w}"] = c.rolling(w).apply(slope, raw=True) / (c + 1e-12)

    # ── ボリューム系（FXではtick volume） ──
    for w in [5, 10, 20]:
        f[f"VMA{w}"] = v.rolling(w).mean() / (v + 1e-12)
        f[f"VRATIO{w}"] = v / (v.rolling(w).mean() + 1e-12)

    # ── ADX（トレンド強度）手動計算 ──
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    for w in [14]:
        atr_w = tr.rolling(w).mean()
        plus_di = 100 * plus_dm.rolling(w).mean() / (atr_w + 1e-12)
        minus_di = 100 * minus_dm.rolling(w).mean() / (atr_w + 1e-12)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        f[f"ADX{w}"] = dx.rolling(w).mean()
        f[f"PLUS_DI{w}"] = plus_di
        f[f"MINUS_DI{w}"] = minus_di

    # ── ボリンジャーバンド幅 ──
    for w in [20]:
        ma = c.rolling(w).mean()
        std = c.rolling(w).std()
        f[f"BB_WIDTH{w}"] = 2 * std / (ma + 1e-12)
        f[f"BB_POS{w}"] = (c - (ma - 2*std)) / (4*std + 1e-12)  # 0-1のBB内位置

    # ── EMA乖離率（v76はEMA20のトレンドだけ。乖離率は別情報） ──
    for w in [10, 20, 50]:
        ema = c.ewm(span=w, adjust=False).mean()
        f[f"EMA_DEV{w}"] = (c - ema) / (ema + 1e-12)

    return f

print("\n4Hファクター計算中...")
factors_4h = compute_factors_4h(data_4h)
print(f"  ファクター数: {len(factors_4h.columns)}")


# ══════════════════════════════════════════════════════════════════
# 各トレードにファクター値を紐付け
# ══════════════════════════════════════════════════════════════════
print("トレードにファクター紐付け中...")
factor_at_trade = []

for _, trade in df_trades.iterrows():
    t = trade["time"]
    # 4h足のファクター: エントリー時点以前の最新値
    h4_before = factors_4h[factors_4h.index <= t]
    if len(h4_before) == 0:
        continue
    fvals = h4_before.iloc[-1].to_dict()
    fvals["pnl"] = trade["pnl"]
    fvals["result"] = trade["result"]
    fvals["dir"] = trade["dir"]
    fvals["time"] = t
    factor_at_trade.append(fvals)

df_factors = pd.DataFrame(factor_at_trade)
print(f"  紐付け完了: {len(df_factors)}トレード")


# ══════════════════════════════════════════════════════════════════
# スクリーニング: 勝敗を分離するファクターを探す
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  ファクタースクリーニング結果")
print("=" * 72)

factor_cols = [c for c in df_factors.columns if c not in ["pnl", "result", "dir", "time"]]

results = []
for col in factor_cols:
    vals = df_factors[col].dropna()
    if len(vals) < 20:
        continue

    valid_idx = vals.index
    wins = df_factors.loc[valid_idx][df_factors.loc[valid_idx, "result"] == "win"][col]
    losses = df_factors.loc[valid_idx][df_factors.loc[valid_idx, "result"] == "loss"][col]

    if len(wins) < 5 or len(losses) < 5:
        continue

    # t検定: 勝ちと負けの平均差
    t_stat, p_val = stats.ttest_ind(wins, losses, equal_var=False)

    # 相関: ファクター値とpnlの相関
    valid = df_factors[[col, "pnl"]].dropna()
    corr = valid[col].corr(valid["pnl"])

    # IC (Information Coefficient): ランク相関
    ic = valid[col].corr(valid["pnl"], method="spearman")

    results.append({
        "factor": col,
        "win_mean": wins.mean(),
        "loss_mean": losses.mean(),
        "t_stat": t_stat,
        "p_value": p_val,
        "corr": corr,
        "ic": ic,
        "n_valid": len(vals),
    })

df_results = pd.DataFrame(results)
df_results["abs_ic"] = df_results["ic"].abs()
df_results = df_results.sort_values("abs_ic", ascending=False)

# ── トップ30表示 ──
print(f"\n  IC (情報係数) ランキング TOP30")
print(f"  ※ ICが高いほどpnlとの相関が強い = フィルターとして有効")
print(f"  {'ファクター':<16s} {'IC':>8s} {'相関':>8s} {'p値':>10s} {'勝ち平均':>10s} {'負け平均':>10s}")
print(f"  {'─'*62}")

for _, row in df_results.head(30).iterrows():
    sig = "★" if row["p_value"] < 0.05 else " "
    print(f"  {row['factor']:<16s} {row['ic']:>+8.3f} {row['corr']:>+8.3f} {row['p_value']:>10.4f}{sig} {row['win_mean']:>10.4f} {row['loss_mean']:>10.4f}")

# ── p < 0.05 のファクター ──
sig_factors = df_results[df_results["p_value"] < 0.05]
print(f"\n  統計的有意 (p<0.05) なファクター: {len(sig_factors)}個")
if len(sig_factors) > 0:
    print(f"  {'ファクター':<16s} {'IC':>8s} {'p値':>10s} {'方向':>10s}")
    print(f"  {'─'*50}")
    for _, row in sig_factors.iterrows():
        direction = "勝ち>負け" if row["win_mean"] > row["loss_mean"] else "負け>勝ち"
        print(f"  {row['factor']:<16s} {row['ic']:>+8.3f} {row['p_value']:>10.4f} {direction:>10s}")

# ── ロング/ショート別分析 ──
for dir_val, dir_name in [(1, "ロング"), (-1, "ショート")]:
    df_dir = df_factors[df_factors["dir"] == dir_val]
    if len(df_dir) < 20:
        continue
    print(f"\n  === {dir_name}のみ ({len(df_dir)}回) ===")
    dir_results = []
    for col in factor_cols:
        vals = df_dir[col].dropna()
        if len(vals) < 10:
            continue
        valid_idx = vals.index
        wins = df_dir.loc[valid_idx][df_dir.loc[valid_idx, "result"] == "win"][col]
        losses = df_dir.loc[valid_idx][df_dir.loc[valid_idx, "result"] == "loss"][col]
        if len(wins) < 3 or len(losses) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(wins, losses, equal_var=False)
        valid = df_dir[[col, "pnl"]].dropna()
        ic = valid[col].corr(valid["pnl"], method="spearman")
        dir_results.append({"factor": col, "ic": ic, "p_value": p_val,
                           "win_mean": wins.mean(), "loss_mean": losses.mean()})
    df_dir_res = pd.DataFrame(dir_results)
    df_dir_res["abs_ic"] = df_dir_res["ic"].abs()
    df_dir_res = df_dir_res.sort_values("abs_ic", ascending=False)
    for _, row in df_dir_res.head(10).iterrows():
        sig = "★" if row["p_value"] < 0.05 else " "
        print(f"    {row['factor']:<16s} IC={row['ic']:>+.3f}  p={row['p_value']:.4f}{sig}")

# ── CSV保存 ──
df_results.to_csv(f"{RESULTS}/qlib_factor_screening.csv", index=False)
print(f"\n  結果保存: {RESULTS}/qlib_factor_screening.csv")

# ── 推奨ファクターまとめ ──
print(f"\n{'='*72}")
print("  推奨ファクター（v76への追加候補）")
print(f"{'='*72}")

# IC上位5 + p<0.05のうちシンプルなもの
top5 = df_results.head(5)
for _, row in top5.iterrows():
    sig = "★有意" if row["p_value"] < 0.05 else "（有意差なし）"
    direction = "高い方が勝ち" if row["ic"] > 0 else "低い方が勝ち"
    print(f"  {row['factor']:<16s}: IC={row['ic']:>+.3f}, p={row['p_value']:.4f} {sig}")
    print(f"    → {direction}")
