"""
ファクタートーナメント: 4候補の総当たり → 1位がv76とバトル
================================================================
出場者:
  L: KMID + KLOW（実体方向一致 + 下ヒゲ小）
  K: KMID + KSFT（実体方向一致 + 重心一致）
  A: KMID単体（ロング時陽線・ショート時陰線）
  B: KSFT単体（ロング時重心上・ショート時重心下）

定量分析 + 計量分析:
  - PF, 勝率, 総損益, MDD, ケリー基準
  - t検定, シャープレシオ, カルマーレシオ
  - 月次安定性, ドローダウン回復, 連敗耐性
"""
import sys, os
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

print("データ読み込み中...")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h = load(f"{DATA}/usdjpy_is_4h.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h = load(f"{DATA}/usdjpy_oos_4h.csv")
data_15m = pd.concat([is_15m, oos_15m]).sort_index()
data_15m = data_15m[~data_15m.index.duplicated(keep="first")]
data_4h = pd.concat([is_4h, oos_4h]).sort_index()
data_4h = data_4h[~data_4h.index.duplicated(keep="first")]

# ── v76シグナル ──
from strategies.current.yagami_mtf_v76 import generate_signals
print("v76シグナル生成中...")
signals = generate_signals(data_15m, data_15m, data_4h, spread_pips=SPREAD)
sig_map = {s["time"]: s for s in signals}

# ── 4Hファクター ──
c, h, l, o = data_4h["close"], data_4h["high"], data_4h["low"], data_4h["open"]
factors_4h = pd.DataFrame(index=data_4h.index)
factors_4h["KMID"] = (c - o) / (o + 1e-12)
factors_4h["KSFT"] = (2*c - h - l) / (o + 1e-12)
factors_4h["KLOW"] = (np.minimum(o, c) - l) / (o + 1e-12)

def get_fv(t):
    h4 = factors_4h[factors_4h.index <= t]
    return h4.iloc[-1] if len(h4) > 0 else None


# ── バックテストエンジン ──
def run_bt(filter_func):
    trades = []
    pos = None
    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name
        if pos is not None:
            d = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["hc"]:
                if (d==1 and bar["high"]>=half_tp) or (d==-1 and bar["low"]<=half_tp):
                    pos["hp"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["hc"] = True
            if (d==1 and bar["low"]<=pos["sl"]) or (d==-1 and bar["high"]>=pos["sl"]):
                total = pos.get("hp", 0) + (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({"pnl": total, "dir": d, "time": pos["et"],
                              "exit_time": t, "exit_type": "HALF+SL" if pos["hc"] else "SL"})
                pos = None; continue
            if (d==1 and bar["high"]>=pos["tp"]) or (d==-1 and bar["low"]<=pos["tp"]):
                total = pos.get("hp", 0) + (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({"pnl": total, "dir": d, "time": pos["et"],
                              "exit_time": t, "exit_type": "HALF+TP" if pos["hc"] else "TP"})
                pos = None; continue
        if pos is None and t in sig_map and START <= t <= END:
            sig = sig_map[t]
            fv = get_fv(t)
            if fv is not None and filter_func(sig, fv):
                pos = {**sig, "et": t, "hc": False}
    return pd.DataFrame(trades)


# ── 計量分析 ──
def full_analysis(df, name):
    if df.empty:
        return {"name": name, "trades": 0}
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    nw, nl = len(wins), len(losses)
    wr = nw / n * 100
    gp = wins["pnl"].sum() if nw > 0 else 0
    gl = abs(losses["pnl"].sum()) if nl > 0 else 0.001
    pf = gp / gl
    avg_w = wins["pnl"].mean() if nw > 0 else 0
    avg_l = losses["pnl"].mean() if nl > 0 else 0
    kelly = wr/100 - (1-wr/100) / (abs(avg_w)/abs(avg_l)) if avg_l != 0 else 0

    # t検定
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0) if n >= 2 else (0, 1)

    # 累積・DD
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    mdd = dd.max()

    # DD回復時間
    in_dd = dd > 0
    dd_periods = []
    count = 0
    for v in in_dd:
        if v:
            count += 1
        else:
            if count > 0:
                dd_periods.append(count)
            count = 0
    avg_dd_recovery = np.mean(dd_periods) if dd_periods else 0

    # 月次
    df_t = df.copy()
    df_t["month"] = df_t["time"].dt.strftime("%Y-%m")
    monthly = df_t.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()
    total_m = len(monthly[monthly != 0])
    m_mean = monthly.mean()
    m_std = monthly.std()
    sharpe = (m_mean / m_std * np.sqrt(12)) if m_std > 0 else 0

    # カルマーレシオ
    annual_ret = df["pnl"].sum()
    calmar = annual_ret / mdd if mdd > 0 else float("inf")

    # 連敗
    streak = 0
    max_streak = 0
    for p in df["pnl"]:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # 決済タイプ別
    et_stats = {}
    for et in df["exit_type"].unique():
        sub = df[df["exit_type"] == et]
        et_stats[et] = {"n": len(sub), "avg": sub["pnl"].mean(), "total": sub["pnl"].sum()}

    return {
        "name": name, "trades": n, "wins": nw, "losses": nl,
        "wr": wr, "pf": pf, "total_pnl": df["pnl"].sum(),
        "avg_pnl": df["pnl"].mean(), "avg_win": avg_w, "avg_loss": avg_l,
        "kelly": kelly, "mdd": mdd, "sharpe": sharpe, "calmar": calmar,
        "t_stat": t_stat, "p_value": p_val,
        "plus_months": f"{plus_m}/{total_m}", "monthly": monthly,
        "max_streak": max_streak, "avg_dd_recovery": avg_dd_recovery,
        "et_stats": et_stats,
    }


# ══════════════════════════════════════════════════════════════════
# 出場者定義
# ══════════════════════════════════════════════════════════════════
contestants = {
    "L: KMID+KLOW": lambda sig, fv: (
        ((sig["dir"]==1 and fv["KMID"]>0) or (sig["dir"]==-1 and fv["KMID"]<0)) and
        fv["KLOW"] < 0.0015
    ),
    "K: KMID+KSFT": lambda sig, fv: (
        ((sig["dir"]==1 and fv["KMID"]>0) or (sig["dir"]==-1 and fv["KMID"]<0)) and
        ((sig["dir"]==1 and fv["KSFT"]>0) or (sig["dir"]==-1 and fv["KSFT"]<0))
    ),
    "A: KMID単体": lambda sig, fv: (
        (sig["dir"]==1 and fv["KMID"]>0) or (sig["dir"]==-1 and fv["KMID"]<0)
    ),
    "B: KSFT単体": lambda sig, fv: (
        (sig["dir"]==1 and fv["KSFT"]>0) or (sig["dir"]==-1 and fv["KSFT"]<0)
    ),
}

# v76ベースライン
baseline_filter = lambda sig, fv: True

# ══════════════════════════════════════════════════════════════════
# トーナメント実行
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  ファクタートーナメント 予選リーグ（4者総当たり）")
print("=" * 80)

results = {}
for name, filt in contestants.items():
    print(f"\n  {name} バックテスト中...")
    df = run_bt(filt)
    s = full_analysis(df, name)
    results[name] = s

# v76も計算
print(f"\n  v76 (ベースライン) バックテスト中...")
df_v76 = run_bt(baseline_filter)
s_v76 = full_analysis(df_v76, "v76 (champion)")


# ── 予選スコアボード ──
print("\n" + "=" * 80)
print("  予選スコアボード")
print("=" * 80)

header = f"  {'選手':<16s} {'トレ数':>6s} {'勝率':>6s} {'PF':>7s} {'総損益':>10s} {'平均損益':>8s} {'MDD':>7s} {'Sharpe':>7s} {'Calmar':>7s} {'Kelly':>7s} {'p値':>8s} {'月+':>6s} {'最大連敗':>6s}"
print(header)
print(f"  {'─'*112}")

# スコアリング: 10指標で順位を付けて総合点
scoring_metrics = [
    ("wr", True), ("pf", True), ("total_pnl", True), ("avg_pnl", True),
    ("mdd", False), ("sharpe", True), ("calmar", True), ("kelly", True),
    ("p_value", False), ("max_streak", False),
]

all_entries = list(results.values())
scores = {s["name"]: 0 for s in all_entries}

for metric, higher_better in scoring_metrics:
    vals = [(s["name"], s[metric]) for s in all_entries]
    vals.sort(key=lambda x: x[1], reverse=higher_better)
    for rank, (name, _) in enumerate(vals):
        scores[name] += (4 - rank)  # 1位=4点, 2位=3点, 3位=2点, 4位=1点

for s in sorted(all_entries, key=lambda x: scores[x["name"]], reverse=True):
    sig = "★" if s["p_value"] < 0.05 else " "
    print(f"  {s['name']:<16s} {s['trades']:>6d} {s['wr']:>5.1f}% {s['pf']:>7.3f} {s['total_pnl']:>+10.1f} {s['avg_pnl']:>+8.1f} {s['mdd']:>7.1f} {s['sharpe']:>7.3f} {s['calmar']:>7.2f} {s['kelly']:>7.3f} {s['p_value']:>7.4f}{sig} {s['plus_months']:>6s} {s['max_streak']:>6d}")

# ── 順位表 ──
print(f"\n  総合順位（10指標×4段階スコア、満点40）")
print(f"  {'─'*40}")
sorted_names = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
for rank, name in enumerate(sorted_names, 1):
    print(f"  {rank}位: {name} ({scores[name]}点/40点)")

champion_name = sorted_names[0]
champion = results[champion_name]

# ── 月次詳細 ──
print(f"\n  月次損益 (pips)")
months_all = sorted(set(m for s in all_entries for m in s["monthly"].index if m.startswith("2025")))
print(f"  {'月':>10s}", end="")
for s in sorted(all_entries, key=lambda x: scores[x["name"]], reverse=True):
    print(f" {s['name'][:8]:>12s}", end="")
print()
print(f"  {'─'*60}")
for m in months_all:
    print(f"  {m:>10s}", end="")
    for s in sorted(all_entries, key=lambda x: scores[x["name"]], reverse=True):
        v = s["monthly"].get(m, 0)
        print(f" {v:>+12.1f}", end="")
    print()

# ── 決済タイプ分析 ──
print(f"\n  決済タイプ分析")
print(f"  {'─'*70}")
for s in sorted(all_entries, key=lambda x: scores[x["name"]], reverse=True):
    print(f"  {s['name']}:")
    for et, info in sorted(s["et_stats"].items()):
        print(f"    {et:<12s}: {info['n']:>4d}回  平均{info['avg']:>+8.1f}pips  計{info['total']:>+10.1f}pips")


# ══════════════════════════════════════════════════════════════════
# 決勝: トーナメント1位 vs v76
# ══════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print(f"  決勝: {champion_name} vs v76 (champion)")
print(f"{'='*80}")

def head_to_head(s1, s2):
    """直接対決スコア"""
    metrics = [
        ("トレード数", "trades", None),
        ("勝率 (%)", "wr", True),
        ("PF", "pf", True),
        ("総損益 (pips)", "total_pnl", True),
        ("平均損益 (pips)", "avg_pnl", True),
        ("平均利益 (pips)", "avg_win", True),
        ("平均損失 (pips)", "avg_loss", False),  # abs小さい方が良い
        ("ケリー基準", "kelly", True),
        ("MDD (pips)", "mdd", False),
        ("月次シャープ", "sharpe", True),
        ("カルマーレシオ", "calmar", True),
        ("t検定 p値", "p_value", False),
        ("最大連敗", "max_streak", False),
        ("DD回復(トレ数)", "avg_dd_recovery", False),
        ("プラス月", "plus_months", None),
    ]
    sc = {s1["name"]: 0, s2["name"]: 0}

    print(f"\n  {'指標':<18s} {s1['name'][:14]:>16s} {s2['name'][:14]:>16s} {'勝者':>8s}")
    print(f"  {'─'*60}")

    for label, key, higher_better in metrics:
        v1 = s1[key]
        v2 = s2[key]
        if higher_better is None:
            mark = ""
            if isinstance(v1, str):
                print(f"  {label:<18s} {v1:>16s} {v2:>16s} {mark:>8s}")
            else:
                print(f"  {label:<18s} {v1:>16d} {v2:>16d} {mark:>8s}")
            continue

        if key == "avg_loss":
            winner = s1["name"] if abs(v1) < abs(v2) else s2["name"]
        elif higher_better:
            winner = s1["name"] if v1 > v2 else s2["name"]
        else:
            winner = s1["name"] if v1 < v2 else s2["name"]
        sc[winner] += 1
        mark = "◀" if winner == s1["name"] else "▶"
        print(f"  {label:<18s} {v1:>16.3f} {v2:>16.3f} {mark:>8s}")

    print(f"\n  {'─'*60}")
    print(f"  スコア: {s1['name'][:14]} {sc[s1['name']]} - {sc[s2['name']]} {s2['name'][:14]}")

    if sc[s1['name']] > sc[s2['name']]:
        winner_name = s1['name']
    elif sc[s2['name']] > sc[s1['name']]:
        winner_name = s2['name']
    else:
        winner_name = "引き分け"

    print(f"\n  {'*'*60}")
    if winner_name == "引き分け":
        print(f"  ★ 引き分け! ({sc[s1['name']]}-{sc[s2['name']]})")
    else:
        print(f"  ★ 勝者: {winner_name} ({max(sc.values())}-{min(sc.values())})")
    print(f"  {'*'*60}")
    return winner_name, sc

winner, sc = head_to_head(champion, s_v76)

# ── 最終まとめ ──
print(f"\n\n{'='*80}")
print(f"  大会結果まとめ")
print(f"{'='*80}")
print(f"  予選1位: {champion_name} ({scores[champion_name]}点/40点)")
print(f"  決勝: {champion_name} vs v76 → {winner}")
print(f"\n  v76への推奨:")
if winner != "v76 (champion)":
    print(f"  → {champion_name} のフィルターを採用してv77を作成すべき")
else:
    print(f"  → v76のまま継続。フィルターは不要")

# CSV保存
rows = []
for s in all_entries:
    rows.append({k: v for k, v in s.items() if k not in ["monthly", "et_stats"]})
rows.append({k: v for k, v in s_v76.items() if k not in ["monthly", "et_stats"]})
pd.DataFrame(rows).to_csv(f"{RESULTS}/factor_tournament.csv", index=False)
print(f"\n  結果保存: {RESULTS}/factor_tournament.csv")
