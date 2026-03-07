"""
なぜKMID+KLOWフィルターでこんなに成績が良くなるのか？
========================================================
フィルターで「除外されたトレード」と「通過したトレード」を比較して
何が起きているかを解剖する。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h = load(f"{DATA}/usdjpy_is_4h.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h = load(f"{DATA}/usdjpy_oos_4h.csv")
data_15m = pd.concat([is_15m, oos_15m]).sort_index()
data_15m = data_15m[~data_15m.index.duplicated(keep="first")]
data_4h = pd.concat([is_4h, oos_4h]).sort_index()
data_4h = data_4h[~data_4h.index.duplicated(keep="first")]

from strategies.current.yagami_mtf_v76 import generate_signals

signals = generate_signals(data_15m, data_15m, data_4h, spread_pips=0.4)
sig_map = {s["time"]: s for s in signals}

c, h, l, o = data_4h["close"], data_4h["high"], data_4h["low"], data_4h["open"]
factors_4h = pd.DataFrame(index=data_4h.index)
factors_4h["KMID"] = (c - o) / (o + 1e-12)
factors_4h["KLOW"] = (np.minimum(o, c) - l) / (o + 1e-12)
factors_4h["KSFT"] = (2*c - h - l) / (o + 1e-12)
factors_4h["body"] = abs(c - o)
factors_4h["upper_wick"] = h - np.maximum(o, c)
factors_4h["lower_wick"] = np.minimum(o, c) - l
factors_4h["range"] = h - l
factors_4h["body_ratio"] = factors_4h["body"] / (factors_4h["range"] + 1e-12)
factors_4h["is_bullish"] = (c > o).astype(int)

START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31", tz="UTC")

# ── 全トレードにファクター値＋フィルター結果を紐付け ──
all_trades = []
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
            all_trades.append({**pos["meta"], "pnl": total,
                              "exit_type": "HALF+SL" if pos["hc"] else "SL"})
            pos = None; continue
        if (d==1 and bar["high"]>=pos["tp"]) or (d==-1 and bar["low"]<=pos["tp"]):
            total = pos.get("hp", 0) + (pos["tp"] - pos["ep"]) * 100 * d
            all_trades.append({**pos["meta"], "pnl": total,
                              "exit_type": "HALF+TP" if pos["hc"] else "TP"})
            pos = None; continue
    if pos is None and t in sig_map and START <= t <= END:
        sig = sig_map[t]
        h4 = factors_4h[factors_4h.index <= t]
        if len(h4) > 0:
            fv = h4.iloc[-1]
            kmid_ok = (sig["dir"]==1 and fv["KMID"]>0) or (sig["dir"]==-1 and fv["KMID"]<0)
            klow_ok = fv["KLOW"] < 0.0015

            meta = {
                "time": t, "dir": sig["dir"], "tf": sig["tf"],
                "pattern": sig["pattern"],
                "KMID": fv["KMID"], "KLOW": fv["KLOW"], "KSFT": fv["KSFT"],
                "body_ratio": fv["body_ratio"], "is_bullish": fv["is_bullish"],
                "lower_wick": fv["lower_wick"], "range": fv["range"],
                "kmid_pass": kmid_ok, "klow_pass": klow_ok,
                "filter_pass": kmid_ok and klow_ok,
            }
            pos = {**sig, "et": t, "hc": False, "meta": meta}

df = pd.DataFrame(all_trades)
df["result"] = np.where(df["pnl"] > 0, "win", "loss")

passed = df[df["filter_pass"] == True]
rejected = df[df["filter_pass"] == False]

print("=" * 80)
print("  なぜKMID+KLOWフィルターで成績が激変するのか？")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════
# 1. 基本: 通過 vs 除外
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  1. 通過トレード vs 除外トレード")
print(f"{'─'*60}")

for label, sub in [("通過 (v77)", passed), ("除外", rejected), ("全体 (v76)", df)]:
    n = len(sub)
    w = (sub["pnl"] > 0).sum()
    gp = sub[sub["pnl"]>0]["pnl"].sum()
    gl = abs(sub[sub["pnl"]<=0]["pnl"].sum())
    pf = gp / (gl + 1e-12)
    print(f"  {label:<14s}: {n:>4d}回  勝率{w/n*100:>5.1f}%  PF{pf:>6.3f}  総損益{sub['pnl'].sum():>+10.1f}  平均{sub['pnl'].mean():>+7.1f}")

print(f"\n  → 除外された{len(rejected)}回のトレードの中身:")
print(f"     勝率: {(rejected['pnl']>0).sum()}/{len(rejected)} = {(rejected['pnl']>0).sum()/len(rejected)*100:.1f}%")
print(f"     総損益: {rejected['pnl'].sum():+.1f}pips")
print(f"     → フィルターが除外したのは「負けトレードの宝庫」")


# ══════════════════════════════════════════════════════════════════
# 2. KMID単体の効果: なぜ実体方向が重要なのか
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  2. KMIDフィルターの効果（実体方向の一致性）")
print(f"{'─'*60}")

# ロングトレードを陽線/陰線で分類
longs = df[df["dir"] == 1]
shorts = df[df["dir"] == -1]

print(f"\n  ロング（{len(longs)}回）:")
l_bull = longs[longs["KMID"] > 0]  # 直前4Hが陽線
l_bear = longs[longs["KMID"] <= 0]  # 直前4Hが陰線
print(f"    直前4H陽線でロング: {len(l_bull):>4d}回  勝率{(l_bull['pnl']>0).sum()/len(l_bull)*100:>5.1f}%  平均{l_bull['pnl'].mean():>+7.1f}pips")
print(f"    直前4H陰線でロング: {len(l_bear):>4d}回  勝率{(l_bear['pnl']>0).sum()/len(l_bear)*100:>5.1f}%  平均{l_bear['pnl'].mean():>+7.1f}pips")
print(f"    → 陰線でロング = 逆張り。勝率{(l_bear['pnl']>0).sum()/len(l_bear)*100:.0f}%は「養分」")

print(f"\n  ショート（{len(shorts)}回）:")
s_bear = shorts[shorts["KMID"] < 0]  # 直前4Hが陰線
s_bull = shorts[shorts["KMID"] >= 0]  # 直前4Hが陽線
print(f"    直前4H陰線でショート: {len(s_bear):>4d}回  勝率{(s_bear['pnl']>0).sum()/len(s_bear)*100:>5.1f}%  平均{s_bear['pnl'].mean():>+7.1f}pips")
print(f"    直前4H陽線でショート: {len(s_bull):>4d}回  勝率{(s_bull['pnl']>0).sum()/len(s_bull)*100:>5.1f}%  平均{s_bull['pnl'].mean():>+7.1f}pips")
print(f"    → 陽線でショート = 逆張り。勝率{(s_bull['pnl']>0).sum()/len(s_bull)*100:.0f}%は「養分」")


# ══════════════════════════════════════════════════════════════════
# 3. KLOWフィルターの効果: 下ヒゲが意味するもの
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  3. KLOWフィルターの効果（下ヒゲの大きさ）")
print(f"{'─'*60}")

# KMID通過したものの中で、KLOWの影響を見る
kmid_passed = df[df["kmid_pass"] == True]
kp_low_ok = kmid_passed[kmid_passed["klow_pass"] == True]
kp_low_ng = kmid_passed[kmid_passed["klow_pass"] == False]

print(f"\n  KMID通過トレードの中での比較:")
print(f"    KLOW<0.0015（下ヒゲ小）: {len(kp_low_ok):>4d}回  勝率{(kp_low_ok['pnl']>0).sum()/len(kp_low_ok)*100:>5.1f}%  平均{kp_low_ok['pnl'].mean():>+7.1f}pips")
if len(kp_low_ng) > 0:
    print(f"    KLOW≥0.0015（下ヒゲ大）: {len(kp_low_ng):>4d}回  勝率{(kp_low_ng['pnl']>0).sum()/len(kp_low_ng)*100:>5.1f}%  平均{kp_low_ng['pnl'].mean():>+7.1f}pips")

print(f"\n  下ヒゲが大きい = 何を意味するか:")
print(f"    → 4H足の安値から反発して実体が形成された")
print(f"    → 反発＝一度下を試してから戻った → ダマシの可能性")
print(f"    → 下ヒゲ小 = 素直に方向通りに動いた → モメンタム純度が高い")


# ══════════════════════════════════════════════════════════════════
# 4. 除外トレードの共通パターン
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  4. 除外されたトレード46本の共通パターン")
print(f"{'─'*60}")

# 除外理由の内訳
kmid_fail = df[(df["kmid_pass"]==False)]
klow_fail = df[(df["kmid_pass"]==True) & (df["klow_pass"]==False)]

print(f"\n  除外理由:")
print(f"    KMID不一致（実体方向が逆）: {len(kmid_fail)}回  平均損益{kmid_fail['pnl'].mean():>+.1f}pips")
if len(klow_fail) > 0:
    print(f"    KLOW超過（下ヒゲ大きい）:   {len(klow_fail)}回  平均損益{klow_fail['pnl'].mean():>+.1f}pips")

# 決済タイプ別
print(f"\n  除外トレードの決済タイプ:")
for et in rejected["exit_type"].unique():
    sub = rejected[rejected["exit_type"] == et]
    print(f"    {et:<12s}: {len(sub):>4d}回  平均{sub['pnl'].mean():>+7.1f}pips")

print(f"\n  通過トレードの決済タイプ:")
for et in passed["exit_type"].unique():
    sub = passed[passed["exit_type"] == et]
    print(f"    {et:<12s}: {len(sub):>4d}回  平均{sub['pnl'].mean():>+7.1f}pips")


# ══════════════════════════════════════════════════════════════════
# 5. body_ratio（実体/レンジ比）の影響
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  5. 実体比率（body/range）と勝敗の関係")
print(f"{'─'*60}")

# 通過 vs 除外の実体比率
print(f"    通過トレードの実体比率: 平均{passed['body_ratio'].mean():.3f}")
print(f"    除外トレードの実体比率: 平均{rejected['body_ratio'].mean():.3f}")
print(f"    → 通過 = 実体がレンジに対して{'大きい' if passed['body_ratio'].mean() > rejected['body_ratio'].mean() else '小さい'}")


# ══════════════════════════════════════════════════════════════════
# 6. やがみメソッドとの関連
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"  6. やがみメソッドとの関連性")
print(f"{'─'*60}")
print(f"""
  やがみ5条件のうち「ローソク足の強弱」がv76では二番底/天井の
  直前足のclose>open（陽線判定）としてのみ実装されていた。

  KMIDフィルターが追加するのは:
    「二番底/天井を検出した次の4H足（= エントリーする足の直前足）
     の実体方向がエントリー方向と一致していること」

  つまり:
    v76: 「二番底形成→即エントリー」（方向転換を確認せず）
    v77: 「二番底形成→次の足で方向転換を確認→エントリー」

  これは「二番底/二番天井を待ってからエントリー」というやがみルールの
  精神をより忠実に実装したもの。二番底を形成しても、次の足がまだ
  陰線なら「まだ底打ちしていない」と判断して見送る。

  KLOWフィルターは:
    「下ヒゲが小さい = 一方的に方向が出た足」を選好
    → やがみの「大陽線/大陰線に逆らうと死ぬ」の逆読み
    → 素直にモメンタムが出ている足の後にエントリーする
""")

# ══════════════════════════════════════════════════════════════════
# 7. 数字で確認: 除外46本が全部負けだったら？
# ══════════════════════════════════════════════════════════════════
print(f"{'─'*60}")
print(f"  7. インパクト分析")
print(f"{'─'*60}")

n_rej = len(rejected)
rej_wins = (rejected["pnl"] > 0).sum()
rej_losses = (rejected["pnl"] <= 0).sum()
rej_total = rejected["pnl"].sum()

print(f"\n  除外{n_rej}本の内訳:")
print(f"    勝ち: {rej_wins}回（{rej_wins/n_rej*100:.1f}%）")
print(f"    負け: {rej_losses}回（{rej_losses/n_rej*100:.1f}%）")
print(f"    総損益: {rej_total:+.1f}pips")
print(f"    平均損益: {rejected['pnl'].mean():+.1f}pips")
print(f"\n  → フィルターは「勝率{rej_wins/n_rej*100:.0f}%のゴミトレード群」を除外している")
print(f"     たった{n_rej}本を除外するだけで、全体の勝率が{(df['pnl']>0).sum()/len(df)*100:.1f}%→{(passed['pnl']>0).sum()/len(passed)*100:.1f}%に跳ね上がる")
