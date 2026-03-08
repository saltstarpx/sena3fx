"""
triple_verification.py
======================
3つの検証を一括実行:
1. 採用報告と同条件（15分足シミュレーション・半決済あり）でv77を再バックテスト
2. 半決済あり vs 全決済の影響を定量分析
3. 1H足シグナル vs v76 15分足シグナルのコード比較・統計

期間: 2025/1-12（採用報告と同一）
スプレッド: 0.4pips
RR: 2.5
"""
import sys, os
sys.path.insert(0, "/home/ubuntu/sena3fx")
sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

SPREAD = 0.4
RR_RATIO = 2.5
START = pd.Timestamp("2025-01-01", tz="UTC")
END   = pd.Timestamp("2025-12-31", tz="UTC")
DATA  = "/home/ubuntu/sena3fx/data"
OUT   = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

print("データ読み込み中...")
is_15m  = load(f"{DATA}/usdjpy_is_15m.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
is_4h   = load(f"{DATA}/usdjpy_is_4h.csv")
oos_4h  = load(f"{DATA}/usdjpy_oos_4h.csv")
is_1m   = load(f"{DATA}/usdjpy_is_1m.csv")  if os.path.exists(f"{DATA}/usdjpy_is_1m.csv") else None
oos_1m  = load(f"{DATA}/usdjpy_oos_1m.csv") if os.path.exists(f"{DATA}/usdjpy_oos_1m.csv") else None

data_15m = pd.concat([is_15m, oos_15m]).sort_index()
data_15m = data_15m[~data_15m.index.duplicated(keep="first")]
data_4h  = pd.concat([is_4h, oos_4h]).sort_index()
data_4h  = data_4h[~data_4h.index.duplicated(keep="first")]

if is_1m is not None and oos_1m is not None:
    data_1m = pd.concat([is_1m, oos_1m]).sort_index()
    data_1m = data_1m[~data_1m.index.duplicated(keep="first")]
elif oos_1m is not None:
    data_1m = oos_1m
else:
    data_1m = None

print(f"  15M: {len(data_15m)}行 {data_15m.index[0]} 〜 {data_15m.index[-1]}")
print(f"  4H:  {len(data_4h)}行")
if data_1m is not None:
    print(f"  1M:  {len(data_1m)}行")

# ─────────────────────────────────────────────
# v76 / v77 シグナル生成
# ─────────────────────────────────────────────
from strategies.current.yagami_mtf_v76 import generate_signals as gen_v76
from strategies.current.yagami_mtf_v77 import generate_signals as gen_v77

print("\nv76シグナル生成中...")
sigs_v76 = gen_v76(data_15m, data_15m, data_4h, spread_pips=SPREAD, rr_ratio=RR_RATIO)
sig_map_v76 = {s["time"]: s for s in sigs_v76}

print("v77シグナル生成中...")
sigs_v77 = gen_v77(data_15m, data_15m, data_4h, spread_pips=SPREAD, rr_ratio=RR_RATIO)
sig_map_v77 = {s["time"]: s for s in sigs_v77}

print(f"  v76シグナル数: {len(sigs_v76)}")
print(f"  v77シグナル数: {len(sigs_v77)}")

# ─────────────────────────────────────────────
# シミュレーション関数（15分足ベース）
# ─────────────────────────────────────────────
def simulate_15m(sig_map, label="", half_exit=True):
    """
    15分足でSL/TP判定。
    half_exit=True: RR1.0でTP半分→SLをBEに移動（採用報告方式）
    half_exit=False: 全決済のみ（シンプル方式）
    """
    trades = []
    pos = None
    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name
        if pos is not None:
            d = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d  # RR1.0のTP（半決済ポイント）

            if half_exit and not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep  # SLをBE（ブレークイーブン）に移動
                    pos["half_closed"] = True

            # SL判定
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + sl_pnl
                trades.append({
                    "pnl": total, "dir": d, "time": pos["entry_time"],
                    "exit": "SL", "tf": pos.get("tf","?"), "pattern": pos.get("pattern","?")
                })
                pos = None
                continue

            # TP判定
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + tp_pnl
                trades.append({
                    "pnl": total, "dir": d, "time": pos["entry_time"],
                    "exit": "TP", "tf": pos.get("tf","?"), "pattern": pos.get("pattern","?")
                })
                pos = None
                continue

        # 新規エントリー
        if pos is None and t in sig_map:
            if START <= t <= END:
                sig = sig_map[t]
                pos = {**sig, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)


def calc_stats(df, label=""):
    """統計計算"""
    if df.empty:
        print(f"[{label}] トレードなし")
        return {}
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
    wr = len(wins) / len(df)
    _, p_val = stats.ttest_1samp(df["pnl"], 0) if len(df) >= 2 else (0, 1)
    monthly = df.set_index("time").resample("ME")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    cumsum = df["pnl"].cumsum()
    mdd = abs((cumsum - cumsum.cummax()).min())
    avg_win  = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly = wr - (1 - wr) / (pf if pf > 0 else 1e-9)
    sharpe = (df["pnl"].mean() / df["pnl"].std() * np.sqrt(12)) if df["pnl"].std() > 0 else 0

    print(f"\n{'='*55}")
    print(f"【{label}】")
    print(f"  トレード数  : {len(df)}")
    print(f"  勝率        : {wr:.1%}")
    print(f"  PF          : {pf:.2f}")
    print(f"  総損益      : {df['pnl'].sum():+.1f}")
    print(f"  平均勝ち    : {avg_win:+.1f}")
    print(f"  平均負け    : {avg_loss:+.1f}")
    print(f"  MDD         : {mdd:.1f}")
    print(f"  月次シャープ: {sharpe:.2f}")
    print(f"  ケリー基準  : {kelly:.3f}")
    print(f"  p値         : {p_val:.2e}")
    print(f"  プラス月    : {plus_months}/{len(monthly)}")
    print(f"{'='*55}")

    return {
        "label": label, "trades": len(df), "wr": wr, "pf": pf,
        "total_pnl": df["pnl"].sum(), "avg_win": avg_win, "avg_loss": avg_loss,
        "mdd": mdd, "sharpe": sharpe, "kelly": kelly, "p_value": p_val,
        "plus_months": f"{plus_months}/{len(monthly)}"
    }


# ─────────────────────────────────────────────
# ① 採用報告と同条件でv77を再バックテスト
# ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("【検証①】採用報告と同条件（15M・半決済あり）でv76/v77比較")
print("="*60)

df_v76_half = simulate_15m(sig_map_v76, "v76_half", half_exit=True)
df_v77_half = simulate_15m(sig_map_v77, "v77_half", half_exit=True)

res_v76_half = calc_stats(df_v76_half, "v76 | 15M・半決済あり（採用報告と同条件）")
res_v77_half = calc_stats(df_v77_half, "v77 | 15M・半決済あり（採用報告と同条件）")

# ─────────────────────────────────────────────
# ② 半決済あり vs 全決済の影響分析
# ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("【検証②】半決済あり vs 全決済の影響分析（v77で比較）")
print("="*60)

df_v77_full = simulate_15m(sig_map_v77, "v77_full", half_exit=False)

res_v77_full = calc_stats(df_v77_full, "v77 | 15M・全決済のみ（シンプル方式）")

# 半決済の影響を詳細分析
print("\n--- 半決済の影響詳細 ---")
if not df_v77_half.empty and not df_v77_full.empty:
    print(f"  トレード数差  : {len(df_v77_half) - len(df_v77_full):+d}")
    print(f"  勝率差        : {(res_v77_half.get('wr',0) - res_v77_full.get('wr',0))*100:+.1f}pt")
    print(f"  PF差          : {res_v77_half.get('pf',0) - res_v77_full.get('pf',0):+.2f}")
    print(f"  総損益差      : {res_v77_half.get('total_pnl',0) - res_v77_full.get('total_pnl',0):+.1f}")
    print(f"  MDD差         : {res_v77_half.get('mdd',0) - res_v77_full.get('mdd',0):+.1f}")

# ─────────────────────────────────────────────
# ③ 1H足シグナル vs v76 15分足シグナルの統計比較
# ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("【検証③】シグナル構造の比較（v76 15M全シグナル vs v77 1H足のみ）")
print("="*60)

# v77の1H足シグナルのみを抽出
sigs_v77_1h = [s for s in sigs_v77 if s.get("tf") == "1h" and START <= s["time"] <= END]
sigs_v77_4h = [s for s in sigs_v77 if s.get("tf") == "4h" and START <= s["time"] <= END]
sigs_v76_all = [s for s in sigs_v76 if START <= s["time"] <= END]
sigs_v76_1h  = [s for s in sigs_v76 if s.get("tf") == "1h" and START <= s["time"] <= END]
sigs_v76_4h  = [s for s in sigs_v76 if s.get("tf") == "4h" and START <= s["time"] <= END]

print(f"\nv76 全シグナル: {len(sigs_v76_all)}本 (4H:{len(sigs_v76_4h)} / 1H:{len(sigs_v76_1h)})")
print(f"v77 全シグナル: {len([s for s in sigs_v77 if START<=s['time']<=END])}本 (4H:{len(sigs_v77_4h)} / 1H:{len(sigs_v77_1h)})")

# TF別の勝率をv77半決済結果から計算
if not df_v77_half.empty and "tf" in df_v77_half.columns:
    print("\nv77（半決済）TF別統計:")
    for tf in ["4h", "1h"]:
        sub = df_v77_half[df_v77_half["tf"] == tf]
        if len(sub) > 0:
            wr = (sub["pnl"] > 0).mean()
            pf_w = sub[sub["pnl"]>0]["pnl"].sum()
            pf_l = abs(sub[sub["pnl"]<=0]["pnl"].sum())
            pf = pf_w/pf_l if pf_l > 0 else float("inf")
            print(f"  {tf}: {len(sub)}本 勝率{wr:.1%} PF{pf:.2f} 総損益{sub['pnl'].sum():+.1f}")

# ─────────────────────────────────────────────
# 全条件の比較表を作成
# ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("【総合比較表】")
print("="*60)

# 採用報告の数値（qlib_factor_backtest.csvから）
reported_v76 = {"label": "v76（採用報告）", "trades": 373, "wr": 0.5657, "pf": 2.17, "total_pnl": 8227, "mdd": 460.9, "sharpe": 5.57, "kelly": 0.305}
reported_v77 = {"label": "v77（採用報告）", "trades": 327, "wr": 0.7615, "pf": 4.96, "total_pnl": 12551, "mdd": 222.6, "sharpe": 10.47, "kelly": 0.608}

all_results = [reported_v76, reported_v77, res_v76_half, res_v77_half, res_v77_full]

print(f"\n{'条件':<35} {'本数':>5} {'勝率':>7} {'PF':>6} {'総損益':>9} {'MDD':>8} {'シャープ':>8} {'ケリー':>7}")
print("-"*90)
for r in all_results:
    if not r:
        continue
    label = r.get("label","")[:34]
    print(f"{label:<35} {r.get('trades',0):>5} {r.get('wr',0):>7.1%} {r.get('pf',0):>6.2f} {r.get('total_pnl',0):>9.0f} {r.get('mdd',0):>8.1f} {r.get('sharpe',0):>8.2f} {r.get('kelly',0):>7.3f}")

# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────
print("\nグラフ生成中...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("v77 3条件比較検証（USDJPY 2025/1-12）", fontsize=14, fontweight="bold")

# 1. 損益曲線比較（半決済あり）
ax = axes[0, 0]
if not df_v76_half.empty:
    ax.plot(df_v76_half["pnl"].cumsum().values, label="v76（半決済）", color="#3b82f6", linewidth=1.5)
if not df_v77_half.empty:
    ax.plot(df_v77_half["pnl"].cumsum().values, label="v77（半決済）", color="#ef4444", linewidth=1.5)
ax.set_title("損益曲線（15M・半決済あり）")
ax.set_xlabel("トレード番号")
ax.set_ylabel("累積損益")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 半決済 vs 全決済（v77）
ax = axes[0, 1]
if not df_v77_half.empty:
    ax.plot(df_v77_half["pnl"].cumsum().values, label="v77 半決済あり", color="#ef4444", linewidth=1.5)
if not df_v77_full.empty:
    ax.plot(df_v77_full["pnl"].cumsum().values, label="v77 全決済のみ", color="#f59e0b", linewidth=1.5, linestyle="--")
ax.set_title("半決済 vs 全決済（v77）")
ax.set_xlabel("トレード番号")
ax.set_ylabel("累積損益")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 指標比較棒グラフ（勝率）
ax = axes[0, 2]
labels = ["v76\n採用報告", "v77\n採用報告", "v76\n今回再現", "v77\n今回再現", "v77\n全決済"]
wrs = [reported_v76["wr"]*100, reported_v77["wr"]*100,
       res_v76_half.get("wr",0)*100 if res_v76_half else 0,
       res_v77_half.get("wr",0)*100 if res_v77_half else 0,
       res_v77_full.get("wr",0)*100 if res_v77_full else 0]
colors = ["#93c5fd", "#fca5a5", "#3b82f6", "#ef4444", "#f59e0b"]
bars = ax.bar(labels, wrs, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, wrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("勝率比較")
ax.set_ylabel("勝率 (%)")
ax.set_ylim(0, 90)
ax.grid(True, alpha=0.3, axis="y")

# 4. PF比較
ax = axes[1, 0]
pfs = [reported_v76["pf"], reported_v77["pf"],
       res_v76_half.get("pf",0) if res_v76_half else 0,
       res_v77_half.get("pf",0) if res_v77_half else 0,
       res_v77_full.get("pf",0) if res_v77_full else 0]
bars = ax.bar(labels, pfs, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(1.5, color="orange", linestyle="--", alpha=0.7, label="PF=1.5")
ax.axhline(2.0, color="green", linestyle="--", alpha=0.7, label="PF=2.0")
for bar, val in zip(bars, pfs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_title("プロフィットファクター比較")
ax.set_ylabel("PF")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# 5. MDD比較
ax = axes[1, 1]
mdds = [reported_v76["mdd"], reported_v77["mdd"],
        res_v76_half.get("mdd",0) if res_v76_half else 0,
        res_v77_half.get("mdd",0) if res_v77_half else 0,
        res_v77_full.get("mdd",0) if res_v77_full else 0]
bars = ax.bar(labels, mdds, color=colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, mdds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_title("最大ドローダウン比較（小さいほど良い）")
ax.set_ylabel("MDD")
ax.grid(True, alpha=0.3, axis="y")

# 6. 月次損益（v77半決済）
ax = axes[1, 2]
if not df_v77_half.empty:
    monthly = df_v77_half.set_index("time").resample("ME")["pnl"].sum()
    colors_m = ["#ef4444" if v < 0 else "#10b981" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors_m, edgecolor="white")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m)[:7] for m in monthly.index], rotation=45, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("v77月次損益（15M・半決済あり）")
    ax.set_ylabel("損益")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(f"{OUT}/triple_verification.png", dpi=150, bbox_inches="tight")
print(f"グラフ保存: {OUT}/triple_verification.png")

# ─────────────────────────────────────────────
# 結果をCSVに保存
# ─────────────────────────────────────────────
summary_rows = []
for r in all_results:
    if r:
        summary_rows.append(r)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{OUT}/triple_verification_summary.csv", index=False)
print(f"CSV保存: {OUT}/triple_verification_summary.csv")

print("\n完了！")
