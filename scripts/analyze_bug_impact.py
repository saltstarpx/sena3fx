"""
analyze_bug_impact.py
=====================
27カラムバグ・月末重複バグの修正が既存バックテスト結果に与える影響を定量分析する

実験設計:
  A. 修正後データ（正規）    → 現行の正しい結果
  B. 月末重複あり            → BUG-002のみ再現（4時間足に重複を注入）
  C. 27カラムバグ（代替）    → BUG-001の影響を近似（4時間足データを意図的に欠損させる）
  D. 両方バグあり            → BUG-001+BUG-002の複合影響

比較指標: トレード数・勝率・PF・総損益・ケリー基準・p値・プラス月数
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies/current")
sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
try:
    import yagami_mtf_v76 as v76
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yagami_mtf_v76",
        "/home/ubuntu/sena3fx/strategies/current/yagami_mtf_v76.py"
    )
    v76 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v76)

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"
SPREAD  = 0.4

# ============================================================
# データ読み込み
# ============================================================
def load(p):
    df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

print("データ読み込み中...")
is_1m  = load(f"{DATA}/usdjpy_is_1m.csv")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h  = load(f"{DATA}/usdjpy_is_4h.csv")
oos_1m  = load(f"{DATA}/usdjpy_oos_1m.csv")
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
oos_4h  = load(f"{DATA}/usdjpy_oos_4h.csv")
print(f"IS  1m: {len(is_1m):,}行  4h: {len(is_4h)}本")
print(f"OOS 1m: {len(oos_1m):,}行  4h: {len(oos_4h)}本")

# ============================================================
# BUG-002再現: 月末重複を注入する
# ============================================================
def inject_month_end_duplicates(df_4h):
    """月末最終足のコピーを直後に挿入して重複を再現する"""
    df = df_4h.copy().reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    inserts = []
    for i, row in df.iterrows():
        ts = row['timestamp']
        # 月末最終足の判定: 翌行が翌月かどうか
        if i + 1 < len(df):
            next_ts = df.iloc[i+1]['timestamp']
            if ts.month != next_ts.month:
                # 翌月最初足を同じタイムスタンプで複製（値は翌行の値）
                dup_row = df.iloc[i+1].copy()
                dup_row['timestamp'] = ts  # 月末のタイムスタンプに書き換え
                inserts.append((i + 1, dup_row))

    # 後ろから挿入（インデックスがずれないように）
    result = df.copy()
    for pos, row in reversed(inserts):
        top = result.iloc[:pos]
        bot = result.iloc[pos:]
        result = pd.concat([top, pd.DataFrame([row]), bot], ignore_index=True)

    dup_count = len(inserts)
    print(f"  月末重複注入: {dup_count}件")
    result = result.set_index('timestamp')
    result.index = pd.to_datetime(result.index, utc=True)
    return result, dup_count

# ============================================================
# BUG-001影響近似: 4時間足データを「ほぼ空」にする
# ============================================================
def simulate_27col_bug(df_4h):
    """
    27カラムバグでは open 列以外が NaN になる。
    4時間足の EMA・パターン判定が機能しなくなるため、
    4時間足の close/high/low を NaN にして影響を近似する。
    """
    df = df_4h.copy()
    # open以外をNaNにする（27カラムバグの実際の症状を近似）
    for col in ['high', 'low', 'close']:
        if col in df.columns:
            df[col] = np.nan
    # volumeやema等も欠損させる
    for col in df.columns:
        if col not in ['open']:
            df[col] = np.nan
    print(f"  27カラムバグ近似: high/low/close/ema等をNaNに設定")
    return df

# ============================================================
# バックテストエンジン
# ============================================================
def run_backtest(data_1m, data_15m, data_4h, spread=0.4, label=""):
    print(f"\n  [{label}] シグナル生成中...", flush=True)
    try:
        signals = v76.generate_signals(data_1m, data_15m, data_4h, spread_pips=spread)
    except Exception as e:
        print(f"  [{label}] シグナル生成エラー: {e}")
        return pd.DataFrame()

    sig_map = {s["time"]: s for s in signals}
    print(f"  [{label}] シグナル数: {len(signals)}")

    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        if pos is not None:
            d      = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + sl_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "SL" if not pos["half_closed"] else "HALF+SL",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": label,
                })
                pos = None; continue
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + tp_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "TP" if not pos["half_closed"] else "HALF+TP",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": label,
                })
                pos = None; continue
        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)

# ============================================================
# 統計計算
# ============================================================
def calc_stats(df, label):
    if df.empty:
        return dict(label=label, trades=0, win_rate=0, pf=0,
                    total_pnl=0, avg_win=0, avg_loss=0,
                    kelly=0, t_stat=0, p_value=1.0,
                    plus_months="0/0", monthly=pd.Series(dtype=float))
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df) * 100
    avg_w  = wins["pnl"].mean()  if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0)
    monthly     = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)
    return dict(label=label, trades=len(df), win_rate=wr, pf=pf,
                total_pnl=df["pnl"].sum(), avg_win=avg_w, avg_loss=avg_l,
                kelly=kelly, t_stat=t_stat, p_value=p_val,
                plus_months=f"{plus_months}/{total_months}", monthly=monthly)

# ============================================================
# 実験実行
# ============================================================
print("\n" + "="*60)
print("実験A: 修正後データ（正規）")
print("="*60)
df_A_is  = run_backtest(is_1m,  is_15m, is_4h,  SPREAD, "A_IS")
df_A_oos = run_backtest(oos_1m, oos_15m, oos_4h, SPREAD, "A_OOS")
df_A = pd.concat([df_A_is, df_A_oos], ignore_index=True)
s_A_is  = calc_stats(df_A_is,  "A_IS")
s_A_oos = calc_stats(df_A_oos, "A_OOS")
s_A     = calc_stats(df_A,     "A_ALL")

print("\n" + "="*60)
print("実験B: 月末重複バグ（BUG-002）あり")
print("="*60)
is_4h_dup,  n_dup_is  = inject_month_end_duplicates(is_4h)
oos_4h_dup, n_dup_oos = inject_month_end_duplicates(oos_4h)
df_B_is  = run_backtest(is_1m,  is_15m, is_4h_dup,  SPREAD, "B_IS")
df_B_oos = run_backtest(oos_1m, oos_15m, oos_4h_dup, SPREAD, "B_OOS")
df_B = pd.concat([df_B_is, df_B_oos], ignore_index=True)
s_B_is  = calc_stats(df_B_is,  "B_IS")
s_B_oos = calc_stats(df_B_oos, "B_OOS")
s_B     = calc_stats(df_B,     "B_ALL")

print("\n" + "="*60)
print("実験C: 27カラムバグ（BUG-001）近似")
print("="*60)
is_4h_bug  = simulate_27col_bug(is_4h)
oos_4h_bug = simulate_27col_bug(oos_4h)
df_C_is  = run_backtest(is_1m,  is_15m, is_4h_bug,  SPREAD, "C_IS")
df_C_oos = run_backtest(oos_1m, oos_15m, oos_4h_bug, SPREAD, "C_OOS")
df_C = pd.concat([df_C_is, df_C_oos], ignore_index=True)
s_C_is  = calc_stats(df_C_is,  "C_IS")
s_C_oos = calc_stats(df_C_oos, "C_OOS")
s_C     = calc_stats(df_C,     "C_ALL")

print("\n" + "="*60)
print("実験D: 両方バグあり（BUG-001 + BUG-002）")
print("="*60)
is_4h_both  = simulate_27col_bug(is_4h_dup)
oos_4h_both = simulate_27col_bug(oos_4h_dup)
df_D_is  = run_backtest(is_1m,  is_15m, is_4h_both,  SPREAD, "D_IS")
df_D_oos = run_backtest(oos_1m, oos_15m, oos_4h_both, SPREAD, "D_OOS")
df_D = pd.concat([df_D_is, df_D_oos], ignore_index=True)
s_D_is  = calc_stats(df_D_is,  "D_IS")
s_D_oos = calc_stats(df_D_oos, "D_OOS")
s_D     = calc_stats(df_D,     "D_ALL")

# ============================================================
# 結果サマリー出力
# ============================================================
print("\n\n" + "="*70)
print("比較サマリー（IS+OOS統合）")
print("="*70)
for s in [s_A, s_B, s_C, s_D]:
    label_map = {
        "A_ALL": "A. 修正後（正規）",
        "B_ALL": "B. 月末重複バグのみ",
        "C_ALL": "C. 27カラムバグのみ",
        "D_ALL": "D. 両方バグあり",
    }
    lbl = label_map.get(s['label'], s['label'])
    print(f"\n{lbl}")
    print(f"  トレード数: {s['trades']:3d}回  勝率: {s['win_rate']:5.1f}%  PF: {s['pf']:6.2f}  "
          f"総損益: {s['total_pnl']:+8.1f}pips  ケリー: {s['kelly']:.3f}  "
          f"p値: {s['p_value']:.4f}  プラス月: {s['plus_months']}")

# ============================================================
# 可視化
# ============================================================
fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor("#0d0d1a")

# --- 上段: 比較バーチャート ---
ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 2)
ax3 = fig.add_subplot(4, 2, 3)
ax4 = fig.add_subplot(4, 2, 4)

labels_short = ["A\n修正後\n(正規)", "B\n月末重複\nバグのみ", "C\n27カラム\nバグのみ", "D\n両方\nバグあり"]
colors_bar   = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

stats_list = [s_A, s_B, s_C, s_D]

def styled_bar(ax, values, title, ylabel, fmt=".1f", ref_line=None):
    bars = ax.bar(labels_short, values, color=colors_bar, edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if ref_line is not None:
        ax.axhline(ref_line, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(bar.get_height())*0.02,
                f"{val:{fmt}}", ha='center', va='bottom', color='white', fontsize=8)

styled_bar(ax1, [s['trades'] for s in stats_list],
           "トレード数（IS+OOS）", "回", fmt=".0f")
styled_bar(ax2, [s['win_rate'] for s in stats_list],
           "勝率（IS+OOS）", "%", ref_line=50)
styled_bar(ax3, [s['pf'] if s['pf'] != float('inf') else 0 for s in stats_list],
           "プロフィットファクター（IS+OOS）", "PF", ref_line=1.0)
styled_bar(ax4, [s['total_pnl'] for s in stats_list],
           "総損益（IS+OOS）", "pips", ref_line=0)

# --- 中段: IS/OOS別比較 ---
ax5 = fig.add_subplot(4, 2, 5)
ax6 = fig.add_subplot(4, 2, 6)

is_stats  = [s_A_is,  s_B_is,  s_C_is,  s_D_is]
oos_stats = [s_A_oos, s_B_oos, s_C_oos, s_D_oos]

styled_bar(ax5, [s['pf'] if s['pf'] != float('inf') else 0 for s in is_stats],
           "PF比較（IS期間）", "PF", ref_line=1.0)
styled_bar(ax6, [s['pf'] if s['pf'] != float('inf') else 0 for s in oos_stats],
           "PF比較（OOS期間）", "PF", ref_line=1.0)

# --- 下段: エクイティカーブ比較 ---
ax7 = fig.add_subplot(4, 1, 3)
ax8 = fig.add_subplot(4, 1, 4)

curve_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
curve_labels = ["A: 修正後（正規）", "B: 月末重複バグのみ", "C: 27カラムバグのみ", "D: 両方バグあり"]

for df_all, color, lbl in zip([df_A, df_B, df_C, df_D], curve_colors, curve_labels):
    if not df_all.empty:
        df_sorted = df_all.sort_values("entry_time")
        cumulative = df_sorted["pnl"].cumsum().values
        ax7.plot(range(len(cumulative)), cumulative, color=color, linewidth=1.5, label=lbl, alpha=0.85)

ax7.axhline(0, color="white", linewidth=0.5, linestyle="--")
ax7.set_title("エクイティカーブ比較（IS+OOS）", color="white", fontsize=11)
ax7.set_xlabel("トレード番号", color="white")
ax7.set_ylabel("累積損益 (pips)", color="white")
ax7.set_facecolor("#1a1a2e")
ax7.tick_params(colors="white")
ax7.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
for spine in ax7.spines.values():
    spine.set_edgecolor("#444")

# 月末重複バグの影響詳細（月別差分）
if not df_A.empty and not df_B.empty:
    monthly_A = df_A.groupby("month")["pnl"].sum()
    monthly_B = df_B.groupby("month")["pnl"].sum()
    diff = (monthly_B - monthly_A).fillna(0)
    colors_diff = ["#e74c3c" if v < 0 else "#2ecc71" for v in diff.values]
    ax8.bar(range(len(diff)), diff.values, color=colors_diff, edgecolor="white", linewidth=0.3)
    ax8.set_xticks(range(len(diff)))
    ax8.set_xticklabels(diff.index, rotation=45, fontsize=8)
    ax8.axhline(0, color="white", linewidth=0.6)
    ax8.set_title("月末重複バグによる月別損益の差分（B - A）", color="white", fontsize=11)
    ax8.set_ylabel("損益差 (pips)", color="white")
    ax8.set_facecolor("#1a1a2e")
    ax8.tick_params(colors="white")
    for spine in ax8.spines.values():
        spine.set_edgecolor("#444")

plt.tight_layout(pad=2.5)
out_img = f"{RESULTS}/v76_bug_impact_analysis.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

# ============================================================
# 詳細サマリーCSV保存
# ============================================================
rows = []
for s, period in [(s_A_is, "IS"), (s_A_oos, "OOS"), (s_A, "ALL"),
                  (s_B_is, "IS"), (s_B_oos, "OOS"), (s_B, "ALL"),
                  (s_C_is, "IS"), (s_C_oos, "OOS"), (s_C, "ALL"),
                  (s_D_is, "IS"), (s_D_oos, "OOS"), (s_D, "ALL")]:
    scenario_map = {"A": "修正後(正規)", "B": "月末重複バグ", "C": "27カラムバグ", "D": "両方バグ"}
    scenario = scenario_map[s['label'][0]]
    rows.append({
        "シナリオ": scenario,
        "期間": period,
        "トレード数": s['trades'],
        "勝率(%)": round(s['win_rate'], 1),
        "PF": round(s['pf'], 2) if s['pf'] != float('inf') else 999,
        "総損益(pips)": round(s['total_pnl'], 1),
        "平均利益(pips)": round(s['avg_win'], 2),
        "平均損失(pips)": round(s['avg_loss'], 2),
        "ケリー基準": round(s['kelly'], 3),
        "t統計量": round(s['t_stat'], 4),
        "p値": round(s['p_value'], 4),
        "プラス月": s['plus_months'],
    })

df_summary = pd.DataFrame(rows)
out_csv = f"{RESULTS}/v76_bug_impact_summary.csv"
df_summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"サマリーCSV保存: {out_csv}")
print("\n完了")
