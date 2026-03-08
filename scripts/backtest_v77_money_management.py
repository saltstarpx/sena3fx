"""
backtest_v77_money_management.py
=================================
v77ロジック + 資金管理（RiskManager / AdaptiveRiskManager）+ PositionManager
統合バックテストスクリプト

【概要】
- v77戦略のシグナル生成 → 円建てロットサイズ計算 → 損益を円で算出
- 固定リスク2% vs AdaptiveRiskManager（資産規模×DD連動型）の比較
- PositionManagerによるポジション制約（同銘柄1ポジ限定）

【データ】
- 1分足データがないため、15分足をエントリーデータとして代用
- エントリーウィンドウ: 足更新後の最初の15分足始値（本来は1分足2分以内）

【実行】
    cd /home/user/sena3fx
    python scripts/backtest_v77_money_management.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# パス設定
ROOT = "/home/user/sena3fx"
sys.path.insert(0, ROOT)

from strategies.current.yagami_mtf_v77 import generate_signals
from utils.risk_manager import RiskManager, AdaptiveRiskManager
from utils.position_manager import PositionManager

# 日本語フォント設定
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# ── 設定 ──────────────────────────────────────────────────────
DATA_DIR    = f"{ROOT}/data"
RESULTS_DIR = f"{ROOT}/results"
SYMBOL      = "USDJPY"
SPREAD_PIPS = 0.42       # Exness ゼロ口座 USDJPY 実測スプレッド
RR_RATIO    = 2.5
INIT_EQUITY = 1_000_000  # 初期資金: 100万円
IS_END      = "2025-03-01"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# データ読み込み
# ============================================================
def load(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

print("=" * 70)
print("v77 + 資金管理 統合バックテスト")
print("=" * 70)
print(f"銘柄: {SYMBOL}  初期資金: {INIT_EQUITY:,}円  スプレッド: {SPREAD_PIPS}pips")
print()

print("データ読み込み・結合中...")
data_15m = pd.concat([
    load(f"{DATA_DIR}/usdjpy_is_15m.csv"),
    load(f"{DATA_DIR}/usdjpy_oos_15m.csv"),
]).sort_index()
data_15m = data_15m[~data_15m.index.duplicated(keep='first')]

data_1h = pd.concat([
    load(f"{DATA_DIR}/usdjpy_is_1h.csv"),
    load(f"{DATA_DIR}/usdjpy_oos_1h.csv"),
]).sort_index()
data_1h = data_1h[~data_1h.index.duplicated(keep='first')]

data_4h = pd.concat([
    load(f"{DATA_DIR}/usdjpy_is_4h.csv"),
    load(f"{DATA_DIR}/usdjpy_oos_4h.csv"),
]).sort_index()
data_4h = data_4h[~data_4h.index.duplicated(keep='first')]

print(f"15分足: {len(data_15m):,}行  {data_15m.index[0]} 〜 {data_15m.index[-1]}")
print(f"1時間足: {len(data_1h):,}行  {data_1h.index[0]} 〜 {data_1h.index[-1]}")
print(f"4時間足: {len(data_4h):,}行  {data_4h.index[0]} 〜 {data_4h.index[-1]}")

# ============================================================
# シグナル生成（15分足を1分足の代わりに使用）
# ============================================================
print("\nシグナル生成中（v77 / 15分足エントリー）...")
# generate_signals は data_1m の引数に15分足を渡す
# エントリーウィンドウは「足更新後2分以内」→15分足では最初の15分足の始値
signals = generate_signals(data_15m, data_15m, data_4h,
                           spread_pips=SPREAD_PIPS, rr_ratio=RR_RATIO)
sig_map = {s["time"]: s for s in signals}
print(f"シグナル数: {len(signals)}")


# ============================================================
# バックテストエンジン（資金管理統合版）
# ============================================================
def run_backtest(signals, data_15m, mode="fixed", init_equity=INIT_EQUITY):
    """
    mode: "fixed" = 固定2% RiskManager
          "adaptive" = AdaptiveRiskManager（資産規模×DD連動型）
    """
    if mode == "fixed":
        rm = RiskManager(SYMBOL, risk_pct=0.02)
    else:
        rm = AdaptiveRiskManager(SYMBOL, base_risk_pct=0.02)

    pm = PositionManager()
    equity = init_equity
    peak_equity = init_equity
    trades = []
    pos = None
    equity_curve = []  # (time, equity) の記録

    sig_map_local = {s["time"]: s for s in signals}

    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name

        if pos is not None:
            d = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d

            # 半利確チェック
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"] <= half_tp):
                    # 半分のロットを1Rで決済
                    half_lot = pos["lot"] / 2
                    half_pnl_jpy = rm.calc_pnl_jpy(
                        direction=d, ep=pos["ep"], exit_price=half_tp,
                        lot=half_lot, usdjpy_rate=bar["close"]
                    )
                    pos["half_pnl_jpy"] = half_pnl_jpy
                    pos["sl"] = raw_ep  # SLをBEへ移動
                    pos["half_closed"] = True
                    pos["remaining_lot"] = half_lot  # 残り半分

            current_lot = pos.get("remaining_lot", pos["lot"])

            # SL到達
            if (d == 1 and bar["low"] <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl_jpy = rm.calc_pnl_jpy(
                    direction=d, ep=pos["ep"], exit_price=pos["sl"],
                    lot=current_lot, usdjpy_rate=bar["close"]
                )
                total_pnl_jpy = pos.get("half_pnl_jpy", 0) + sl_pnl_jpy
                equity += total_pnl_jpy

                # pips計算（参照用）
                sl_pnl_pips = (pos["sl"] - pos["ep"]) * 100 * d
                half_pnl_pips = pos.get("half_pnl_pips", 0)
                if pos["half_closed"] and "half_pnl_pips" not in pos:
                    half_pnl_pips = (half_tp - pos["ep"]) * 100 * d

                period = "IS" if str(pos["entry_time"]) < IS_END else "OOS"
                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "dir": d,
                    "ep": pos["ep"],
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "lot": pos["lot"],
                    "risk_pct": pos["risk_pct"],
                    "pnl_jpy": total_pnl_jpy,
                    "pnl_pips": half_pnl_pips + sl_pnl_pips,
                    "equity_after": equity,
                    "result": "win" if total_pnl_jpy > 0 else "loss",
                    "exit_type": "HALF+SL" if pos["half_closed"] else "SL",
                    "tf": pos.get("tf", "?"),
                    "pattern": pos.get("pattern", "?"),
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": period,
                })

                # AdaptiveRiskManager: ピーク更新
                if mode == "adaptive":
                    rm.update_peak(equity)
                peak_equity = max(peak_equity, equity)

                pm.close_position(SYMBOL)
                pos = None
                equity_curve.append((t, equity))
                continue

            # TP到達
            if (d == 1 and bar["high"] >= pos["tp"]) or \
               (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl_jpy = rm.calc_pnl_jpy(
                    direction=d, ep=pos["ep"], exit_price=pos["tp"],
                    lot=current_lot, usdjpy_rate=bar["close"]
                )
                total_pnl_jpy = pos.get("half_pnl_jpy", 0) + tp_pnl_jpy
                equity += total_pnl_jpy

                tp_pnl_pips = (pos["tp"] - pos["ep"]) * 100 * d
                half_pnl_pips = pos.get("half_pnl_pips", 0)
                if pos["half_closed"] and "half_pnl_pips" not in pos:
                    half_pnl_pips = (half_tp - pos["ep"]) * 100 * d

                period = "IS" if str(pos["entry_time"]) < IS_END else "OOS"
                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "dir": d,
                    "ep": pos["ep"],
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "lot": pos["lot"],
                    "risk_pct": pos["risk_pct"],
                    "pnl_jpy": total_pnl_jpy,
                    "pnl_pips": half_pnl_pips + tp_pnl_pips,
                    "equity_after": equity,
                    "result": "win" if total_pnl_jpy > 0 else "loss",
                    "exit_type": "HALF+TP" if pos["half_closed"] else "TP",
                    "tf": pos.get("tf", "?"),
                    "pattern": pos.get("pattern", "?"),
                    "month": pos["entry_time"].strftime("%Y-%m"),
                    "period": period,
                })

                if mode == "adaptive":
                    rm.update_peak(equity)
                peak_equity = max(peak_equity, equity)

                pm.close_position(SYMBOL)
                pos = None
                equity_curve.append((t, equity))
                continue

        # 新規エントリー
        if pos is None and t in sig_map_local:
            sig = sig_map_local[t]

            # PositionManagerチェック（USDJPYは1ポジのみ）
            can, reason = pm.can_enter(SYMBOL, risk_pct=0.02)
            if not can:
                continue

            sl_distance = sig["risk"]  # チャートレベルのSLまでの距離

            # ロットサイズ計算
            if mode == "adaptive":
                lot, eff_risk, risk_reason = rm.calc_lot_adaptive(
                    equity=equity,
                    sl_distance=sl_distance,
                    ref_price=sig["ep"],
                    usdjpy_rate=bar["close"],
                )
                risk_pct_used = eff_risk
            else:
                lot = rm.calc_lot(
                    equity=equity,
                    sl_distance=sl_distance,
                    ref_price=sig["ep"],
                    usdjpy_rate=bar["close"],
                )
                risk_pct_used = rm.risk_pct

            if lot <= 0:
                continue

            pos = {
                **sig,
                "entry_time": t,
                "half_closed": False,
                "lot": lot,
                "risk_pct": risk_pct_used,
            }
            pm.open_position(SYMBOL, risk_pct=risk_pct_used, entry_time=t)

    return pd.DataFrame(trades), equity_curve


# ============================================================
# 両モードで実行
# ============================================================
print("\n--- 固定リスク2%モード ---")
df_fixed, ec_fixed = run_backtest(signals, data_15m, mode="fixed")
print(f"完了: {len(df_fixed)}トレード  最終資産: {df_fixed['equity_after'].iloc[-1]:,.0f}円" if len(df_fixed) > 0 else "トレードなし")

print("\n--- AdaptiveRiskManagerモード ---")
df_adaptive, ec_adaptive = run_backtest(signals, data_15m, mode="adaptive")
print(f"完了: {len(df_adaptive)}トレード  最終資産: {df_adaptive['equity_after'].iloc[-1]:,.0f}円" if len(df_adaptive) > 0 else "トレードなし")


# ============================================================
# 統計計算
# ============================================================
def calc_stats(df_sub, label=""):
    if df_sub.empty:
        return {}
    wins   = df_sub[df_sub["pnl_jpy"] > 0]
    losses = df_sub[df_sub["pnl_jpy"] < 0]
    pf     = wins["pnl_jpy"].sum() / abs(losses["pnl_jpy"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df_sub) * 100
    avg_w  = wins["pnl_jpy"].mean() if len(wins) > 0 else 0
    avg_l  = losses["pnl_jpy"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df_sub["pnl_jpy"], 0) if len(df_sub) > 1 else (0, 1)
    monthly = df_sub.groupby("month")["pnl_jpy"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)

    # 最大ドローダウン（円）
    cumulative = df_sub.sort_values("entry_time")["pnl_jpy"].cumsum()
    rolling_max = cumulative.cummax()
    max_dd_jpy = (rolling_max - cumulative).max()

    # 最大DD率（%）
    equity_series = df_sub.sort_values("entry_time")["equity_after"]
    eq_peak = equity_series.cummax()
    dd_rate = ((eq_peak - equity_series) / eq_peak * 100).max()

    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0

    return dict(
        label=label, trades=len(df_sub), win_rate=wr, pf=pf,
        total_pnl=df_sub["pnl_jpy"].sum(),
        final_equity=df_sub["equity_after"].iloc[-1] if len(df_sub) > 0 else INIT_EQUITY,
        avg_win=avg_w, avg_loss=avg_l,
        kelly=kelly, t_stat=t_stat, p_value=p_val,
        plus_months=f"{plus_months}/{total_months}",
        max_dd_jpy=max_dd_jpy, max_dd_pct=dd_rate, sharpe=sharpe,
        monthly=monthly,
    )


def print_stats(s):
    if not s:
        return
    pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "∞"
    print(f"[{s['label']}]")
    print(f"  トレード数: {s['trades']}回  勝率: {s['win_rate']:.1f}%  PF: {pf_str}")
    print(f"  総損益: {s['total_pnl']:+,.0f}円  最終資産: {s['final_equity']:,.0f}円")
    print(f"  平均利益: {s['avg_win']:+,.0f}円  平均損失: {s['avg_loss']:+,.0f}円")
    print(f"  ケリー: {s['kelly']:.3f}  最大DD: {s['max_dd_jpy']:,.0f}円 ({s['max_dd_pct']:.1f}%)")
    print(f"  月次シャープ: {s['sharpe']:.2f}  p値: {s['p_value']:.4f}  プラス月: {s['plus_months']}")
    print()


# 統計計算
s_fixed_all = calc_stats(df_fixed, "固定2% 全期間")
s_fixed_is  = calc_stats(df_fixed[df_fixed["period"] == "IS"], "固定2% IS期間")
s_fixed_oos = calc_stats(df_fixed[df_fixed["period"] == "OOS"], "固定2% OOS期間")

s_adapt_all = calc_stats(df_adaptive, "Adaptive 全期間")
s_adapt_is  = calc_stats(df_adaptive[df_adaptive["period"] == "IS"], "Adaptive IS期間")
s_adapt_oos = calc_stats(df_adaptive[df_adaptive["period"] == "OOS"], "Adaptive OOS期間")

print("\n" + "=" * 70)
print(f"v77 + 資金管理 バックテスト結果（{SYMBOL} / spread={SPREAD_PIPS}pips / 初期資金{INIT_EQUITY:,}円）")
print("=" * 70)

print("\n━━━ 固定リスク2% ━━━")
for s in [s_fixed_all, s_fixed_is, s_fixed_oos]:
    print_stats(s)

print("\n━━━ AdaptiveRiskManager（資産規模×DD連動型）━━━")
for s in [s_adapt_all, s_adapt_is, s_adapt_oos]:
    print_stats(s)

# ── 比較テーブル ──
print("━━━ 比較サマリー ━━━")
print(f"{'指標':<20} {'固定2%':>14} {'Adaptive':>14}")
print("-" * 50)
if s_fixed_all and s_adapt_all:
    rows = [
        ("トレード数", f"{s_fixed_all['trades']}回", f"{s_adapt_all['trades']}回"),
        ("勝率", f"{s_fixed_all['win_rate']:.1f}%", f"{s_adapt_all['win_rate']:.1f}%"),
        ("PF", f"{s_fixed_all['pf']:.2f}", f"{s_adapt_all['pf']:.2f}"),
        ("総損益", f"{s_fixed_all['total_pnl']:+,.0f}円", f"{s_adapt_all['total_pnl']:+,.0f}円"),
        ("最終資産", f"{s_fixed_all['final_equity']:,.0f}円", f"{s_adapt_all['final_equity']:,.0f}円"),
        ("最大DD(円)", f"{s_fixed_all['max_dd_jpy']:,.0f}円", f"{s_adapt_all['max_dd_jpy']:,.0f}円"),
        ("最大DD(%)", f"{s_fixed_all['max_dd_pct']:.1f}%", f"{s_adapt_all['max_dd_pct']:.1f}%"),
        ("ケリー基準", f"{s_fixed_all['kelly']:.3f}", f"{s_adapt_all['kelly']:.3f}"),
        ("月次シャープ", f"{s_fixed_all['sharpe']:.2f}", f"{s_adapt_all['sharpe']:.2f}"),
        ("プラス月", s_fixed_all['plus_months'], s_adapt_all['plus_months']),
    ]
    for label, v1, v2 in rows:
        print(f"{label:<20} {v1:>14} {v2:>14}")

# ============================================================
# 可視化
# ============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0d0d1a")

C_FIXED   = "#3498db"   # 青: 固定2%
C_ADAPT   = "#e74c3c"   # 赤: Adaptive
C_IS      = "#2ecc71"   # 緑
C_OOS     = "#f39c12"   # 黄

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=9)

# ── 1. エクイティカーブ比較（円建て）──
ax1 = fig.add_subplot(4, 2, (1, 2))
if len(df_fixed) > 0:
    df_f = df_fixed.sort_values("entry_time").reset_index(drop=True)
    ax1.plot(range(len(df_f)), df_f["equity_after"],
             color=C_FIXED, linewidth=1.8,
             label=f"固定2%  最終: {df_f['equity_after'].iloc[-1]:,.0f}円")
if len(df_adaptive) > 0:
    df_a = df_adaptive.sort_values("entry_time").reset_index(drop=True)
    ax1.plot(range(len(df_a)), df_a["equity_after"],
             color=C_ADAPT, linewidth=1.8,
             label=f"Adaptive  最終: {df_a['equity_after'].iloc[-1]:,.0f}円")

ax1.axhline(INIT_EQUITY, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
styled_ax(ax1,
          f"エクイティカーブ比較（v77 / {SYMBOL} / 初期{INIT_EQUITY:,}円）",
          xlabel="トレード番号", ylabel="資産額 (円)")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# ── 2. 月別損益比較（円建て）──
ax2 = fig.add_subplot(4, 2, (3, 4))
if len(df_fixed) > 0 and len(df_adaptive) > 0:
    monthly_f = df_fixed.groupby("month")["pnl_jpy"].sum()
    monthly_a = df_adaptive.groupby("month")["pnl_jpy"].sum()
    all_months = sorted(set(monthly_f.index) | set(monthly_a.index))
    x_pos = np.arange(len(all_months))
    w = 0.35
    vals_f = [monthly_f.get(m, 0) for m in all_months]
    vals_a = [monthly_a.get(m, 0) for m in all_months]
    ax2.bar(x_pos - w/2, vals_f, w, label="固定2%", color=C_FIXED, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.bar(x_pos + w/2, vals_a, w, label="Adaptive", color=C_ADAPT, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axhline(0, color="white", linewidth=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(all_months, rotation=45, fontsize=7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax2, "月別損益比較（円）", ylabel="損益 (円)")

# ── 3. ドローダウン比較（%）──
ax3 = fig.add_subplot(4, 2, (5, 6))
for df_bt, color, label in [
    (df_fixed, C_FIXED, "固定2%"),
    (df_adaptive, C_ADAPT, "Adaptive"),
]:
    if len(df_bt) > 0:
        eq = df_bt.sort_values("entry_time")["equity_after"].reset_index(drop=True)
        peak = eq.cummax()
        dd_pct = (peak - eq) / peak * 100
        ax3.fill_between(range(len(dd_pct)), -dd_pct.values, 0,
                         color=color, alpha=0.4, label=f"{label} 最大DD: {dd_pct.max():.1f}%")
ax3.axhline(0, color="white", linewidth=0.5)
styled_ax(ax3, "ドローダウン推移（%）", xlabel="トレード番号", ylabel="ドローダウン (%)")
ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── 4. リスク%推移（Adaptiveのみ）──
ax4 = fig.add_subplot(4, 2, 7)
if len(df_adaptive) > 0:
    risk_series = df_adaptive.sort_values("entry_time")["risk_pct"].reset_index(drop=True) * 100
    ax4.plot(range(len(risk_series)), risk_series, color=C_ADAPT, linewidth=1.2)
    ax4.axhline(2.0, color="white", linewidth=0.5, linestyle="--", alpha=0.4, label="ベース 2%")
styled_ax(ax4, "Adaptive リスク%推移", xlabel="トレード番号", ylabel="リスク率 (%)")
ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── 5. ロットサイズ推移比較 ──
ax5 = fig.add_subplot(4, 2, 8)
for df_bt, color, label in [
    (df_fixed, C_FIXED, "固定2%"),
    (df_adaptive, C_ADAPT, "Adaptive"),
]:
    if len(df_bt) > 0:
        lots = df_bt.sort_values("entry_time")["lot"].reset_index(drop=True)
        ax5.plot(range(len(lots)), lots, color=color, linewidth=1.0, alpha=0.8, label=label)
styled_ax(ax5, "ロットサイズ推移（通貨数）", xlabel="トレード番号", ylabel="ロット (通貨)")
ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout(pad=2.5)
out_img = f"{RESULTS_DIR}/v77_money_management_backtest.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

# ============================================================
# CSV出力
# ============================================================
# 全トレード詳細
for mode_name, df_bt in [("fixed", df_fixed), ("adaptive", df_adaptive)]:
    if len(df_bt) > 0:
        out_csv = f"{RESULTS_DIR}/v77_mm_{mode_name}_trades.csv"
        df_bt.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"トレード詳細: {out_csv}")

# 比較サマリーCSV
summary_rows = []
for s in [s_fixed_all, s_fixed_is, s_fixed_oos, s_adapt_all, s_adapt_is, s_adapt_oos]:
    if not s:
        continue
    summary_rows.append({
        "区分": s["label"],
        "トレード数": s["trades"],
        "勝率(%)": round(s["win_rate"], 1),
        "PF": round(s["pf"], 2) if s["pf"] != float("inf") else 999,
        "総損益(円)": round(s["total_pnl"]),
        "最終資産(円)": round(s["final_equity"]),
        "平均利益(円)": round(s["avg_win"]),
        "平均損失(円)": round(s["avg_loss"]),
        "ケリー基準": round(s["kelly"], 3),
        "最大DD(円)": round(s["max_dd_jpy"]),
        "最大DD(%)": round(s["max_dd_pct"], 1),
        "月次シャープ": round(s["sharpe"], 2),
        "p値": round(s["p_value"], 4),
        "プラス月": s["plus_months"],
    })
pd.DataFrame(summary_rows).to_csv(
    f"{RESULTS_DIR}/v77_mm_comparison_summary.csv", index=False, encoding="utf-8-sig")
print(f"比較サマリー: {RESULTS_DIR}/v77_mm_comparison_summary.csv")

print("\n完了")
