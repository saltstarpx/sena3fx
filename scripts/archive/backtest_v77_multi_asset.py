"""
backtest_v77_multi_asset.py
===========================
v77ロジック（KMID+KLOWフィルター）を異なる市場構造のアセットで検証する。

対象アセット（相関の低い7種）:
  FX主要ペア: EURUSD, GBPUSD, AUDUSD
  コモディティ: XAUUSD（金）
  株価指数: SPX500, US30（ダウ）, NAS100（ナスダック）

比較ベース（JPYクロス）:
  USDJPY, EURJPY, GBPJPY（既存結果から参照）

スプレッド設定（実環境に近い値）:
  EURUSD: 0.2pips（0.00002）
  GBPUSD: 0.3pips（0.00003）
  AUDUSD: 0.3pips（0.00003）
  XAUUSD: 0.5USD
  SPX500: 1.0pt
  US30:   2.0pt
  NAS100: 2.0pt

pip換算係数（USD建て統一）:
  FXペア: 1pip = 0.0001（4桁）
  XAUUSD: 1pip = 1USD
  指数:   1pt = 1USD
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

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies/current")
from yagami_mtf_v77 import generate_signals

# ─── 設定 ──────────────────────────────────────────────────────────────────
DATA_DIR = "/home/ubuntu/sena3fx/data"
OUT_DIR  = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT_DIR, exist_ok=True)

# アセット設定
# spread: pips単位（FX）またはポイント単位（指数・金）
# pip_size: 1pipsの価格幅
# category: 市場分類
ASSETS = {
    "EURUSD": {
        "data_1m":  f"{DATA_DIR}/eurusd_1m.csv",
        "data_15m": f"{DATA_DIR}/eurusd_15m.csv",
        "data_4h":  f"{DATA_DIR}/eurusd_4h.csv",
        "spread":   0.2,
        "pip_size": 0.0001,
        "category": "FX主要ペア",
        "color":    "#3b82f6",
    },
    "GBPUSD": {
        "data_1m":  f"{DATA_DIR}/gbpusd_1m.csv",
        "data_15m": f"{DATA_DIR}/gbpusd_15m.csv",
        "data_4h":  f"{DATA_DIR}/gbpusd_4h.csv",
        "spread":   0.3,
        "pip_size": 0.0001,
        "category": "FX主要ペア",
        "color":    "#8b5cf6",
    },
    "AUDUSD": {
        "data_1m":  f"{DATA_DIR}/audusd_1m.csv",
        "data_15m": f"{DATA_DIR}/audusd_15m.csv",
        "data_4h":  f"{DATA_DIR}/audusd_4h.csv",
        "spread":   0.3,
        "pip_size": 0.0001,
        "category": "FX主要ペア",
        "color":    "#06b6d4",
    },
    "XAUUSD": {
        "data_1m":  f"{DATA_DIR}/xauusd_1m.csv",
        "data_15m": f"{DATA_DIR}/xauusd_15m.csv",
        "data_4h":  f"{DATA_DIR}/xauusd_4h.csv",
        "spread":   0.5,
        "pip_size": 1.0,
        "category": "コモディティ",
        "color":    "#f59e0b",
    },
    "SPX500": {
        "data_1m":  f"{DATA_DIR}/spx500_1m.csv",
        "data_15m": f"{DATA_DIR}/spx500_15m.csv",
        "data_4h":  f"{DATA_DIR}/spx500_4h.csv",
        "spread":   1.0,
        "pip_size": 1.0,
        "category": "株価指数",
        "color":    "#10b981",
    },
    "US30": {
        "data_1m":  f"{DATA_DIR}/us30_1m.csv",
        "data_15m": f"{DATA_DIR}/us30_15m.csv",
        "data_4h":  f"{DATA_DIR}/us30_4h.csv",
        "spread":   2.0,
        "pip_size": 1.0,
        "category": "株価指数",
        "color":    "#ef4444",
    },
    "NAS100": {
        "data_1m":  f"{DATA_DIR}/nas100_1m.csv",
        "data_15m": f"{DATA_DIR}/nas100_15m.csv",
        "data_4h":  f"{DATA_DIR}/nas100_4h.csv",
        "spread":   2.0,
        "pip_size": 1.0,
        "category": "株価指数",
        "color":    "#ec4899",
    },
}

# 共通期間
START_DATE = "2025-01-01"
END_DATE   = "2026-02-28"

# 半利確設定
HALF_CLOSE_R = 1.0   # 1R到達で半分決済

# 日本語フォント設定
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf",
]
jp_font = None
for f in font_candidates:
    if os.path.exists(f):
        jp_font = fm.FontProperties(fname=f)
        plt.rcParams["font.family"] = jp_font.get_name()
        break
if jp_font is None:
    plt.rcParams["font.family"] = "DejaVu Sans"


def load_data(path, start=None, end=None):
    """CSVを読み込んでDataFrameを返す"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # タイムゾーン情報を除去
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.set_index("timestamp").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def simulate_trades(signals, data_1m):
    """シグナルリストからトレードを模擬実行（半利確あり）"""
    trades = []
    for sig in signals:
        ep    = sig["ep"]
        sl    = sig["sl"]
        tp    = sig["tp"]
        risk  = sig["risk"]
        d     = sig["dir"]
        entry_time = sig["time"]
        half_tp = ep + d * risk * HALF_CLOSE_R

        # エントリー後の1分足を検索
        future = data_1m[data_1m.index > entry_time]
        if len(future) == 0:
            continue

        half_done = False
        exit_price = None
        exit_time  = None
        exit_reason = None

        for bar_time, bar in future.iterrows():
            h = bar["high"]
            l = bar["low"]

            if d == 1:  # ロング
                # 半利確チェック
                if not half_done and h >= half_tp:
                    half_done = True

                # SL判定
                if l <= sl:
                    if half_done:
                        # 半利確済み: 残り半分をSLで決済
                        exit_price  = sl
                        exit_time   = bar_time
                        exit_reason = "SL(half)"
                    else:
                        exit_price  = sl
                        exit_time   = bar_time
                        exit_reason = "SL"
                    break

                # TP判定
                if h >= tp:
                    exit_price  = tp
                    exit_time   = bar_time
                    exit_reason = "TP"
                    break

            else:  # ショート
                # 半利確チェック
                if not half_done and l <= half_tp:
                    half_done = True

                # SL判定
                if h >= sl:
                    if half_done:
                        exit_price  = sl
                        exit_time   = bar_time
                        exit_reason = "SL(half)"
                    else:
                        exit_price  = sl
                        exit_time   = bar_time
                        exit_reason = "SL"
                    break

                # TP判定
                if l <= tp:
                    exit_price  = tp
                    exit_time   = bar_time
                    exit_reason = "TP"
                    break

        if exit_price is None:
            continue

        # 損益計算（pips単位）
        raw_pnl_price = (exit_price - ep) * d
        pnl_pips = raw_pnl_price / sig.get("pip_size", 0.0001)

        # 半利確の場合の損益調整
        if exit_reason == "SL(half)":
            # 半分はhalf_tpで決済済み（+1R相当）、残り半分はSLで決済（-1R相当）
            half_pnl = (half_tp - ep) * d / sig.get("pip_size", 0.0001)
            sl_pnl   = (sl - ep) * d / sig.get("pip_size", 0.0001)
            pnl_pips = (half_pnl * 0.5) + (sl_pnl * 0.5)

        trades.append({
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "dir":         d,
            "ep":          ep,
            "sl":          sl,
            "tp":          tp,
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "pnl_pips":    pnl_pips,
            "risk_pips":   risk / sig.get("pip_size", 0.0001),
            "tf":          sig.get("tf", "?"),
            "pattern":     sig.get("pattern", "?"),
        })

    return pd.DataFrame(trades)


def calc_stats(trades_df, asset_name):
    """トレード統計を計算する"""
    if len(trades_df) == 0:
        return {
            "asset": asset_name,
            "trades": 0,
            "win_rate": 0,
            "pf": 0,
            "total_pips": 0,
            "mdd_pips": 0,
            "sharpe": 0,
            "kelly": 0,
            "p_value": 1.0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_consec_loss": 0,
        }

    wins  = trades_df[trades_df["pnl_pips"] > 0]["pnl_pips"]
    losses = trades_df[trades_df["pnl_pips"] <= 0]["pnl_pips"]

    win_rate = len(wins) / len(trades_df)
    gross_win  = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    total_pips = trades_df["pnl_pips"].sum()

    # 最大ドローダウン
    cumsum = trades_df["pnl_pips"].cumsum()
    rolling_max = cumsum.cummax()
    dd = rolling_max - cumsum
    mdd = dd.max()

    # シャープレシオ（月次）
    trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl_pips"].sum()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0

    # ケリー基準
    avg_win  = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0

    # p値（ブートストラップ）
    n_boot = 1000
    null_pf = []
    pnl_arr = trades_df["pnl_pips"].values
    for _ in range(n_boot):
        shuffled = np.random.choice([-1, 1], size=len(pnl_arr)) * np.abs(pnl_arr)
        w = shuffled[shuffled > 0].sum()
        l = abs(shuffled[shuffled <= 0].sum())
        null_pf.append(w / l if l > 0 else 1.0)
    p_value = np.mean(np.array(null_pf) >= pf)

    # 最大連敗
    consec = 0
    max_consec = 0
    for p in trades_df["pnl_pips"]:
        if p <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    return {
        "asset": asset_name,
        "trades": len(trades_df),
        "win_rate": round(win_rate * 100, 1),
        "pf": round(pf, 2),
        "total_pips": round(total_pips, 1),
        "mdd_pips": round(mdd, 1),
        "sharpe": round(sharpe, 2),
        "kelly": round(kelly, 3),
        "p_value": round(p_value, 4),
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
        "max_consec_loss": max_consec,
    }


def run_backtest_for_asset(asset_name, cfg):
    """1アセットのバックテストを実行する"""
    print(f"\n{'='*55}")
    print(f"[{asset_name}] ({cfg['category']})")

    # データファイルの存在確認
    for key in ["data_1m", "data_15m", "data_4h"]:
        if not os.path.exists(cfg[key]):
            print(f"  ✗ データファイルなし: {cfg[key]}")
            return None, None

    print(f"  データ読み込み中...", end="", flush=True)
    try:
        d1m  = load_data(cfg["data_1m"],  START_DATE, END_DATE)
        d15m = load_data(cfg["data_15m"], START_DATE, END_DATE)
        d4h  = load_data(cfg["data_4h"],  START_DATE, END_DATE)
    except Exception as e:
        print(f"\n  ✗ 読み込みエラー: {e}")
        return None, None

    print(f" 1m:{len(d1m)}行, 15m:{len(d15m)}行, 4h:{len(d4h)}行")

    # スプレッドをprice単位に変換
    spread_price = cfg["spread"] * cfg["pip_size"]

    print(f"  シグナル生成中...", end="", flush=True)
    try:
        signals = generate_signals(d1m, d15m, d4h, spread_pips=spread_price)
    except Exception as e:
        print(f"\n  ✗ シグナル生成エラー: {e}")
        return None, None

    # pip_sizeをシグナルに付加
    for s in signals:
        s["pip_size"] = cfg["pip_size"]

    print(f" {len(signals)}シグナル")

    if len(signals) == 0:
        print(f"  ✗ シグナルなし")
        return None, None

    print(f"  トレードシミュレーション中...", end="", flush=True)
    trades = simulate_trades(signals, d1m)
    print(f" {len(trades)}トレード")

    if len(trades) == 0:
        print(f"  ✗ トレードなし")
        return None, None

    stats = calc_stats(trades, asset_name)
    stats["category"] = cfg["category"]

    print(f"  結果: PF={stats['pf']}, 勝率={stats['win_rate']}%, "
          f"総損益={stats['total_pips']}pips, p={stats['p_value']}")

    return trades, stats


# ─── メイン処理 ─────────────────────────────────────────────────────────────
print("v77 多アセットバックテスト開始")
print(f"期間: {START_DATE} 〜 {END_DATE}")
print(f"対象: {list(ASSETS.keys())}")

all_stats = []
all_trades = {}

for asset_name, cfg in ASSETS.items():
    trades, stats = run_backtest_for_asset(asset_name, cfg)
    if stats is not None:
        all_stats.append(stats)
        all_trades[asset_name] = trades

# ─── 結果サマリー ──────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("=== v77 多アセットバックテスト結果サマリー ===")
print("="*70)

if all_stats:
    df_stats = pd.DataFrame(all_stats)
    df_stats = df_stats.sort_values("pf", ascending=False)

    print(df_stats[[
        "asset", "category", "trades", "win_rate", "pf",
        "total_pips", "mdd_pips", "sharpe", "kelly", "p_value"
    ]].to_string(index=False))

    # CSV保存
    out_csv = f"{OUT_DIR}/v77_multi_asset_summary.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"\n→ 結果CSV: {out_csv}")

    # ─── グラフ作成 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("v77 多アセットバックテスト結果 (2025/1〜2026/2)", fontsize=14, fontweight="bold")

    # 1. PF比較棒グラフ
    ax1 = axes[0, 0]
    colors = [ASSETS.get(a, {}).get("color", "#888") for a in df_stats["asset"]]
    bars = ax1.bar(df_stats["asset"], df_stats["pf"], color=colors, alpha=0.85)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="PF=1.0")
    ax1.axhline(y=1.5, color="orange", linestyle="--", linewidth=1, label="PF=1.5")
    ax1.set_title("プロフィットファクター比較")
    ax1.set_ylabel("PF")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, df_stats["pf"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # 2. 勝率比較
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df_stats["asset"], df_stats["win_rate"], color=colors, alpha=0.85)
    ax2.axhline(y=50, color="red", linestyle="--", linewidth=1, label="50%")
    ax2.set_title("勝率比較 (%)")
    ax2.set_ylabel("勝率 (%)")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars2, df_stats["win_rate"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # 3. 累積損益曲線
    ax3 = axes[1, 0]
    for asset_name, trades_df in all_trades.items():
        if trades_df is not None and len(trades_df) > 0:
            cumsum = trades_df["pnl_pips"].cumsum()
            color = ASSETS.get(asset_name, {}).get("color", "#888")
            ax3.plot(range(len(cumsum)), cumsum.values, label=asset_name,
                     color=color, linewidth=1.5)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_title("累積損益曲線 (pips)")
    ax3.set_xlabel("トレード番号")
    ax3.set_ylabel("累積損益 (pips)")
    ax3.legend(fontsize=8)

    # 4. ケリー基準 vs シャープレシオ散布図
    ax4 = axes[1, 1]
    for _, row in df_stats.iterrows():
        color = ASSETS.get(row["asset"], {}).get("color", "#888")
        ax4.scatter(row["kelly"], row["sharpe"], color=color, s=100, zorder=5)
        ax4.annotate(row["asset"], (row["kelly"], row["sharpe"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.axvline(x=0, color="gray", linewidth=0.5)
    ax4.set_title("ケリー基準 vs シャープレシオ")
    ax4.set_xlabel("ケリー基準")
    ax4.set_ylabel("シャープレシオ（月次）")

    plt.tight_layout()
    out_img = f"{OUT_DIR}/v77_multi_asset_chart.png"
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ グラフ: {out_img}")

    # ─── カテゴリ別分析 ──────────────────────────────────────────────────
    print("\n=== カテゴリ別サマリー ===")
    cat_summary = df_stats.groupby("category").agg({
        "pf": "mean",
        "win_rate": "mean",
        "trades": "sum",
        "total_pips": "sum",
    }).round(2)
    print(cat_summary)

    # ─── 月次分析 ────────────────────────────────────────────────────────
    print("\n=== 月次勝率（各アセット） ===")
    for asset_name, trades_df in all_trades.items():
        if trades_df is not None and len(trades_df) > 0:
            trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
            monthly = trades_df.groupby("month")["pnl_pips"].sum()
            pos_months = (monthly > 0).sum()
            total_months = len(monthly)
            print(f"  {asset_name}: {pos_months}/{total_months}月プラス")

print("\n完了")
