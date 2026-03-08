"""
US30・US500のティックデータからスプレッドを計算してIS/OOS期間で比較する。
スプレッド = ASK - BID（ポイント単位）
"""
import os
import zipfile
from io import StringIO

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

UPLOAD_DIR = "/home/ubuntu/upload"
OUT_DIR    = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT_DIR, exist_ok=True)

IS_START  = pd.Timestamp("2025-01-01", tz="UTC")
IS_END    = pd.Timestamp("2025-03-01", tz="UTC")
OOS_START = pd.Timestamp("2025-03-03", tz="UTC")
OOS_END   = pd.Timestamp("2026-02-28", tz="UTC")

INSTRUMENTS = {
    "US30":  "us30",
    "US500": "spx500",
}


def load_ticks(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            content = f.read().decode("utf-8", errors="replace")

    df = pd.read_csv(
        StringIO(content),
        sep="\t",
        header=0,
        names=["date", "time", "bid", "ask", "last", "volume"],
        dtype={"bid": float, "ask": float},
        na_values=["", " "],
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["spread"] = df["ask"] - df["bid"]
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S.%f",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    return df[["timestamp", "bid", "ask", "spread"]]


def calc_stats(df: pd.DataFrame) -> dict:
    s = df["spread"]
    return {
        "count":    len(s),
        "mean":     s.mean(),
        "median":   s.median(),
        "std":      s.std(),
        "p25":      s.quantile(0.25),
        "p75":      s.quantile(0.75),
        "p95":      s.quantile(0.95),
        "min":      s.min(),
        "max":      s.max(),
    }


def main():
    results = {}

    for instrument_key, prefix in INSTRUMENTS.items():
        print(f"\n{'='*50}")
        print(f"銘柄: {instrument_key}")
        print(f"{'='*50}")

        zip_files = sorted([
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.startswith(f"ticks_{instrument_key}_") and f.endswith(".zip")
        ])

        all_ticks = []
        for zf in zip_files:
            print(f"  読み込み中: {os.path.basename(zf)}")
            ticks = load_ticks(zf)
            all_ticks.append(ticks)

        df_all = pd.concat(all_ticks, ignore_index=True)
        df_all = df_all.sort_values("timestamp").reset_index(drop=True)

        # IS / OOS 分割
        df_is  = df_all[(df_all["timestamp"] >= IS_START)  & (df_all["timestamp"] < IS_END)]
        df_oos = df_all[(df_all["timestamp"] >= OOS_START) & (df_all["timestamp"] < OOS_END)]

        stats_is  = calc_stats(df_is)
        stats_oos = calc_stats(df_oos)

        results[instrument_key] = {"IS": stats_is, "OOS": stats_oos}

        print(f"  IS  ({IS_START.date()}〜{IS_END.date()}): {stats_is['count']:,}ティック | 平均={stats_is['mean']:.2f} | 中央値={stats_is['median']:.2f}")
        print(f"  OOS ({OOS_START.date()}〜{OOS_END.date()}): {stats_oos['count']:,}ティック | 平均={stats_oos['mean']:.2f} | 中央値={stats_oos['median']:.2f}")

        # 月次平均スプレッド
        df_all["ym"] = df_all["timestamp"].dt.to_period("M")
        monthly = df_all.groupby("ym")["spread"].mean()
        results[instrument_key]["monthly"] = monthly

    # ── 可視化 ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")
    plt.suptitle("US30 / US500 スプレッド分析（IS vs OOS）", color="white", fontsize=15, fontweight="bold", y=0.98)

    colors = {"IS": "#60a5fa", "OOS": "#34d399"}
    instruments = list(results.keys())

    # ── 上段: 棒グラフ（平均・中央値・P95比較） ──
    for idx, inst in enumerate(instruments):
        ax = axes[0][idx]
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        periods = ["IS", "OOS"]
        metrics = ["mean", "median", "p95"]
        labels  = ["平均", "中央値", "P95"]
        x = np.arange(len(metrics))
        width = 0.35

        for i, period in enumerate(periods):
            vals = [results[inst][period][m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=period,
                          color=colors[period], alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=9)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels, color="white", fontsize=10)
        ax.set_title(f"{inst} スプレッド統計（ポイント）", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("スプレッド（ポイント）", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── 下段: 月次スプレッド推移 ──
    for idx, inst in enumerate(instruments):
        ax = axes[1][idx]
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        monthly = results[inst]["monthly"]
        xs = [str(p) for p in monthly.index]
        ys = monthly.values

        # IS/OOS 色分け
        is_mask  = [pd.Period(p) >= pd.Period(IS_START.strftime("%Y-%m"))  and
                    pd.Period(p) <  pd.Period(IS_END.strftime("%Y-%m"))
                    for p in monthly.index]
        oos_mask = [pd.Period(p) >= pd.Period(OOS_START.strftime("%Y-%m")) and
                    pd.Period(p) <  pd.Period(OOS_END.strftime("%Y-%m"))
                    for p in monthly.index]

        ax.plot(xs, ys, color="#94a3b8", linewidth=1, zorder=1)
        ax.scatter([x for x, m in zip(xs, is_mask)  if m],
                   [y for y, m in zip(ys, is_mask)  if m],
                   color=colors["IS"],  s=50, zorder=2, label="IS")
        ax.scatter([x for x, m in zip(xs, oos_mask) if m],
                   [y for y, m in zip(ys, oos_mask) if m],
                   color=colors["OOS"], s=50, zorder=2, label="OOS")

        ax.set_title(f"{inst} 月次平均スプレッド推移", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("スプレッド（ポイント）", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "spread_analysis_us30_us500.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nチャート保存: {out_png}")

    # ── サマリーテーブル出力 ──
    print("\n" + "="*60)
    print("スプレッド比較サマリー（ポイント単位）")
    print("="*60)
    header = f"{'銘柄':<8} {'期間':<5} {'ティック数':>12} {'平均':>8} {'中央値':>8} {'P95':>8} {'最大':>8}"
    print(header)
    print("-" * 60)
    for inst in instruments:
        for period in ["IS", "OOS"]:
            s = results[inst][period]
            print(f"{inst:<8} {period:<5} {s['count']:>12,} {s['mean']:>8.2f} {s['median']:>8.2f} {s['p95']:>8.2f} {s['max']:>8.2f}")

    return results


if __name__ == "__main__":
    main()
