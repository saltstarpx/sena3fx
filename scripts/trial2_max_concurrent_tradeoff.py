#!/usr/bin/env python3
"""
同時ポジション制限 トレードオフ分析
====================================
全期間トレードログを使い、同時MAX制限値ごとに
PnL/PF/MDD/スキップ数を計算する。

制限ルール: 同一4Hバケットで既にN銘柄がエントリー済みなら、
それ以降のエントリーをスキップする（entry_time順で先着優先）。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(OUT_DIR, "backtest_portfolio_fullperiod.csv")

INIT_CASH = 1_000_000

def load_trades():
    df = pd.read_csv(CSV_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"]  = pd.to_datetime(df["exit_time"])
    return df.sort_values("entry_time").reset_index(drop=True)

def pf(pnl):
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return gw / gl if gl > 0 else float("inf")

def calc_mdd(equity_series):
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return abs(dd.min()) * 100

def apply_max_concurrent(df, max_sym):
    """同一4Hバケットで先着max_sym銘柄まで、残りスキップ"""
    if max_sym is None:
        return df.copy()

    kept = []
    bucket_syms = {}  # h4_key -> set of symbols already entered

    for _, row in df.iterrows():
        key = row["h4_key"]
        if key not in bucket_syms:
            bucket_syms[key] = set()

        if len(bucket_syms[key]) < max_sym or row["sym"] in bucket_syms[key]:
            bucket_syms[key].add(row["sym"])
            kept.append(row)
        # else: skip this trade

    return pd.DataFrame(kept).reset_index(drop=True)

def apply_max_concurrent_priority(df, max_sym, priority_order):
    """同一4Hバケットで優先度順にmax_sym銘柄まで選択"""
    if max_sym is None:
        return df.copy()

    # Group by h4_key, then select top-priority symbols
    kept = []
    sym_priority = {s: i for i, s in enumerate(priority_order)}

    for key, grp in df.groupby("h4_key"):
        syms_in_bucket = grp["sym"].unique()
        if len(syms_in_bucket) <= max_sym:
            kept.append(grp)
        else:
            # Sort by priority, keep top max_sym symbols
            sorted_syms = sorted(syms_in_bucket, key=lambda s: sym_priority.get(s, 99))
            keep_syms = set(sorted_syms[:max_sym])
            kept.append(grp[grp["sym"].isin(keep_syms)])

    return pd.concat(kept).sort_values("entry_time").reset_index(drop=True)

def analyze(df_filtered, label):
    n = len(df_filtered)
    if n == 0:
        return {"label": label, "n": 0, "pnl": 0, "pf": 0, "wr": 0, "mdd": 0, "months_minus": 0}

    total_pnl = df_filtered["pnl"].sum()
    pf_val = pf(df_filtered["pnl"])
    wr = len(df_filtered[df_filtered["result"] == "win"]) / n

    # Portfolio MDD from cumulative PnL
    portfolio_eq = INIT_CASH + df_filtered.sort_values("exit_time")["pnl"].cumsum()
    mdd_val = calc_mdd(portfolio_eq)

    # Monthly analysis
    monthly = df_filtered.groupby(df_filtered["entry_time"].dt.strftime("%Y-%m"))["pnl"].sum()
    months_minus = (monthly < 0).sum()
    total_months = len(monthly)

    # Per-symbol breakdown
    sym_pnl = df_filtered.groupby("sym")["pnl"].sum()

    return {
        "label": label, "n": n, "pnl": total_pnl, "pf": pf_val,
        "wr": wr, "mdd": mdd_val, "months_minus": months_minus,
        "total_months": total_months, "sym_pnl": sym_pnl,
    }

def main():
    df = load_trades()
    total_n = len(df)
    print(f"全トレード数: {total_n}")
    print(f"期間: {df['entry_time'].min()} 〜 {df['entry_time'].max()}")

    # Per-symbol PF for priority ranking (higher PF = higher priority)
    sym_pfs = {}
    for sym, grp in df.groupby("sym"):
        sym_pfs[sym] = pf(grp["pnl"])
    priority_by_pf = sorted(sym_pfs, key=lambda s: sym_pfs[s], reverse=True)
    print(f"\n銘柄PF順（優先度）: {', '.join(f'{s}({sym_pfs[s]:.2f})' for s in priority_by_pf)}")

    # ── Method A: 先着順（entry_time順）──────────────────────────
    print("\n" + "=" * 90)
    print("  Method A: 先着順（entry_time順で先にエントリーした銘柄を優先）")
    print("=" * 90)

    limits = [None, 7, 6, 5, 4, 3, 2]
    results_a = []

    for lim in limits:
        label = f"MAX={lim}" if lim else "制限なし(8)"
        filtered = apply_max_concurrent(df, lim)
        r = analyze(filtered, label)
        results_a.append(r)
        skipped = total_n - r["n"]
        print(f"  {label:>12}  n={r['n']:>5} (skip={skipped:>4})  "
              f"PnL={r['pnl']:>14,.0f}  PF={r['pf']:>5.2f}  WR={r['wr']:.1%}  "
              f"MDD={r['mdd']:>5.1f}%  マイナス月={r['months_minus']}/{r['total_months']}")

    # ── Method B: PF優先順 ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Method B: PF優先順（PFが高い銘柄を優先的にキープ）")
    print("=" * 90)

    results_b = []
    for lim in limits:
        label = f"MAX={lim}" if lim else "制限なし(8)"
        filtered = apply_max_concurrent_priority(df, lim, priority_by_pf)
        r = analyze(filtered, label)
        results_b.append(r)
        skipped = total_n - r["n"]
        print(f"  {label:>12}  n={r['n']:>5} (skip={skipped:>4})  "
              f"PnL={r['pnl']:>14,.0f}  PF={r['pf']:>5.2f}  WR={r['wr']:.1%}  "
              f"MDD={r['mdd']:>5.1f}%  マイナス月={r['months_minus']}/{r['total_months']}")

    # ── 銘柄別影響分析 ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  銘柄別: 各制限値でスキップされるトレード数")
    print("=" * 90)

    all_syms = sorted(df["sym"].unique())
    header = f"{'制限値':>10}"
    for s in all_syms:
        header += f" {s:>8}"
    header += f" {'合計skip':>10}"
    print(header)
    print("-" * len(header))

    for lim in limits:
        label = f"MAX={lim}" if lim else "無制限"
        filtered = apply_max_concurrent(df, lim)
        row = f"{label:>10}"
        total_skip = 0
        for s in all_syms:
            orig = len(df[df["sym"] == s])
            kept = len(filtered[filtered["sym"] == s])
            skipped = orig - kept
            total_skip += skipped
            row += f" {skipped:>8}"
        row += f" {total_skip:>10}"
        print(row)

    # ── 2026年1-3月の集中損失分析 ───────────────────────────────
    print("\n" + "=" * 90)
    print("  2026年1-3月 限定分析（集中損失期間）")
    print("=" * 90)

    df_2026q1 = df[df["entry_time"] >= "2026-01-01"].copy()
    if len(df_2026q1) > 0:
        print(f"  2026Q1 トレード数: {len(df_2026q1)}")
        for lim in [None, 6, 5, 4, 3]:
            label = f"MAX={lim}" if lim else "制限なし"
            filtered = apply_max_concurrent(df_2026q1, lim)
            r = analyze(filtered, label)
            skipped = len(df_2026q1) - r["n"]
            print(f"  {label:>12}  n={r['n']:>5} (skip={skipped:>3})  "
                  f"PnL={r['pnl']:>14,.0f}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>5.1f}%")

    # ── 同時エントリー頻度の詳細 ───────────────────────────────
    print("\n" + "=" * 90)
    print("  同時エントリー頻度分布")
    print("=" * 90)

    h4_counts = df.groupby("h4_key")["sym"].nunique()
    total_buckets = len(h4_counts)
    for n in range(1, 9):
        cnt = (h4_counts == n).sum()
        pct = cnt / total_buckets * 100
        cum = (h4_counts >= n).sum()
        cum_pct = cum / total_buckets * 100
        print(f"  {n}銘柄: {cnt:>4}回 ({pct:>5.1f}%)  累積≥{n}: {cum:>4}回 ({cum_pct:>5.1f}%)")

    # ── 結論サマリー ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  結論サマリー")
    print("=" * 90)

    base = results_a[0]  # 制限なし
    print(f"\n  制限なし: PnL={base['pnl']:,.0f}  PF={base['pf']:.2f}  MDD={base['mdd']:.1f}%")
    print()
    for r in results_a[1:]:
        delta_pnl = r["pnl"] - base["pnl"]
        delta_pnl_pct = delta_pnl / base["pnl"] * 100
        delta_mdd = r["mdd"] - base["mdd"]
        print(f"  {r['label']:>12}  PnL差={delta_pnl:>+14,.0f} ({delta_pnl_pct:>+5.1f}%)  "
              f"MDD差={delta_mdd:>+5.1f}pp  PF={r['pf']:.2f}")

if __name__ == "__main__":
    main()
