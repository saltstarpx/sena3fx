#!/usr/bin/env python3
"""
同時ポジション制限 トレードオフ分析 v2
======================================
穴①②③を潰す:
  穴①: スキップされたトレードの勝率 vs 採用トレードの勝率
  穴②: 2026Q1単独でのMDD・月次分析（好調期間で薄めない）
  穴③: 月次単位で制限の効果を見る（最悪月の改善度）

制限ルール:
  - 先着順: entry_time順で先にエントリーした銘柄を優先
  - PF優先順: 全期間PFが高い銘柄を優先
  - 最悪銘柄除外: そのバケットでPFが最低の銘柄をスキップ
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

def calc_mdd_details(equity_series):
    eq = equity_series.reset_index(drop=True).astype(float)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    trough_idx = dd.idxmin()
    mdd_pct = abs(dd.iloc[trough_idx]) * 100
    return mdd_pct

def apply_max_concurrent(df, max_sym):
    """先着順: entry_time順でmax_sym銘柄まで"""
    if max_sym is None:
        return df.copy(), pd.DataFrame()
    kept = []; skipped = []
    bucket_syms = {}
    for _, row in df.iterrows():
        key = row["h4_key"]
        if key not in bucket_syms:
            bucket_syms[key] = set()
        if len(bucket_syms[key]) < max_sym or row["sym"] in bucket_syms[key]:
            bucket_syms[key].add(row["sym"])
            kept.append(row)
        else:
            skipped.append(row)
    return (pd.DataFrame(kept).reset_index(drop=True),
            pd.DataFrame(skipped).reset_index(drop=True))

def apply_max_concurrent_priority(df, max_sym, priority_order):
    """PF優先順: PFが高い銘柄を優先的にキープ"""
    if max_sym is None:
        return df.copy(), pd.DataFrame()
    sym_priority = {s: i for i, s in enumerate(priority_order)}
    kept = []; skipped = []
    for key, grp in df.groupby("h4_key"):
        syms_in_bucket = grp["sym"].unique()
        if len(syms_in_bucket) <= max_sym:
            kept.append(grp)
        else:
            sorted_syms = sorted(syms_in_bucket, key=lambda s: sym_priority.get(s, 99))
            keep_syms = set(sorted_syms[:max_sym])
            kept.append(grp[grp["sym"].isin(keep_syms)])
            skipped.append(grp[~grp["sym"].isin(keep_syms)])
    kept_df = pd.concat(kept).sort_values("entry_time").reset_index(drop=True) if kept else pd.DataFrame()
    skip_df = pd.concat(skipped).sort_values("entry_time").reset_index(drop=True) if skipped else pd.DataFrame()
    return kept_df, skip_df

def apply_max_worst_skip(df, max_sym, sym_pfs):
    """最悪銘柄スキップ: そのバケットでPFが最低の銘柄を除外"""
    if max_sym is None:
        return df.copy(), pd.DataFrame()
    kept = []; skipped = []
    for key, grp in df.groupby("h4_key"):
        syms_in_bucket = grp["sym"].unique()
        if len(syms_in_bucket) <= max_sym:
            kept.append(grp)
        else:
            # PFが低い順にスキップ
            sorted_syms = sorted(syms_in_bucket, key=lambda s: sym_pfs.get(s, 0))
            n_skip = len(syms_in_bucket) - max_sym
            skip_syms = set(sorted_syms[:n_skip])
            kept.append(grp[~grp["sym"].isin(skip_syms)])
            skipped.append(grp[grp["sym"].isin(skip_syms)])
    kept_df = pd.concat(kept).sort_values("entry_time").reset_index(drop=True) if kept else pd.DataFrame()
    skip_df = pd.concat(skipped).sort_values("entry_time").reset_index(drop=True) if skipped else pd.DataFrame()
    return kept_df, skip_df

def analyze_period(df_filtered, label, init_cash=INIT_CASH):
    n = len(df_filtered)
    if n == 0:
        return {"label": label, "n": 0, "pnl": 0, "pf": 0, "wr": 0,
                "mdd": 0, "months_minus": 0, "total_months": 0,
                "worst_month_pnl": 0, "worst_month": "N/A"}

    total_pnl = df_filtered["pnl"].sum()
    pf_val = pf(df_filtered["pnl"])
    wr = len(df_filtered[df_filtered["result"] == "win"]) / n

    portfolio_eq = init_cash + df_filtered.sort_values("exit_time")["pnl"].cumsum()
    mdd_val = calc_mdd_details(portfolio_eq)

    monthly = df_filtered.groupby(df_filtered["entry_time"].dt.strftime("%Y-%m"))["pnl"].sum()
    months_minus = (monthly < 0).sum()
    total_months = len(monthly)
    worst_month = monthly.idxmin() if len(monthly) > 0 else "N/A"
    worst_month_pnl = monthly.min() if len(monthly) > 0 else 0

    return {
        "label": label, "n": n, "pnl": total_pnl, "pf": pf_val,
        "wr": wr, "mdd": mdd_val, "months_minus": months_minus,
        "total_months": total_months, "worst_month": worst_month,
        "worst_month_pnl": worst_month_pnl, "monthly": monthly,
    }

def print_sep(title):
    line = "=" * 90
    print(f"\n{line}\n  {title}\n{line}\n")

def main():
    df = load_trades()
    total_n = len(df)

    # 全期間PF（優先順位用）
    sym_pfs_full = {}
    for sym, grp in df.groupby("sym"):
        sym_pfs_full[sym] = pf(grp["pnl"])
    priority_by_pf = sorted(sym_pfs_full, key=lambda s: sym_pfs_full[s], reverse=True)

    # 2026Q1抽出
    df_q1 = df[df["entry_time"] >= "2026-01-01"].sort_values("entry_time").reset_index(drop=True)
    # 2026Q1のPF
    sym_pfs_q1 = {}
    for sym, grp in df_q1.groupby("sym"):
        sym_pfs_q1[sym] = pf(grp["pnl"])

    # ═══════════════════════════════════════════════════════════════
    #  穴① スキップされたトレードの勝率分析
    # ═══════════════════════════════════════════════════════════════
    print_sep("穴① スキップされたトレードの質（全期間 + 2026Q1）")

    limits = [7, 6, 5, 4, 3]

    for period_name, period_df in [("全期間", df), ("2026Q1", df_q1)]:
        print(f"--- {period_name} ---")
        print(f"{'制限':>8} {'採用WR':>8} {'スキップWR':>10} {'スキップPF':>10} "
              f"{'skip_n':>7} {'skip勝ち寄り?':>14}")
        print("-" * 70)

        for lim in limits:
            kept, skipped = apply_max_concurrent(period_df, lim)
            if len(skipped) == 0:
                print(f"  MAX={lim}   スキップなし")
                continue
            kept_wr = len(kept[kept["result"] == "win"]) / len(kept) if len(kept) > 0 else 0
            skip_wr = len(skipped[skipped["result"] == "win"]) / len(skipped)
            skip_pf = pf(skipped["pnl"])
            bias = "YES ★" if skip_wr > kept_wr else "no"
            print(f"  MAX={lim}   {kept_wr:>7.1%}   {skip_wr:>9.1%}   {skip_pf:>9.2f}   "
                  f"{len(skipped):>6}   {bias:>12}")

        # スキップされた銘柄の内訳
        print(f"\n  MAX=4 スキップ内訳（{period_name}）:")
        _, skipped_4 = apply_max_concurrent(period_df, 4)
        if len(skipped_4) > 0:
            for sym in sorted(skipped_4["sym"].unique()):
                sg = skipped_4[skipped_4["sym"] == sym]
                sw = len(sg[sg["result"] == "win"])
                print(f"    {sym:>8}: {len(sg):>4}件  勝率={sw/len(sg):.1%}  "
                      f"PnL={sg['pnl'].sum():>+14,.0f}")
        print()

    # ═══════════════════════════════════════════════════════════════
    #  穴② 2026Q1単独のトレードオフ表（3つの方式で比較）
    # ═══════════════════════════════════════════════════════════════
    print_sep("穴② 2026Q1単独トレードオフ表（好調期間で薄めない）")

    print(f"2026Q1トレード数: {len(df_q1)}")
    print(f"2026Q1銘柄PF: {', '.join(f'{s}({sym_pfs_q1[s]:.2f})' for s in sorted(sym_pfs_q1, key=lambda x: sym_pfs_q1[x], reverse=True))}")
    print()

    all_limits = [None, 7, 6, 5, 4, 3, 2]

    for method_name, apply_fn, extra_args in [
        ("A: 先着順",
         lambda d, l: apply_max_concurrent(d, l), {}),
        ("B: 全期間PF優先",
         lambda d, l: apply_max_concurrent_priority(d, l, priority_by_pf), {}),
        ("C: Q1期間PF最悪除外",
         lambda d, l: apply_max_worst_skip(d, l, sym_pfs_q1), {}),
    ]:
        print(f"--- {method_name} ---")
        print(f"{'制限':>12} {'n':>5} {'skip':>5} {'PnL':>16} {'PF':>6} "
              f"{'WR':>7} {'MDD':>7} {'月-':>4} {'最悪月PnL':>16} {'最悪月':>8}")
        print("-" * 100)

        for lim in all_limits:
            label = f"MAX={lim}" if lim else "制限なし"
            kept, _ = apply_fn(df_q1, lim)
            r = analyze_period(kept, label)
            skipped = len(df_q1) - r["n"]
            print(f"{label:>12} {r['n']:>5} {skipped:>5} {r['pnl']:>16,.0f} "
                  f"{r['pf']:>6.2f} {r['wr']:>6.1%} {r['mdd']:>6.1f}% "
                  f"{r['months_minus']:>3}/{r['total_months']} "
                  f"{r['worst_month_pnl']:>16,.0f} {r['worst_month']:>8}")
        print()

    # ═══════════════════════════════════════════════════════════════
    #  穴③ 月次詳細: 制限値ごとの月次PnL（2026Q1全月）
    # ═══════════════════════════════════════════════════════════════
    print_sep("穴③ 月次詳細（2026Q1: 制限値ごとの月次PnL）")

    print("--- 先着順 ---")
    months_q1 = sorted(df_q1.groupby(df_q1["entry_time"].dt.strftime("%Y-%m")).groups.keys())
    header = f"{'月':>8}"
    for lim in [None, 6, 5, 4, 3]:
        label = f"MAX={lim}" if lim else "無制限"
        header += f" {label:>14}"
    print(header)
    print("-" * (8 + 15 * 5))

    monthly_data = {}
    for lim in [None, 6, 5, 4, 3]:
        kept, _ = apply_max_concurrent(df_q1, lim)
        monthly = kept.groupby(kept["entry_time"].dt.strftime("%Y-%m"))["pnl"].sum()
        monthly_data[lim] = monthly

    for m in months_q1:
        row = f"{m:>8}"
        for lim in [None, 6, 5, 4, 3]:
            val = monthly_data[lim].get(m, 0)
            mark = " ★" if val < 0 else ""
            row += f" {val:>12,.0f}{mark}"
        print(row)

    # 合計
    print("-" * (8 + 15 * 5))
    row = f"{'合計':>8}"
    for lim in [None, 6, 5, 4, 3]:
        row += f" {monthly_data[lim].sum():>14,.0f}"
    print(row)

    print("\n--- PF優先順 ---")
    header = f"{'月':>8}"
    for lim in [None, 6, 5, 4, 3]:
        label = f"MAX={lim}" if lim else "無制限"
        header += f" {label:>14}"
    print(header)
    print("-" * (8 + 15 * 5))

    monthly_data_b = {}
    for lim in [None, 6, 5, 4, 3]:
        kept, _ = apply_max_concurrent_priority(df_q1, lim, priority_by_pf)
        monthly = kept.groupby(kept["entry_time"].dt.strftime("%Y-%m"))["pnl"].sum()
        monthly_data_b[lim] = monthly

    for m in months_q1:
        row = f"{m:>8}"
        for lim in [None, 6, 5, 4, 3]:
            val = monthly_data_b[lim].get(m, 0)
            mark = " ★" if val < 0 else ""
            row += f" {val:>12,.0f}{mark}"
        print(row)

    print("-" * (8 + 15 * 5))
    row = f"{'合計':>8}"
    for lim in [None, 6, 5, 4, 3]:
        row += f" {monthly_data_b[lim].sum():>14,.0f}"
    print(row)

    print("\n--- Q1最悪銘柄除外 ---")
    header = f"{'月':>8}"
    for lim in [None, 6, 5, 4, 3]:
        label = f"MAX={lim}" if lim else "無制限"
        header += f" {label:>14}"
    print(header)
    print("-" * (8 + 15 * 5))

    monthly_data_c = {}
    for lim in [None, 6, 5, 4, 3]:
        kept, _ = apply_max_worst_skip(df_q1, lim, sym_pfs_q1)
        monthly = kept.groupby(kept["entry_time"].dt.strftime("%Y-%m"))["pnl"].sum()
        monthly_data_c[lim] = monthly

    for m in months_q1:
        row = f"{m:>8}"
        for lim in [None, 6, 5, 4, 3]:
            val = monthly_data_c[lim].get(m, 0)
            mark = " ★" if val < 0 else ""
            row += f" {val:>12,.0f}{mark}"
        print(row)

    print("-" * (8 + 15 * 5))
    row = f"{'合計':>8}"
    for lim in [None, 6, 5, 4, 3]:
        row += f" {monthly_data_c[lim].sum():>14,.0f}"
    print(row)

    # ═══════════════════════════════════════════════════════════════
    #  週次DD分析: 2026Q1の最悪週間
    # ═══════════════════════════════════════════════════════════════
    print_sep("2026Q1 週次損益（最悪の週を特定）")

    for lim in [None, 5, 4, 3]:
        label = f"MAX={lim}" if lim else "無制限"
        kept, _ = apply_max_concurrent(df_q1, lim)
        weekly = kept.groupby(kept["entry_time"].dt.isocalendar().week)["pnl"].sum()
        worst_w = weekly.idxmin()
        worst_v = weekly.min()
        print(f"  {label:>8}  最悪週={worst_w}  PnL={worst_v:>+14,.0f}  "
              f"週マイナス数={len(weekly[weekly < 0])}/{len(weekly)}")

    # ═══════════════════════════════════════════════════════════════
    #  2025（好調期間）vs 2026Q1（不調期間）での制限効果比較
    # ═══════════════════════════════════════════════════════════════
    print_sep("好調期間(2025) vs 不調期間(2026Q1) 制限効果比較")

    df_2025 = df[df["entry_time"] < "2026-01-01"].sort_values("entry_time").reset_index(drop=True)

    print(f"{'':>12} {'───── 2025（好調）─────':>40} {'───── 2026Q1（不調）─────':>40}")
    print(f"{'制限':>12} {'PnL':>16} {'PF':>6} {'MDD':>7} {'PnL':>16} {'PF':>6} {'MDD':>7}")
    print("-" * 80)

    for lim in all_limits:
        label = f"MAX={lim}" if lim else "制限なし"
        kept_25, _ = apply_max_concurrent(df_2025, lim)
        kept_q1, _ = apply_max_concurrent(df_q1, lim)
        r25 = analyze_period(kept_25, label)
        rq1 = analyze_period(kept_q1, label)
        print(f"{label:>12} {r25['pnl']:>16,.0f} {r25['pf']:>6.2f} {r25['mdd']:>6.1f}% "
              f"{rq1['pnl']:>16,.0f} {rq1['pf']:>6.2f} {rq1['mdd']:>6.1f}%")

    # 好調/不調での制限効果の方向
    print("\n制限効果の方向:")
    base_25 = analyze_period(df_2025, "base")
    base_q1 = analyze_period(df_q1, "base")
    for lim in [6, 5, 4, 3]:
        kept_25, _ = apply_max_concurrent(df_2025, lim)
        kept_q1, _ = apply_max_concurrent(df_q1, lim)
        r25 = analyze_period(kept_25, f"MAX={lim}")
        rq1 = analyze_period(kept_q1, f"MAX={lim}")
        d25_pnl = (r25["pnl"] - base_25["pnl"]) / abs(base_25["pnl"]) * 100
        dq1_pnl = (rq1["pnl"] - base_q1["pnl"]) / abs(base_q1["pnl"]) * 100
        d25_mdd = r25["mdd"] - base_25["mdd"]
        dq1_mdd = rq1["mdd"] - base_q1["mdd"]
        print(f"  MAX={lim}  2025: PnL{d25_pnl:>+6.1f}% MDD{d25_mdd:>+5.1f}pp  "
              f"2026Q1: PnL{dq1_pnl:>+6.1f}% MDD{dq1_mdd:>+5.1f}pp  "
              f"{'効果逆転 ★' if (d25_mdd > 0 and dq1_mdd < 0) or (d25_mdd < 0 and dq1_mdd > 0) else ''}")

if __name__ == "__main__":
    main()
