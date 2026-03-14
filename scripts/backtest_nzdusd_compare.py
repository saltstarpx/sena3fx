"""
NZDUSD専用: 現行ロジック vs F3+F7 比較 + IS/OOS過学習チェック
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# backtest_new_features.py の関数を再利用
from scripts.backtest_new_features import (
    load_all, build_4h, build_1h, calc_atr,
    generate_signals_with_meta, filter_signals, simulate, calc_stats,
    SYMBOL_CONFIG, INIT_CASH, FEATURE_NAMES
)

SYM = "NZDUSD"

def run_sim(signals, d1m, features=None):
    sigs = filter_signals(signals, features) if features else signals
    trades, eq, mdd = simulate(sigs, d1m, SYM)
    st = calc_stats(trades)
    if st:
        st["mdd"] = mdd
        st["n_signals"] = len(sigs)
    return st, trades

def monthly_equity(trades, init=INIT_CASH):
    """月次エクイティカーブを返す"""
    if not trades:
        return {}
    combined = {}
    for t in trades:
        combined[t["month"]] = combined.get(t["month"], 0) + t["pnl"]
    months = sorted(combined.keys())
    eq = init
    curve = {}
    for m in months:
        eq += combined[m]
        curve[m] = eq
    return curve

def print_stats(label, st):
    if not st:
        print(f"    {label:20} トレード数不足")
        return
    pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
    print(f"    {label:20} n={st['n']:>4}  WR={st['wr']*100:.1f}%  PF={pf_s:>6}  "
          f"Sharpe={st['sharpe']:.2f}  MDD={st['mdd']:.1f}%  "
          f"Kelly={st['kelly']:.3f}  月+={st['plus_m']}/{st['total_m']}")

def main():
    t0 = time.time()
    print("\n" + "="*100)
    print("  NZDUSD 現行ロジック vs F3+F7 比較 + IS/OOS過学習チェック")
    print("="*100)

    # データロード
    d1m, d4h = load_all(SYM)
    if d1m is None:
        print("  ❌ データ未発見"); return

    cfg = SYMBOL_CONFIG[SYM]
    spread = cfg["spread"] * cfg["pip"]
    atr_d = calc_atr(d1m, 10).to_dict()
    m1c = {"idx": d1m.index, "opens": d1m["open"].values,
           "closes": d1m["close"].values,
           "highs": d1m["high"].values, "lows": d1m["low"].values}

    print(f"\n  データ: 1m={len(d1m):,}行  4h={len(d4h):,}行")
    print(f"  期間: {d1m.index[0].strftime('%Y-%m-%d')} 〜 {d1m.index[-1].strftime('%Y-%m-%d')}")
    print(f"  スプレッド: {cfg['spread']}pips = {spread:.6f}")

    # シグナル生成（メタデータ付き）
    sigs = generate_signals_with_meta(d1m, d4h, spread, "A", atr_d, m1c)
    print(f"  ベースシグナル数: {len(sigs)}")

    # ─── 1. 全期間比較 ───
    print(f"\n  [1] 全期間比較")
    print("  " + "-"*90)

    # ベースライン（現行: Logic-A のみ）
    st_base, tr_base = run_sim(sigs, d1m)
    print_stats("現行 Logic-A", st_base)

    # F3のみ
    st_f3, tr_f3 = run_sim(sigs, d1m, features={"F3"})
    print_stats("+ F3(EMA傾き)", st_f3)

    # F7のみ
    st_f7, tr_f7 = run_sim(sigs, d1m, features={"F7"})
    print_stats("+ F7(パターン品質)", st_f7)

    # F3+F7
    st_f37, tr_f37 = run_sim(sigs, d1m, features={"F3", "F7"})
    print_stats("+ F3+F7", st_f37)

    # F3+F6+F7
    st_f367, tr_f367 = run_sim(sigs, d1m, features={"F3", "F6", "F7"})
    print_stats("+ F3+F6+F7", st_f367)

    # F3+F7+F9
    st_f379, tr_f379 = run_sim(sigs, d1m, features={"F3", "F7", "F9"})
    print_stats("+ F3+F7+F9", st_f379)

    # ─── 2. IS/OOS分割比較 ───
    print(f"\n  [2] IS/OOS過学習チェック（40/60分割）")
    print("  " + "-"*90)

    n_split = int(len(d1m) * 0.4)
    ts_split = d1m.index[n_split]
    print(f"  分割点: {ts_split.strftime('%Y-%m-%d %H:%M')} (IS: 先頭40% / OOS: 後方60%)")

    d1m_is  = d1m[d1m.index < ts_split]
    d1m_oos = d1m[d1m.index >= ts_split]
    sigs_is  = [s for s in sigs if s["time"] < ts_split]
    sigs_oos = [s for s in sigs if s["time"] >= ts_split]
    print(f"  IS シグナル: {len(sigs_is)}  /  OOS シグナル: {len(sigs_oos)}")

    variants = [
        ("現行 Logic-A", None),
        ("+ F3", {"F3"}),
        ("+ F7", {"F7"}),
        ("+ F3+F7", {"F3", "F7"}),
        ("+ F3+F6+F7", {"F3", "F6", "F7"}),
        ("+ F3+F7+F9", {"F3", "F7", "F9"}),
    ]

    print(f"\n  {'バリアント':20} {'IS_n':>5} {'IS_PF':>7} {'IS_Sh':>7} {'OOS_n':>6} {'OOS_PF':>7} {'OOS_Sh':>7} {'OOS/IS':>7} {'判定':>4}")
    print("  " + "-"*90)

    for label, feats in variants:
        is_sigs = filter_signals(sigs_is, feats) if feats else sigs_is
        oos_sigs = filter_signals(sigs_oos, feats) if feats else sigs_oos

        _, is_trades = run_sim(is_sigs, d1m_is, features=None)
        is_st = calc_stats(is_trades)

        _, oos_trades = run_sim(oos_sigs, d1m_oos, features=None)
        oos_st = calc_stats(oos_trades)

        if is_st and oos_st:
            is_pf = is_st['pf'] if is_st['pf'] < 99 else 99.9
            oos_pf = oos_st['pf'] if oos_st['pf'] < 99 else 99.9
            ratio_pf = oos_pf / is_pf if is_pf > 0 else 0
            ratio_sh = oos_st['sharpe'] / is_st['sharpe'] if is_st['sharpe'] > 0 else 0
            flag_pf = "✅" if ratio_pf >= 0.70 else "❌"
            flag_sh = "✅" if ratio_sh >= 0.70 else "⚠️"
            print(f"  {label:20} {is_st['n']:>5} {is_pf:>7.2f} {is_st['sharpe']:>7.2f} "
                  f"{oos_st['n']:>6} {oos_pf:>7.2f} {oos_st['sharpe']:>7.2f} "
                  f"PF:{ratio_pf:.2f}{flag_pf} Sh:{ratio_sh:.2f}{flag_sh}")
        elif oos_st:
            oos_pf = oos_st['pf'] if oos_st['pf'] < 99 else 99.9
            print(f"  {label:20}   IS不足           "
                  f"{oos_st['n']:>6} {oos_pf:>7.2f} {oos_st['sharpe']:>7.2f}   ---")
        else:
            print(f"  {label:20}   データ不足")

    # ─── 3. 月次エクイティカーブ比較 ───
    print(f"\n  [3] 月次損益比較（全期間）")
    print("  " + "-"*90)

    curves = {
        "Base": tr_base,
        "F3+F7": tr_f37,
    }

    # 月次損益
    all_months = set()
    monthly = {}
    for name, trades in curves.items():
        m = {}
        for t in trades:
            m[t["month"]] = m.get(t["month"], 0) + t["pnl"]
        monthly[name] = m
        all_months.update(m.keys())

    months_sorted = sorted(all_months)
    print(f"  {'月':>8}  {'Base損益':>12}  {'F3+F7損益':>12}  {'差分':>12}  {'Base累計':>12}  {'F3+F7累計':>12}")
    print("  " + "-"*80)

    cum_base = 0
    cum_f37 = 0
    for m in months_sorted:
        b = monthly.get("Base", {}).get(m, 0)
        f = monthly.get("F3+F7", {}).get(m, 0)
        cum_base += b
        cum_f37 += f
        print(f"  {m:>8}  {b:>12,.0f}  {f:>12,.0f}  {f-b:>+12,.0f}  {cum_base:>12,.0f}  {cum_f37:>12,.0f}")

    # ─── 4. フィルター通過率 ───
    print(f"\n  [4] 特徴量フィルター通過率")
    print("  " + "-"*60)

    for fn in ["F3", "F6", "F7", "F9"]:
        passed = sum(1 for s in sigs if s["meta"].get(fn, True))
        rate = passed / len(sigs) * 100 if sigs else 0
        print(f"    {fn:4} {FEATURE_NAMES[fn]:22}  通過: {passed:>4}/{len(sigs)} ({rate:.1f}%)")

    f37_passed = sum(1 for s in sigs if s["meta"].get("F3", True) and s["meta"].get("F7", True))
    rate37 = f37_passed / len(sigs) * 100 if sigs else 0
    print(f"    F3+F7 組み合わせ              通過: {f37_passed:>4}/{len(sigs)} ({rate37:.1f}%)")

    # ─── 5. 勝敗別フィルター分析 ───
    print(f"\n  [5] 勝敗別: F3+F7で除外されたトレードの内訳")
    print("  " + "-"*60)

    # ベースのトレード結果とシグナルを対応させる
    base_wins = sum(1 for t in tr_base if t["result"] == "win")
    base_losses = sum(1 for t in tr_base if t["result"] == "loss")
    f37_wins = sum(1 for t in tr_f37 if t["result"] == "win")
    f37_losses = sum(1 for t in tr_f37 if t["result"] == "loss")

    print(f"    ベース:  勝={base_wins}  負={base_losses}  計={len(tr_base)}")
    print(f"    F3+F7:  勝={f37_wins}  負={f37_losses}  計={len(tr_f37)}")
    print(f"    除外:   勝={base_wins-f37_wins}  負={base_losses-f37_losses}  計={len(tr_base)-len(tr_f37)}")

    removed_total = len(tr_base) - len(tr_f37)
    if removed_total > 0:
        removed_losses = base_losses - f37_losses
        print(f"    除外中の負け比率: {removed_losses}/{removed_total} = {removed_losses/removed_total*100:.1f}%")
        print(f"    → {'負けを多く除外 ✅' if removed_losses/removed_total > 0.5 else '勝ちも多く除外 ⚠️'}")

    print(f"\n  実行時間: {time.time()-t0:.0f}秒")
    print("="*100)

if __name__ == "__main__":
    main()
