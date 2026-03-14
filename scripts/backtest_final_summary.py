"""
backtest_final_summary.py
=========================
現行採用構成の最終バックテスト結果サマリー（PnL含む）
変更前（全銘柄tol=0.30）vs 変更後（NZDUSD/XAUUSD tol=0.20）を比較
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# backtest_mdd_reduction から再利用
from scripts.backtest_mdd_reduction import (
    load_all, run_sym, calc_stats, simulate, INIT_CASH,
    portfolio_sharpe, portfolio_mdd, TARGETS, LOGIC_NAMES,
    _exit_numba
)

# ── 構成定義 ──────────────────────────────────────────────────────
# 変更前: 全銘柄 tol=0.30
CONFIG_BEFORE = {t["sym"]: 0.30 for t in TARGETS}

# 変更後: NZDUSD/XAUUSD のみ tol=0.20
CONFIG_AFTER = {t["sym"]: 0.30 for t in TARGETS}
CONFIG_AFTER["NZDUSD"] = 0.20
CONFIG_AFTER["XAUUSD"] = 0.20

def run_config(sym_data, config, label):
    """指定構成でバックテスト実行、全詳細を返す"""
    all_stats = {}
    all_trades = {}
    all_equity = {}

    for tgt in TARGETS:
        sym = tgt["sym"]
        if sym not in sym_data:
            continue
        data = sym_data[sym]
        tol = config[sym]
        st, trades = run_sym(data["d1m"], data["d4h"], sym, data["logic"], tol)

        # simulate を再実行して最終equity取得
        from scripts.backtest_mdd_reduction import generate_signals, SYMBOL_CONFIG, calc_atr
        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d = calc_atr(data["d1m"], 10).to_dict()
        m1c = {"idx": data["d1m"].index, "opens": data["d1m"]["open"].values,
               "closes": data["d1m"]["close"].values,
               "highs": data["d1m"]["high"].values, "lows": data["d1m"]["low"].values}
        sigs = generate_signals(data["d1m"], data["d4h"], spread, data["logic"], atr_d, m1c, tol_factor=tol)
        _, final_eq, mdd = simulate(sigs, data["d1m"], sym)

        if st:
            st["mdd"] = mdd
            st["final_equity"] = final_eq
            st["total_pnl"] = final_eq - INIT_CASH
            st["total_pnl_pct"] = (final_eq - INIT_CASH) / INIT_CASH * 100
            st["tol"] = tol
            st["logic"] = tgt["logic"]

            # 月別PnL
            if trades:
                df_t = pd.DataFrame(trades)
                monthly = df_t.groupby("month")["pnl"].sum()
                st["monthly_pnl"] = monthly

                # 勝ちトレード/負けトレード詳細
                wins = [t for t in trades if t["pnl"] > 0]
                losses = [t for t in trades if t["pnl"] <= 0]
                st["avg_win"] = np.mean([t["pnl"] for t in wins]) if wins else 0
                st["avg_loss"] = np.mean([t["pnl"] for t in losses]) if losses else 0
                st["max_win"] = max([t["pnl"] for t in wins]) if wins else 0
                st["max_loss"] = min([t["pnl"] for t in losses]) if losses else 0
                st["win_count"] = len(wins)
                st["loss_count"] = len(losses)

        all_stats[sym] = st
        all_trades[sym] = trades
        all_equity[sym] = final_eq

    return all_stats, all_trades, all_equity

def print_results(stats, trades, equities, label, config):
    """詳細結果を表示"""
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")

    # ── 銘柄別サマリー ──
    print(f"\n  {'銘柄':<10} {'Logic':<15} {'tol':>5} {'取引数':>6} {'勝率':>7} {'PF':>7} "
          f"{'Sharpe':>7} {'MDD':>6} {'Kelly':>7} {'月+':>5} {'総損益(¥)':>14} {'利益率':>8}")
    print("  " + "-" * 115)

    total_pnl = 0
    total_init = 0
    for tgt in TARGETS:
        sym = tgt["sym"]
        st = stats.get(sym)
        if not st:
            print(f"  {sym:<10} データなし")
            continue
        logic_name = LOGIC_NAMES[st["logic"]]
        pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
        tol_mark = " *" if config[sym] != 0.30 else ""
        print(f"  {sym:<10} {logic_name:<15} {st['tol']:.2f}{tol_mark} {st['n']:>5} "
              f"{st['wr']*100:>6.1f}% {pf_s:>7} {st['sharpe']:>7.2f} {st['mdd']:>5.1f}% "
              f"{st['kelly']:>7.3f} {st['plus_m']:>2}/{st['total_m']:<2} "
              f"{st['total_pnl']:>+13,.0f} {st['total_pnl_pct']:>+6.1f}%")
        total_pnl += st['total_pnl']
        total_init += INIT_CASH

    print("  " + "-" * 115)
    print(f"  {'ポートフォリオ合計':<26} {'':>5} {'':>6} {'':>7} {'':>7} {'':>7} "
          f"{'':>6} {'':>7} {'':>5} {total_pnl:>+13,.0f} {total_pnl/total_init*100:>+6.1f}%")

    # ── ポートフォリオ指標 ──
    p_sharpe = portfolio_sharpe(trades)
    p_mdd = portfolio_mdd(trades)
    print(f"\n  ポートフォリオ Sharpe: {p_sharpe:.2f}  |  ポートフォリオ MDD: {p_mdd:.1f}%")

    # ── 銘柄別詳細 ──
    print(f"\n  --- 銘柄別 詳細統計 ---")
    for tgt in TARGETS:
        sym = tgt["sym"]
        st = stats.get(sym)
        if not st or "avg_win" not in st:
            continue
        print(f"\n  [{sym}] Logic-{st['logic']} ({LOGIC_NAMES[st['logic']]})  tol={st['tol']:.2f}")
        print(f"    取引数: {st['n']}  (勝ち {st['win_count']} / 負け {st['loss_count']})")
        print(f"    勝率:   {st['wr']*100:.1f}%")
        print(f"    PF:     {st['pf']:.2f}  |  Sharpe: {st['sharpe']:.2f}  |  Kelly: {st['kelly']:.3f}")
        print(f"    MDD:    {st['mdd']:.1f}%")
        print(f"    平均勝ち: ¥{st['avg_win']:>+,.0f}  |  平均負け: ¥{st['avg_loss']:>+,.0f}  |  比率: {abs(st['avg_win']/st['avg_loss']) if st['avg_loss'] != 0 else 0:.2f}")
        print(f"    最大勝ち: ¥{st['max_win']:>+,.0f}  |  最大負け: ¥{st['max_loss']:>+,.0f}")
        print(f"    総損益:   ¥{st['total_pnl']:>+,.0f}  ({st['total_pnl_pct']:>+.1f}%)")
        print(f"    最終資産: ¥{st['final_equity']:>,.0f}")

        # 月別PnL
        if "monthly_pnl" in st:
            mp = st["monthly_pnl"]
            print(f"    月別PnL:")
            for m in mp.index:
                bar = "+" * max(1, int(abs(mp[m]) / 50000)) if mp[m] > 0 else "-" * max(1, int(abs(mp[m]) / 50000))
                print(f"      {m}: ¥{mp[m]:>+12,.0f}  {bar}")

    return total_pnl, p_sharpe, p_mdd


def main():
    t0 = time.time()

    # Numba ウォームアップ
    print("  [Warmup] Numba JIT コンパイル中...")
    _exit_numba(np.array([1.0, 2.0]), np.array([0.5, 0.5]), 1.0, 0.5, 2.0, 0.5, 1, 1.0, 2)
    print(f"  [Warmup] 完了")

    # データロード
    print("\n  [データロード中...]")
    sym_data = {}
    for tgt in TARGETS:
        sym = tgt["sym"]
        d1m, d4h = load_all(sym)
        if d1m is None:
            print(f"    ❌ {sym}: データ未発見"); continue
        sym_data[sym] = {"d1m": d1m, "d4h": d4h, "logic": tgt["logic"], "cat": tgt["cat"]}
        print(f"    ✅ {sym}: 1m={len(d1m):,}行  4h={len(d4h):,}行")

    # ── 変更前 ──
    print("\n" + "#" * 120)
    print("  変更前（全銘柄 tol_factor=0.30）vs 変更後（NZDUSD/XAUUSD tol=0.20）")
    print("#" * 120)

    stats_b, trades_b, eq_b = run_config(sym_data, CONFIG_BEFORE, "BEFORE")
    pnl_b, sh_b, mdd_b = print_results(stats_b, trades_b, eq_b,
                                         "【変更前】全銘柄 tol_factor=0.30", CONFIG_BEFORE)

    # ── 変更後 ──
    stats_a, trades_a, eq_a = run_config(sym_data, CONFIG_AFTER, "AFTER")
    pnl_a, sh_a, mdd_a = print_results(stats_a, trades_a, eq_a,
                                         "【変更後】NZDUSD/XAUUSD tol=0.20（現行採用構成）", CONFIG_AFTER)

    # ── 比較サマリー ──
    print(f"\n{'='*120}")
    print(f"  変更前後 比較サマリー")
    print(f"{'='*120}")
    print(f"  {'指標':<25} {'変更前':>15} {'変更後':>15} {'差分':>15}")
    print(f"  {'-'*70}")
    print(f"  {'ポートフォリオ総損益':.<25} ¥{pnl_b:>+12,.0f} ¥{pnl_a:>+12,.0f} ¥{pnl_a-pnl_b:>+12,.0f}")
    print(f"  {'ポートフォリオ利益率':.<25} {pnl_b/(INIT_CASH*7)*100:>+12.1f}% {pnl_a/(INIT_CASH*7)*100:>+12.1f}% {(pnl_a-pnl_b)/(INIT_CASH*7)*100:>+12.1f}%")
    print(f"  {'ポートフォリオ Sharpe':.<25} {sh_b:>15.2f} {sh_a:>15.2f} {sh_a-sh_b:>+15.2f}")
    print(f"  {'ポートフォリオ MDD':.<25} {mdd_b:>14.1f}% {mdd_a:>14.1f}% {mdd_a-mdd_b:>+14.1f}%")

    # 変更銘柄の比較
    print(f"\n  変更銘柄のみの比較:")
    for sym in ["NZDUSD", "XAUUSD"]:
        sb = stats_b.get(sym, {})
        sa = stats_a.get(sym, {})
        if sb and sa:
            print(f"\n    {sym}:")
            print(f"      PF:     {sb['pf']:.2f} → {sa['pf']:.2f} ({sa['pf']-sb['pf']:+.2f})")
            print(f"      MDD:    {sb['mdd']:.1f}% → {sa['mdd']:.1f}% ({sa['mdd']-sb['mdd']:+.1f}pp)")
            print(f"      Sharpe: {sb['sharpe']:.2f} → {sa['sharpe']:.2f} ({sa['sharpe']-sb['sharpe']:+.2f})")
            print(f"      総損益: ¥{sb['total_pnl']:+,.0f} → ¥{sa['total_pnl']:+,.0f} ({sa['total_pnl']-sb['total_pnl']:+,.0f})")
            print(f"      取引数: {sb['n']} → {sa['n']} ({sa['n']-sb['n']:+d})")

    elapsed = time.time() - t0
    print(f"\n  実行時間: {elapsed:.1f}秒")
    print()


if __name__ == "__main__":
    main()
