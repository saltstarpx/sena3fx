"""
RUN-010 CC: やがみ式3層MTF v3.0 バックテスト + Walk-Forward
============================================================
Manus RUN-010 (USDJPY M1, PF=3.137) の3層ロジックをXAUUSDで検証。

検証バリアント:
  1. H4+M15 (3層): XAUUSD_4h + XAUUSD_2025_15m
  2. H4+H1 (2層近似): XAUUSD_4h + XAUUSD_1h (より長期データ)

パラメータグリッド:
  - rr_min: [1.5, 2.0, 3.0]
  - h4_lookback: [15, 20, 30]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v3 import sig_yagami_mtf_v3, sig_yagami_mtf_v3_h1


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')


def load(pair, tf):
    path = os.path.join(DATA_DIR, f'{pair}_{tf}.csv')
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_h1_backtest():
    """2層近似 (H4+H1) バックテスト — 長期データ利用可能"""
    print("=" * 60)
    print("2層近似: H4(環境) + H1(パターン+執行)")
    print("=" * 60)

    bars_4h = load('XAUUSD', '4h')
    bars_h1 = load('XAUUSD', '1h')
    print(f"H4: {bars_4h.index[0]} -> {bars_4h.index[-1]}, {len(bars_4h)} bars")
    print(f"H1: {bars_h1.index[0]} -> {bars_h1.index[-1]}, {len(bars_h1)} bars")

    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=1.0,   # タイトSL (3層の核心)
        default_tp_atr=6.0,   # 高RR
        trail_start_atr=3.0,
        trail_dist_atr=2.0,
        exit_on_signal=False,  # SL/TPのみ決済
        target_min_trades=0,
        target_min_wr=0.0,
    )

    results = []
    for rr_min in [1.5, 2.0, 3.0]:
        for h4_lb in [15, 20, 30]:
            name = f'H4H1_rr{rr_min}_lb{h4_lb}'
            print(f"\n--- {name} ---")

            sig_func = sig_yagami_mtf_v3_h1(bars_4h, rr_min=rr_min, h4_lookback=h4_lb)
            engine = BacktestEngine(**engine_params)

            try:
                res = engine.run(data=bars_h1, signal_func=sig_func, freq='1h', name=name)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'name': name, 'error': str(e)})
                continue

            if res is None:
                print("  No trades")
                results.append({'name': name, 'total_trades': 0})
                continue

            pf = res.get('profit_factor', 0)
            wr = res.get('win_rate_pct', 0)
            mdd = res.get('max_drawdown_pct', 0)
            n = res.get('total_trades', 0)
            net = res.get('net_pnl', 0)
            print(f"  N={n}, PF={pf:.4f}, WR={wr:.1f}%, MDD={mdd:.1f}%")
            results.append({
                'name': name, 'total_trades': n,
                'pf': round(pf, 4), 'wr': round(wr, 2),
                'mdd': round(mdd, 2), 'net_pnl': round(net, 2),
            })

    return pd.DataFrame(results)


def run_m15_backtest():
    """3層 (H4+H1+M15) バックテスト — 2025年データのみ"""
    print("\n" + "=" * 60)
    print("3層: H4(環境) + H1(パターン) + M15(執行)")
    print("=" * 60)

    bars_4h = load('XAUUSD', '4h')
    bars_h1 = load('XAUUSD', '1h')

    # M15は2025年データのみ
    path_15m = os.path.join(DATA_DIR, 'XAUUSD_2025_15m.csv')
    bars_m15 = pd.read_csv(path_15m, parse_dates=['datetime'], index_col='datetime')
    bars_m15.columns = [c.lower() for c in bars_m15.columns]
    bars_m15.index = pd.to_datetime(bars_m15.index, utc=True)

    print(f"H4: {bars_4h.index[0]} -> {bars_4h.index[-1]}, {len(bars_4h)} bars")
    print(f"M15: {bars_m15.index[0]} -> {bars_m15.index[-1]}, {len(bars_m15)} bars")

    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=0.5,   # M15はよりタイト
        default_tp_atr=8.0,   # 高RR
        trail_start_atr=3.0,
        trail_dist_atr=2.0,
        exit_on_signal=False,
        target_min_trades=0,
        target_min_wr=0.0,
    )

    results = []
    for rr_min in [1.5, 2.0, 3.0]:
        for h4_lb in [15, 20]:
            name = f'M15_rr{rr_min}_lb{h4_lb}'
            print(f"\n--- {name} ---")

            # Filter H1 to overlap with M15 period
            h1_overlap = bars_h1[bars_h1.index >= bars_m15.index[0]]
            sig_func = sig_yagami_mtf_v3(bars_4h, bars_h1=h1_overlap,
                                          rr_min=rr_min, h4_lookback=h4_lb)
            engine = BacktestEngine(**engine_params)

            try:
                res = engine.run(data=bars_m15, signal_func=sig_func,
                                 freq='15min', name=name)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'name': name, 'error': str(e)})
                continue

            if res is None:
                print("  No trades")
                results.append({'name': name, 'total_trades': 0})
                continue

            pf = res.get('profit_factor', 0)
            wr = res.get('win_rate_pct', 0)
            mdd = res.get('max_drawdown_pct', 0)
            n = res.get('total_trades', 0)
            print(f"  N={n}, PF={pf:.4f}, WR={wr:.1f}%, MDD={mdd:.1f}%")
            results.append({
                'name': name, 'total_trades': n,
                'pf': round(pf, 4), 'wr': round(wr, 2),
                'mdd': round(mdd, 2), 'net_pnl': round(net if 'net' in dir() else 0, 2),
            })

    return pd.DataFrame(results)


def run_walk_forward_h1(best_rr, best_lb):
    """H4+H1バリアントのWalk-Forward検証"""
    print(f"\n{'='*60}")
    print(f"Walk-Forward: H4+H1, rr={best_rr}, lb={best_lb}")
    print(f"{'='*60}")

    bars_4h = load('XAUUSD', '4h')
    bars_h1 = load('XAUUSD', '1h')

    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=1.0,
        default_tp_atr=6.0,
        trail_start_atr=3.0,
        trail_dist_atr=2.0,
        exit_on_signal=False,
        target_min_trades=0,
        target_min_wr=0.0,
    )

    # H1: ~6bars/day, about 6*250=1500 bars/year
    train_months = 6
    oos_months = 3
    step_months = 3

    data_start = bars_h1.index[0]
    data_end = bars_h1.index[-1]

    folds = []
    fold_num = 0
    oos_start = data_start + pd.DateOffset(months=train_months)

    while True:
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > data_end:
            if (data_end - oos_start).days >= 30:
                oos_end = data_end
            else:
                break
        folds.append({
            'fold': fold_num,
            'train_start': data_start,
            'train_end': oos_start,
            'oos_start': oos_start,
            'oos_end': oos_end,
        })
        fold_num += 1
        oos_start += pd.DateOffset(months=step_months)
        if oos_start >= data_end:
            break

    print(f"Total folds: {len(folds)}")

    results = []
    for fold in folds:
        train_h1 = bars_h1[(bars_h1.index >= fold['train_start']) &
                           (bars_h1.index < fold['train_end'])]
        oos_h1 = bars_h1[(bars_h1.index >= fold['oos_start']) &
                         (bars_h1.index < fold['oos_end'])]

        sig_is = sig_yagami_mtf_v3_h1(bars_4h, rr_min=best_rr, h4_lookback=best_lb)
        sig_oos = sig_yagami_mtf_v3_h1(bars_4h, rr_min=best_rr, h4_lookback=best_lb)

        engine_is = BacktestEngine(**engine_params)
        res_is = engine_is.run(data=train_h1, signal_func=sig_is,
                               freq='1h', name=f'MTF_IS_{fold["fold"]}')

        engine_oos = BacktestEngine(**engine_params)
        res_oos = engine_oos.run(data=oos_h1, signal_func=sig_oos,
                                 freq='1h', name=f'MTF_OOS_{fold["fold"]}')

        n_is = res_is.get('total_trades', 0) if res_is else 0
        pf_is = res_is.get('profit_factor', 0) if res_is else 0
        n_oos = res_oos.get('total_trades', 0) if res_oos else 0
        pf_oos = res_oos.get('profit_factor', 0) if res_oos else 0
        wr_oos = res_oos.get('win_rate_pct', 0) if res_oos else 0
        mdd_oos = res_oos.get('max_drawdown_pct', 0) if res_oos else 0

        passed = (pf_oos >= 1.2 and n_oos >= 3 and mdd_oos <= 20.0)

        row = {
            'fold': fold['fold'],
            'oos_start': fold['oos_start'].strftime('%Y-%m-%d'),
            'oos_end': fold['oos_end'].strftime('%Y-%m-%d'),
            'n_is': n_is, 'pf_is': round(pf_is, 4),
            'n_oos': n_oos, 'pf_oos': round(pf_oos, 4),
            'wr_oos': round(wr_oos, 2), 'mdd_oos': round(mdd_oos, 2),
            'passed': passed,
        }
        results.append(row)
        mark = "PASS" if passed else "FAIL"
        print(f"  Fold {fold['fold']}: IS N={n_is} PF={pf_is:.2f} | "
              f"OOS N={n_oos} PF={pf_oos:.2f} WR={wr_oos:.1f}% MDD={mdd_oos:.1f}% [{mark}]")

    df_wf = pd.DataFrame(results)

    # Summary
    valid = df_wf[df_wf['n_oos'] >= 3]
    if len(valid) > 0:
        avg_pf = valid['pf_oos'].mean()
        avg_wr = valid['wr_oos'].mean()
        avg_mdd = valid['mdd_oos'].mean()
        passed_n = int(valid['passed'].sum())
        print(f"\n=== WF Summary ===")
        print(f"Valid folds: {len(valid)}/{len(df_wf)}")
        print(f"Passed: {passed_n}/{len(valid)} ({passed_n/len(valid)*100:.0f}%)")
        print(f"OOS avg PF: {avg_pf:.4f}")
        print(f"OOS avg WR: {avg_wr:.1f}%")
        print(f"OOS avg MDD: {avg_mdd:.1f}%")

    return df_wf


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Phase 1: H4+H1 full backtest (longer data)
    df_h1 = run_h1_backtest()
    df_h1.to_csv(os.path.join(RESULTS_DIR, 'run010cc_h1_backtest.csv'), index=False)

    # Phase 2: H4+H1+M15 full backtest (2025 only)
    df_m15 = run_m15_backtest()
    df_m15.to_csv(os.path.join(RESULTS_DIR, 'run010cc_m15_backtest.csv'), index=False)

    # Phase 3: Walk-Forward on best H1 variant
    valid_h1 = df_h1[(df_h1.get('total_trades', pd.Series(dtype=int)) >= 5) &
                      df_h1.get('pf', pd.Series(dtype=float)).notna()]
    if len(valid_h1) > 0:
        best = valid_h1.sort_values('pf', ascending=False).iloc[0]
        print(f"\n=== Best H1: {best['name']} PF={best.get('pf', 'N/A')} N={best.get('total_trades', 0)} ===")
        # Parse params
        parts = best['name'].split('_')
        rr = float(parts[1].replace('rr', ''))
        lb = int(parts[2].replace('lb', ''))

        df_wf = run_walk_forward_h1(rr, lb)
        df_wf.to_csv(os.path.join(RESULTS_DIR, 'run010cc_wf.csv'), index=False)
    else:
        print("\nNo valid H1 strategies. Trying default params for WF.")
        df_wf = run_walk_forward_h1(2.0, 20)
        df_wf.to_csv(os.path.join(RESULTS_DIR, 'run010cc_wf.csv'), index=False)
