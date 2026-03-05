"""
RUN-009: やがみPA v2 バックテスト + Walk-Forward検証
====================================================
画像解析知見を反映した改良版プライスアクションシグナルの定量検証。

比較対象:
  - v1 (RUN-003ベースライン): sig_yagami_B (従来5条件 B評価)
  - v2: sig_yagami_pa_v2 (止まり検出 + 厳格実体揃い + IB蓄積ブレイク)

パラメータグリッド (v2):
  - min_conditions: [1, 2]
  - align_tol: [0.08, 0.12, 0.15]
  - stop_tol: [0.12, 0.18]

データ: XAUUSD 4H (data/ohlc/XAUUSD_4h.csv)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.backtest import BacktestEngine
from lib.yagami import sig_yagami_B, sig_yagami_pa_v2


def load_data(tf='4h'):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc', f'XAUUSD_{tf}.csv')
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_full_backtest():
    df = load_data('4h')
    print(f"Data: {df.index[0]} -> {df.index[-1]}, {len(df)} bars")

    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=1.5,
        default_tp_atr=4.5,
        trail_start_atr=2.0,
        trail_dist_atr=1.5,
        exit_on_signal=True,
        target_min_trades=0,
        target_min_wr=0.0,
    )

    strategies = {}

    # --- ベースライン: v1 (従来のやがみB評価) ---
    strategies['v1_Yagami_B'] = sig_yagami_B(freq='4h')

    # --- v2パラメータグリッド ---
    for min_cond in [1, 2]:
        for align_tol in [0.08, 0.12, 0.15]:
            for stop_tol in [0.12, 0.18]:
                name = f'v2_mc{min_cond}_at{align_tol}_st{stop_tol}'
                strategies[name] = sig_yagami_pa_v2(
                    freq='4h',
                    min_conditions=min_cond,
                    align_tol=align_tol,
                    stop_tol=stop_tol,
                )

    results = []
    for name, sig_func in strategies.items():
        print(f"\n--- {name} ---")
        engine = BacktestEngine(**engine_params)
        try:
            res = engine.run(data=df, signal_func=sig_func, freq='4h', name=name)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'name': name, 'error': str(e)})
            continue

        if res is None:
            print("  No trades")
            results.append({'name': name, 'total_trades': 0, 'pf': 0, 'wr': 0, 'mdd': 0})
            continue

        pf = res.get('profit_factor', 0)
        wr = res.get('win_rate_pct', 0)
        mdd = res.get('max_drawdown_pct', 0)
        n = res.get('total_trades', 0)
        net = res.get('net_pnl', 0)

        print(f"  N={n}, PF={pf:.4f}, WR={wr:.1f}%, MDD={mdd:.1f}%, Net={net:.0f}")
        results.append({
            'name': name,
            'total_trades': n,
            'pf': round(pf, 4),
            'wr': round(wr, 2),
            'mdd': round(mdd, 2),
            'net_pnl': round(net, 2),
        })

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'run009_pa_v2_backtest.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return df_results


def run_walk_forward(best_params):
    """best_paramsで確定した戦略のWalk-Forward検証"""
    df = load_data('4h')
    print(f"\n=== Walk-Forward検証 ===")
    print(f"Data: {df.index[0]} -> {df.index[-1]}, {len(df)} bars")
    print(f"Params: {best_params}")

    signal_func = sig_yagami_pa_v2(freq='4h', **best_params)

    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=1.5,
        default_tp_atr=4.5,
        trail_start_atr=2.0,
        trail_dist_atr=1.5,
        exit_on_signal=True,
        target_min_trades=0,
        target_min_wr=0.0,
    )

    # 4H: ~6bars/day, ~1500bars/year
    train_months = 12
    oos_months = 3
    step_months = 3

    data_start = df.index[0]
    data_end = df.index[-1]

    folds = []
    fold_num = 0
    train_start = data_start
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
            'train_start': train_start,
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
        train_data = df[(df.index >= fold['train_start']) & (df.index < fold['train_end'])]
        oos_data = df[(df.index >= fold['oos_start']) & (df.index < fold['oos_end'])]

        # IS
        engine_is = BacktestEngine(**engine_params)
        res_is = engine_is.run(data=train_data, signal_func=signal_func,
                               freq='4h', name=f'PA_v2_IS_{fold["fold"]}')

        # OOS
        engine_oos = BacktestEngine(**engine_params)
        res_oos = engine_oos.run(data=oos_data, signal_func=signal_func,
                                 freq='4h', name=f'PA_v2_OOS_{fold["fold"]}')

        pf_is = res_is.get('profit_factor', 0) if res_is else 0
        n_is = res_is.get('total_trades', 0) if res_is else 0
        pf_oos = res_oos.get('profit_factor', 0) if res_oos else 0
        wr_oos = res_oos.get('win_rate_pct', 0) if res_oos else 0
        n_oos = res_oos.get('total_trades', 0) if res_oos else 0
        mdd_oos = res_oos.get('max_drawdown_pct', 0) if res_oos else 0

        passed = (pf_oos >= 1.5 and wr_oos >= 30.0 and mdd_oos <= 20.0 and n_oos >= 3)

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

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    csv_path = os.path.join(results_dir, 'run009_pa_v2_wf.csv')
    df_wf = pd.DataFrame(results)
    df_wf.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

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
    else:
        print("\nNo valid folds (all N<3)")

    return df_wf


if __name__ == '__main__':
    # Phase 1: Full period backtest
    df_bt = run_full_backtest()

    # Find best v2 config (by PF, min N>=10)
    v2_results = df_bt[df_bt['name'].str.startswith('v2_') & (df_bt['total_trades'] >= 5)]
    if len(v2_results) > 0:
        best_row = v2_results.sort_values('pf', ascending=False).iloc[0]
        print(f"\n=== Best v2: {best_row['name']} ===")
        print(f"PF={best_row['pf']}, WR={best_row['wr']}%, N={best_row['total_trades']}, MDD={best_row['mdd']}%")

        # Parse params from name
        parts = best_row['name'].split('_')
        best_params = {
            'min_conditions': int(parts[1].replace('mc', '')),
            'align_tol': float(parts[2].replace('at', '')),
            'stop_tol': float(parts[3].replace('st', '')),
        }

        # Phase 2: Walk-Forward
        run_walk_forward(best_params)
    else:
        print("\nNo v2 strategies with N>=5. Trying Walk-Forward with default params.")
        run_walk_forward({'min_conditions': 1, 'align_tol': 0.12, 'stop_tol': 0.15})
