"""
RUN-20260305-002: 4時間足フィルター付き バックテスト
=====================================================
PA1_Reversal_TightSL に4時間足レジサポフィルターを追加し、
RUN-20260305-001 との比較を行う。

検証パラメータ:
  - HTF lookback: 20 / 30 本
  - HTF zone_atr: 1.0 / 1.5 / 2.0
  - SL: ATR×1.0 / ATR×1.5
  合計 12 パラメータセット
"""
import sys
import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.backtest import BacktestEngine
from strategies.yagami_pa import signal_pa1_reversal
from strategies.htf_filter import build_htf_filter, align_htf_to_ltf, apply_htf_filter

DATA_1H = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'usdjpy_1h.csv')
DATA_4H = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ohlc', 'USDJPY_4h.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_1h():
    df = pd.read_csv(DATA_1H, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_4h():
    df = pd.read_csv(DATA_4H, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    return df


def run_all(bars_1h, bars_4h):
    results = []

    # ベースライン（フィルターなし、RUN-001の最良）
    baseline_params = dict(default_sl_atr=1.0, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=1, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35)

    print("\n[ベースライン] PA1_Reversal_TightSL (フィルターなし)")
    try:
        engine = BacktestEngine(init_cash=5_000_000, risk_pct=0.02, pip=0.01,
                                slippage_pips=0.5, **baseline_params)
        raw_sig = signal_pa1_reversal(bars_1h, zone_atr=1.5, lookback=20)
        result = engine.run(data=bars_1h, signal_func=lambda b: raw_sig, freq='1h',
                            name='PA1_TightSL_NoFilter')
        if result:
            results.append(_extract(result, 'PA1_TightSL_NoFilter', 'なし', '-', '-', '-'))
            _print_result(results[-1])
    except Exception as e:
        print(f"  エラー: {e}")

    # 4時間足フィルター付きパラメータセット
    htf_params_list = [
        # (htf_lookback, htf_zone_atr, sl_atr, tp_atr, label)
        (20, 1.0, 1.0, 3.0, 'HTF_lb20_z1.0_sl1.0_tp3.0'),
        (20, 1.5, 1.0, 3.0, 'HTF_lb20_z1.5_sl1.0_tp3.0'),
        (20, 2.0, 1.0, 3.0, 'HTF_lb20_z2.0_sl1.0_tp3.0'),
        (30, 1.0, 1.0, 3.0, 'HTF_lb30_z1.0_sl1.0_tp3.0'),
        (30, 1.5, 1.0, 3.0, 'HTF_lb30_z1.5_sl1.0_tp3.0'),
        (30, 2.0, 1.0, 3.0, 'HTF_lb30_z2.0_sl1.0_tp3.0'),
        (20, 1.5, 1.5, 3.0, 'HTF_lb20_z1.5_sl1.5_tp3.0'),
        (20, 1.5, 1.0, 4.0, 'HTF_lb20_z1.5_sl1.0_tp4.0'),
        (30, 1.5, 1.0, 4.0, 'HTF_lb30_z1.5_sl1.0_tp4.0'),
        (20, 1.0, 1.0, 4.0, 'HTF_lb20_z1.0_sl1.0_tp4.0'),
        (30, 1.0, 1.0, 4.0, 'HTF_lb30_z1.0_sl1.0_tp4.0'),
        (20, 2.0, 1.0, 4.0, 'HTF_lb20_z2.0_sl1.0_tp4.0'),
    ]

    for htf_lb, htf_z, sl_atr, tp_atr, label in htf_params_list:
        print(f"\n[HTFフィルター] {label}")
        try:
            # 4時間足フィルター構築
            htf_filter_df = build_htf_filter(bars_4h, lookback=htf_lb, zone_atr=htf_z)
            htf_aligned = align_htf_to_ltf(htf_filter_df, bars_1h.index)

            # 1時間足シグナル生成 → HTFフィルター適用
            raw_sig = signal_pa1_reversal(bars_1h, zone_atr=1.5, lookback=20)
            filtered_sig = apply_htf_filter(raw_sig, htf_aligned)

            n_raw = raw_sig.isin(['long', 'short']).sum()
            n_filtered = filtered_sig.isin(['long', 'short']).sum()
            print(f"  シグナル数: 元={n_raw} → フィルター後={n_filtered} (除外率={100*(1-n_filtered/max(n_raw,1)):.0f}%)")

            if n_filtered == 0:
                print("  → シグナルなし、スキップ")
                continue

            engine_params = dict(default_sl_atr=sl_atr, default_tp_atr=tp_atr,
                                 use_dynamic_sl=True, sl_n_confirm=1, pyramid_entries=0,
                                 trail_start_atr=0.0, target_max_dd=0.20, target_min_wr=0.35)
            engine = BacktestEngine(init_cash=5_000_000, risk_pct=0.02, pip=0.01,
                                    slippage_pips=0.5, **engine_params)
            result = engine.run(data=bars_1h, signal_func=lambda b: filtered_sig,
                                freq='1h', name=label)
            if result:
                results.append(_extract(result, label, 'あり', htf_lb, htf_z, sl_atr))
                _print_result(results[-1])
        except Exception as e:
            print(f"  エラー: {e}")
            import traceback; traceback.print_exc()

    return results


def _extract(result, name, htf_filter, htf_lb, htf_z, sl_atr):
    return {
        'name': name,
        'htf_filter': htf_filter,
        'htf_lookback': htf_lb,
        'htf_zone_atr': htf_z,
        'sl_atr': sl_atr,
        'total_trades': result['total_trades'],
        'win_rate_pct': result['win_rate_pct'],
        'profit_factor': result['profit_factor'],
        'max_drawdown_pct': result['max_drawdown_pct'],
        'total_return_pct': result['total_return_pct'],
        'rr_ratio': result['rr_ratio'],
        'passed': result['passed'],
        'trades': result.get('trades', []),
    }


def _print_result(r):
    print(f"  トレード数: {r['total_trades']} | 勝率: {r['win_rate_pct']:.1f}% | "
          f"PF: {r['profit_factor']:.3f} | DD: {r['max_drawdown_pct']:.1f}% | "
          f"リターン: {r['total_return_pct']:.2f}%")


def find_best(results):
    def score(r):
        pf = min(r['profit_factor'], 5.0)
        wr = r['win_rate_pct'] / 100
        dd = r['max_drawdown_pct'] / 100
        n = r['total_trades']
        if n < 3:
            return 0
        return pf * wr * (1 - dd) * np.log(n + 1)

    scored = sorted([(score(r), r) for r in results], key=lambda x: x[0], reverse=True)
    print("\n=== ランキング（上位5） ===")
    for i, (s, r) in enumerate(scored[:5]):
        print(f"  {i+1}. {r['name']}: score={s:.3f} | PF={r['profit_factor']:.3f} | "
              f"WR={r['win_rate_pct']:.1f}% | DD={r['max_drawdown_pct']:.1f}% | N={r['total_trades']}")
    return scored[0][1] if scored else None


def save_results(results, best):
    # サマリーCSV
    path = os.path.join(RESULTS_DIR, 'yagami_htf_backtest_summary.csv')
    fieldnames = ['name', 'htf_filter', 'htf_lookback', 'htf_zone_atr', 'sl_atr',
                  'total_trades', 'win_rate_pct', 'profit_factor',
                  'max_drawdown_pct', 'total_return_pct', 'rr_ratio', 'passed']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nサマリー保存: {path}")

    # 最良トレード詳細
    if best and best.get('trades'):
        trades_path = os.path.join(RESULTS_DIR, f"htf_best_trades_{best['name']}.csv")
        pd.DataFrame(best['trades']).to_csv(trades_path, index=False)
        print(f"最良トレード詳細: {trades_path}")

    # 最良名保存
    with open(os.path.join(RESULTS_DIR, 'htf_best_strategy_name.txt'), 'w') as f:
        f.write(best['name'] if best else '')

    return path


if __name__ == '__main__':
    print("=== RUN-20260305-002: 4時間足フィルター バックテスト ===")
    bars_1h = load_1h()
    bars_4h = load_4h()
    print(f"1時間足: {len(bars_1h)} bars ({bars_1h.index[0]} 〜 {bars_1h.index[-1]})")
    print(f"4時間足: {len(bars_4h)} bars ({bars_4h.index[0]} 〜 {bars_4h.index[-1]})")

    results = run_all(bars_1h, bars_4h)
    best = find_best(results)

    if best:
        print(f"\n=== 最良ロジック: {best['name']} ===")
        print(f"  PF: {best['profit_factor']:.3f} | 勝率: {best['win_rate_pct']:.1f}% | "
              f"DD: {best['max_drawdown_pct']:.1f}% | リターン: {best['total_return_pct']:.2f}%")

    save_results(results, best)
    print("\n完了。")
