"""
やがみプライスアクション バックテスト実行スクリプト
================================================
複数パラメータセットを検証し、最良ロジックを特定する。
"""
import sys
import os
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.backtest import BacktestEngine
from strategies.yagami_pa import (
    signal_pa1_reversal,
    signal_pa2_pinbar,
    signal_pa3_engulf,
    signal_pa4_combined,
    signal_pa5_combined_strict,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'usdjpy_1h.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """USD/JPY 1時間足データを読み込む。"""
    df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    # tick_countカラムがない場合は追加
    if 'tick_count' not in df.columns:
        df['tick_count'] = 100
    return df


def run_all_backtests(bars):
    """全パラメータセットでバックテストを実行。"""
    results = []

    # パラメータセット定義
    param_sets = [
        # (名前, signal_func, engine_params)
        {
            'name': 'PA1_Reversal_Base',
            'signal': lambda b: signal_pa1_reversal(b, zone_atr=1.5, lookback=20),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA1_Reversal_TightSL',
            'signal': lambda b: signal_pa1_reversal(b, zone_atr=1.5, lookback=20),
            'engine': dict(default_sl_atr=1.0, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=1, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA2_Pinbar_Base',
            'signal': lambda b: signal_pa2_pinbar(b, zone_atr=1.5, lookback=20, wick_body_ratio=2.0),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA2_Pinbar_StrictWick',
            'signal': lambda b: signal_pa2_pinbar(b, zone_atr=1.5, lookback=20, wick_body_ratio=3.0),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA3_Engulf_Base',
            'signal': lambda b: signal_pa3_engulf(b, zone_atr=2.0, lookback=20),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA4_Combined_Base',
            'signal': lambda b: signal_pa4_combined(b, zone_atr=1.5, lookback=20),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA4_Combined_Trail',
            'signal': lambda b: signal_pa4_combined(b, zone_atr=1.5, lookback=20),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=5.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=2.0, trail_dist_atr=1.5,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA5_Strict_Base',
            'signal': lambda b: signal_pa5_combined_strict(b, zone_atr=1.2, lookback=20),
            'engine': dict(default_sl_atr=1.5, default_tp_atr=3.0, use_dynamic_sl=True,
                           sl_n_confirm=2, pyramid_entries=0, trail_start_atr=0.0,
                           target_max_dd=0.20, target_min_wr=0.35),
        },
        {
            'name': 'PA5_Strict_HighRR',
            'signal': lambda b: signal_pa5_combined_strict(b, zone_atr=1.2, lookback=20),
            'engine': dict(default_sl_atr=1.0, default_tp_atr=4.0, use_dynamic_sl=True,
                           sl_n_confirm=1, pyramid_entries=0, trail_start_atr=2.5, trail_dist_atr=1.5,
                           target_max_dd=0.20, target_min_wr=0.30),
        },
    ]

    for ps in param_sets:
        print(f"\n実行中: {ps['name']}")
        try:
            engine = BacktestEngine(
                init_cash=5_000_000,
                risk_pct=0.02,
                pip=0.01,
                slippage_pips=0.5,
                **ps['engine']
            )
            result = engine.run(
                data=bars,
                signal_func=ps['signal'],
                freq='1h',
                name=ps['name'],
            )
            if result is None:
                print(f"  → トレードなし")
                continue

            # 結果サマリー
            r = result
            print(f"  トレード数: {r['total_trades']}")
            print(f"  勝率: {r['win_rate_pct']:.1f}%")
            print(f"  PF: {r['profit_factor']:.3f}")
            print(f"  最大DD: {r['max_drawdown_pct']:.1f}%")
            print(f"  総リターン: {r['total_return_pct']:.1f}%")
            print(f"  RR比: {r['rr_ratio']:.2f}")
            print(f"  合格: {r['passed']}")

            results.append({
                'name': ps['name'],
                'total_trades': r['total_trades'],
                'win_rate_pct': r['win_rate_pct'],
                'profit_factor': r['profit_factor'],
                'max_drawdown_pct': r['max_drawdown_pct'],
                'total_return_pct': r['total_return_pct'],
                'rr_ratio': r['rr_ratio'],
                'passed': r['passed'],
                'trades': r['trades'],
            })

        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()

    return results


def find_best(results):
    """最良ロジックを特定する（スコアリング）。"""
    if not results:
        return None

    def score(r):
        # スコア = PF × 勝率 × (1 - DD/100) × log(トレード数+1)
        pf = min(r['profit_factor'], 5.0)
        wr = r['win_rate_pct'] / 100
        dd = r['max_drawdown_pct'] / 100
        n = r['total_trades']
        if n < 3:
            return 0
        return pf * wr * (1 - dd) * np.log(n + 1)

    scored = [(score(r), r) for r in results]
    scored.sort(key=lambda x: x[0], reverse=True)
    print("\n=== ランキング ===")
    for i, (s, r) in enumerate(scored[:5]):
        print(f"  {i+1}. {r['name']}: score={s:.3f} | PF={r['profit_factor']:.3f} | WR={r['win_rate_pct']:.1f}% | DD={r['max_drawdown_pct']:.1f}% | N={r['total_trades']}")
    return scored[0][1]


def save_results(results, best):
    """バックテスト結果をCSVに保存。"""
    output_path = os.path.join(RESULTS_DIR, 'yagami_backtest_summary.csv')
    fieldnames = ['name', 'total_trades', 'win_rate_pct', 'profit_factor',
                  'max_drawdown_pct', 'total_return_pct', 'rr_ratio', 'passed']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            w.writerow(row)
    print(f"\nサマリー保存: {output_path}")

    # 最良ロジックのトレード詳細を保存
    if best and best.get('trades'):
        trades_path = os.path.join(RESULTS_DIR, f"best_trades_{best['name']}.csv")
        df_trades = pd.DataFrame(best['trades'])
        df_trades.to_csv(trades_path, index=False)
        print(f"最良トレード詳細保存: {trades_path}")

    return output_path


if __name__ == '__main__':
    print("=== やがみプライスアクション バックテスト ===")
    print(f"データ読み込み中: {DATA_PATH}")
    bars = load_data()
    print(f"バー数: {len(bars)}, 期間: {bars.index[0]} 〜 {bars.index[-1]}")

    results = run_all_backtests(bars)
    best = find_best(results)

    if best:
        print(f"\n=== 最良ロジック: {best['name']} ===")
        print(f"  PF: {best['profit_factor']:.3f}")
        print(f"  勝率: {best['win_rate_pct']:.1f}%")
        print(f"  最大DD: {best['max_drawdown_pct']:.1f}%")
        print(f"  総リターン: {best['total_return_pct']:.1f}%")
        print(f"  トレード数: {best['total_trades']}")

    save_results(results, best)

    # 最良ロジック名をファイルに保存（次フェーズで使用）
    best_name_path = os.path.join(RESULTS_DIR, 'best_strategy_name.txt')
    if best:
        with open(best_name_path, 'w') as f:
            f.write(best['name'])
        print(f"\n最良ロジック名保存: {best_name_path}")
