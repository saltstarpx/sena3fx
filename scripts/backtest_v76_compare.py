"""
v76 vs v76改qlib 比較バックテスト
===================================
v76現行版と定量改善版を同一データ・同一条件で比較。
対象: USDJPY / EURJPY / GBPJPY (1Hデータ)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.backtest import BacktestEngine
from strategies.v76_mtf import (make_v76_signal, resample_to_4h,
                                 V76_ENGINE_PARAMS, PAIR_CONFIGS as V76_PAIRS)
from strategies.v76_qlib import (make_v76_qlib_signal,
                                  V76_QLIB_ENGINE_PARAMS,
                                  PAIR_CONFIGS as QLIB_PAIRS)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load(pair, tf):
    path = os.path.join(DATA_DIR, f'{pair}_{tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fmt_metric(val, fmt='.3f'):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    return f'{val:{fmt}}'


def extract_metrics(res):
    """BacktestEngine結果からメトリクスを抽出"""
    if not res or res.get('total_trades', 0) == 0:
        return {
            'trades': 0, 'pf': 0, 'wr': 0, 'mdd': 0,
            'net_pnl': 0, 'avg_rr': 0, 'ret_pct': 0,
        }
    return {
        'trades': res.get('total_trades', 0),
        'pf': res.get('profit_factor', 0),
        'wr': res.get('win_rate_pct', 0),
        'mdd': res.get('max_drawdown_pct', 0),
        'net_pnl': res.get('total_pnl', 0),
        'avg_rr': res.get('rr_ratio', 0),
        'ret_pct': res.get('total_return_pct', 0),
    }


def run_pair(pair, bars_1h, bars_4h):
    """1ペアのv76 vs v76改qlib比較"""
    pair_cfg = V76_PAIRS.get(pair, {'pip': 0.01, 'slippage_pips': 0.5})

    results = {}

    # ===== v76 現行版 =====
    eng_params = V76_ENGINE_PARAMS.copy()
    eng_params['pip'] = pair_cfg['pip']
    eng_params['slippage_pips'] = pair_cfg['slippage_pips']
    eng_v76 = BacktestEngine(**eng_params)

    sig_fn_v76 = make_v76_signal(bars_4h)
    res_v76 = eng_v76.run(bars_1h, sig_fn_v76, name=f'v76_{pair}')
    results['v76'] = extract_metrics(res_v76)

    # ===== v76改qlib =====
    qlib_params = V76_QLIB_ENGINE_PARAMS.copy()
    qlib_pair_cfg = QLIB_PAIRS.get(pair, pair_cfg)
    qlib_params['pip'] = qlib_pair_cfg['pip']
    qlib_params['slippage_pips'] = qlib_pair_cfg['slippage_pips']
    eng_qlib = BacktestEngine(**qlib_params)

    sig_fn_qlib = make_v76_qlib_signal(bars_4h)
    res_qlib = eng_qlib.run(bars_1h, sig_fn_qlib, name=f'v76qlib_{pair}')
    results['v76_qlib'] = extract_metrics(res_qlib)

    return results


def print_comparison(all_results):
    """全ペア比較テーブルを表示"""
    print("\n" + "=" * 100)
    print("v76 vs v76改qlib 比較結果")
    print("=" * 100)

    header = f"{'ペア':<10} {'戦略':<12} {'トレード数':>10} {'PF':>8} {'勝率%':>8} {'MDD%':>8} {'純損益':>14} {'Avg RR':>8}"
    print(header)
    print("-" * 100)

    summary_v76 = {'trades': 0, 'pnl': 0}
    summary_qlib = {'trades': 0, 'pnl': 0}

    for pair, results in all_results.items():
        for strategy, m in results.items():
            label = 'v76' if strategy == 'v76' else 'v76改qlib'
            print(f"{pair:<10} {label:<12} {m['trades']:>10} {m['pf']:>8.3f} {m['wr']:>8.1f} {m['mdd']:>8.1f} {m['net_pnl']:>14,.0f} {m['avg_rr']:>8.2f}")

            if strategy == 'v76':
                summary_v76['trades'] += m['trades']
                summary_v76['pnl'] += m['net_pnl']
            else:
                summary_qlib['trades'] += m['trades']
                summary_qlib['pnl'] += m['net_pnl']
        print()

    print("-" * 100)
    print(f"\n合計サマリー:")
    print(f"  v76:      トレード={summary_v76['trades']:>5}, 純損益={summary_v76['pnl']:>14,.0f}")
    print(f"  v76改qlib: トレード={summary_qlib['trades']:>5}, 純損益={summary_qlib['pnl']:>14,.0f}")

    # 改善率
    if summary_v76['pnl'] != 0:
        improvement = (summary_qlib['pnl'] - summary_v76['pnl']) / abs(summary_v76['pnl']) * 100
        print(f"\n  純損益改善率: {improvement:+.1f}%")

    print("\n" + "=" * 100)
    print("\nv76改qlib 改善点:")
    print("  1. ADXフィルター (>20): レンジ相場のフェイクシグナル除去")
    print("  2. RSI確認 (二番底<40, 二番天井>60): 逆張りタイミング精度向上")
    print("  3. ボラレジーム (ATR比率>0.6): 低ボラ期の小幅損切り回避")
    print("  4. 確認足 (陽線/陰線): パターン確定前のダマシ排除")
    print("  5. 動的SL (スウィングベース): ATR固定→実際のサポレジに基づくSL")
    print("  6. RR 1:3.0 (v76は1:2.5): 高RRで少ない勝率でも利益確保")
    print("  7. トレーリングストップ (2ATR発動): 含み益を最大化")
    print("  8. EMA20+EMA50クロス: トレンド確認の二重フィルター")


def save_results_csv(all_results):
    """結果をCSV保存"""
    rows = []
    for pair, results in all_results.items():
        for strategy, m in results.items():
            rows.append({
                'pair': pair,
                'strategy': strategy,
                'trades': m['trades'],
                'profit_factor': m['pf'],
                'win_rate_pct': m['wr'],
                'max_drawdown_pct': m['mdd'],
                'net_pnl': m['net_pnl'],
                'avg_rr': m['avg_rr'],
            })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'v76_comparison.csv')
    df.to_csv(path, index=False)
    print(f"\n結果CSV保存: {path}")


def main():
    pairs = ['USDJPY', 'EURJPY', 'GBPJPY']
    all_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair} バックテスト実行中...")
        print(f"{'='*60}")

        # データロード
        bars_1h = load(pair, '1h')
        if bars_1h is None:
            print(f"  ⚠ {pair}_1h.csv が見つかりません。スキップ。")
            continue

        # 4Hデータ: 直接ロードまたは1Hからリサンプル
        bars_4h = load(pair, '4h')
        if bars_4h is None:
            print(f"  {pair}_4h.csv なし → 1Hからリサンプル")
            bars_4h = resample_to_4h(bars_1h)

        print(f"  1Hバー数: {len(bars_1h)}, 4Hバー数: {len(bars_4h)}")
        print(f"  期間: {bars_1h.index[0]} ~ {bars_1h.index[-1]}")

        results = run_pair(pair, bars_1h, bars_4h)
        all_results[pair] = results

        # ペア単位の簡易表示
        for strategy, m in results.items():
            label = 'v76' if strategy == 'v76' else 'v76改qlib'
            print(f"  {label}: N={m['trades']}, PF={m['pf']:.3f}, WR={m['wr']:.1f}%, MDD={m['mdd']:.1f}%")

    # 全体比較テーブル
    print_comparison(all_results)
    save_results_csv(all_results)


if __name__ == '__main__':
    main()
