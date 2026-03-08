"""
RUN-009: 改善版やがみ式プライスアクション (v2) バックテスト
======================================================
画像解析から得られた「実体の揃い」と「インサイドバー」を組み込んだロジックを検証。
"""
import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.backtest import BacktestEngine
from strategies.yagami_pa_v2 import signal_pa_v2_improved

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'usdjpy_1h.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    if 'tick_count' not in df.columns:
        df['tick_count'] = 100
    return df

def run_backtest():
    print("=== RUN-009: 改善版やがみ式 v2 バックテスト開始 ===")
    bars = load_data()
    print(f"データ期間: {bars.index[0]} 〜 {bars.index[-1]} ({len(bars)} bars)")

    # パラメータ設定
    thresholds = [0.10, 0.15, 0.20]
    all_results = []

    for th in thresholds:
        name = f"Yagami_v2_Align_{th}"
        print(f"\n検証中: {name}")
        
        engine = BacktestEngine(
            init_cash=5_000_000,
            risk_pct=0.02,
            pip=0.01,
            slippage_pips=0.5,
            default_sl_atr=1.5,
            default_tp_atr=3.0,
            use_dynamic_sl=True,
            sl_n_confirm=2
        )
        
        # signal_funcにパラメータを渡すためのクロージャ
        def custom_signal(b, current_th=th):
            return signal_pa_v2_improved(b, alignment_threshold=current_th)

        result = engine.run(
            data=bars,
            signal_func=custom_signal,
            freq='1h',
            name=name
        )
        
        if result:
            print(f"  トレード数: {result['total_trades']}")
            print(f"  勝率: {result['win_rate_pct']:.1f}%")
            print(f"  PF: {result['profit_factor']:.3f}")
            print(f"  最大DD: {result['max_drawdown_pct']:.1f}%")
            all_results.append(result)

    # 最もPFの高い結果を特定
    best_result = None
    if all_results:
        best_result = max(all_results, key=lambda x: x['profit_factor'])

    # 結果の保存
    if all_results:
        summary_df = pd.DataFrame([{
            'name': r['strategy'],
            'total_trades': r['total_trades'],
            'win_rate_pct': r['win_rate_pct'],
            'profit_factor': r['profit_factor'],
            'max_drawdown_pct': r['max_drawdown_pct'],
            'total_return_pct': r['total_return_pct']
        } for r in all_results])
        
        summary_path = os.path.join(RESULTS_DIR, 'run009_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nRUN-009 サマリーを保存しました: {summary_path}")

    # 最良ロジックのトレード詳細を保存
    if best_result and best_result.get('trades'):
        trades_path = os.path.join(RESULTS_DIR, f"run009_best_trades_{best_result['strategy']}.csv")
        df_trades = pd.DataFrame(best_result['trades'])
        df_trades.to_csv(trades_path, index=False)
        print(f"最良トレード詳細保存: {trades_path}")

if __name__ == '__main__':
    run_backtest()
