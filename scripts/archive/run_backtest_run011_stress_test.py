#!/usr/bin/env python3
"""
RUN-011: やがみ式3層MTF戦略 - スプレッド1.0pips負荷テスト
2026年1月データを用いた「ストレス・バックテスト」
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v3 import signal_yagami_mtf_v3

def resample_ohlc(df, freq):
    if df.index.name != 'timestamp':
        df = df.reset_index()
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns: agg_dict['volume'] = 'sum'
    elif 'tick_count' in df.columns: agg_dict['tick_count'] = 'sum'
    resampled = df.resample(freq).agg(agg_dict)
    return resampled.dropna(subset=['open'])

def apply_spread_cost(entry_price, exit_price, spread_pips=1.0, direction='long'):
    """
    スプレッドコストを適用する
    spread_pips: スプレッド幅 (pips)
    """
    # ドル円は0.01が1pips
    spread_cost = spread_pips * 0.01
    
    if direction == 'long':
        # ロング: エントリー時に買値、決済時に売値
        # スプレッドにより、実質的なエントリー価格が上昇、決済価格が低下
        adjusted_entry = entry_price + spread_cost / 2
        adjusted_exit = exit_price - spread_cost / 2
    else:
        # ショート: エントリー時に売値、決済時に買値
        adjusted_entry = entry_price - spread_cost / 2
        adjusted_exit = exit_price + spread_cost / 2
    
    return adjusted_entry, adjusted_exit

def main():
    print("=" * 80)
    print("RUN-011: やがみ式3層MTF戦略 - スプレッド1.0pips負荷テスト")
    print("=" * 80)
    
    m1_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m_2026_Jan.csv')
    m1_df = pd.read_csv(m1_path, index_col=0, parse_dates=True)
    m1_df.index.name = 'timestamp'
    
    m15_df = resample_ohlc(m1_df, '15min')
    h4_df = resample_ohlc(m1_df, '4h')
    
    print(f"\nデータ期間: {m1_df.index[0]} 〜 {m1_df.index[-1]}")
    print(f"1分足バー数: {len(m1_df)}")
    print(f"15分足バー数: {len(m15_df)}")
    print(f"4時間足バー数: {len(h4_df)}")
    
    print("\nシグナル生成中...")
    all_signals = pd.Series(index=m1_df.index, dtype=object)
    sl_map = {}
    tp_map = {}
    
    for i in range(100, len(m1_df)):
        current_time = m1_df.index[i]
        h4_slice = h4_df[h4_df.index <= current_time].tail(20)
        m15_slice = m15_df[m15_df.index <= current_time].tail(20)
        m1_slice = m1_df[m1_df.index <= current_time].tail(20)
        
        if len(h4_slice) < 5 or len(m15_slice) < 5 or len(m1_slice) < 5: continue
        
        signal = signal_yagami_mtf_v3(h4_slice, m15_slice, m1_slice)
        if signal.get('signal'):
            all_signals.iloc[i] = signal['signal']
            sl_map[current_time] = signal['stop_loss']
            tp_map[current_time] = signal['take_profit']
    
    signal_count = all_signals.count()
    print(f"生成されたシグナル: {signal_count} 件")
    if signal_count == 0:
        print("シグナルが生成されませんでした。")
        return

    # ========== RUN-010 (スプレッドなし) ==========
    print("\n" + "=" * 80)
    print("【RUN-010】スプレッドなし（参考値）")
    print("=" * 80)
    
    engine_no_spread = BacktestEngine(init_cash=1000000, risk_pct=0.02, min_sl_atr_mult=0.0, min_sl_price_pct=0.0)
    engine_no_spread._find_swing_low = lambda bars, idx, htf_bars=None, n_confirm=2: sl_map.get(bars.index[idx], bars['close'].iloc[idx] * 0.99)
    engine_no_spread._find_swing_high = lambda bars, idx, htf_bars=None, n_confirm=2: sl_map.get(bars.index[idx], bars['close'].iloc[idx] * 1.01)
    engine_no_spread.use_dynamic_sl = True
    
    def signal_func_no_spread(bars):
        return all_signals
    
    results_no_spread = engine_no_spread.run(data=m1_df, signal_func=signal_func_no_spread, name="Yagami_MTF_v3_RUN010", freq="1m")
    
    if results_no_spread:
        print(f"プロフィットファクター (PF): {results_no_spread.get('profit_factor', 'N/A'):.4f}")
        print(f"勝率: {results_no_spread.get('win_rate_pct', 'N/A'):.2f}%")
        print(f"総取引数: {results_no_spread.get('total_trades', 'N/A')}")
        print(f"最終利益: {results_no_spread.get('total_pnl', 'N/A'):.2f}")
        print(f"最大ドローダウン: {results_no_spread.get('max_drawdown_pct', 'N/A'):.2f}%")

    # ========== RUN-011 (スプレッド1.0pips) ==========
    print("\n" + "=" * 80)
    print("【RUN-011】スプレッド1.0pips負荷テスト")
    print("=" * 80)
    
    # スプレッドを適用したトレード履歴を作成
    if results_no_spread and results_no_spread.get('trades'):
        trades_with_spread = []
        for trade in results_no_spread['trades']:
            adjusted_entry, adjusted_exit = apply_spread_cost(
                trade['entry'], trade['exit'], spread_pips=1.0, direction=trade['dir']
            )
            
            # スプレッド適用後の損益を再計算
            if trade['dir'] == 'long':
                pnl_with_spread = (adjusted_exit - adjusted_entry) * 1000  # 1000ロット仮定
            else:
                pnl_with_spread = (adjusted_entry - adjusted_exit) * 1000
            
            trades_with_spread.append({
                'dir': trade['dir'],
                'entry': adjusted_entry,
                'exit': adjusted_exit,
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'pnl': pnl_with_spread,
                'reason': trade['reason'],
                'pyramids': trade.get('pyramids', 0)
            })
        
        # スプレッド適用後の統計を計算
        trades_df = pd.DataFrame(trades_with_spread)
        total_pnl = trades_df['pnl'].sum()
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0
        total_wins = winning_trades['pnl'].sum()
        total_losses = abs(losing_trades['pnl'].sum())
        pf = total_wins / total_losses if total_losses > 0 else 0
        
        print(f"プロフィットファクター (PF): {pf:.4f}")
        print(f"勝率: {win_rate:.2f}%")
        print(f"総取引数: {len(trades_df)}")
        print(f"最終利益: {total_pnl:.2f}")
        print(f"総利益: {total_wins:.2f}")
        print(f"総損失: {total_losses:.2f}")
        
        # 結果を保存
        results_dir = os.path.join(BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        trades_df.to_csv(os.path.join(results_dir, 'run011_trades_with_spread.csv'), index=False)
        
        # ========== 比較レポート ==========
        print("\n" + "=" * 80)
        print("【比較レポート】RUN-010 vs RUN-011")
        print("=" * 80)
        
        comparison = {
            '項目': ['PF', '勝率 (%)', '総取引数', '最終利益'],
            'RUN-010 (スプレッドなし)': [
                f"{results_no_spread.get('profit_factor', 0):.4f}",
                f"{results_no_spread.get('win_rate_pct', 0):.2f}",
                f"{results_no_spread.get('total_trades', 0)}",
                f"{results_no_spread.get('total_pnl', 0):.2f}"
            ],
            'RUN-011 (スプレッド1.0pips)': [
                f"{pf:.4f}",
                f"{win_rate:.2f}",
                f"{len(trades_df)}",
                f"{total_pnl:.2f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(os.path.join(results_dir, 'run010_vs_run011_comparison.csv'), index=False)
        
        print("\n✓ 比較レポートを保存しました。")

if __name__ == '__main__':
    main()
