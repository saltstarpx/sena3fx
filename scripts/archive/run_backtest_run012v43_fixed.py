#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v43 import generate_signals

class FixedBacktestEngine(BacktestEngine):
    def __init__(self, init_cash=1000000, slippage_pips=1.0, **kwargs):
        super().__init__(init_cash=init_cash, slippage_pips=slippage_pips, **kwargs)
        self.init_cash = init_cash
        self.slippage_pips = slippage_pips

    def run(self, data, signal_series, tp_series, sl_series, name='Strategy'):
        self.trades = []
        self.cash = self.init_cash
        
        print(f"Starting Backtest: {name} ({len(data)} bars)")
        
        for i in range(1, len(data)):
            sig = signal_series.iloc[i]
            
            if sig != 0:
                direction = 'LONG' if sig > 0 else 'SHORT'
                entry_price = data.iloc[i]['close']
                if direction == 'LONG':
                    entry_price += self.slippage_pips / 100
                else:
                    entry_price -= self.slippage_pips / 100
                
                tp_price = tp_series.iloc[i]
                sl_price = sl_series.iloc[i]
                
                self.trades.append({
                    'entry_time': data.index[i],
                    'direction': direction,
                    'entry_price': entry_price,
                    'tp': tp_price,
                    'sl': sl_price,
                    'status': 'OPEN'
                })
            
            for trade in self.trades:
                if trade['status'] == 'OPEN':
                    curr_price = data.iloc[i]['close']
                    should_exit = False
                    
                    # 損切り判定
                    if trade['direction'] == 'LONG' and curr_price <= trade['sl']:
                        should_exit = True
                    elif trade['direction'] == 'SHORT' and curr_price >= trade['sl']:
                        should_exit = True
                    
                    # 利益確定判定
                    if trade['direction'] == 'LONG' and curr_price >= trade['tp']:
                        should_exit = True
                    elif trade['direction'] == 'SHORT' and curr_price <= trade['tp']:
                        should_exit = True
                    
                    # 24時間経過しても決済されない場合は強制決済（安全策）
                    time_diff = (data.index[i] - trade['entry_time']).total_seconds() / 3600
                    if time_diff >= 24:
                        should_exit = True
                        
                    if should_exit:
                        exit_price = curr_price
                        if trade['direction'] == 'LONG':
                            pnl = (exit_price - trade['entry_price']) * 10000
                        else:
                            pnl = (trade['entry_price'] - exit_price) * 10000
                        
                        trade['exit_time'] = data.index[i]
                        trade['exit_price'] = exit_price
                        trade['pnl'] = pnl
                        trade['status'] = 'CLOSED'
                        self.cash += pnl
            
            if i % 10000 == 0:
                print(f"Progress: {i}/{len(data)} bars...")

        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        wins = [t for t in closed_trades if t['pnl'] > 0]
        losses = [t for t in closed_trades if t['pnl'] <= 0]
        
        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        pf = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'pf': pf,
            'win_rate': len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            'total_trades': len(closed_trades),
            'final_pnl': self.cash - self.init_cash,
            'trades': closed_trades
        }

def main():
    data_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m_2026_Jan.csv')
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    print("シグナル生成中...")
    signal_series, tp_series, sl_series, signals_detail = generate_signals(df, df_15m, df_4h, spread=0.01)
    print(f"生成されたシグナル: {len(signals_detail)} 件")
    
    engine = FixedBacktestEngine(init_cash=1000000, slippage_pips=1.0)
    
    results = engine.run(df, signal_series, tp_series, sl_series, name="Yagami_v43_Fixed")
    
    print("\n" + "=" * 80)
    print("【RUN-012v43-Fixed】ストレス・テスト結果")
    print("=" * 80)
    print(f"プロフィットファクター (PF): {results['pf']:.4f}")
    print(f"勝率: {results['win_rate']:.2f}%")
    print(f"総取引数: {results['total_trades']}")
    print(f"最終利益: {results['final_pnl']:.2f}")
    
    if results['total_trades'] > 0:
        trades_df = pd.DataFrame(results['trades'])
        output_csv = os.path.join(BASE_DIR, 'results', 'run012v43_trades.csv')
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        trades_df.to_csv(output_csv, index=False)
        print(f"\n取引履歴を保存しました: {output_csv}")

if __name__ == '__main__':
    main()
