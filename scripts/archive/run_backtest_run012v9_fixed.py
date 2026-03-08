#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v9 import generate_signals

class FixedBacktestEngine(BacktestEngine):
    def __init__(self, init_cash=1000000, slippage_pips=1.0, **kwargs):
        super().__init__(init_cash=init_cash, slippage_pips=slippage_pips, **kwargs)
        self.init_cash = init_cash
        self.slippage_pips = slippage_pips

    def run(self, data, signal_func, freq='1min', name='Strategy', **kwargs):
        self.trades = []
        self.cash = self.init_cash
        
        print(f"Starting Backtest: {name} ({len(data)} bars)")
        
        for i in range(1, len(data)):
            sig = signal_func(data.iloc[:i+1])
            
            if sig != 0:
                direction = 'LONG' if sig > 0 else 'SHORT'
                entry_price = data.iloc[i]['close']
                if direction == 'LONG':
                    entry_price += 0.01
                else:
                    entry_price -= 0.01
                
                self.trades.append({
                    'entry_time': data.index[i],
                    'direction': direction,
                    'entry_price': entry_price,
                    'status': 'OPEN'
                })
            
            # 決済ロジック（4時間ホールド）
            for trade in self.trades:
                if trade['status'] == 'OPEN':
                    time_diff = (data.index[i] - trade['entry_time']).total_seconds() / 3600
                    if time_diff >= 4:
                        exit_price = data.iloc[i]['close']
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
    signal_series, signals_detail = generate_signals(df, df_15m, df_4h, spread=0.01)
    print(f"生成されたシグナル: {len(signals_detail)} 件")
    
    engine = FixedBacktestEngine(init_cash=1000000, slippage_pips=1.0)
    
    def signal_func(data):
        try:
            return signal_series.loc[data.index[-1]]
        except:
            return 0
            
    results = engine.run(df, signal_func, name="Yagami_v9_Fixed")
    
    print("\n" + "=" * 80)
    print("【RUN-012v9-Fixed】ストレス・テスト結果")
    print("=" * 80)
    print(f"プロフィットファクター (PF): {results['pf']:.4f}")
    print(f"勝率: {results['win_rate']:.2f}%")
    print(f"総取引数: {results['total_trades']}")
    print(f"最終利益: {results['final_pnl']:.2f}")
    
    if results['total_trades'] > 0:
        pnl_series = pd.Series([t['pnl'] for t in results['trades']])
        print(f"\n損益統計:")
        print(pnl_series.describe())

if __name__ == '__main__':
    main()
