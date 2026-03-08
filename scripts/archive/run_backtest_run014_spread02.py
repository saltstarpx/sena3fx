import pandas as pd
import numpy as np
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v58 import generate_signals

def run_backtest(csv_path, spread_pips=1.0):
    spread = spread_pips * 0.0001
    
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    print("Generating signals...")
    signal_series, tp_series, sl_series, signals_list = generate_signals(df, data_15m, data_4h, spread)
    print(f"Signals generated: {len(signals_list)}")

    if not signals_list:
        print("No signals to test.")
        return

    balance = 0
    trades = []
    
    for signal in signals_list:
        entry_time = signal['time']
        direction = signal['direction']
        entry_price = df.loc[entry_time, 'close']
        tp = tp_series.loc[entry_time]
        sl = sl_series.loc[entry_time]
        
        post_entry_data = df.loc[entry_time:]
        pnl = 0
        exit_time = post_entry_data.index[-1]
        exit_price = post_entry_data.iloc[-1]['close']
        
        for t, bar in post_entry_data.iterrows():
            if direction == 'LONG':
                if bar['low'] <= sl:
                    pnl = (sl - entry_price) - spread
                    exit_time = t
                    exit_price = sl
                    break
                elif bar['high'] >= tp:
                    pnl = (tp - entry_price) - spread
                    exit_time = t
                    exit_price = tp
                    break
            else: # SHORT
                if bar['high'] >= sl:
                    pnl = (entry_price - sl) - spread
                    exit_time = t
                    exit_price = sl
                    break
                elif bar['low'] <= tp:
                    pnl = (entry_price - tp) - spread
                    exit_time = t
                    exit_price = tp
                    break
        
        trades.append({
            "entry_time": entry_time,
            "direction": direction,
            "entry_price": entry_price,
            "tp": tp,
            "sl": sl,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl
        })
        balance += pnl

    total_trades = len(trades)
    win_trades = [t for t in trades if t['pnl'] > 0]
    loss_trades = [t for t in trades if t['pnl'] <= 0]
    
    total_profit = sum(t['pnl'] for t in win_trades)
    total_loss = abs(sum(t['pnl'] for t in loss_trades))
    pf = total_profit / total_loss if total_loss > 0 else 0
    win_rate = (len(win_trades) / total_trades) * 100 if total_trades > 0 else 0

    print("="*80)
    print(f"【RUN-012v58】結果")
    print(f"プロフィットファクター (PF): {pf:.4f}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"総取引数: {total_trades}")
    print(f"最終損益 (pips): {balance / 0.0001:.2f}")
    print("="*80)

    os.makedirs("/home/ubuntu/sena3fx/results", exist_ok=True)
    pd.DataFrame(trades).to_csv(f"/home/ubuntu/sena3fx/results/run012v58_{os.path.basename(csv_path)}_trades.csv", index=False)

if __name__ == "__main__":
    print("\n--- Testing January ---")
    run_backtest("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Jan.csv", spread_pips=0.2)
    print("\n--- Testing February ---")
    run_backtest("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Feb.csv", spread_pips=0.2)
