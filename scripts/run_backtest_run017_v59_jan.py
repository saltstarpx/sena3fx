import pandas as pd
import numpy as np
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v59 import generate_signals

def run_backtest(csv_path, spread_pips=1.0):
    spread = spread_pips * 0.0001
    
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    print("Generating signals...")
    signal_series, tp_series, sl_series, _ = generate_signals(df, data_15m, data_4h, spread)
    active_signals = signal_series[signal_series != 0].index
    print(f"Signals generated: {len(active_signals)}")

    trades = []
    
    # タイム・ディケイの期間 (15分)
    time_decay_minutes = 15

    for entry_time in active_signals:
        direction = signal_series.loc[entry_time]
        entry_price = df.loc[entry_time, "close"]
        tp = tp_series.loc[entry_time]
        sl = sl_series.loc[entry_time]
        
        # エントリー後のデータフレーム
        post_entry_data = df.loc[entry_time:].copy()
        
        pnl = 0
        exit_time = None
        exit_price = None
        
        # タイム・ディケイの終了時刻
        time_decay_exit_time = entry_time + pd.Timedelta(minutes=time_decay_minutes)

        for t, bar in post_entry_data.iterrows():
            # タイム・ディケイによる強制決済
            if t >= time_decay_exit_time:
                if direction == 1: # LONG
                    pnl = (bar["close"] - entry_price) - spread
                else: # SHORT
                    pnl = (entry_price - bar["close"]) - spread
                exit_time = t
                exit_price = bar["close"]
                break

            if direction == 1: # LONG
                if bar["low"] <= sl:
                    pnl = (sl - entry_price) - spread
                    exit_time = t
                    exit_price = sl
                    break
                elif bar["high"] >= tp:
                    pnl = (tp - entry_price) - spread
                    exit_time = t
                    exit_price = tp
                    break
            else: # SHORT
                if bar["high"] >= sl:
                    pnl = (entry_price - sl) - spread
                    exit_time = t
                    exit_price = sl
                    break
                elif bar["low"] <= tp:
                    pnl = (entry_price - tp) - spread
                    exit_time = t
                    exit_price = tp
                    break
        
        if exit_time is not None:
            trades.append({
                "entry_time": entry_time,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "tp": tp,
                "sl": sl,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl": pnl
            })

    total_trades = len(trades)
    if total_trades == 0:
        print("No trades executed.")
        return

    win_trades = [t for t in trades if t["pnl"] > 0]
    loss_trades = [t for t in trades if t["pnl"] <= 0]
    
    total_profit = sum(t["pnl"] for t in win_trades)
    total_loss = abs(sum(t["pnl"] for t in loss_trades))
    pf = total_profit / total_loss if total_loss > 0 else 0
    win_rate = (len(win_trades) / total_trades) * 100 if total_trades > 0 else 0
    # NaNを除外して合計を計算
    balance = sum(t["pnl"] for t in trades if not pd.isna(t["pnl"]))

    print("="*80)
    # デバッグ用に最初の5件のトレードを表示
    for i, t in enumerate(trades[:5]):
        print(f"Trade {i+1}: Entry={t['entry_time']}, PnL={t['pnl']:.5f}")
    print(f"【RUN-017_v59_Jan】結果")
    print(f"プロフィットファクター (PF): {pf:.4f}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"総取引数: {total_trades}")
    print(f"最終損益 (pips): {balance / 0.0001:.2f}")
    print("="*80)

    os.makedirs("/home/ubuntu/sena3fx/results", exist_ok=True)
    pd.DataFrame(trades).to_csv(f"/home/ubuntu/sena3fx/results/run017_v59_{os.path.basename(csv_path)}_trades.csv", index=False)

if __name__ == "__main__":
    print("\n--- Testing January (v59) ---")
    run_backtest("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Jan.csv", spread_pips=0.2)
