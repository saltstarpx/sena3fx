import pandas as pd
import sys
sys.path.append('/home/ubuntu/sena3fx/strategies')
import numpy as np
import os
from datetime import datetime
from yagami_mtf_v64 import generate_signals, calculate_atr

# データ読み込み関数
def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df

# バックテスト実行関数
def run_backtest(symbol, start_date, end_date, spread_pips):
    print(f"Running backtest for {symbol} from {start_date} to {end_date} with spread {spread_pips} pips")

    # データのロード
    data_1m = load_data(symbol.lower(), '1m', start_date, end_date)
    data_15m = load_data(symbol.lower(), '15m', start_date, end_date)
    data_1h = load_data(symbol.lower(), '1h', start_date, end_date)
    data_4h = load_data(symbol.lower(), '4h', start_date, end_date)

    # シグナル生成
    signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series = generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips)

    trades = []
    current_position = 0 # 0:なし, 1:ロング, -1:ショート
    entry_price = 0
    entry_time = None
    take_profit = 0
    stop_loss = 0
    risk_amount = 0
    original_risk_amount = 0 
    half_position_closed = False
    half_tp = 0

    for i in range(len(data_1m)):
        current_bar = data_1m.iloc[i]
        current_time = current_bar.name
        current_price = current_bar['close']

        if current_position != 0:
            # 半利確ロジック (リスク幅の1.0倍)
            if not half_position_closed:
                if current_position == 1 and current_bar['high'] >= half_tp:
                    pnl_pips = (half_tp - entry_price) * 100 
                    trades.append({
                        'EntryTime': entry_time,
                        'ExitTime': current_time,
                        'Position': 'LONG_HALF',
                        'EntryPrice': entry_price,
                        'ExitPrice': half_tp,
                        'PnL_pips': pnl_pips,
                        'Risk_pips': original_risk_amount * 100,
                        'Type': 'HALF_TP'
                    })
                    stop_loss = entry_price
                    half_position_closed = True
                elif current_position == -1 and current_bar['low'] <= half_tp:
                    pnl_pips = (entry_price - half_tp) * 100
                    trades.append({
                        'EntryTime': entry_time,
                        'ExitTime': current_time,
                        'Position': 'SHORT_HALF',
                        'EntryPrice': entry_price,
                        'ExitPrice': half_tp,
                        'PnL_pips': pnl_pips,
                        'Risk_pips': original_risk_amount * 100,
                        'Type': 'HALF_TP'
                    })
                    stop_loss = entry_price
                    half_position_closed = True

            # 損切り判定
            if (current_position == 1 and current_bar['low'] <= stop_loss) or \
               (current_position == -1 and current_bar['high'] >= stop_loss):
                exit_price = stop_loss
                pnl_pips = (exit_price - entry_price) * 100 * current_position
                trades.append({
                    'EntryTime': entry_time,
                    'ExitTime': current_time,
                    'Position': 'LONG' if current_position == 1 else 'SHORT',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL_pips': pnl_pips,
                    'Risk_pips': original_risk_amount * 100,
                    'Type': 'SL'
                })
                current_position = 0
                entry_price = 0
                entry_time = None
                take_profit = 0
                stop_loss = 0
                risk_amount = 0
                original_risk_amount = 0
                half_position_closed = False
                half_tp = 0
                continue

            # 利確判定 (残り半分)
            if (current_position == 1 and current_bar['high'] >= take_profit) or \
               (current_position == -1 and current_bar['low'] <= take_profit):
                exit_price = take_profit
                pnl_pips = (exit_price - entry_price) * 100 * current_position
                trades.append({
                    'EntryTime': entry_time,
                    'ExitTime': current_time,
                    'Position': 'LONG' if current_position == 1 else 'SHORT',
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL_pips': pnl_pips,
                    'Risk_pips': original_risk_amount * 100,
                    'Type': 'TP'
                })
                current_position = 0
                entry_price = 0
                entry_time = None
                take_profit = 0
                stop_loss = 0
                risk_amount = 0
                original_risk_amount = 0
                half_position_closed = False
                half_tp = 0
                continue

        if current_position == 0:
            signal = signal_series.loc[current_time]
            if signal != 0:
                entry_price = current_price
                entry_time = current_time
                current_position = signal
                take_profit = tp_series.loc[current_time]
                stop_loss = sl_series.loc[current_time]
                risk_amount = abs(entry_price - stop_loss)
                original_risk_amount = risk_amount
                half_position_closed = False

                if current_position == 1:
                    half_tp = entry_price + risk_amount
                else:
                    half_tp = entry_price - risk_amount

    trades_df = pd.DataFrame(trades)
    return trades_df

if __name__ == "__main__":
    SYMBOL = 'USDJPY'
    START_DATE = '2024-07-01'
    END_DATE = '2024-08-06'
    SPREAD_PIPS = 0.2

    trades_v64 = run_backtest(SYMBOL, START_DATE, END_DATE, SPREAD_PIPS)

    output_dir = "/home/ubuntu/sena3fx/results"
    os.makedirs(output_dir, exist_ok=True)
    trades_v64.to_csv(os.path.join(output_dir, "trades_v64.csv"), index=False)

    print(f"Backtest completed. Results saved to {os.path.join(output_dir, 'trades_v64.csv')}")

    if not trades_v64.empty:
        total_pnl = trades_v63 = trades_v64['PnL_pips'].sum()
        num_trades = len(trades_v64[~trades_v64['Position'].str.contains('HALF')]) 
        winning_trades = trades_v64[trades_v64['PnL_pips'] > 0]
        losing_trades = trades_v64[trades_v64['PnL_pips'] < 0]
        
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        avg_win = winning_trades['PnL_pips'].mean() if num_wins > 0 else 0
        avg_loss = losing_trades['PnL_pips'].mean() if num_losses > 0 else 0
        
        total_profit = winning_trades['PnL_pips'].sum()
        total_loss = abs(losing_trades['PnL_pips'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        print(f"\n--- Backtest Statistics (v64) ---")
        print(f"Total PnL: {total_pnl:.2f} pips")
        print(f"Number of Entries: {num_trades}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: {avg_win:.2f} pips")
        print(f"Average Loss: {avg_loss:.2f} pips")
        
        if avg_loss != 0:
            b = abs(avg_win / avg_loss)
            p = num_wins / (num_wins + num_losses)
            q = 1 - p
            kelly_fraction = (b * p - q) / b if b != 0 else 0
            print(f"Kelly Criterion (Fraction): {kelly_fraction:.4f}")
            print(f"Win Rate (Excl. Break-even): {p:.2%}")
    else:
        print("No trades were executed.")
