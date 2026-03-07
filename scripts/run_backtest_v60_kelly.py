import pandas as pd
import numpy as np
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v60 import generate_signals

def run_backtest_kelly(df, spread_pips=0.2):
    # 15分足、1時間足、4時間足の作成
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

    # シグナル生成
    signal_series, tp_series, sl_series, entry_time_series, time_decay_minutes_series, atr_at_entry_series = generate_signals(df, data_15m, data_1h, data_4h, spread_pips=spread_pips)
    active_signals = signal_series[signal_series != 0].index

    trades = []
    open_positions = {}
    trade_id_counter = 0

    for current_time, current_bar in df.iterrows():
        positions_to_remove = []
        for pos_id, pos in list(open_positions.items()):
            exit_reason = None
            exit_price = 0

            # トレイリングストップ
            if pos['partial_closed']:
                if pos["direction"] == 1:
                    new_sl = current_bar['high'] - pos['atr_at_entry'] * 2.0
                    if new_sl > pos['sl']: pos['sl'] = new_sl
                else:
                    new_sl = current_bar['low'] + pos['atr_at_entry'] * 2.0
                    if new_sl < pos['sl']: pos['sl'] = new_sl

            # SL/TPヒット
            if pos["direction"] == 1:
                if current_bar["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"
            else:
                if current_bar["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"

            if exit_reason:
                pnl = (exit_price - pos["entry_price"]) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"] if pos["direction"] == 1 else (pos["entry_price"] - exit_price) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"]
                trades.append({"exit_time": current_time, "pnl": pnl, "exit_reason": exit_reason})
                positions_to_remove.append(pos_id)
            
            # タイムディケイ（修正版：半決済でもスプレッドを引く）
            elif current_time >= pos["time_decay_exit_time"] and not pos["partial_closed"]:
                current_pnl_raw = (current_bar['close'] - pos['entry_price']) if pos['direction'] == 1 else (pos['entry_price'] - current_bar['close'])
                if current_pnl_raw > (spread_pips * 0.01):
                    # 利益が出ている場合 -> 半決済
                    partial_pnl = (current_pnl_raw * 0.5) - (spread_pips * 0.01 * 0.5)
                    trades.append({"exit_time": current_time, "pnl": partial_pnl, "exit_reason": "Partial Close (Time Decay)"})
                    pos["position_size"] = 0.5
                    pos["partial_closed"] = True
                    pos["sl"] = pos["entry_price"]
                else:
                    # 利益が出ていない場合 -> 全決済
                    exit_price = current_bar['close']
                    pnl = (exit_price - pos["entry_price"]) - (spread_pips * 0.01) if pos["direction"] == 1 else (pos["entry_price"] - exit_price) - (spread_pips * 0.01)
                    trades.append({"exit_time": current_time, "pnl": pnl, "exit_reason": "Full Close (Time Decay)"})
                    positions_to_remove.append(pos_id)

        for pos_id in positions_to_remove:
            del open_positions[pos_id]

        if current_time in active_signals:
            direction = signal_series.loc[current_time]
            if direction != 0:
                trade_id_counter += 1
                if not any(pd.isna([tp_series.loc[current_time], sl_series.loc[current_time]])):
                    open_positions[trade_id_counter] = {
                        "trade_id": trade_id_counter, "entry_time": current_time, "direction": direction,
                        "entry_price": current_bar['close'], "tp": tp_series.loc[current_time], "sl": sl_series.loc[current_time],
                        "time_decay_exit_time": current_time + pd.Timedelta(minutes=time_decay_minutes_series.loc[current_time]),
                        "position_size": 1.0, "partial_closed": False, "atr_at_entry": atr_at_entry_series.loc[current_time]
                    }

    return pd.DataFrame(trades)

if __name__ == "__main__":
    df_q1 = pd.read_csv("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv")
    df_q1["timestamp"] = pd.to_datetime(df_q1["timestamp"])
    df_feb = df_q1[(df_q1["timestamp"].dt.month == 2) & (df_q1["timestamp"].dt.year == 2026)].copy()
    df_feb.set_index('timestamp', inplace=True)
    df_feb.dropna(inplace=True)

    trades_df = run_backtest_kelly(df_feb)
    trades_df['pnl_pips'] = trades_df['pnl'] / 0.01
    
    # ケリー計算
    win_trades = trades_df[trades_df['pnl'] > 0]
    loss_trades = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(win_trades) / len(trades_df)
    avg_win = win_trades['pnl_pips'].mean()
    avg_loss = abs(loss_trades['pnl_pips'].mean())
    odds = avg_win / avg_loss
    kelly_f = (odds * win_rate - (1 - win_rate)) / odds
    
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Win: {avg_win:.4f} pips, Avg Loss: {avg_loss:.4f} pips")
    print(f"Odds: {odds:.4f}")
    print(f"Kelly Fraction: {kelly_f:.4f}")
    print(f"Total PnL: {trades_df['pnl_pips'].sum():.2f} pips")
    trades_df.to_csv("/home/ubuntu/sena3fx/results/trades_v60_kelly.csv", index=False)
