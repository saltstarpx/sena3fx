import pandas as pd
import numpy as np
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v60 import generate_signals

def run_backtest(data_source, spread_pips=0.2):
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        df.set_index('timestamp', inplace=True)
    else:
        print(f"Loading data: {data_source}")
        df = pd.read_csv(data_source)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

    spread = spread_pips * 0.01 # 1pips = 0.01円
    print("Generating signals...")
    signal_series, tp_series, sl_series, entry_time_series, time_decay_minutes_series, atr_at_entry_series = generate_signals(df, data_15m, data_1h, data_4h, spread_pips=spread_pips)
    active_signals = signal_series[signal_series != 0].index
    print(f"Signals generated: {len(active_signals)}")

    trades = []
    open_positions = {}
    trade_id_counter = 0

    debug_log_file = open("/home/ubuntu/sena3fx/results/run018_v60_feb_debug.log", "w")

    for current_time, current_bar in df.iterrows():
        debug_log_file.write(f"\n--- Processing Bar: {current_time} --- Bar: O:{current_bar.open}, H:{current_bar.high}, L:{current_bar.low}, C:{current_bar.close}\n")

        # --- オープンポジションの決済チェック ---
        positions_to_remove = []
        for pos_id, pos in list(open_positions.items()):
            pnl = 0
            exit_reason = None
            exit_price = 0

            # --- トレイリングストップの更新ロジック ---
            if pos['partial_closed']: # 半決済後にトレイリング開始
                if pos["direction"] == 1: # LONG
                    new_sl = current_bar['high'] - pos['atr_at_entry'] * 2.0
                    if new_sl > pos['sl']:
                        debug_log_file.write(f"  [Trailing SL Update] ID {pos_id}: SL moved from {pos['sl']:.5f} to {new_sl:.5f}\n")
                        pos['sl'] = new_sl
                else: # SHORT
                    new_sl = current_bar['low'] + pos['atr_at_entry'] * 2.0
                    if new_sl < pos['sl']:
                        debug_log_file.write(f"  [Trailing SL Update] ID {pos_id}: SL moved from {pos['sl']:.5f} to {new_sl:.5f}\n")
                        pos['sl'] = new_sl

            # --- SL/TPヒットチェック ---
            if pos["direction"] == 1: # LONG
                if current_bar["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"
            else: # SHORT
                if current_bar["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"

            if exit_reason:
                if pos["direction"] == 1:
                    pnl = (exit_price - pos["entry_price"]) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"]
                debug_log_file.write(f"  [CLOSING] ID {pos_id} ({pos['direction']}) by {exit_reason} at {exit_price:.5f}. PnL: {pnl:.5f}\n")

            # --- タイムディケイ・チェック ---
            if not exit_reason and current_time >= pos["time_decay_exit_time"]:
                if not pos["partial_closed"]:
                    current_pnl = (current_bar['close'] - pos['entry_price']) if pos['direction'] == 1 else (pos['entry_price'] - current_bar['close'])
                    if current_pnl > (spread_pips * 0.01): # 利益が出ている場合 -> 半決済
                        partial_pnl = (current_pnl - (spread_pips * 0.01)) * 0.5
                        trades.append({
                            "trade_id": pos["trade_id"], "entry_time": pos["entry_time"], "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                            "entry_price": pos["entry_price"], "tp": pos["tp"], "sl": pos["initial_sl"], "exit_time": current_time,
                            "exit_price": current_bar['close'], "pnl": partial_pnl, "exit_reason": "Partial Close (Time Decay)"
                        })
                        debug_log_file.write(f"  [PARTIAL CLOSE] ID {pos_id} at {current_bar['close']:.5f}. PnL: {partial_pnl:.5f}\n")
                        pos["position_size"] = 0.5
                        pos["partial_closed"] = True
                        pos["sl"] = pos["entry_price"] # SLを建値に移動
                        debug_log_file.write(f"  [SL to Breakeven] ID {pos_id}: SL moved to {pos['sl']:.5f}\n")
                    else: # 利益が出ていない場合 -> 全決済
                        exit_price = current_bar['close']
                        exit_reason = "Full Close (Time Decay)"
                        if pos["direction"] == 1:
                            pnl = (exit_price - pos["entry_price"]) - (spread_pips * 0.01)
                        else:
                            pnl = (pos["entry_price"] - exit_price) - (spread_pips * 0.01)
                        debug_log_file.write(f"  [CLOSING] ID {pos_id} by {exit_reason} at {exit_price:.5f}. PnL: {pnl:.5f}\n")
                # else: 半決済後はタイムディケイでは決済しない（TP/SL or トレイリングSLに任せる）

            if exit_reason:
                trades.append({
                    "trade_id": pos["trade_id"], "entry_time": pos["entry_time"], "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                    "entry_price": pos["entry_price"], "tp": pos["tp"], "sl": pos["initial_sl"], "exit_time": current_time,
                    "exit_price": exit_price, "pnl": pnl, "exit_reason": exit_reason
                })
                positions_to_remove.append(pos_id)

        for pos_id in positions_to_remove:
            del open_positions[pos_id]

        # --- 新規シグナルによるエントリー ---
        if current_time in active_signals:
            direction = signal_series.loc[current_time]
            if direction != 0:
                trade_id_counter += 1
                entry_price = current_bar['close']
                tp = tp_series.loc[current_time]
                sl = sl_series.loc[current_time]
                time_decay_minutes = time_decay_minutes_series.loc[current_time]
                atr_at_entry = atr_at_entry_series.loc[current_time]

                if not any(pd.isna([tp, sl, time_decay_minutes, atr_at_entry])):
                    new_pos = {
                        "trade_id": trade_id_counter, "entry_time": current_time, "direction": direction,
                        "entry_price": entry_price, "tp": tp, "sl": sl, "initial_sl": sl,
                        "time_decay_exit_time": current_time + pd.Timedelta(minutes=time_decay_minutes),
                        "position_size": 1.0, "partial_closed": False, "atr_at_entry": atr_at_entry
                    }
                    open_positions[trade_id_counter] = new_pos
                    debug_log_file.write(f"[NEW ENTRY] ID {trade_id_counter} ({direction}) at {entry_price:.5f}. TP:{tp:.5f}, SL:{sl:.5f}, Decay:{time_decay_minutes}min\n")

    debug_log_file.close()

    # --- バックテスト結果の集計 ---
    if not trades:
        print("No trades were executed.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['pnl_pips'] = trades_df['pnl'] / 0.01

    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['pnl'] > 0]
    loss_trades = trades_df[trades_df['pnl'] <= 0]

    total_profit = win_trades['pnl'].sum()
    total_loss = abs(loss_trades['pnl'].sum())
    
    pf = total_profit / total_loss if total_loss > 0 else float('inf')
    win_rate = (len(win_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl_pips = trades_df['pnl_pips'].sum()

    print("="*80)
    print(f"【RUN-018_v60】結果")
    print(f"期間: {df.index.min()} to {df.index.max()}")
    print(f"スプレッド: {spread_pips} pips")
    print("-" * 30)
    print(f"プロフィットファクター (PF): {pf:.4f}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"総取引数: {total_trades}")
    print(f"最終損益: {total_pnl_pips:.2f} pips")
    print("="*80)

    os.makedirs("/home/ubuntu/sena3fx/results", exist_ok=True)
    trades_df.to_csv(f"/home/ubuntu/sena3fx/results/run018_v60_feb_trades.csv", index=False)

if __name__ == "__main__":
    print("\n--- Testing February 2026 ---")
    df_q1 = pd.read_csv("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv")
    df_q1["timestamp"] = pd.to_datetime(df_q1["timestamp"])
    df_feb = df_q1[(df_q1["timestamp"].dt.month == 2) & (df_q1["timestamp"].dt.year == 2026)].copy()
    if df_feb.isnull().values.any():
        print("Warning: df_feb contains NaN values before backtest.")
        df_feb.dropna(inplace=True)
        print("NaN values dropped from df_feb.")
    
    run_backtest(df_feb, spread_pips=0.2)

