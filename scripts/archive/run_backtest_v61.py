
print("--- SCRIPT START ---")
import pandas as pd
import numpy as np
import sys

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v61 import generate_signals

def run_backtest_v61(df, spread_pips=0.2):
    # 15分足、1時間足、4時間足の作成
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

    # シグナル生成 (v61ではtime_decay_minutes_seriesを返さない)
    signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series = generate_signals(df, data_15m, data_1h, data_4h, spread_pips=spread_pips)
    active_signals = signal_series[signal_series != 0].index

    trades = []
    open_positions = {}
    trade_id_counter = 0

    for current_time, current_bar in df.iterrows():
        positions_to_remove = []
        for pos_id, pos in list(open_positions.items()):
            exit_reason = None
            exit_price = 0

            # v61: 半利確ロジック (RR 1:1達成時)
            if not pos['partial_closed']:
                risk_amount = abs(pos['entry_price'] - pos['sl'])
                partial_tp_price = pos['entry_price'] + risk_amount if pos["direction"] == 1 else pos['entry_price'] - risk_amount

                if (pos["direction"] == 1 and current_bar["high"] >= partial_tp_price) or \
                   (pos["direction"] == -1 and current_bar["low"] <= partial_tp_price):
                    
                    # 半分を利確
                    pnl = (partial_tp_price - pos["entry_price"]) * 0.5 if pos["direction"] == 1 else (pos["entry_price"] - partial_tp_price) * 0.5
                    trades.append({
                        "trade_id": pos["trade_id"],
                        "exit_time": current_time, 
                        "pnl": pnl - (spread_pips * 0.01 * 0.5), # スプレッドコストを引く
                        "exit_reason": "Partial TP (RR 1:1)"
                    })
                    
                    # ポジション情報を更新
                    pos["position_size"] = 0.5
                    pos["partial_closed"] = True
                    pos["sl"] = pos["entry_price"] # SLを建値に移動
                    # 残りのポジションのTPを伸ばす（例: 2N計算値）
                    # 現在のTPはN計算値なので、さらにN計算値分伸ばす
                    if pos["direction"] == 1:
                        pos["tp"] = pos["entry_price"] + (pos["tp"] - pos["entry_price"]) * 2 # N計算値の2倍に設定
                    else:
                        pos["tp"] = pos["entry_price"] - (pos["entry_price"] - pos["tp"]) * 2 # N計算値の2倍に設定

            # SL/TPヒット
            if pos["direction"] == 1:
                if current_bar["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"
            else: # direction == -1
                if current_bar["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                    exit_reason = "SL"
                elif current_bar["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                    exit_reason = "TP"

            if exit_reason:
                pnl = (exit_price - pos["entry_price"]) * pos["position_size"] if pos["direction"] == 1 else (pos["entry_price"] - exit_price) * pos["position_size"]
                # 半利確済みでない場合のみスプレッドを引く
                if not pos['partial_closed']:
                    pnl -= (spread_pips * 0.01)

                trades.append({
                    "trade_id": pos['trade_id'],
                    "exit_time": current_time, 
                    "pnl": pnl, 
                    "exit_reason": exit_reason
                })
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
                        "position_size": 1.0, "partial_closed": False, "atr_at_entry": atr_at_entry_series.loc[current_time]
                    }

    return pd.DataFrame(trades)

if __name__ == "__main__":
    print("Running backtest v61...")
    print("Loading data...")
    df_q1 = pd.read_csv("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv")
    df_q1["timestamp"] = pd.to_datetime(df_q1["timestamp"])
    df_feb = df_q1[(df_q1["timestamp"].dt.month == 2) & (df_q1["timestamp"].dt.year == 2026)].copy()
    df_feb.set_index('timestamp', inplace=True)
    df_feb.dropna(inplace=True)

    print("Generating signals...")
    trades_df = run_backtest_v61(df_feb)
    print("Signals generated.")
    
    # pipsに変換
    trades_df['pnl_pips'] = trades_df['pnl'] / 0.01
    
    # ケリー計算用の集計
    final_trades = trades_df.groupby('trade_id')['pnl_pips'].sum()

    win_trades = final_trades[final_trades > 0]
    loss_trades = final_trades[final_trades <= 0]
    
    if not final_trades.empty:
        win_rate = len(win_trades) / len(final_trades)
        avg_win = win_trades.mean()
        avg_loss = abs(loss_trades.mean())
        odds = avg_win / avg_loss if avg_loss > 0 else float('inf')
        kelly_f = (odds * win_rate - (1 - win_rate)) / odds if odds > 0 else 0

        print(f"Total Trades: {len(final_trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: {avg_win:.4f} pips, Avg Loss: {avg_loss:.4f} pips")
        print(f"Odds (Risk/Reward Ratio): {odds:.4f}")
        print(f"Kelly Fraction: {kelly_f:.4f}")
        print(f"Total PnL: {final_trades.sum():.2f} pips")
    else:
        print("No trades were executed.")

    trades_df.to_csv("/home/ubuntu/sena3fx/results/trades_v61.csv", index=False)
    print("Trades saved to trades_v61.csv")
    print("--- SCRIPT END ---")

