import pandas as pd
import numpy as np
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')
from strategies.yagami_mtf_v60 import generate_signals

def run_backtest_internal(df, spread_pips=0.2, volatility_multiplier=0.5):
    # 15分足、1時間足、4時間足の作成
    data_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    data_4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

    # 戦略側の閾値を一時的に変更するために、generate_signalsをラップするか、直接引数で渡せるように改造されている必要があるが、
    # 現在のyagami_mtf_v60.pyはvolatility_threshold = spread_value * 0.5 とハードコードされている。
    # そのため、一時的にファイルを書き換えて実行する方式をとる。
    
    # シグナル生成
    signal_series, tp_series, sl_series, entry_time_series, time_decay_minutes_series, atr_at_entry_series = generate_signals(df, data_15m, data_1h, data_4h, spread_pips=spread_pips)
    active_signals = signal_series[signal_series != 0].index

    trades = []
    open_positions = {}
    trade_id_counter = 0

    for current_time, current_bar in df.iterrows():
        # オープンポジションの決済チェック
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
                if pos["direction"] == 1:
                    pnl = (exit_price - pos["entry_price"]) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["position_size"] - (spread_pips * 0.01) * pos["position_size"]
            
            # タイムディケイ
            if not exit_reason and current_time >= pos["time_decay_exit_time"]:
                if not pos["partial_closed"]:
                    current_pnl = (current_bar['close'] - pos['entry_price']) if pos['direction'] == 1 else (pos['entry_price'] - current_bar['close'])
                    if current_pnl > (spread_pips * 0.01):
                        partial_pnl = (current_pnl - (spread_pips * 0.01)) * 0.5
                        trades.append({"exit_time": current_time, "pnl": partial_pnl})
                        pos["position_size"] = 0.5
                        pos["partial_closed"] = True
                        pos["sl"] = pos["entry_price"]
                    else:
                        exit_price = current_bar['close']
                        exit_reason = "Full Close (Time Decay)"
                        pnl = (exit_price - pos["entry_price"]) - (spread_pips * 0.01) if pos["direction"] == 1 else (pos["entry_price"] - exit_price) - (spread_pips * 0.01)

            if exit_reason:
                trades.append({"exit_time": current_time, "pnl": pnl})
                positions_to_remove.append(pos_id)

        for pos_id in positions_to_remove:
            del open_positions[pos_id]

        # エントリー
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
                    open_positions[trade_id_counter] = {
                        "trade_id": trade_id_counter, "entry_time": current_time, "direction": direction,
                        "entry_price": entry_price, "tp": tp, "sl": sl, "initial_sl": sl,
                        "time_decay_exit_time": current_time + pd.Timedelta(minutes=time_decay_minutes),
                        "position_size": 1.0, "partial_closed": False, "atr_at_entry": atr_at_entry
                    }

    return pd.DataFrame(trades)

if __name__ == "__main__":
    df_q1 = pd.read_csv("/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv")
    df_q1["timestamp"] = pd.to_datetime(df_q1["timestamp"])
    df_feb = df_q1[(df_q1["timestamp"].dt.month == 2) & (df_q1["timestamp"].dt.year == 2026)].copy()
    df_feb.set_index('timestamp', inplace=True)
    df_feb.dropna(inplace=True)

    # 1. フィルター緩和 (0.5倍) - 現在の設定
    print("Running backtest for Volatility 0.5x...")
    trades_05 = run_backtest_internal(df_feb, volatility_multiplier=0.5)
    trades_05.to_csv("/home/ubuntu/sena3fx/results/trades_v60_05x.csv", index=False)

    # 2. フィルター無効 (0.0倍)
    # yagami_mtf_v60.py を書き換える
    strategy_path = "/home/ubuntu/sena3fx/strategies/yagami_mtf_v60.py"
    with open(strategy_path, "r") as f:
        original_code = f.read()
    
    modified_code = original_code.replace("volatility_threshold = spread_value * 0.5", "volatility_threshold = 0.0")
    with open(strategy_path, "w") as f:
        f.write(modified_code)
    
    print("Running backtest for Volatility 0.0x...")
    # モジュールをリロードする必要がある
    import importlib
    import strategies.yagami_mtf_v60
    importlib.reload(strategies.yagami_mtf_v60)
    from strategies.yagami_mtf_v60 import generate_signals as generate_signals_00
    
    # run_backtest_internal内で使用されるgenerate_signalsを差し替えるのは面倒なので、
    # 実行環境を分けて、最終的にCSVを統合する
    trades_00 = run_backtest_internal(df_feb, volatility_multiplier=0.0)
    trades_00.to_csv("/home/ubuntu/sena3fx/results/trades_v60_00x.csv", index=False)

    # 元に戻す
    with open(strategy_path, "w") as f:
        f.write(original_code)
    print("Done.")
