#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v51 - 高精度・動的決済・RR 5.0目安型

改善点：
1.  **精度の追求**: 15分足の「明確な壁（実体ベースの壁）」と「ミスプライス埋め」を重視。
2.  **時間軸の連続性**: 15分足の反転パターンが上位足（1H/4H）の「髭」や「推進波」を形成する初期段階であることを考慮。
3.  **利確の柔軟性**: RR 5.0は目安とし、15分足の形状が崩れない限りは握り続ける「動的ホールド」。
4.  **損切りの根拠**: 15分足のチャートパターンが崩れた場所（髭先や実体の壁の外側）に設定。
"""

import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    m15_atr = calculate_atr(data_15m)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    signals_list = []

    # 15分足の各バーをループ
    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        prev_m15_bar = data_15m.iloc[i-1] if i > 0 else None
        atr_val = m15_atr.iloc[i] if not pd.isna(m15_atr.iloc[i]) else 0.05 # ATRがNaNの場合はデフォルト値

        if prev_m15_bar is None or atr_val == 0:
            print(f"Skipping M15 bar {current_m15_bar.name}: prev_m15_bar is None or atr_val is 0")
            continue

        # 15分足の髭の定義と「明確な壁」の判断
        # 上髭の長さ = current_m15_bar['high'] - max(current_m15_bar['open'], current_m15_bar['close'])
        # 下髭の長さ = min(current_m15_bar['open'], current_m15_bar['close']) - current_m15_bar['low']
        
        # 髭が実体に対して一定の割合以上であること、かつATRに対して一定の長さがあることを条件とする
        # ここでは簡略化のため、ATRの0.5倍以上の髭を「意味のある髭」と仮定
        
        # ロングエントリーの条件
        # 下髭が長く、かつその後の1分足でミスプライス埋め（髭の方向への押し目）を確認
        # 陰線であること、下髭がATRの0.5倍以上、実体がATRの0.5倍未満
        long_condition_1 = (current_m15_bar["open"] > current_m15_bar["close"])
        long_condition_2 = ((current_m15_bar["open"] - current_m15_bar["low"]) > (atr_val * 0.2)) # 髭の条件を緩和 (0.5 -> 0.2)
        long_condition_3 = ((current_m15_bar["open"] - current_m15_bar["close"]) < (atr_val * 0.5))
        print(f'M15 Bar: {current_m15_bar.name}, Open: {current_m15_bar["open"]:.3f}, Close: {current_m15_bar["close"]:.3f}, Low: {current_m15_bar["low"]:.3f}, High: {current_m15_bar["high"]:.3f}, ATR: {atr_val:.3f}')
        print(f'Long Conditions: C1={long_condition_1}, C2={long_condition_2}, C3={long_condition_3}')

        if long_condition_1 and long_condition_2 and long_condition_3:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            
            # 1分足でミスプライス埋め（髭の方向への押し目）を確認
            # 15分足の髭の安値（low）に近づき、反発する1分足を探す
            entry_1m_bars_in_range = data_1m.loc[start_1m_time:end_1m_time]
            entry_1m_bar = None
            # 15分足の髭の範囲内で、かつトレンド方向への反発を確認
            for _, bar in entry_1m_bars_in_range.iterrows():
                if bar["low"] <= current_m15_bar["low"] and bar["close"] > bar["open"] and bar["close"] > current_m15_bar["low"]:
                    entry_1m_bar = bar
                    break
            
            if entry_1m_bar is not None:
                # 損切りは15分足の髭先からATRの0.2倍程度離す
                sl_price = current_m15_bar['low'] - (atr_val * 0.2)
                risk = entry_1m_bar['close'] - sl_price
                
                print(f'  Long Entry Candidate: Time={entry_1m_bar.name}, EntryPrice={entry_1m_bar["close"]:.3f}, SL={sl_price:.3f}, Risk={risk:.3f}')

                # 最小リスクのチェックを削除し、リスクをそのまま採用
                if risk > 0: # リスクが正であることを確認
                    signal_series.loc[entry_1m_bar.name] = 1
                    tp_series.loc[entry_1m_bar.name] = entry_1m_bar['close'] + risk * 5.0 # RR 5.0は目安
                    sl_series.loc[entry_1m_bar.name] = sl_price
                    signals_list.append({"time": entry_1m_bar.name, "direction": "LONG"})
                    print(f'  LONG Signal Generated: Time={entry_1m_bar.name}')
                else:
                    print(f'  Long Entry Rejected: Risk too small ({risk:.3f})')
        # ショートエントリーの条件
        # 上髭が長く、かつその後の1分足でミスプライス埋め（髭の方向への戻り）を確認
        # 陽線であること、上髭がATRの0.5倍以上、実体がATRの0.5倍未満
        short_condition_1 = (current_m15_bar["open"] < current_m15_bar["close"])
        short_condition_2 = ((current_m15_bar["high"] - current_m15_bar["open"]) > (atr_val * 0.2)) # 髭の条件を緩和 (0.5 -> 0.2)
        short_condition_3 = ((current_m15_bar["close"] - current_m15_bar["open"]) < (atr_val * 0.5))

        print(f'Short Conditions: C1={short_condition_1}, C2={short_condition_2}, C3={short_condition_3}')

        if short_condition_1 and short_condition_2 and short_condition_3:
            
            # 15分足の開始時刻から次の15分足の開始時刻までを1分足で探索
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            
            # 1分足でミスプライス埋め（髭の方向への戻り）を確認
            # 15分足の髭の高値（high）に近づき、反発する1分足を探す
            entry_1m_bars_in_range = data_1m.loc[start_1m_time:end_1m_time]
            entry_1m_bar = None
            # 15分足の髭の範囲内で、かつトレンド方向への反発を確認
            for _, bar in entry_1m_bars_in_range.iterrows():
                if bar["high"] >= current_m15_bar["high"] and bar["close"] < bar["open"] and bar["close"] < current_m15_bar["high"]:
                    entry_1m_bar = bar
                    break
            
            if entry_1m_bar is not None:
                # 損切りは15分足の髭先からATRの0.2倍程度離す
                sl_price = current_m15_bar['high'] + (atr_val * 0.2)
                risk = sl_price - entry_1m_bar['close']
                
                print(f'  Short Entry Candidate: Time={entry_1m_bar.name}, EntryPrice={entry_1m_bar["close"]:.3f}, SL={sl_price:.3f}, Risk={risk:.3f}')

                # 最小リスクのチェックを削除し、リスクをそのまま採用
                if risk > 0: # リスクが正であることを確認
                    signal_series.loc[entry_1m_bar.name] = -1
                    tp_series.loc[entry_1m_bar.name] = entry_1m_bar['close'] - risk * 5.0 # RR 5.0は目安
                    sl_series.loc[entry_1m_bar.name] = sl_price
                    signals_list.append({"time": entry_1m_bar.name, "direction": "SHORT"})
                    print(f'  SHORT Signal Generated: Time={entry_1m_bar.name}')
                else:
                    print(f'  Short Entry Rejected: Risk too small ({risk:.3f})')

    return signal_series, tp_series, sl_series, signals_list
