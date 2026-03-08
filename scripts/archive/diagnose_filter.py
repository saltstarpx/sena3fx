"""
フィルター段階別診断スクリプト
各フィルターで何件のシグナルが除外されているかを計測し、
取引回数が少ない原因を特定する。
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/ubuntu/sena3fx/strategies')

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def load_data(symbol, timeframe, start_date, end_date):
    file_path = f"/home/ubuntu/sena3fx/data/{symbol}_{timeframe}.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df

def diagnose(symbol, start_date, end_date, spread_pips=0.2):
    spread = spread_pips * 0.01

    data_1m  = load_data(symbol.lower(), '1m',  start_date, end_date)
    data_15m = load_data(symbol.lower(), '15m', start_date, end_date)
    data_1h  = load_data(symbol.lower(), '1h',  start_date, end_date)
    data_4h  = load_data(symbol.lower(), '4h',  start_date, end_date)

    m15_atr = calculate_atr(data_15m)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)
    data_1h["ema20"] = data_1h["close"].ewm(span=20, adjust=False).mean()

    volatility_threshold = spread * 3.0

    # カウンター
    cnt_total_15m_bars   = 0
    cnt_atr_nan          = 0
    cnt_low_volatility   = 0   # ボラティリティフィルターで除外
    cnt_h4_not_found     = 0   # 4時間足データなし
    cnt_h1_not_found     = 0   # 1時間足データなし
    cnt_no_wick          = 0   # 髭が短すぎ
    cnt_h1_misalign      = 0   # 1時間足トレンドが逆向き
    cnt_wick_ok          = 0   # 髭条件クリア（エントリー候補）
    cnt_entry_triggered  = 0   # 実際に1分足でエントリー条件成立
    cnt_no_entry_in_bar  = 0   # 15分足内で1分足がエントリー条件を満たさなかった

    # 詳細ログ
    wick_details = []

    for i in range(len(data_15m)):
        cnt_total_15m_bars += 1
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]

        if pd.isna(atr_val):
            cnt_atr_nan += 1
            atr_val = 0.0005
            continue

        if atr_val < volatility_threshold:
            cnt_low_volatility += 1
            continue

        h4_time = current_m15_bar.name.floor("4h")
        if h4_time not in data_4h.index:
            cnt_h4_not_found += 1
            continue
        current_4h_trend = data_4h.loc[h4_time]["trend"]

        h1_time = current_m15_bar.name.floor("1h")
        if h1_time not in data_1h.index:
            cnt_h1_not_found += 1
            continue
        current_1h_close  = data_1h.loc[h1_time]["close"]
        current_1h_ema20  = data_1h.loc[h1_time]["ema20"]

        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low  = min(current_m15_bar["open"], current_m15_bar["close"])

        lower_wick = body_low - current_m15_bar["low"]
        upper_wick = current_m15_bar["high"] - body_high
        wick_threshold = atr_val * 0.5

        long_wick_ok  = (lower_wick > wick_threshold)
        short_wick_ok = (upper_wick > wick_threshold)

        h1_long_ok  = (current_1h_close > current_1h_ema20)
        h1_short_ok = (current_1h_close < current_1h_ema20)

        # 4時間足トレンド方向に髭があるか
        if current_4h_trend == 1 and not long_wick_ok:
            cnt_no_wick += 1
            continue
        if current_4h_trend == -1 and not short_wick_ok:
            cnt_no_wick += 1
            continue

        # 1時間足の方向確認
        if current_4h_trend == 1 and not h1_long_ok:
            cnt_h1_misalign += 1
            wick_details.append({
                "time": current_m15_bar.name,
                "direction": "LONG候補→1H逆向き除外",
                "4h_trend": current_4h_trend,
                "lower_wick_pips": round(lower_wick * 100, 2),
                "atr_pips": round(atr_val * 100, 2),
                "1h_close": current_1h_close,
                "1h_ema20": current_1h_ema20,
            })
            continue
        if current_4h_trend == -1 and not h1_short_ok:
            cnt_h1_misalign += 1
            wick_details.append({
                "time": current_m15_bar.name,
                "direction": "SHORT候補→1H逆向き除外",
                "4h_trend": current_4h_trend,
                "upper_wick_pips": round(upper_wick * 100, 2),
                "atr_pips": round(atr_val * 100, 2),
                "1h_close": current_1h_close,
                "1h_ema20": current_1h_ema20,
            })
            continue

        # ここまで来たら「エントリー候補」
        cnt_wick_ok += 1

        # 1分足でエントリー条件が成立するか確認
        start_1m_time = current_m15_bar.name
        end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
        entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]

        entry_found = False
        if current_4h_trend == 1:
            mid_point = current_m15_bar["low"] + lower_wick * 0.5
            for _, bar in entry_1m_bars.iterrows():
                if bar["low"] <= mid_point:
                    entry_found = True
                    break
        else:
            mid_point = current_m15_bar["high"] - upper_wick * 0.5
            for _, bar in entry_1m_bars.iterrows():
                if bar["high"] >= mid_point:
                    entry_found = True
                    break

        if entry_found:
            cnt_entry_triggered += 1
        else:
            cnt_no_entry_in_bar += 1

    print("\n" + "="*65)
    print("  フィルター段階別 診断レポート")
    print(f"  期間: {start_date} 〜 {end_date}  スプレッド: {spread_pips}pips")
    print("="*65)
    print(f"  15分足バー総数                    : {cnt_total_15m_bars:>6} 本")
    print(f"  ATR NaN（ウォームアップ）で除外    : {cnt_atr_nan:>6} 本")
    print(f"  ❶ ボラティリティフィルターで除外   : {cnt_low_volatility:>6} 本  ← spread×3 = {spread*3*100:.3f}pips")
    print(f"  ❷ 4時間足データなし               : {cnt_h4_not_found:>6} 本")
    print(f"  ❸ 1時間足データなし               : {cnt_h1_not_found:>6} 本")
    print(f"  ❹ 髭が短すぎ（ATR×0.5未満）      : {cnt_no_wick:>6} 本  ← 最大の除外要因?")
    print(f"  ❺ 1時間足トレンドが逆向き          : {cnt_h1_misalign:>6} 本  ← 2番目の除外要因?")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  エントリー候補（全フィルター通過）  : {cnt_wick_ok:>6} 本")
    print(f"    └ 1分足でエントリー条件成立       : {cnt_entry_triggered:>6} 本  ← 実際の取引数")
    print(f"    └ 1分足が50%戻しに届かず          : {cnt_no_entry_in_bar:>6} 本")
    print("="*65)

    total_filtered = cnt_low_volatility + cnt_h4_not_found + cnt_h1_not_found + cnt_no_wick + cnt_h1_misalign + cnt_atr_nan
    print(f"\n  【除外率の内訳】")
    print(f"  ボラティリティフィルター : {cnt_low_volatility/cnt_total_15m_bars*100:.1f}%")
    print(f"  髭が短すぎ              : {cnt_no_wick/cnt_total_15m_bars*100:.1f}%")
    print(f"  1時間足トレンド逆向き   : {cnt_h1_misalign/cnt_total_15m_bars*100:.1f}%")
    print(f"  50%戻しに届かず        : {cnt_no_entry_in_bar/(cnt_wick_ok) *100:.1f}% (候補比)")
    print()

    # 1時間足除外の詳細
    if wick_details:
        df_detail = pd.DataFrame(wick_details)
        df_detail.to_csv("/home/ubuntu/sena3fx/results/filter_diagnosis_h1_excluded.csv", index=False)
        print(f"  1時間足除外の詳細: /home/ubuntu/sena3fx/results/filter_diagnosis_h1_excluded.csv ({len(df_detail)}件)")

    return {
        "total_15m": cnt_total_15m_bars,
        "low_vol": cnt_low_volatility,
        "no_wick": cnt_no_wick,
        "h1_misalign": cnt_h1_misalign,
        "candidates": cnt_wick_ok,
        "entries": cnt_entry_triggered,
        "no_entry": cnt_no_entry_in_bar,
    }

if __name__ == "__main__":
    diagnose("USDJPY", "2024-07-01", "2024-08-06", spread_pips=0.2)
