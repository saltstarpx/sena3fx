
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_data import fetch_ticks, ticks_to_ohlc

OHLC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ohlc')
os.makedirs(OHLC_DIR, exist_ok=True)

def fetch_and_save_usdjpy_1m(days=7):
    print(f"USD/JPY 1分足データをDukascopyから取得中 (過去 {days} 日分)... ")
    end_date = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    # ティックデータを取得
    ticks_df = fetch_ticks(symbol='USDJPY', start_date=start_date, end_date=end_date)

    if ticks_df is None or ticks_df.empty:
        print("ティックデータの取得に失敗しました。")
        return

    # timestampカラムをインデックスに設定し、DatetimeIndexであることを確認
    ticks_df["timestamp"] = pd.to_datetime(ticks_df["timestamp"])
    ticks_df = ticks_df.set_index("timestamp")

    # ティックデータを1分足OHLCに変換
    ohlc_df = ticks_to_ohlc(ticks_df, freq='1min')

    if ohlc_df is None or ohlc_df.empty:
        print("1分足OHLCデータの生成に失敗しました。")
        return

    # CSVとして保存
    output_path = os.path.join(OHLC_DIR, 'USDJPY_1m.csv')
    ohlc_df.to_csv(output_path)
    print(f"USD/JPY 1分足データを {output_path} に保存しました。バー数: {len(ohlc_df)}")

if __name__ == '__main__':
    fetch_and_save_usdjpy_1m(days=1) # まずは1日分取得
