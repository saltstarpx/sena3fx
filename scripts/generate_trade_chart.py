
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import os

def plot_trade_example(data_path, trade_details, output_dir, filename):
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    entry_time = pd.to_datetime(trade_details['entry_time'], utc=True)
    exit_time = pd.to_datetime(trade_details['exit_time'], utc=True)

    # エントリー・決済時刻の前後数時間分のデータを抽出
    start_time = entry_time - pd.Timedelta(hours=24)
    end_time = exit_time + pd.Timedelta(hours=24)
    plot_df = df.loc[start_time:end_time].copy()

    # エントリー・決済価格をプロットするためにSeriesを作成
    add_plots = []

    # エントリーポイント
    entry_price = trade_details['entry']
    entry_series = pd.Series(np.nan, index=plot_df.index)
    entry_series.loc[plot_df.index[np.argmin(np.abs(plot_df.index - entry_time))]] = entry_price
    add_plots.append(mpf.make_addplot(entry_series, type='scatter', marker='^', markersize=200, color='green', panel=0))

    # 決済ポイント
    exit_price = trade_details['exit']
    exit_series = pd.Series(np.nan, index=plot_df.index)
    exit_series.loc[plot_df.index[np.argmin(np.abs(plot_df.index - exit_time))]] = exit_price
    add_plots.append(mpf.make_addplot(exit_series, type='scatter', marker='v', markersize=200, color='red', panel=0))

    # タイトル
    title = f"{filename.replace('_', ' ').title()} ({trade_details['dir'].upper()})\nEntry: {entry_time} @ {entry_price:.3f}, Exit: {exit_time} @ {exit_price:.3f}"

    # プロット
    fig, axes = mpf.plot(plot_df, type='candle', style='yahoo', title=title, volume=True, returnfig=True, figsize=(15, 8), addplot=add_plots)
    
    # チャートを保存
    output_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

if __name__ == '__main__':
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'usdjpy_1h.csv')
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'trade_examples')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 成功したショートトレードの例
    short_trade = {
        'dir': 'short',
        'entry': 161.453,
        'exit': 161.034,
        'entry_time': '2024-07-03 20:00:00+00:00',
        'exit_time': '2024-07-04 11:00:00+00:00'
    }
    plot_trade_example(DATA_PATH, short_trade, OUTPUT_DIR, 'yagami_short_trade_example')

    # 成功したロングトレードの例
    long_trade = {
        'dir': 'long',
        'entry': 142.36,
        'exit': 143.722,
        'entry_time': '2024-08-05 10:00:00+00:00',
        'exit_time': '2024-08-05 14:00:00+00:00'
    }
    plot_trade_example(DATA_PATH, long_trade, OUTPUT_DIR, 'yagami_long_trade_example')

    print(f"トレード例チャートを {OUTPUT_DIR} に保存しました。")
