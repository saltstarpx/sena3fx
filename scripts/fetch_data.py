"""
Dukascopyティックデータ取得スクリプト
======================================
直接Dukascopy BIバイナリフォーマットからデータを取得。
dukascopy_pythonライブラリが利用可能ならそちらを使用。
フォールバック: OHLCデータをYahoo Finance等から取得。
"""
import os
import sys
import struct
import lzma
import io
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TICK_DIR = os.path.join(DATA_DIR, 'tick')
os.makedirs(TICK_DIR, exist_ok=True)


def fetch_dukascopy_hour(symbol, dt_hour):
    """
    Dukascopyから1時間分のティックデータをバイナリで取得。
    URLフォーマット: datafeed/XAUUSD/YYYY/MM-1/DD/HHh_ticks.bi5
    """
    sym = symbol.upper()
    y = dt_hour.year
    m = dt_hour.month - 1  # Dukascopyは0-indexed month
    d = dt_hour.day
    h = dt_hour.hour

    url = f"https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"

    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) == 0:
            return None

        decompressed = lzma.decompress(data)
        n_ticks = len(decompressed) // 20

        ticks = []
        for i in range(n_ticks):
            offset = i * 20
            ms, ask_i, bid_i, ask_vol, bid_vol = struct.unpack(
                '>IIIff', decompressed[offset:offset+20])

            tick_time = dt_hour + timedelta(milliseconds=ms)
            # XAUUSDの場合: 価格は整数/100000 ではなく /1000
            if 'XAU' in sym or 'XAG' in sym:
                ask = ask_i / 1000.0
                bid = bid_i / 1000.0
            else:
                ask = ask_i / 100000.0
                bid = bid_i / 100000.0

            ticks.append({
                'timestamp': tick_time,
                'askPrice': ask,
                'bidPrice': bid,
                'askVolume': ask_vol,
                'bidVolume': bid_vol,
            })

        if ticks:
            return pd.DataFrame(ticks)
        return None

    except (URLError, HTTPError, lzma.LZMAError):
        return None
    except Exception:
        return None


def fetch_ticks(symbol='XAUUSD', start_date=None, end_date=None, days=28):
    """
    指定期間のティックデータを取得してCSV保存。
    """
    if end_date is None:
        end_date = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    print(f"ティックデータ取得: {symbol}")
    print(f"期間: {start_date} ~ {end_date}")

    current = start_date
    all_dfs = []
    total_ticks = 0
    hours_processed = 0
    hours_total = int((end_date - start_date).total_seconds() / 3600)

    while current < end_date:
        df = fetch_dukascopy_hour(symbol, current)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            total_ticks += len(df)

        hours_processed += 1
        if hours_processed % 24 == 0:
            print(f"  {hours_processed}/{hours_total}h | {total_ticks:,} ticks", flush=True)

        current += timedelta(hours=1)

    if not all_dfs:
        print("データ取得失敗")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

    # 週ごとにCSV保存
    combined['week'] = combined['timestamp'].dt.isocalendar().week
    combined['year'] = combined['timestamp'].dt.year

    saved_files = []
    for (year, week), group in combined.groupby(['year', 'week']):
        filename = f"{symbol}_tick_{year}_W{week:02d}.csv"
        filepath = os.path.join(TICK_DIR, filename)
        group.drop(columns=['week', 'year']).to_csv(filepath, index=False)
        saved_files.append(filepath)
        print(f"  保存: {filename} ({len(group):,} ticks)")

    print(f"合計: {total_ticks:,} ticks, {len(saved_files)} files")
    return combined


def load_ticks(symbol='XAUUSD'):
    """保存済みティックデータを読み込み"""
    csv_files = sorted([
        os.path.join(TICK_DIR, f) for f in os.listdir(TICK_DIR)
        if f.startswith(f'{symbol}_tick_') and f.endswith('.csv')
    ])
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], format='ISO8601', utc=True)
        df = df.set_index(ts_col)
        df.index = df.index.tz_convert(None)

        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'bid' in cl and 'price' in cl: col_map[c] = 'bidPrice'
            elif 'ask' in cl and 'price' in cl: col_map[c] = 'askPrice'
        if col_map:
            df = df.rename(columns=col_map)
        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def ticks_to_ohlc(ticks, freq='1h'):
    """ティック→OHLCバー変換"""
    bid = ticks['bidPrice']
    bars = bid.resample(freq).agg(open='first', high='max', low='min', close='last')
    if 'askPrice' in ticks.columns:
        bars['spread'] = (ticks['askPrice'] - ticks['bidPrice']).resample(freq).mean()
    bars['tick_count'] = bid.resample(freq).count()
    bars = bars.dropna(subset=['open'])
    return bars


def generate_sample_ohlc(n_bars=500, freq='1h', seed=42):
    """テスト用のサンプルOHLCデータ生成（XAUUSDライク）"""
    np.random.seed(seed)
    base_price = 2650.0
    returns = np.random.normal(0.0001, 0.003, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range('2025-12-01', periods=n_bars, freq=freq)
    volatility = np.random.uniform(0.001, 0.008, n_bars)

    bars = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 1, n_bars)) * volatility),
        'low': prices * (1 - np.abs(np.random.normal(0, 1, n_bars)) * volatility),
        'close': prices * (1 + np.random.normal(0, 1, n_bars) * volatility),
    }, index=dates)

    # high >= max(open, close), low <= min(open, close)
    bars['high'] = bars[['open', 'high', 'close']].max(axis=1)
    bars['low'] = bars[['open', 'low', 'close']].min(axis=1)

    return bars


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='XAUUSD')
    parser.add_argument('--days', type=int, default=28)
    parser.add_argument('--sample', action='store_true',
                       help='テスト用サンプルデータを生成')
    args = parser.parse_args()

    if args.sample:
        for freq in ['1h', '4h']:
            bars = generate_sample_ohlc(500, freq)
            path = os.path.join(DATA_DIR, f'{args.symbol}_{freq}_sample.csv')
            bars.to_csv(path)
            print(f"サンプル保存: {path} ({len(bars)} bars)")
    else:
        fetch_ticks(args.symbol, days=args.days)
