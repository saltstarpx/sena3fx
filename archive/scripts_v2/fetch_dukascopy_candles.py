"""
Dukascopy 並列OHLCデータ取得スクリプト
===========================================
ティックデータ(bi5)を並列ダウンロードして1H/4H OHLCに変換・保存。

使用方法:
  python scripts/fetch_dukascopy_candles.py --days 365
  python scripts/fetch_dukascopy_candles.py --days 730 --workers 30
  python scripts/fetch_dukascopy_candles.py --year 2024

仕組み:
  - Dukascopy の bi5 ファイルは 1時間単位のティックデータ
  - 1ファイル = 1時間分のティック → そのまま 1H バーに集計
  - ThreadPoolExecutor で並列ダウンロード (デフォルト: 20スレッド)
  - 保存先:
      data/XAUUSD_1h_dukascopy.csv
      data/XAUUSD_4h_dukascopy.csv
"""
import os
import sys
import struct
import lzma
import time
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def _fetch_hour_ohlc(symbol: str, dt_hour: datetime):
    """
    Dukascopy bi5 から 1時間分ティックを取得し、OHLCを返す。
    Returns: dict(timestamp, open, high, low, close, tick_count) or None
    """
    sym = symbol.upper()
    y, m, d, h = dt_hour.year, dt_hour.month - 1, dt_hour.day, dt_hour.hour
    url = (f"https://datafeed.dukascopy.com/datafeed/{sym}"
           f"/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5")

    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) == 0:
            return None

        raw = lzma.decompress(data)
        n = len(raw) // 20
        if n == 0:
            return None

        bids = []
        for i in range(n):
            off = i * 20
            _, _, bid_i, _, _ = struct.unpack('>IIIff', raw[off:off + 20])
            if 'XAU' in sym or 'XAG' in sym:
                bids.append(bid_i / 1000.0)
            else:
                bids.append(bid_i / 100000.0)

        bids = np.array(bids)
        return {
            'timestamp': dt_hour,
            'open':  float(bids[0]),
            'high':  float(bids.max()),
            'low':   float(bids.min()),
            'close': float(bids[-1]),
            'tick_count': n,
        }
    except Exception:
        return None


def fetch_dukascopy_ohlc(symbol='XAUUSD', days=365, workers=20,
                          start_dt=None, end_dt=None):
    """
    指定期間のOHLCデータを並列取得。

    Args:
        symbol: 通貨ペア
        days: 取得日数 (start_dt 未指定時)
        workers: 並列スレッド数
        start_dt: 取得開始 (datetime, UTC)
        end_dt:   取得終了 (datetime, UTC)

    Returns:
        pd.DataFrame: 1H OHLC, DatetimeIndex
    """
    if end_dt is None:
        end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=days)

    # 時間リスト生成 (週末・夜間は Dukascopy が空ファイルを返す → スキップ)
    hours = []
    cur = start_dt
    while cur < end_dt:
        hours.append(cur)
        cur += timedelta(hours=1)

    total = len(hours)
    print(f"[Dukascopy] {symbol} {start_dt.date()} ~ {end_dt.date()}")
    print(f"  時間数: {total}h / スレッド数: {workers}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_hour_ohlc, symbol, h): h for h in hours}
        for fut in as_completed(futs):
            dt = futs[fut]
            row = fut.result()
            if row is not None:
                results[dt] = row
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t0
                speed = done / elapsed
                eta = (total - done) / speed
                print(f"  {done}/{total} ({done*100//total}%) "
                      f"| {len(results)} bars "
                      f"| {speed:.0f}req/s "
                      f"| ETA {eta:.0f}s",
                      flush=True)

    if not results:
        print("[Dukascopy] データ取得失敗 (0 bars)")
        return None

    bars = pd.DataFrame(list(results.values()))
    bars = bars.sort_values('timestamp').set_index('timestamp')
    bars.index = pd.to_datetime(bars.index)

    elapsed = time.time() - t0
    print(f"[Dukascopy] 完了: {len(bars)} bars in {elapsed:.0f}s")
    return bars


def save_ohlc(bars_1h, symbol='XAUUSD'):
    """1H→4H集計してCSV保存"""
    path_1h = os.path.join(DATA_DIR, f'{symbol}_1h_dukascopy.csv')
    bars_1h.to_csv(path_1h)
    print(f"  保存: {path_1h} ({len(bars_1h)} bars)")

    # 4H集計
    bars_4h = bars_1h.resample('4h').agg(
        open='first', high='max', low='min', close='last',
        tick_count='sum'
    ).dropna(subset=['open'])
    path_4h = os.path.join(DATA_DIR, f'{symbol}_4h_dukascopy.csv')
    bars_4h.to_csv(path_4h)
    print(f"  保存: {path_4h} ({len(bars_4h)} bars)")

    return bars_1h, bars_4h


def load_dukascopy_ohlc(symbol='XAUUSD'):
    """保存済みDukascopy OHLCを読み込み"""
    path_1h = os.path.join(DATA_DIR, f'{symbol}_1h_dukascopy.csv')
    path_4h = os.path.join(DATA_DIR, f'{symbol}_4h_dukascopy.csv')

    if not os.path.exists(path_1h):
        return None, None

    bars_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True)
    bars_1h.index = pd.to_datetime(bars_1h.index)

    if os.path.exists(path_4h):
        bars_4h = pd.read_csv(path_4h, index_col=0, parse_dates=True)
        bars_4h.index = pd.to_datetime(bars_4h.index)
    else:
        bars_4h = bars_1h.resample('4h').agg(
            open='first', high='max', low='min', close='last'
        ).dropna(subset=['open'])

    return bars_1h, bars_4h


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dukascopy OHLC取得')
    parser.add_argument('--symbol', default='XAUUSD')
    parser.add_argument('--days', type=int, default=365, help='取得日数')
    parser.add_argument('--workers', type=int, default=20, help='並列スレッド数')
    parser.add_argument('--year', type=int, default=None,
                        help='特定年のみ取得 (例: --year 2024)')
    args = parser.parse_args()

    if args.year:
        start = datetime(args.year, 1, 1)
        end   = datetime(args.year, 12, 31, 23)
    else:
        end   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)

    bars_1h = fetch_dukascopy_ohlc(
        symbol=args.symbol,
        start_dt=start,
        end_dt=end,
        workers=args.workers,
    )

    if bars_1h is not None:
        save_ohlc(bars_1h, symbol=args.symbol)
        print(f"\n取得完了:")
        print(f"  1H bars: {len(bars_1h)}")
        print(f"  期間: {bars_1h.index[0]} ~ {bars_1h.index[-1]}")
        print(bars_1h.tail(3).to_string())
    else:
        print("失敗")
        sys.exit(1)
