"""
XAGUSD (銀) ローソク足データ取得スクリプト
============================================
このスクリプトをローカル環境で実行してください。
Dukascopyからティックデータを取得→4H/1H OHLCバーに変換します。

使い方:
  python scripts/fetch_silver.py --days 400
  # => data/ohlc/XAGUSD_4h.csv, XAGUSD_1h.csv を生成

推奨期間:
  --days 400  (2025-01-01 から 2026-02-27 をカバー)
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
OHLC_DIR = os.path.join(DATA_DIR, 'ohlc')
os.makedirs(TICK_DIR, exist_ok=True)
os.makedirs(OHLC_DIR, exist_ok=True)

SYMBOL = 'XAGUSD'


def fetch_dukascopy_hour(symbol, dt_hour):
    """Dukascopy 1時間ティックデータ取得"""
    sym = symbol.upper()
    y = dt_hour.year
    m = dt_hour.month - 1  # 0-indexed
    d = dt_hour.day
    h = dt_hour.hour
    url = f"https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) == 0:
            return None
        dec = lzma.decompress(data)
        n_ticks = len(dec) // 20
        rows = []
        for i in range(n_ticks):
            ms, ask_i, bid_i, ask_vol, bid_vol = struct.unpack('>IIIff', dec[i*20:(i+1)*20])
            tick_time = dt_hour + timedelta(milliseconds=ms)
            # XAGUSDの価格スケール: /1000
            ask = ask_i / 1000.0
            bid = bid_i / 1000.0
            rows.append({'timestamp': tick_time, 'askPrice': ask, 'bidPrice': bid})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def ticks_to_ohlc(df_ticks, freq='1h'):
    """ティック→OHLCバー変換 (mid price使用)"""
    df_ticks = df_ticks.copy()
    df_ticks['mid'] = (df_ticks['askPrice'] + df_ticks['bidPrice']) / 2
    df_ticks = df_ticks.set_index('timestamp')
    df_ticks.index = pd.to_datetime(df_ticks.index)
    bars = df_ticks['mid'].resample(freq).agg(
        open='first', high='max', low='min', close='last'
    )
    bars['spread'] = (df_ticks['askPrice'] - df_ticks['bidPrice']).resample(freq).mean()
    bars['tick_count'] = df_ticks['mid'].resample(freq).count()
    bars = bars.dropna(subset=['open'])
    bars.index.name = 'datetime'
    return bars


def fetch_silver_ohlc(days=400, verbose=True):
    """
    指定日数分のXAGUSDデータをDukascopyから取得し、
    1H/4H OHLCバーCSVを保存する。

    Returns:
        dict: {'1h': df_1h, '4h': df_4h} or None
    """
    end_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    if verbose:
        print(f"XAGUSD ティックデータ取得開始")
        print(f"  期間: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
        print(f"  総時間数: {int((end_dt-start_dt).total_seconds()/3600)}h")
        print()

    all_dfs = []
    current = start_dt
    total_hours = int((end_dt - start_dt).total_seconds() / 3600)
    processed = 0
    total_ticks = 0

    while current < end_dt:
        # 週末(土曜全日・日曜全日)はスキップ
        if current.weekday() == 5:      # Saturday
            current += timedelta(hours=1)
            continue
        if current.weekday() == 6:      # Sunday
            current += timedelta(hours=1)
            continue

        df = fetch_dukascopy_hour(SYMBOL, current)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            total_ticks += len(df)

        processed += 1
        if verbose and processed % 120 == 0:
            pct = processed / total_hours * 100
            print(f"  進捗: {pct:.1f}% ({processed}/{total_hours}h) | {total_ticks:,} ticks", flush=True)

        current += timedelta(hours=1)

    if not all_dfs:
        print("[ERROR] データが取得できませんでした")
        print("  Dukascopyへのアクセスを確認してください")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

    if verbose:
        print(f"\n合計 {total_ticks:,} ticks 取得完了")
        print(f"  価格範囲: ${combined[['askPrice','bidPrice']].mean(axis=1).min():.3f} ~ ${combined[['askPrice','bidPrice']].mean(axis=1).max():.3f}")

    # OHLC変換
    ohlc_1h = ticks_to_ohlc(combined, '1h')
    ohlc_4h = ticks_to_ohlc(combined, '4h')

    # 保存
    path_1h = os.path.join(OHLC_DIR, 'XAGUSD_1h.csv')
    path_4h = os.path.join(OHLC_DIR, 'XAGUSD_4h.csv')
    ohlc_1h.to_csv(path_1h)
    ohlc_4h.to_csv(path_4h)

    if verbose:
        print(f"\n保存完了:")
        print(f"  1H: {path_1h} ({len(ohlc_1h)} bars)")
        print(f"  4H: {path_4h} ({len(ohlc_4h)} bars)")
        print()
        print("次のステップ:")
        print("  python scripts/main_loop.py  # XAGUSD対応後にバックテスト実行")

    return {'1h': ohlc_1h, '4h': ohlc_4h}


def check_silver_price_range(df_4h):
    """Silver価格帯の確認 (XAUUSD比較用)"""
    print("\n=== XAGUSD 価格分析 ===")
    monthly = df_4h['close'].resample('ME').agg(['first', 'last'])
    monthly['chg_pct'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
    for dt, row in monthly.iterrows():
        print(f"  {dt.strftime('%Y-%m')}: ${row['first']:.3f} -> ${row['last']:.3f} "
              f"({row['chg_pct']:+.1f}%)")

    # Gold/Silver Ratio (GSR) ヒント
    print()
    print("GSR (Gold/Silver Ratio) 参考:")
    print("  GSR < 70 = Silver bullish relative to Gold")
    print("  GSR > 90 = Silver undervalued vs Gold")
    print("  2025-04 Gold=$3176, if GSR=80 -> Silver=$39.7")
    print("  2026-02 Gold=$5200, if GSR=80 -> Silver=$65.0")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='XAGUSD データ取得')
    parser.add_argument('--days', type=int, default=400, help='取得日数 (デフォルト: 400日)')
    parser.add_argument('--check', action='store_true', help='既存CSVの確認のみ')
    args = parser.parse_args()

    if args.check:
        path = os.path.join(OHLC_DIR, 'XAGUSD_4h.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"XAGUSD_4h.csv: {len(df)} bars")
            print(f"  期間: {df.index[0]} ~ {df.index[-1]}")
            check_silver_price_range(df)
        else:
            print("XAGUSD_4h.csv が見つかりません。--days オプションで取得してください")
    else:
        result = fetch_silver_ohlc(days=args.days)
        if result:
            check_silver_price_range(result['4h'])
