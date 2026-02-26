"""
データ取得スクリプト
======================================
1. OANDA v20 REST API (推奨 / 実データ)
2. Dukascopy BIバイナリフォーマット (ティックデータ)
3. サンプル生成 (フォールバック)

OANDA API 利用方法:
  環境変数を設定するか --oanda-key / --oanda-account で指定。
    OANDA_API_KEY   : API アクセストークン
    OANDA_ACCOUNT   : "practice" または "live" (デフォルト: practice)

  例:
    export OANDA_API_KEY="your-api-key-here"
    python scripts/fetch_data.py --oanda --days 90

  またはコマンドラインで直接指定:
    python scripts/fetch_data.py --oanda --oanda-key YOUR_KEY --oanda-account practice
"""
import os
import sys
import struct
import lzma
import io
import json
from datetime import datetime, timedelta, timezone
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


# ===== OANDA v20 REST API =====

OANDA_GRANULARITY_MAP = {
    '1m': 'M1',
    '5m': 'M5',
    '15m': 'M15',
    '30m': 'M30',
    '1h': 'H1',
    '4h': 'H4',
    '1d': 'D',
    'D': 'D',
    'H4': 'H4',
    'H1': 'H1',
}

OANDA_PRACTICE_HOST = 'api-fxpractice.oanda.com'
OANDA_LIVE_HOST = 'api-fxtrade.oanda.com'


def fetch_oanda_candles(instrument='XAU_USD',
                        granularity='H1',
                        count=500,
                        api_key=None,
                        account_type='practice',
                        start_dt=None,
                        end_dt=None):
    """
    OANDA v20 REST API からローソク足データを取得。

    Args:
        instrument: OANDA 通貨ペアコード (例: 'XAU_USD', 'EUR_USD')
        granularity: 時間軸 ('M1','M5','M15','M30','H1','H4','D')
        count: 取得本数 (最大5000)
        api_key: OANDA API アクセストークン (None の場合は環境変数 OANDA_API_KEY を使用)
        account_type: 'practice' または 'live'
        start_dt: 取得開始日時 (datetime, UTC)。指定時は count より優先
        end_dt:   取得終了日時 (datetime, UTC)

    Returns:
        pd.DataFrame: open/high/low/close/volume/spread 列, DatetimeIndex (UTC naive)
        None: 取得失敗時
    """
    if api_key is None:
        api_key = os.environ.get('OANDA_API_KEY')
    if not api_key:
        print("[OANDA] API キーが設定されていません。")
        print("  環境変数 OANDA_API_KEY を設定するか --oanda-key オプションを使用してください。")
        return None

    host = OANDA_PRACTICE_HOST if account_type == 'practice' else OANDA_LIVE_HOST

    # クエリパラメータ構築
    params = [f'granularity={granularity}', 'price=BA']  # BA = bid+ask midpoint

    if start_dt is not None:
        from_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        params.append(f'from={from_str}')
        if end_dt is not None:
            to_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            params.append(f'to={to_str}')
        else:
            params.append(f'count={min(count, 5000)}')
    else:
        params.append(f'count={min(count, 5000)}')

    url = f'https://{host}/v3/instruments/{instrument}/candles?{"&".join(params)}'

    try:
        req = Request(url, headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        })
        with urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        print(f"[OANDA] HTTP エラー {e.code}: {body}")
        return None
    except URLError as e:
        print(f"[OANDA] 接続エラー: {e.reason}")
        return None

    candles = raw.get('candles', [])
    if not candles:
        print("[OANDA] ローソク足データが空です。")
        return None

    rows = []
    for c in candles:
        if not c.get('complete', True):
            continue  # 未確定足は除外
        ts = datetime.strptime(c['time'][:19], '%Y-%m-%dT%H:%M:%S')

        # bid/ask の中間価格 (mid) または直接 mid を使用
        if 'mid' in c:
            o = float(c['mid']['o'])
            h = float(c['mid']['h'])
            l = float(c['mid']['l'])
            cl = float(c['mid']['c'])
            spread = 0.0
        elif 'bid' in c and 'ask' in c:
            bid_c = float(c['bid']['c'])
            ask_c = float(c['ask']['c'])
            o = (float(c['bid']['o']) + float(c['ask']['o'])) / 2
            h = (float(c['bid']['h']) + float(c['ask']['h'])) / 2
            l = (float(c['bid']['l']) + float(c['ask']['l'])) / 2
            cl = (bid_c + ask_c) / 2
            spread = ask_c - bid_c
        else:
            continue

        rows.append({
            'timestamp': ts,
            'open': o,
            'high': h,
            'low': l,
            'close': cl,
            'volume': int(c.get('volume', 0)),
            'spread': spread,
        })

    if not rows:
        print("[OANDA] 有効なローソク足が0本です。")
        return None

    df = pd.DataFrame(rows).set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    print(f"[OANDA] {instrument} {granularity}: {len(df)} bars 取得完了")
    return df


def fetch_oanda_long(instrument='XAU_USD',
                     granularity='H1',
                     days=180,
                     api_key=None,
                     account_type='practice'):
    """
    OANDA から長期データを 5000 本ずつ分割取得して結合。

    Returns:
        pd.DataFrame | None
    """
    if api_key is None:
        api_key = os.environ.get('OANDA_API_KEY')
    if not api_key:
        print("[OANDA] API キーが設定されていません。")
        return None

    # 時間軸ごとのバー数/日を算出
    bars_per_day = {'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48,
                    'H1': 24, 'H4': 6, 'D': 1}.get(granularity, 24)
    total_bars = days * bars_per_day
    chunk = 5000

    end_dt = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    all_dfs = []
    cursor = start_dt

    print(f"[OANDA] {instrument} {granularity} 取得開始: {start_dt} ~ {end_dt} ({total_bars} bars)")

    while cursor < end_dt:
        df = fetch_oanda_candles(
            instrument=instrument,
            granularity=granularity,
            count=chunk,
            api_key=api_key,
            account_type=account_type,
            start_dt=cursor,
            end_dt=end_dt,
        )
        if df is None or len(df) == 0:
            break
        all_dfs.append(df)
        cursor = df.index[-1] + timedelta(seconds=1)
        if len(df) < chunk:
            break

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    # CSV保存
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = f"{instrument.replace('_','')}_{granularity}_oanda.csv"
    fpath = os.path.join(DATA_DIR, fname)
    combined.to_csv(fpath)
    print(f"[OANDA] 保存: {fpath} ({len(combined)} bars)")
    return combined


def load_oanda_csv(instrument='XAUUSD', granularity='H1'):
    """保存済み OANDA CSV を読み込み"""
    fname = f"{instrument}_{granularity}_oanda.csv"
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


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
    parser = argparse.ArgumentParser(description='市場データ取得ツール')
    parser.add_argument('--symbol', default='XAUUSD', help='通貨ペア (例: XAUUSD)')
    parser.add_argument('--days', type=int, default=28, help='取得日数')
    parser.add_argument('--sample', action='store_true', help='テスト用サンプルデータを生成')

    # OANDA オプション
    oanda_grp = parser.add_argument_group('OANDA API')
    oanda_grp.add_argument('--oanda', action='store_true', help='OANDA API からデータ取得')
    oanda_grp.add_argument('--oanda-key', default=None,
                           help='OANDA API キー (未指定時は環境変数 OANDA_API_KEY を使用)')
    oanda_grp.add_argument('--oanda-account', default='practice',
                           choices=['practice', 'live'], help='アカウントタイプ')
    oanda_grp.add_argument('--oanda-tf', default='H1',
                           help='時間軸 (M1/M5/M15/M30/H1/H4/D, デフォルト: H1)')

    args = parser.parse_args()

    if args.oanda:
        # XAUUSD → XAU_USD に変換
        sym = args.symbol
        if '_' not in sym and len(sym) == 6:
            instrument = sym[:3] + '_' + sym[3:]
        else:
            instrument = sym

        gran = OANDA_GRANULARITY_MAP.get(args.oanda_tf, args.oanda_tf)
        key = args.oanda_key or os.environ.get('OANDA_API_KEY')

        bars = fetch_oanda_long(
            instrument=instrument,
            granularity=gran,
            days=args.days,
            api_key=key,
            account_type=args.oanda_account,
        )
        if bars is not None:
            print(f"\n取得完了: {len(bars)} bars")
            print(bars.tail(3).to_string())
        else:
            print("OANDA データ取得失敗。API キーを確認してください。")
            sys.exit(1)

    elif args.sample:
        for freq in ['1h', '4h']:
            bars = generate_sample_ohlc(500, freq)
            path = os.path.join(DATA_DIR, f'{args.symbol}_{freq}_sample.csv')
            bars.to_csv(path)
            print(f"サンプル保存: {path} ({len(bars)} bars)")
    else:
        fetch_ticks(args.symbol, days=args.days)
