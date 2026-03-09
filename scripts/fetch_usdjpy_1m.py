"""
fetch_usdjpy_1m.py
==================
USDJPY 1分足データを OANDA API から取得して data/ohlc/USDJPY_1m.csv に保存する。

IS + OOS 全期間: 2024/7/1 〜 現在 (約20ヶ月)

使用方法:
    export OANDA_API_KEY="your-key"
    python scripts/fetch_usdjpy_1m.py

    # ライブアカウントの場合
    python scripts/fetch_usdjpy_1m.py --account live

    # 期間を指定
    python scripts/fetch_usdjpy_1m.py --start 2024-07-01 --end 2026-03-09

注意:
    OANDA M1 の最大取得本数: 5000本/リクエスト = 約3.5日分
    約20ヶ月 = 約600,000本 → 120回以上のリクエストが必要
    取得に数分かかる場合があります。
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd

# プロジェクトルートをパスに追加
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.fetch_data import fetch_oanda_candles

OUTDIR   = os.path.join(BASE_DIR, 'data', 'ohlc')
OUTFILE  = os.path.join(OUTDIR, 'USDJPY_1m.csv')
CHUNK    = 5000      # OANDA API 最大取得本数
INTERVAL = 0.3       # リクエスト間隔（秒）


def fetch_usdjpy_1m(start_dt: datetime, end_dt: datetime,
                    api_key: str, account_type: str = 'practice') -> pd.DataFrame | None:
    """
    USDJPY M1 を期間分割して取得し、結合して返す。
    """
    os.makedirs(OUTDIR, exist_ok=True)

    all_dfs  = []
    cursor   = start_dt
    total    = 0
    req_num  = 0

    print(f"[fetch_usdjpy_1m] 取得開始: {start_dt.date()} ~ {end_dt.date()}")
    print(f"[fetch_usdjpy_1m] 推定リクエスト数: {int((end_dt - start_dt).days * 1440 / CHUNK) + 1} 回")

    while cursor < end_dt:
        df = fetch_oanda_candles(
            instrument='USD_JPY',
            granularity='M1',
            count=CHUNK,
            api_key=api_key,
            account_type=account_type,
            start_dt=cursor,
            end_dt=end_dt,
        )
        req_num += 1

        if df is None or len(df) == 0:
            print(f"  [req#{req_num}] データなし。終了。")
            break

        all_dfs.append(df)
        total += len(df)
        cursor = df.index[-1] + timedelta(seconds=1)

        if req_num % 20 == 0:
            print(f"  [req#{req_num}] {cursor.date()} | 累計 {total:,} bars", flush=True)

        # 最後のチャンクならループ終了
        if len(df) < CHUNK:
            break

        time.sleep(INTERVAL)

    if not all_dfs:
        print("[fetch_usdjpy_1m] 取得失敗")
        return None

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    # 既存ファイルがあれば結合（差分取得対応）
    if os.path.exists(OUTFILE):
        existing = pd.read_csv(OUTFILE, index_col=0, parse_dates=True)
        existing.index = pd.to_datetime(existing.index, utc=False)
        combined = pd.concat([existing, combined])
        combined = combined[~combined.index.duplicated(keep='first')].sort_index()
        print(f"[fetch_usdjpy_1m] 既存データと結合: {len(existing):,} + {total:,} → {len(combined):,} bars")

    combined.to_csv(OUTFILE)
    print(f"[fetch_usdjpy_1m] 保存完了: {OUTFILE}")
    print(f"  期間: {combined.index[0]} ~ {combined.index[-1]}")
    print(f"  総行数: {len(combined):,} bars")

    return combined


def main():
    parser = argparse.ArgumentParser(description='USDJPY 1m データ取得 (OANDA API)')
    parser.add_argument('--start',   default='2024-07-01',
                        help='取得開始日 (YYYY-MM-DD, デフォルト: 2024-07-01)')
    parser.add_argument('--end',     default=None,
                        help='取得終了日 (YYYY-MM-DD, デフォルト: 本日)')
    parser.add_argument('--account', default='practice',
                        choices=['practice', 'live'],
                        help='OANDA アカウントタイプ (デフォルト: practice)')
    parser.add_argument('--key',     default=None,
                        help='OANDA API キー (未指定時は環境変数 OANDA_API_KEY)')
    args = parser.parse_args()

    api_key = args.key or os.environ.get('OANDA_API_KEY')
    if not api_key:
        print('[ERROR] OANDA_API_KEY が設定されていません。')
        print('  export OANDA_API_KEY="your-api-key"')
        sys.exit(1)

    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt   = (datetime.strptime(args.end, '%Y-%m-%d')
                if args.end else datetime.utcnow().replace(second=0, microsecond=0))

    fetch_usdjpy_1m(start_dt, end_dt, api_key, args.account)


if __name__ == '__main__':
    main()
