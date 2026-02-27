"""
2025年〜現在の XAUUSD ティックデータ取得スクリプト
(改善指令 v2.0 Mission2)
===================================================
Dukascopy から 2025-01-01 〜 現在のティックデータを取得し、
バックテストで利用可能な形式で保存する。

大量データの分割取得と OHLC 変換に対応。

実行例:
  # 2025-01-01 〜 今日まで全部取得（デフォルト）
  python scripts/fetch_tick_data.py

  # 特定期間を指定
  python scripts/fetch_tick_data.py --from 2025-06-01 --to 2025-12-31

  # 直近 30 日だけ（軽量テスト）
  python scripts/fetch_tick_data.py --days 30

  # 取得後に 1H/4H/8H/12H OHLC CSV を生成
  python scripts/fetch_tick_data.py --convert-ohlc

  # 既存キャッシュから続けて取得（中断再開）
  python scripts/fetch_tick_data.py --resume

注意:
  2025 年全体 (8,760 時間) の取得は数時間かかります。
  中断した場合は --resume で続きから再開できます。
"""
import os
import sys
import argparse
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
TICK_DIR = os.path.join(DATA_DIR, 'tick')

# デフォルト取得開始日: 2025-01-01
DEFAULT_START = datetime(2025, 1, 1)


def main():
    parser = argparse.ArgumentParser(
        description='2025年〜現在の XAUUSD ティックデータ取得 (Dukascopy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--from', dest='start', default=None, metavar='YYYY-MM-DD',
        help='取得開始日 (デフォルト: 2025-01-01)',
    )
    parser.add_argument(
        '--to', dest='end', default=None, metavar='YYYY-MM-DD',
        help='取得終了日 (デフォルト: 今日)',
    )
    parser.add_argument(
        '--days', type=int, default=None,
        help='直近 N 日分だけ取得 (--from/--to より優先)',
    )
    parser.add_argument(
        '--symbol', default='XAUUSD',
        help='通貨ペア (デフォルト: XAUUSD)',
    )
    parser.add_argument(
        '--convert-ohlc', action='store_true',
        help='取得後に 1H/4H/8H/12H OHLC CSV を生成し、既存 dukascopy CSV にマージ',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='既存キャッシュの最終時刻以降から取得（中断再開）',
    )
    args = parser.parse_args()

    from scripts.fetch_data import fetch_ticks, load_ticks, ticks_to_ohlc
    import pandas as pd

    # ── 期間決定 ──
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    if args.days:
        start_dt = now - timedelta(days=args.days)
        end_dt   = now
    else:
        start_dt = (datetime.strptime(args.start, '%Y-%m-%d')
                    if args.start else DEFAULT_START)
        end_dt   = (datetime.strptime(args.end, '%Y-%m-%d')
                    if args.end   else now)

    # ── --resume: 既存データの最終時刻以降から再開 ──
    if args.resume:
        existing = load_ticks(args.symbol)
        if existing is not None and len(existing) > 0:
            last_ts = existing.index.max()
            resume_start = last_ts.to_pydatetime() - timedelta(hours=1)
            if resume_start > start_dt:
                start_dt = resume_start
                print(f"[Resume] 既存データ最終: {last_ts}  → {start_dt.strftime('%Y-%m-%d %H:%M')} から再開")

    total_hours = int((end_dt - start_dt).total_seconds() / 3600)

    print(f"{'='*65}")
    print(f"XAUUSD ティックデータ取得 (Dukascopy)")
    print(f"期間  : {start_dt.strftime('%Y-%m-%d')} 〜 {end_dt.strftime('%Y-%m-%d')}")
    print(f"取得量: 約 {total_hours:,} 時間分")
    print(f"保存先: {TICK_DIR}")
    print(f"{'='*65}\n")

    if total_hours > 5000:
        print(f"[注意] {total_hours:,} 時間分は大量です。")
        print(f"       2025 年全体 ≒ 8,760 時間 = 数時間〜半日かかる場合があります。")
        print(f"       中断した場合は --resume で続きから再開できます。\n")

    # ── ティックデータ取得 ──
    ticks = fetch_ticks(
        symbol     = args.symbol,
        start_date = start_dt,
        end_date   = end_dt,
    )

    if ticks is None or len(ticks) == 0:
        print("\n[失敗] ティックデータを取得できませんでした。")
        print("ネットワーク接続と Dukascopy の利用可否を確認してください。")
        sys.exit(1)

    print(f"\n[完了] {len(ticks):,} ticks 取得")
    print(f"  期間: {ticks.index.min()} 〜 {ticks.index.max()}")

    # ── OHLC 変換 ──
    if args.convert_ohlc:
        print("\n[OHLC 変換中...]")
        os.makedirs(DATA_DIR, exist_ok=True)

        freqs = [('1h', '1H'), ('4h', '4H'), ('8h', '8H'), ('12h', '12H')]
        for freq, label in freqs:
            ohlc = ticks_to_ohlc(ticks, freq)
            out_path = os.path.join(DATA_DIR, f'XAUUSD_{label}_2025plus.csv')
            ohlc.to_csv(out_path)
            print(f"  {label}: {len(ohlc):,} bars → {out_path}")

        # 全期間 1H OHLC の更新（既存 dukascopy CSV とマージ）
        full_path_1h = os.path.join(DATA_DIR, 'XAUUSD_1h_dukascopy.csv')
        full_path_4h = os.path.join(DATA_DIR, 'XAUUSD_4h_dukascopy.csv')

        for tf_freq, tf_label, full_path in [
            ('1h', '1H', full_path_1h),
            ('4h', '4H', full_path_4h),
        ]:
            new_bars = ticks_to_ohlc(ticks, tf_freq)
            if os.path.exists(full_path):
                try:
                    existing_bars = pd.read_csv(full_path, index_col=0, parse_dates=True)
                    combined = pd.concat([existing_bars, new_bars])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    combined.to_csv(full_path)
                    print(f"\n[更新] {full_path}")
                    print(f"  既存: {len(existing_bars):,} bars → 統合後: {len(combined):,} bars")
                    print(f"  範囲: {combined.index.min().date()} 〜 {combined.index.max().date()}")
                except Exception as e:
                    print(f"  [警告] {tf_label} マージ失敗: {e}")
                    print(f"  新規保存: {full_path}")
                    new_bars.to_csv(full_path)
            else:
                new_bars.to_csv(full_path)
                print(f"\n[新規] {full_path} ({len(new_bars):,} bars)")

    print(f"\n{'='*65}")
    print(f"次のステップ:")
    print(f"  バックテスト実行:")
    print(f"    python scripts/backtest_maedai.py --dukascopy")
    print(f"    python scripts/backtest_portfolio.py")
    if not args.convert_ohlc:
        print(f"\n  OHLC 変換（バックテストに必要）:")
        print(f"    python scripts/fetch_tick_data.py --convert-ohlc --days 30")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
