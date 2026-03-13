"""
Union戦略 フォワードテスト監視スクリプト
==========================================
最新OHLCデータを読み込み、Union戦略のシグナルをログ出力する。
実際の取引は行わない — シグナル発生の監視のみ。

使用方法:
  # 1回実行 (最新データを確認してシグナルを出力)
  python monitors/forward_union.py

  # 無限ループで1時間ごとに監視 (cronなしで常駐)
  python monitors/forward_union.py --watch

  # cronジョブとして1時間ごとに実行
  # crontab -e に以下を追記:
  # 0 * * * * cd /path/to/sena3fx && python monitors/forward_union.py >> trade_logs/forward_union.log 2>&1

シグナルログ保存先: trade_logs/forward_union_signals.csv
"""
import os
import sys
import csv
import time
import argparse
import logging
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from lib.yagami import sig_maedai_yagami_union
from strategies.market_filters import make_usd_filtered_signal, SEASON_ALL

# ──────────────────────────────────────────────────────
#  設定
# ──────────────────────────────────────────────────────

# データファイル (最新を優先)
DATA_CANDIDATES = [
    os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv'),
    os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_4h.csv'),
]

# ログファイル
LOG_DIR = os.path.join(ROOT, 'trade_logs')
SIGNAL_LOG_CSV = os.path.join(LOG_DIR, 'forward_union_signals.csv')

# Union戦略パラメータ (v9 確定値)
UNION_PARAMS = dict(freq='4h', lookback_days=15, ema_days=200, confirm_bars=2, rsi_oversold=45)

# ウォッチモードのポーリング間隔 (秒)
WATCH_INTERVAL_SEC = 3600  # 1時間

# ──────────────────────────────────────────────────────
#  ロギング設定
# ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('forward_union')


# ──────────────────────────────────────────────────────
#  データ読み込み
# ──────────────────────────────────────────────────────

def load_latest_ohlc():
    """利用可能な最新OHLCファイルを読み込む"""
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            try:
                dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
            except Exception:
                dt = pd.to_datetime(df['datetime'])
                if dt.dt.tz is not None:
                    dt = dt.dt.tz_localize(None)
            df['datetime'] = dt
            df = df.set_index('datetime').sort_index()
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[cols].astype(float)
            logger.info(f'データ読み込み: {path} ({len(df)} 本, {df.index[-1].date()} まで)')
            return df
    raise FileNotFoundError(f'4Hデータが見つかりません: {DATA_CANDIDATES}')


# ──────────────────────────────────────────────────────
#  シグナルCSV管理
# ──────────────────────────────────────────────────────

CSV_HEADER = ['logged_at', 'bar_datetime', 'signal', 'close_price',
              'strategy', 'timeframe', 'note']


def _ensure_csv():
    """シグナルログCSVのヘッダーを初期化"""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(SIGNAL_LOG_CSV):
        with open(SIGNAL_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)
        logger.info(f'シグナルログCSV新規作成: {SIGNAL_LOG_CSV}')


def _append_signal(bar_dt, signal, close_price, note=''):
    """シグナルをCSVに追記"""
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with open(SIGNAL_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            now_str,
            str(bar_dt)[:19],
            signal,
            f'{close_price:.3f}',
            'Union_4H',
            '4h',
            note,
        ])


def _load_logged_bars():
    """既にログに記録されたバー日時のセット (重複シグナル防止)"""
    if not os.path.exists(SIGNAL_LOG_CSV):
        return set()
    df = pd.read_csv(SIGNAL_LOG_CSV)
    if df.empty:
        return set()
    return set(df['bar_datetime'].astype(str).tolist())


# ──────────────────────────────────────────────────────
#  シグナル検出
# ──────────────────────────────────────────────────────

def detect_signals(df, n_bars_check=3):
    """
    Union戦略のシグナルを検出し、新規シグナルのみ返す。

    Args:
        df: OHLCデータフレーム
        n_bars_check: 最新N本のバーを確認対象とする

    Returns:
        list of dict: 新規シグナル情報
    """
    sig_fn = sig_maedai_yagami_union(**UNION_PARAMS)
    sig_fn_usd = make_usd_filtered_signal(sig_maedai_yagami_union, 75)(**UNION_PARAMS)

    sigs_raw = sig_fn(df)
    sigs_usd = sig_fn_usd(df)

    logged_bars = _load_logged_bars()
    new_signals = []

    # 最新N本のみ確認
    recent_idx = df.index[-n_bars_check:]
    for bar_dt in recent_idx:
        sig_r = sigs_raw.get(bar_dt, 'flat')
        sig_u = sigs_usd.get(bar_dt, 'flat')
        close_p = df.loc[bar_dt, 'close']
        bar_str = str(bar_dt)[:19]

        if sig_r in ('long', 'short') and bar_str not in logged_bars:
            usd_note = 'USD-filtered_BLOCKED' if sig_u == 'flat' else 'USD-OK'
            new_signals.append({
                'bar_datetime': bar_dt,
                'signal': sig_r,
                'close_price': close_p,
                'usd_filter': usd_note,
            })

    return new_signals


# ──────────────────────────────────────────────────────
#  メイン処理
# ──────────────────────────────────────────────────────

def run_once():
    """1回のシグナルチェックを実行"""
    _ensure_csv()
    df = load_latest_ohlc()

    new_signals = detect_signals(df)

    if new_signals:
        logger.info(f'新規シグナル検出: {len(new_signals)} 件')
        for s in new_signals:
            direction = s['signal'].upper()
            note = s['usd_filter']
            logger.info(
                f'  [{direction}] {s["bar_datetime"]} | '
                f'close={s["close_price"]:.3f} | {note}'
            )
            _append_signal(s['bar_datetime'], s['signal'], s['close_price'], note)
        print(f'\n{"-" * 50}')
        print(f'Union戦略シグナル: {len(new_signals)} 件を記録')
        print(f'ログ保存先: {SIGNAL_LOG_CSV}')
        print(f'{"-" * 50}')
    else:
        logger.info('新規シグナルなし')
        print('Union戦略: シグナルなし (最新3本のバーを確認)')

    # 最新シグナルログを表示
    if os.path.exists(SIGNAL_LOG_CSV):
        df_log = pd.read_csv(SIGNAL_LOG_CSV)
        if not df_log.empty:
            print(f'\n--- 直近シグナル履歴 (最新5件) ---')
            print(df_log.tail(5).to_string(index=False))

    return new_signals


def run_watch():
    """無限ループで WATCH_INTERVAL_SEC ごとにシグナルチェック"""
    logger.info(f'ウォッチモード開始 (間隔: {WATCH_INTERVAL_SEC // 60} 分)')
    logger.info('停止するには Ctrl+C を押してください')
    try:
        while True:
            logger.info('--- シグナルチェック開始 ---')
            run_once()
            next_run = datetime.now().strftime('%H:%M:%S')
            logger.info(f'次回チェック: {WATCH_INTERVAL_SEC // 60} 分後 ({next_run} 基点)')
            time.sleep(WATCH_INTERVAL_SEC)
    except KeyboardInterrupt:
        logger.info('ウォッチモード終了')


# ──────────────────────────────────────────────────────
#  cronジョブ設定例
# ──────────────────────────────────────────────────────

CRON_EXAMPLE = """
cronジョブ設定例 (1時間ごとに実行):
-------------------------------------------
crontab -e に以下を追記:

# Union戦略 フォワードシグナル監視 (毎時0分)
0 * * * * cd {root} && python monitors/forward_union.py >> trade_logs/forward_union.log 2>&1

-------------------------------------------
""".format(root=ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Union戦略 フォワードテスト監視スクリプト'
    )
    parser.add_argument(
        '--watch', action='store_true',
        help=f'ウォッチモード: {WATCH_INTERVAL_SEC // 60}分ごとに監視を継続'
    )
    parser.add_argument(
        '--cron', action='store_true',
        help='cronジョブの設定例を表示'
    )
    args = parser.parse_args()

    if args.cron:
        print(CRON_EXAMPLE)
    elif args.watch:
        run_watch()
    else:
        run_once()
