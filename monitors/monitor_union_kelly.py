"""
Union + Kelly シグナルモニター — OANDA v20 API 連携
=====================================================
Union_4H 戦略のリアルタイムシグナルを監視し、ログ出力する。
実際の注文発注は行わない（監視専用）。

実行方法:
  python monitors/monitor_union_kelly.py

環境変数 (config.yaml または .env で設定):
  OANDA_API_TOKEN : OANDA v20 APIトークン
  OANDA_ACCOUNT_ID: OANDAアカウントID
  OANDA_ENV       : 'practice' (デフォルト) または 'live'

出力例:
  [Signal] 2025-10-15 09:00 JST | XAUUSD | LONG | Price: 2650.40
  [Sizing] Kelly Multiplier: 2.4x | Final Risk: 11.0% of Equity
  [NoTrade] 2025-10-15 13:00 JST | XAUUSD | FLAT (no signal)
"""

import os
import sys
import time
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────
#  設定
# ──────────────────────────────────────────────────────

# OANDA接続設定 (環境変数 > config.yaml > デフォルト)
OANDA_ENV       = os.environ.get('OANDA_ENV', 'practice')
OANDA_API_TOKEN = os.environ.get('OANDA_API_TOKEN', '')
OANDA_ACCOUNT_ID= os.environ.get('OANDA_ACCOUNT_ID', '')
INSTRUMENT      = 'XAU_USD'   # OANDA形式 (スラッシュなし)
GRANULARITY     = 'H4'        # 4時間足
CANDLE_COUNT    = 300         # 取得バー数 (SMA200 + バッファ)

# Kelly乗数設定 (v13実績値)
KELLY_MULTIPLIER  = 2.4       # v12実績
BASE_RISK_PCT     = 0.05      # 5%
FINAL_RISK_PCT    = BASE_RISK_PCT * KELLY_MULTIPLIER  # 12%

# ポーリング間隔 (秒)
POLL_INTERVAL_SEC = 60 * 30   # 30分ごとにチェック (4H足なので余裕あり)

# ログ設定
LOG_DIR  = os.path.join(ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'union_kelly_monitor.log')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger('union_kelly_monitor')


# ──────────────────────────────────────────────────────
#  Union シグナル (バックテスト実装と同一ロジック)
# ──────────────────────────────────────────────────────

from lib.yagami import sig_maedai_yagami_union

_SIGNAL_FUNC = sig_maedai_yagami_union(
    freq='4h',
    lookback_days=15,
    ema_days=200,
    confirm_bars=2,
    rsi_oversold=45,
)


# ──────────────────────────────────────────────────────
#  OANDA データ取得
# ──────────────────────────────────────────────────────

def _oanda_base_url() -> str:
    if OANDA_ENV == 'live':
        return 'https://api-fxtrade.oanda.com'
    return 'https://api-fxpractice.oanda.com'


def fetch_ohlc_from_oanda(count: int = CANDLE_COUNT) -> pd.DataFrame | None:
    """
    OANDA v20 REST API から XAU_USD 4時間足 OHLC を取得する。

    Returns:
        pd.DataFrame or None (取得失敗時)
    """
    if not OANDA_API_TOKEN:
        logger.warning('OANDA_API_TOKEN が未設定。ローカルCSVにフォールバック。')
        return _load_local_ohlc()

    try:
        import urllib.request, json as _json
        url = (f'{_oanda_base_url()}/v3/instruments/{INSTRUMENT}/candles'
               f'?count={count}&granularity={GRANULARITY}&price=M')
        req = urllib.request.Request(url, headers={
            'Authorization': f'Bearer {OANDA_API_TOKEN}',
            'Content-Type': 'application/json',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())

        candles = data.get('candles', [])
        rows = []
        for c in candles:
            if not c.get('complete', True):
                continue
            mid = c.get('mid', {})
            rows.append({
                'datetime': pd.Timestamp(c['time']).tz_localize(None),
                'open':  float(mid.get('o', 0)),
                'high':  float(mid.get('h', 0)),
                'low':   float(mid.get('l', 0)),
                'close': float(mid.get('c', 0)),
                'volume': int(c.get('volume', 0)),
            })

        if not rows:
            logger.warning('OANDAから空のデータ。')
            return None

        df = pd.DataFrame(rows).set_index('datetime').sort_index()
        logger.debug(f'OANDA OHLC取得: {len(df)}バー ({df.index[0]} ~ {df.index[-1]})')
        return df

    except Exception as e:
        logger.error(f'OANDAデータ取得エラー: {e}')
        return _load_local_ohlc()


def _load_local_ohlc() -> pd.DataFrame | None:
    """ローカルCSVからOHLCを読み込む (フォールバック)。"""
    path = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    if not os.path.exists(path):
        logger.error(f'ローカルCSVが見つかりません: {path}')
        return None
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df.set_index('datetime').sort_index()[cols].astype(float)


# ──────────────────────────────────────────────────────
#  シグナル評価 & ログ出力
# ──────────────────────────────────────────────────────

def _jst(ts) -> str:
    """UTCタイムスタンプを JST 文字列に変換。"""
    try:
        t = pd.Timestamp(ts)
        jst = t + pd.Timedelta(hours=9)
        return jst.strftime('%Y-%m-%d %H:%M JST')
    except Exception:
        return str(ts)


def evaluate_signal(df: pd.DataFrame) -> dict:
    """
    最新バーのシグナルを評価する。

    Returns:
        dict: {signal, price, bar_time, equity_pct}
    """
    signals = _SIGNAL_FUNC(df)
    latest_bar = signals.index[-1]
    signal_val = signals.iloc[-1]

    latest_price = float(df['close'].iloc[-1])

    if signal_val is None or signal_val == 'flat' or pd.isna(signal_val):
        sig = 'flat'
    else:
        sig = str(signal_val)

    return {
        'signal':     sig,
        'price':      latest_price,
        'bar_time':   latest_bar,
        'equity_pct': FINAL_RISK_PCT if sig == 'long' else 0.0,
    }


def log_signal(result: dict) -> None:
    """シグナル結果をフォーマットしてログ出力。"""
    bar_jst = _jst(result['bar_time'])
    sig = result['signal'].upper()
    price = result['price']

    if result['signal'] == 'long':
        logger.info(
            f'[Signal] {bar_jst} | XAUUSD | {sig} | Price: {price:.2f}'
        )
        logger.info(
            f'[Sizing] Kelly Multiplier: {KELLY_MULTIPLIER:.1f}x '
            f'| Final Risk: {result["equity_pct"]*100:.1f}% of Equity'
        )
    elif result['signal'] == 'close':
        logger.info(
            f'[Signal] {bar_jst} | XAUUSD | CLOSE | Price: {price:.2f}'
        )
    else:
        logger.debug(
            f'[NoTrade] {bar_jst} | XAUUSD | FLAT (no signal)'
        )


# ──────────────────────────────────────────────────────
#  メインループ
# ──────────────────────────────────────────────────────

def run_once() -> dict:
    """1回のシグナルチェックを実行して結果を返す。"""
    df = fetch_ohlc_from_oanda()
    if df is None or len(df) < 50:
        logger.warning('データ不足でシグナル評価をスキップ。')
        return {'signal': 'error', 'price': 0.0, 'bar_time': None, 'equity_pct': 0.0}

    result = evaluate_signal(df)
    log_signal(result)
    return result


def run_monitor(poll_interval: int = POLL_INTERVAL_SEC) -> None:
    """
    継続監視ループ。Ctrl+C で停止。

    Args:
        poll_interval: ポーリング間隔（秒）
    """
    logger.info('=' * 60)
    logger.info('Union + Kelly シグナルモニター 起動')
    logger.info(f'  対象: {INSTRUMENT} | 時間足: {GRANULARITY}')
    logger.info(f'  Kelly乗数: {KELLY_MULTIPLIER}x | 最終リスク: {FINAL_RISK_PCT*100:.0f}%')
    logger.info(f'  ポーリング間隔: {poll_interval}秒')
    logger.info(f'  モード: 監視のみ（発注なし）')
    logger.info('=' * 60)

    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            logger.info('モニター停止 (KeyboardInterrupt)')
            break
        except Exception as e:
            logger.error(f'予期しないエラー: {e}')

        logger.debug(f'次のチェックまで {poll_interval}秒待機...')
        time.sleep(poll_interval)


# ──────────────────────────────────────────────────────
#  エントリーポイント
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Union + Kelly シグナルモニター')
    parser.add_argument('--once', action='store_true',
                        help='1回だけ実行して終了 (デフォルト: ループ監視)')
    parser.add_argument('--interval', type=int, default=POLL_INTERVAL_SEC,
                        help=f'ポーリング間隔（秒）デフォルト: {POLL_INTERVAL_SEC}')
    args = parser.parse_args()

    if args.once:
        result = run_once()
        print(f'\n結果: signal={result["signal"]}, price={result["price"]:.2f}')
    else:
        run_monitor(poll_interval=args.interval)
