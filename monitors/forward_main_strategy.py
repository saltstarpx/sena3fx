"""
メイン戦略 フォワードテスト・シグナルモニター (v18)
====================================================
XAUUSD+Kelly(f=0.25)+ADX(14)>25 の最新シグナルをリアルタイムで監視し、
trade_logs/forward_signals.csv に記録する。
実際の注文発注は行わない（監視専用）。

実行方法:
  python monitors/forward_main_strategy.py             # ループ監視
  python monitors/forward_main_strategy.py --once      # 1回実行して終了

環境変数 (config.yaml または .env で設定):
  OANDA_API_TOKEN : OANDA v20 APIトークン
  OANDA_ACCOUNT_ID: OANDAアカウントID
  OANDA_ENV       : 'practice' (デフォルト) または 'live'

出力例:
  [Signal] 2025-10-15 09:00 JST | XAUUSD | LONG | Price: 2650.40 | ADX: 28.3
  [NoTrade] 2025-10-15 13:00 JST | XAUUSD | FLAT (ADX=22.1 < 25.0)
"""

import os
import sys
import csv
import time
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from strategies.main_strategy import (
    make_signal_func,
    compute_adx,
    SYMBOL, KELLY_FRACTION, ADX_THRESH, ADX_PERIOD,
    ENGINE_CFG,
)

# ── 設定 ────────────────────────────────────────────────

OANDA_ENV        = os.environ.get('OANDA_ENV', 'practice')
OANDA_API_TOKEN  = os.environ.get('OANDA_API_TOKEN', '')
OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID', '')
INSTRUMENT       = 'XAU_USD'   # OANDA形式
GRANULARITY      = 'H4'
CANDLE_COUNT     = 300         # SMA200 + バッファ

# Kelly 最終リスク (v17実績: 乗数1.13x × base 5%)
KELLY_MULTIPLIER = 1.13
FINAL_RISK_PCT   = ENGINE_CFG['risk_pct'] * KELLY_MULTIPLIER

POLL_INTERVAL_SEC = 60 * 30   # 30分

# ログ
LOG_DIR  = os.path.join(ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'forward_main_strategy.log')
os.makedirs(LOG_DIR, exist_ok=True)

# シグナルCSV
SIGNAL_CSV_DIR  = os.path.join(ROOT, 'trade_logs')
SIGNAL_CSV_PATH = os.path.join(SIGNAL_CSV_DIR, 'forward_signals.csv')
os.makedirs(SIGNAL_CSV_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger('forward_main_strategy')

# メインシグナル関数 (ADX(14)>25 込み)
_SIGNAL_FUNC = make_signal_func(adx_thresh=ADX_THRESH)


# ── OANDA データ取得 ─────────────────────────────────────

def _oanda_base_url() -> str:
    if OANDA_ENV == 'live':
        return 'https://api-fxtrade.oanda.com'
    return 'https://api-fxpractice.oanda.com'


def fetch_ohlc_from_oanda(count: int = CANDLE_COUNT) -> pd.DataFrame | None:
    """OANDA v20 REST API から XAU_USD 4時間足 OHLC を取得する。"""
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
                'open':   float(mid.get('o', 0)),
                'high':   float(mid.get('h', 0)),
                'low':    float(mid.get('l', 0)),
                'close':  float(mid.get('c', 0)),
                'volume': int(c.get('volume', 0)),
            })

        if not rows:
            logger.warning('OANDAから空のデータ。')
            return None

        df = pd.DataFrame(rows).set_index('datetime').sort_index()
        logger.debug(f'OANDA OHLC取得: {len(df)}バー')
        return df

    except Exception as e:
        logger.error(f'OANDAデータ取得エラー: {e}')
        return _load_local_ohlc()


def _load_local_ohlc() -> pd.DataFrame | None:
    """ローカルCSVからOHLCを読み込む (フォールバック)。"""
    import glob as _glob
    pat = os.path.join(ROOT, 'data', 'ohlc', f'{SYMBOL}*4h.csv')
    matches = sorted(_glob.glob(pat))
    if not matches:
        logger.error(f'ローカルCSVが見つかりません: {pat}')
        return None
    path = matches[-1]
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


# ── シグナル評価 ─────────────────────────────────────────

def _jst(ts) -> str:
    try:
        t   = pd.Timestamp(ts)
        jst = t + pd.Timedelta(hours=9)
        return jst.strftime('%Y-%m-%d %H:%M JST')
    except Exception:
        return str(ts)


def evaluate_signal(df: pd.DataFrame) -> dict:
    """最新バーのメイン戦略シグナルを評価する。"""
    signals  = _SIGNAL_FUNC(df)
    adx_vals = compute_adx(df['high'], df['low'], df['close'], p=ADX_PERIOD)

    bar_time  = signals.index[-1]
    sig_val   = signals.iloc[-1]
    adx_val   = float(adx_vals.iloc[-1]) if adx_vals.iloc[-1] == adx_vals.iloc[-1] else None
    price     = float(df['close'].iloc[-1])

    signal = 'flat' if (sig_val is None or sig_val == 'flat' or
                        (isinstance(sig_val, float) and sig_val != sig_val)) \
             else str(sig_val)

    return {
        'signal':   signal,
        'price':    price,
        'bar_time': bar_time,
        'adx':      adx_val,
    }


# ── CSV ロギング ─────────────────────────────────────────

_CSV_HEADER = ['datetime_utc', 'symbol', 'direction', 'price', 'adx']


def _ensure_csv_header():
    """forward_signals.csv が存在しない場合はヘッダー行を作成する。"""
    if not os.path.exists(SIGNAL_CSV_PATH):
        with open(SIGNAL_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(_CSV_HEADER)


def append_signal_csv(result: dict) -> None:
    """シグナル結果を forward_signals.csv に追記する。"""
    _ensure_csv_header()
    bar_time = result['bar_time']
    utc_str  = pd.Timestamp(bar_time).strftime('%Y-%m-%dT%H:%M:%SZ') \
               if bar_time is not None else ''
    adx_str  = f'{result["adx"]:.2f}' if result['adx'] is not None else ''
    row = [utc_str, SYMBOL, result['signal'], f'{result["price"]:.5f}', adx_str]
    with open(SIGNAL_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


# ── ログ出力 ─────────────────────────────────────────────

def log_signal(result: dict) -> None:
    bar_jst = _jst(result['bar_time'])
    sig     = result['signal'].upper()
    price   = result['price']
    adx_str = f'ADX={result["adx"]:.1f}' if result['adx'] is not None else 'ADX=N/A'

    if result['signal'] == 'long':
        logger.info(
            f'[Signal] {bar_jst} | {SYMBOL} | {sig} | '
            f'Price: {price:.2f} | {adx_str}'
        )
        logger.info(
            f'[Sizing] Kelly(f={KELLY_FRACTION}) x{KELLY_MULTIPLIER:.2f} = '
            f'{FINAL_RISK_PCT*100:.1f}% risk'
        )
    elif result['signal'] == 'close':
        logger.info(
            f'[Signal] {bar_jst} | {SYMBOL} | CLOSE | '
            f'Price: {price:.2f} | {adx_str}'
        )
    else:
        reason = (f'ADX={result["adx"]:.1f} < {ADX_THRESH:.0f}'
                  if result['adx'] is not None and result['adx'] < ADX_THRESH
                  else 'no signal')
        logger.debug(f'[NoTrade] {bar_jst} | {SYMBOL} | FLAT ({reason})')


# ── メインループ ─────────────────────────────────────────

def run_once() -> dict:
    """1回のシグナルチェックを実行して結果を返す。"""
    df = fetch_ohlc_from_oanda()
    if df is None or len(df) < 50:
        logger.warning('データ不足でシグナル評価をスキップ。')
        return {'signal': 'error', 'price': 0.0, 'bar_time': None, 'adx': None}

    result = evaluate_signal(df)
    log_signal(result)
    append_signal_csv(result)
    return result


def run_monitor(poll_interval: int = POLL_INTERVAL_SEC) -> None:
    """継続監視ループ。Ctrl+C で停止。"""
    logger.info('=' * 60)
    logger.info(f'メイン戦略 フォワードモニター 起動')
    logger.info(f'  戦略: XAUUSD+Kelly(f={KELLY_FRACTION})+ADX(>{ADX_THRESH:.0f})')
    logger.info(f'  対象: {INSTRUMENT} | 時間足: {GRANULARITY}')
    logger.info(f'  Kelly乗数: {KELLY_MULTIPLIER}x | 最終リスク: {FINAL_RISK_PCT*100:.1f}%')
    logger.info(f'  シグナルCSV: {SIGNAL_CSV_PATH}')
    logger.info(f'  ポーリング間隔: {poll_interval}秒')
    logger.info('=' * 60)

    _ensure_csv_header()

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


# ── エントリーポイント ────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='メイン戦略 フォワードテスト・シグナルモニター'
    )
    parser.add_argument('--once', action='store_true',
                        help='1回だけ実行して終了 (デフォルト: ループ監視)')
    parser.add_argument('--interval', type=int, default=POLL_INTERVAL_SEC,
                        help=f'ポーリング間隔（秒）デフォルト: {POLL_INTERVAL_SEC}')
    args = parser.parse_args()

    if args.once:
        result = run_once()
        adx_str = f'{result["adx"]:.1f}' if result['adx'] is not None else 'N/A'
        print(f'\n結果: signal={result["signal"]}, '
              f'price={result["price"]:.2f}, ADX={adx_str}')
        print(f'シグナルCSV: {SIGNAL_CSV_PATH}')
    else:
        run_monitor(poll_interval=args.interval)
