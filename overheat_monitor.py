"""
XAUT過熱度アラートモニター
============================
XAUTとXAUUSDの価格乖離を監視し、過熱・冷却サインを検知する。

XAUTとXAUUSDは理論上ほぼ同価格であるため、乖離が大きい場合は
市場の過熱または冷却を示すシグナルとして活用できる。

データソース:
    - XAUT価格: Bybit 公開API (認証不要)
    - XAUUSD価格: OANDA APIまたはローカルデータ (フォールバック)

使い方:
    # スタンドアロン監視ループ
    python overheat_monitor.py

    # main.py / bot_v2.py から利用
    from overheat_monitor import OverheatMonitor
    monitor = OverheatMonitor()
    if monitor.is_overheated():
        logger.warning("XAUT過熱警戒中 - 新規ロング停止")
        entry_filter.pause_long_entries = True
    else:
        entry_filter.pause_long_entries = False
"""
import os
import sys
import time
import logging
import csv
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger('sena3fx.overheat')

# ------------------------------------------------------------------ #
#  設定                                                               #
# ------------------------------------------------------------------ #
XAUT_API_URL          = "https://api.bybit.com/v5/market/tickers?category=spot&symbol=XAUTUSDT"
OVERHEAT_THRESHOLD    = +0.3    # 乖離率 +0.3% 以上 → 過熱警戒
COOLDOWN_THRESHOLD    = -0.3    # 乖離率 -0.3% 以下 → 冷却サイン
CHECK_INTERVAL_MINUTES = 15     # 監視間隔 (分)
LOG_PATH              = os.path.join(ROOT, 'data', 'overheat_log.csv')
REQUEST_TIMEOUT       = 10      # API タイムアウト (秒)
CSV_HEADERS           = [
    'timestamp', 'xaut_price', 'xauusd_price', 'divergence_pct',
    'status', 'alert',
]


# ------------------------------------------------------------------ #
#  XAUT価格取得 (Bybit公開API)                                        #
# ------------------------------------------------------------------ #
def get_xaut_price(timeout: int = REQUEST_TIMEOUT) -> Optional[float]:
    """
    BybitからXAUT/USDT現在価格を取得する。

    Returns:
        float: XAUT価格 (USD), 取得失敗時は None
    """
    try:
        resp = requests.get(XAUT_API_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        result_list = data.get('result', {}).get('list', [])
        if result_list:
            return float(result_list[0]['lastPrice'])
        log.warning("Bybit API: XAUTデータが空")
        return None
    except requests.exceptions.Timeout:
        log.warning("Bybit API タイムアウト")
        return None
    except requests.exceptions.RequestException as e:
        log.warning(f"Bybit API エラー: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        log.warning(f"Bybit API レスポンス解析エラー: {e}")
        return None


# ------------------------------------------------------------------ #
#  XAUUSD価格取得                                                      #
# ------------------------------------------------------------------ #
def get_xauusd_price_from_oanda(account_id: str,
                                 api_key: str,
                                 environment: str = 'practice',
                                 timeout: int = REQUEST_TIMEOUT) -> Optional[float]:
    """
    OANDA APIからXAUUSDの現在Mid価格を取得する。

    Args:
        account_id:  OANDA口座ID
        api_key:     OANDAのAPIキー
        environment: 'practice' または 'live'
    """
    base = ('https://api-fxpractice.oanda.com'
            if environment == 'practice'
            else 'https://api-fxtrade.oanda.com')
    url     = f"{base}/v3/instruments/XAU_USD/candles"
    headers = {'Authorization': f'Bearer {api_key}'}
    params  = {'count': 1, 'granularity': 'M1', 'price': 'M'}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        candles = resp.json().get('candles', [])
        if candles:
            mid = candles[-1]['mid']
            return float(mid['c'])
    except Exception as e:
        log.warning(f"OANDA API エラー: {e}")
    return None


def get_xauusd_price_from_csv() -> Optional[float]:
    """
    ローカルCSVからXAUUSDの最新価格を取得する (フォールバック)。
    ライブ環境でOANDA APIが使えない場合に利用する。
    """
    path = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return float(df['close'].iloc[-1])
    except Exception as e:
        log.warning(f"CSV読み込みエラー: {e}")
        return None


# ------------------------------------------------------------------ #
#  乖離率計算                                                         #
# ------------------------------------------------------------------ #
def calc_divergence(xaut_price: float, xauusd_price: float) -> float:
    """
    XAUTとXAUUSDの価格乖離率を計算する。

    乖離率 = (XAUT - XAUUSD) / XAUUSD × 100

    プラス = XAUTが割高 = 先物過熱 → ロング新規参入に注意
    マイナス = XAUTが割安 = 冷却サイン → ロング再開の可能性

    Args:
        xaut_price:   BybitのXAUT/USDT価格
        xauusd_price: OANDAのXAU/USD価格

    Returns:
        float: 乖離率 (%)
    """
    if xauusd_price == 0:
        return 0.0
    return (xaut_price - xauusd_price) / xauusd_price * 100


def get_alert_status(divergence_pct: float) -> Tuple[str, bool]:
    """
    乖離率からアラートステータスを判定する。

    Returns:
        Tuple[str, bool]: (status_str, is_alert)
            status_str: 'OVERHEAT' / 'COOLDOWN' / 'NORMAL'
            is_alert:   True = アラート発生
    """
    if divergence_pct >= OVERHEAT_THRESHOLD:
        return 'OVERHEAT', True
    elif divergence_pct <= COOLDOWN_THRESHOLD:
        return 'COOLDOWN', True
    else:
        return 'NORMAL', False


# ------------------------------------------------------------------ #
#  ログ記録                                                           #
# ------------------------------------------------------------------ #
def log_to_csv(timestamp: datetime,
               xaut_price: float,
               xauusd_price: float,
               divergence_pct: float,
               status: str,
               alert: bool,
               path: str = LOG_PATH) -> None:
    """乖離データをCSVに追記する"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp':      timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'xaut_price':     f'{xaut_price:.2f}',
            'xauusd_price':   f'{xauusd_price:.2f}',
            'divergence_pct': f'{divergence_pct:.4f}',
            'status':         status,
            'alert':          '1' if alert else '0',
        })


# ================================================================== #
#  OverheatMonitor クラス                                             #
# ================================================================== #
class OverheatMonitor:
    """
    XAUT/XAUUSD 過熱度モニター

    使い方:
        monitor = OverheatMonitor()
        monitor.update()  # 最新データを取得

        if monitor.is_overheated():
            # 過熱中 → ロング停止
        if monitor.is_cooled_down():
            # 冷却中 → ロング再開検討
    """

    def __init__(self,
                 oanda_account_id: Optional[str] = None,
                 oanda_api_key: Optional[str] = None,
                 oanda_environment: str = 'practice',
                 log_path: str = LOG_PATH):
        self.oanda_account_id  = oanda_account_id
        self.oanda_api_key     = oanda_api_key
        self.oanda_environment = oanda_environment
        self.log_path          = log_path

        # 最新の状態
        self.xaut_price:      Optional[float] = None
        self.xauusd_price:    Optional[float] = None
        self.divergence_pct:  Optional[float] = None
        self.status:          str = 'UNKNOWN'
        self.last_updated:    Optional[datetime] = None

    def _get_xauusd(self) -> Optional[float]:
        """XAUUSD価格を取得 (OANDA優先、フォールバックはCSV)"""
        if self.oanda_account_id and self.oanda_api_key:
            price = get_xauusd_price_from_oanda(
                self.oanda_account_id,
                self.oanda_api_key,
                self.oanda_environment,
            )
            if price:
                return price
        return get_xauusd_price_from_csv()

    def update(self) -> bool:
        """
        最新の価格データを取得して状態を更新する。

        Returns:
            bool: 更新成功なら True
        """
        xaut  = get_xaut_price()
        xauusd = self._get_xauusd()

        if xaut is None or xauusd is None:
            log.warning(
                f"価格取得失敗 (XAUT={xaut}, XAUUSD={xauusd}) → スキップ"
            )
            return False

        div = calc_divergence(xaut, xauusd)
        status, is_alert = get_alert_status(div)

        self.xaut_price     = xaut
        self.xauusd_price   = xauusd
        self.divergence_pct = div
        self.status         = status
        self.last_updated   = datetime.now(timezone.utc)

        # CSVログ記録
        log_to_csv(self.last_updated, xaut, xauusd, div, status, is_alert, self.log_path)

        # コンソール表示
        symbol = '🔴' if status == 'OVERHEAT' else '🔵' if status == 'COOLDOWN' else '✅'
        msg = (
            f"[XAUT監視] {symbol} {status}  "
            f"XAUT=${xaut:.2f}  XAUUSD=${xauusd:.2f}  "
            f"乖離={div:+.4f}%"
        )
        if is_alert:
            print(f"\n{'!'*60}")
            print(f"  {msg}")
            if status == 'OVERHEAT':
                print(f"  ⚠ 過熱警戒: 乖離率が +{OVERHEAT_THRESHOLD}% 超え")
                print(f"  ⚠ 新規ロングエントリーを一時停止することを推奨")
            elif status == 'COOLDOWN':
                print(f"  ℹ 冷却サイン: 乖離率が {COOLDOWN_THRESHOLD}% 以下")
                print(f"  ℹ ロング再開の可能性あり")
            print(f"{'!'*60}\n")
        else:
            print(f"[XAUT監視] {msg}")

        log.info(msg)
        return True

    def is_overheated(self) -> bool:
        """
        現在過熱状態かどうかを返す。

        Returns:
            bool: True = 過熱中 (新規ロング停止を推奨)
        """
        return self.status == 'OVERHEAT'

    def is_cooled_down(self) -> bool:
        """
        現在冷却状態かどうかを返す。

        Returns:
            bool: True = 冷却中 (ロング再開を検討)
        """
        return self.status == 'COOLDOWN'

    def is_normal(self) -> bool:
        """通常状態かどうかを返す"""
        return self.status == 'NORMAL'

    def get_summary(self) -> dict:
        """現在の状態サマリーを辞書で返す"""
        return {
            'xaut_price':     self.xaut_price,
            'xauusd_price':   self.xauusd_price,
            'divergence_pct': self.divergence_pct,
            'status':         self.status,
            'is_overheated':  self.is_overheated(),
            'is_cooled_down': self.is_cooled_down(),
            'last_updated':   self.last_updated.isoformat() if self.last_updated else None,
        }


# ------------------------------------------------------------------ #
#  スタンドアロン監視ループ                                          #
# ------------------------------------------------------------------ #
def run_monitor_loop(account_id: Optional[str] = None,
                     api_key: Optional[str] = None,
                     environment: str = 'practice',
                     interval_minutes: int = CHECK_INTERVAL_MINUTES) -> None:
    """
    15分ごとにXAUT/XAUUSD乖離を監視するループ。

    Ctrl+C で停止。

    Args:
        account_id:        OANDA口座ID (省略可 → CSVフォールバック)
        api_key:           OANDA APIキー (省略可)
        environment:       'practice' または 'live'
        interval_minutes:  監視間隔 (分)
    """
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    monitor = OverheatMonitor(
        oanda_account_id  = account_id,
        oanda_api_key     = api_key,
        oanda_environment = environment,
    )

    print("=" * 60)
    print("  XAUT過熱度モニター 起動")
    print(f"  監視間隔: {interval_minutes} 分")
    print(f"  過熱閾値: ≥ +{OVERHEAT_THRESHOLD}%")
    print(f"  冷却閾値: ≤ {COOLDOWN_THRESHOLD}%")
    print(f"  ログ: {LOG_PATH}")
    print("  Ctrl+C で停止")
    print("=" * 60)

    interval_seconds = interval_minutes * 60

    try:
        while True:
            monitor.update()
            print(f"  次回チェック: {interval_minutes}分後")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\nモニター停止")


# ------------------------------------------------------------------ #
#  ワンショット取得 (テスト・デバッグ用)                              #
# ------------------------------------------------------------------ #
def check_once(account_id: Optional[str] = None,
               api_key: Optional[str] = None,
               environment: str = 'practice') -> dict:
    """
    1回だけ価格を取得して乖離率を返す。

    Returns:
        dict: {xaut_price, xauusd_price, divergence_pct, status, ...}
    """
    monitor = OverheatMonitor(account_id, api_key, environment)
    monitor.update()
    return monitor.get_summary()


# ------------------------------------------------------------------ #
#  メイン                                                             #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='XAUT過熱度モニター')
    parser.add_argument('--account',     default=None,       help='OANDA口座ID')
    parser.add_argument('--api-key',     default=None,       help='OANDA APIキー')
    parser.add_argument('--env',         default='practice', help='practice / live')
    parser.add_argument('--interval',    type=int, default=CHECK_INTERVAL_MINUTES,
                        help=f'監視間隔(分) デフォルト={CHECK_INTERVAL_MINUTES}')
    parser.add_argument('--once',        action='store_true', help='1回だけ取得して終了')
    args = parser.parse_args()

    if args.once:
        result = check_once(args.account, args.api_key, args.env)
        print("\n--- 現在の過熱度 ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        run_monitor_loop(args.account, args.api_key, args.env, args.interval)
