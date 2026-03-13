"""
sena3fx Live Trading Bot
========================
OANDA v20 API を使用した24時間自律トレードボット。
やがみメソッド準拠 (SEAS_DC2d / UNION3d50_SEAS 戦略)。

起動方法 (Windows):
  start_bot.bat  を実行してください。

直接起動 (Python):
  python live\\bot.py

停止方法:
  stop_bot.bat または Ctrl+C
"""

import os
import sys
import time
import yaml
import signal
import logging
import traceback
from datetime import datetime
from pathlib import Path

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.broker_oanda import OandaBroker
from live.signal_engine import get_signal
from live.risk_manager import (
    calc_position_size,
    check_drawdown_limit,
    calc_sl_tp_prices,
)


# ------------------------------------------------------------------ #
#  ロギング設定                                                       #
# ------------------------------------------------------------------ #

def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """ファイル + コンソール両方に出力するロガー設定"""
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'bot_{today}.log')

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt   = '%(asctime)s [%(levelname)-5s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    return logging.getLogger('sena3fx')


# ------------------------------------------------------------------ #
#  設定読み込み                                                       #
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> dict:
    """config.yaml を読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  メインボットクラス                                                 #
# ------------------------------------------------------------------ #

class TradingBot:
    """
    24時間稼働の自律トレードボット。
    4H足のクローズ時に新しいシグナルを確認し、エントリーを判断する。
    """

    def __init__(self, config: dict):
        self.config = config
        self.log = logging.getLogger('sena3fx')

        # ブローカー接続
        oanda_cfg = config['oanda']
        self.broker = OandaBroker(
            account_id=oanda_cfg['account_id'],
            api_key=oanda_cfg['api_key'],
            environment=oanda_cfg.get('environment', 'practice'),
        )

        # 最後に処理した4Hバーの時刻 {instrument: datetime}
        self.last_bar_time: dict = {}

        # ピーク残高 (ドローダウン監視用)
        self.peak_balance: float = None

        # 停止フラグ
        self.running = True

    # ---- ライフサイクル -------------------------------------------- #

    def run(self):
        """メインループを開始する"""
        self.log.info("=" * 60)
        self.log.info("  sena3fx Live Trading Bot 起動")
        self.log.info(f"  環境: {self.config['oanda'].get('environment', 'practice').upper()}")
        self.log.info(f"  日時: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.log.info("=" * 60)

        # Ctrl+C / SIGTERM でのグレースフル停止
        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

        # 初期残高取得
        try:
            balance = self.broker.get_balance()
            self.peak_balance = balance
            self.log.info(f"口座残高: ${balance:,.2f}")
        except Exception as e:
            self.log.error(f"口座接続エラー: {e}")
            self.log.error("APIキーと口座IDを config.yaml で確認してください。")
            return

        # 有効な通貨ペアのログ
        instruments = self.config.get('instruments', {})
        enabled = [k for k, v in instruments.items() if v.get('enabled', False)]
        self.log.info(f"監視通貨ペア: {enabled}")

        interval = int(self.config.get('bot', {}).get('check_interval_seconds', 60))
        self.log.info(f"チェック間隔: {interval}秒")
        self.log.info("監視開始... (停止: Ctrl+C または stop_bot.bat)")
        self.log.info("-" * 60)

        # メインループ
        while self.running:
            try:
                self._process_tick()
            except Exception as e:
                self.log.error(f"ティック処理エラー: {e}")
                self.log.debug(traceback.format_exc())

            # 次のチェックまで待機
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)

        self.log.info("ボットを停止しました。")

    def _on_shutdown(self, signum, frame):
        """Ctrl+C / kill シグナル受信時の処理"""
        self.log.info("停止シグナル受信 → ボット停止中...")
        self.running = False

    # ---- メイン処理 ------------------------------------------------ #

    def _process_tick(self):
        """1チェックサイクルの処理"""

        # 残高取得
        try:
            balance = self.broker.get_balance()
        except Exception as e:
            self.log.warning(f"残高取得失敗 (ネットワーク問題?) : {e}")
            return

        # ピーク残高更新
        if self.peak_balance is None or balance > self.peak_balance:
            self.peak_balance = balance

        # ドローダウンチェック
        max_dd = float(self.config.get('bot', {}).get('max_drawdown_pct', 0.15))
        if not check_drawdown_limit(balance, self.peak_balance, max_dd):
            self.log.critical(
                f"最大ドローダウン {max_dd:.0%} 超過 → 自動停止します！\n"
                f"  残高: ${balance:,.2f}  ピーク: ${self.peak_balance:,.2f}\n"
                f"  再起動前に market condition を確認してください。"
            )
            self.running = False
            return

        # 各通貨ペアを処理
        for instrument, inst_cfg in self.config.get('instruments', {}).items():
            if not inst_cfg.get('enabled', False):
                continue
            try:
                self._process_instrument(instrument, inst_cfg, balance)
            except Exception as e:
                self.log.error(f"[{instrument}] エラー: {e}")
                self.log.debug(traceback.format_exc())

    def _process_instrument(self, instrument: str, cfg: dict, balance: float):
        """通貨ペアごとのシグナル確認 & 注文処理"""

        strategy = cfg.get('strategy', 'SEAS_DC2d')

        # 4H完成バーを200本取得
        bars = self.broker.get_candles(instrument, granularity='H4', count=200)
        if bars is None or len(bars) < 30:
            n = len(bars) if bars is not None else 0
            self.log.warning(f"[{instrument}] データ不足 ({n} bars) → スキップ")
            return

        # 最新完成バーの時刻
        last_bar_time = bars.index[-1]

        # 前回と同じバーなら処理済み → スキップ
        prev_time = self.last_bar_time.get(instrument)
        if prev_time is not None and last_bar_time <= prev_time:
            return  # 新しいバーなし

        self.log.info(
            f"[{instrument}] 新4Hバー確定: {last_bar_time} | 戦略: {strategy}"
        )
        self.last_bar_time[instrument] = last_bar_time

        # ---- シグナル計算 ---- #
        result = get_signal(bars, strategy)
        sig       = result['signal']
        atr       = result['atr']
        close_px  = result['close']

        self.log.info(
            f"[{instrument}] シグナル: [{sig.upper():4s}] | "
            f"Close: {close_px:.3f} | ATR14: {atr:.3f if atr else 'N/A'}"
        )

        if sig == 'flat' or atr is None or atr <= 0:
            return  # エントリー条件なし

        # ---- 現在ポジション確認 ---- #
        long_units, short_units = self.broker.get_position(instrument)
        has_long  = long_units  > 0
        has_short = short_units < 0  # OANDAはshortを負数で返す

        # ---- ロングエントリー ---- #
        if sig == 'long':
            if has_long:
                self.log.info(f"[{instrument}] ロング済み ({long_units} units) → スキップ")
                return

            sl_atr   = float(cfg.get('sl_atr', 1.5))
            tp_atr   = float(cfg.get('tp_atr', 4.5))
            risk_pct = float(cfg.get('risk_pct', 0.05))
            max_u    = cfg.get('max_units', None)

            sl_distance = sl_atr * atr
            units = calc_position_size(balance, risk_pct, sl_distance, max_u)

            if units <= 0:
                self.log.warning(f"[{instrument}] 計算ユニット0 → スキップ")
                return

            # 現在のask価格でエントリー
            bid, ask = self.broker.get_current_price(instrument)
            entry_price = ask

            sl_price, tp_price = calc_sl_tp_prices(
                entry_price, atr, sl_atr, tp_atr, side='long'
            )

            self.log.info(
                f"[{instrument}] ロング発注:\n"
                f"  Units:  {units:,}\n"
                f"  Entry:  {entry_price:.3f}\n"
                f"  SL:     {sl_price:.3f}  (-{sl_distance:.3f} / -{sl_atr}×ATR)\n"
                f"  TP:     {tp_price:.3f}  (+{sl_distance * (tp_atr/sl_atr):.3f} / +{tp_atr}×ATR)\n"
                f"  Risk:   ${balance * risk_pct:,.0f} ({risk_pct:.0%} of ${balance:,.0f})"
            )

            try:
                resp = self.broker.place_market_order(
                    instrument=instrument,
                    units=units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                )
                tx = resp.get('orderFillTransaction') or resp.get('orderCreateTransaction', {})
                order_id = tx.get('id', 'N/A')
                fill_px  = tx.get('price', 'N/A')
                self.log.info(
                    f"[{instrument}] 発注完了 ✓ "
                    f"OrderID={order_id}, 約定価格={fill_px}"
                )
            except Exception as e:
                self.log.error(f"[{instrument}] 発注エラー: {e}")


# ------------------------------------------------------------------ #
#  エントリーポイント                                                 #
# ------------------------------------------------------------------ #

def main():
    # プロジェクトルート
    root = Path(__file__).parent.parent

    # config.yaml 確認
    config_path = root / 'config.yaml'
    if not config_path.exists():
        print(f"[ERROR] config.yaml が見つかりません: {config_path}")
        print("        config.example.yaml をコピーして config.yaml を作成してください。")
        print("        Windowsなら setup.bat を実行すると自動でコピーされます。")
        sys.exit(1)

    # 設定読み込み
    config = load_config(str(config_path))

    # ロギング設定
    log_dir   = config.get('bot', {}).get('log_dir',   'logs')
    log_level = config.get('bot', {}).get('log_level', 'INFO')
    setup_logging(str(root / log_dir), log_level)

    # PIDファイル書き込み (stop_bot.bat がプロセスを終了するために使用)
    pid_file = root / 'bot.pid'
    pid_file.write_text(str(os.getpid()))

    try:
        bot = TradingBot(config)
        bot.run()
    finally:
        # 終了時にPIDファイルを削除
        if pid_file.exists():
            pid_file.unlink()


if __name__ == '__main__':
    main()
