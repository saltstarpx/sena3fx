"""
sena3fx Live Trading Bot v2.0
==============================
デュアルタイムフレーム (4H × 15m) 自律トレードボット。

実装ルール:
  - ブレイク確認: 4H足実体確定後のみ有効
  - トレンド方向: 4H足 EMA21 (順張りのみ)
  - エントリータイミング: 15m足 DC ブレイクアウト (4H方向一致)
  - 損切り: 直近スイング安値 (ロング) / 高値 (ショート) @ 4H足
  - RRフィルター: RR < 2.0 はスキップ
  - ポジション管理: XAU/XAG 排他制御
  - 禁止期間: 年末年始 / 米・日祝日 / 土曜日 (CMEクローズ)
  - 監視足: 4H足 (トレンド) + 15m足 (エントリー) のみ

起動方法:
  start_bot_v2.bat  または  python live\\bot_v2.py
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

from live.broker_oanda   import OandaBroker
from live.time_filter    import is_trading_allowed, get_block_reason
from live.position_manager import PositionManager
from lib.swing           import get_sl_for_long, get_sl_for_short
from lib.dual_tf         import get_4h_trend, get_15m_signal


# ------------------------------------------------------------------ #
#  ロギング設定                                                       #
# ------------------------------------------------------------------ #

def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'bot_v2_{today}.log')

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt   = '%(asctime)s [%(levelname)-5s] %(message)s'
    dfmt  = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=dfmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    return logging.getLogger('sena3fx')


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  ポジションサイズ計算                                               #
# ------------------------------------------------------------------ #

def calc_units(balance: float, risk_pct: float,
               entry: float, sl: float,
               lot_divisor: float = 1.0,
               max_units: int = None) -> int:
    """
    スイングSLベースのポジションサイズ計算。

    仕様書記載の計算式:
      lot = (総資金 × 0.02) / (エントリー価格 - 損切り価格) / lot_divisor

    OANDA XAU_USD: 1 unit = 1 troy ounce
      → lot_divisor = 1.0 (そのままユニット数)

    Args:
        balance:     口座残高 (USD)
        risk_pct:    リスク率 (0.02 = 2%)
        entry:       エントリー価格
        sl:          損切り価格
        lot_divisor: ロット換算除数 (OANDA: 1.0)
        max_units:   最大ユニット上限

    Returns:
        int: 発注ユニット数 (0 以上)
    """
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0

    risk_amount = balance * risk_pct
    units = int((risk_amount / sl_dist) / lot_divisor)
    units = max(1, units)

    if max_units is not None:
        units = min(units, int(max_units))

    return units


# ------------------------------------------------------------------ #
#  メインボットクラス                                                 #
# ------------------------------------------------------------------ #

class TradingBotV2:
    """
    4H × 15m デュアルTFトレードボット。

    動作サイクル (check_interval_seconds 毎):
      1. DD チェック → 超過時は自動停止
      2. 時間フィルター → 禁止期間はスキップ
      3. 各通貨ペア:
           a. 15m足の新バー確定を確認 (未確定ならスキップ)
           b. 4H足取得 → EMA21でトレンド判定
           c. 15m足取得 → DC ブレイクアウトでシグナル確認
           d. 排他制御チェック (XAU/XAG)
           e. スイング安値/高値から SL 計算
           f. RR = (TP-entry) / (entry-SL) >= min_rr チェック
           g. 全条件OK → 成行注文 + SL/TP アタッチ
    """

    def __init__(self, config: dict):
        self.config = config
        self.log = logging.getLogger('sena3fx')

        oanda_cfg = config['oanda']
        self.broker = OandaBroker(
            account_id=oanda_cfg['account_id'],
            api_key=oanda_cfg['api_key'],
            environment=oanda_cfg.get('environment', 'practice'),
        )
        self.pos_mgr = PositionManager(self.broker)

        # 最後に処理した15mバーの時刻 {instrument: datetime}
        self.last_15m_bar: dict = {}

        # ピーク残高 (ドローダウン監視)
        self.peak_balance: float = None

        self.running = True

    # ---- ライフサイクル -------------------------------------------- #

    def run(self):
        self.log.info("=" * 65)
        self.log.info("  sena3fx Live Trading Bot v2.0")
        self.log.info("  4H × 15m デュアルTFシステム")
        self.log.info(f"  環境: {self.config['oanda'].get('environment','practice').upper()}")
        self.log.info(f"  起動: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.log.info("=" * 65)

        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

        # 接続確認
        try:
            balance = self.broker.get_balance()
            self.peak_balance = balance
            self.log.info(f"口座残高: ${balance:,.2f}")
        except Exception as e:
            self.log.error(f"口座接続エラー: {e}")
            self.log.error("config.yaml の account_id / api_key / environment を確認してください。")
            return

        # 有効通貨ペアをログ
        instruments = self.config.get('instruments', {})
        enabled = [k for k, v in instruments.items() if v.get('enabled', False)]
        self.log.info(f"監視通貨ペア: {enabled}")

        interval = int(self.config.get('bot', {}).get('check_interval_seconds', 60))
        self.log.info(f"チェック間隔: {interval}秒")
        self.log.info("")
        self.log.info("[戦略]")
        self.log.info("  トレンド: 4H足 EMA21")
        self.log.info("  エントリー: 15m足 DC ブレイクアウト (4H方向と一致)")
        self.log.info("  SL: 4H足スイング安値/高値")
        self.log.info("  禁止期間: 年末年始/祝日/土曜")
        self.log.info("")
        self.log.info("監視開始... (停止: Ctrl+C または stop_bot_v2.bat)")
        self.log.info("-" * 65)

        while self.running:
            try:
                self._process_tick()
            except Exception as e:
                self.log.error(f"ティック処理エラー: {e}")
                self.log.debug(traceback.format_exc())

            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)

        self.log.info("ボット停止完了。")

    def _on_shutdown(self, signum, frame):
        self.log.info("停止シグナル受信 → ボット停止中...")
        self.running = False

    # ---- メイン処理 ------------------------------------------------ #

    def _process_tick(self):
        """1チェックサイクルの処理"""

        # --- 残高取得 --- #
        try:
            balance = self.broker.get_balance()
        except Exception as e:
            self.log.warning(f"残高取得失敗 (ネットワーク?): {e}")
            return

        if balance > (self.peak_balance or 0):
            self.peak_balance = balance

        # --- ドローダウン監視 --- #
        max_dd = float(self.config.get('bot', {}).get('max_drawdown_pct', 0.20))
        if self.peak_balance and self.peak_balance > 0:
            dd = (self.peak_balance - balance) / self.peak_balance
            if dd >= max_dd:
                self.log.critical(
                    f"最大DD {max_dd:.0%} 超過 → 自動停止！\n"
                    f"  現在残高: ${balance:,.2f}  ピーク: ${self.peak_balance:,.2f}\n"
                    f"  DD: {dd:.1%}\n"
                    f"  OANDAダッシュボードでポジションを確認してください。"
                )
                self.running = False
                return

        # --- 時間フィルター --- #
        now_utc = datetime.utcnow()
        if not is_trading_allowed(now_utc):
            reason = get_block_reason(now_utc)
            self.log.debug(f"エントリー禁止期間: {reason}")
            return

        # --- 各通貨ペアを処理 --- #
        for instrument, inst_cfg in self.config.get('instruments', {}).items():
            if not inst_cfg.get('enabled', False):
                continue
            try:
                self._process_instrument(instrument, inst_cfg, balance)
            except Exception as e:
                self.log.error(f"[{instrument}] 処理エラー: {e}")
                self.log.debug(traceback.format_exc())

    def _process_instrument(self, instrument: str, cfg: dict, balance: float):
        """1通貨ペアのシグナル確認と注文処理"""

        # --- 15m足取得 (完成バーのみ) --- #
        bars_15m = self.broker.get_candles(instrument, granularity='M15', count=300)
        if bars_15m is None or len(bars_15m) < 30:
            self.log.debug(f"[{instrument}] 15m データ不足")
            return

        # --- 新しい15mバーが確定したか確認 --- #
        last_15m_time = bars_15m.index[-1]
        prev_15m = self.last_15m_bar.get(instrument)
        if prev_15m is not None and last_15m_time <= prev_15m:
            return  # 新バーなし → スキップ

        self.last_15m_bar[instrument] = last_15m_time

        # --- 4H足取得 (完成バーのみ) --- #
        # ルール: 「4H足の実体確定後のみシグナル有効」
        #         → OANDA API は complete=True のバーのみ返す (既に実装済み)
        bars_4h = self.broker.get_candles(instrument, granularity='H4', count=200)
        if bars_4h is None or len(bars_4h) < 50:
            self.log.debug(f"[{instrument}] 4H データ不足")
            return

        # --- 4H トレンド判定 --- #
        ema_days = int(cfg.get('ema_days', 21))
        trend = get_4h_trend(bars_4h, ema_days)

        # --- 15m エントリーシグナル --- #
        dc_lookback = int(cfg.get('dc_lookback_15m', 20))
        sig = get_15m_signal(bars_15m, trend, dc_lookback)

        self.log.debug(
            f"[{instrument}] 15m={last_15m_time} | 4H trend={trend} | signal={sig}"
        )

        if sig is None:
            return  # シグナルなし

        self.log.info(
            f"[{instrument}] ★ シグナル: {sig.upper()} | "
            f"4H trend: {trend} | 15m: {last_15m_time}"
        )

        # --- XAU/XAG 排他制御チェック --- #
        if not self.pos_mgr.can_enter(instrument):
            return

        # --- SL 計算 (4H スイング安値 / 高値) --- #
        swing_win = int(cfg.get('swing_window',  3))
        swing_lb  = int(cfg.get('swing_lookback', 10))

        if sig == 'long':
            sl_price = get_sl_for_long(bars_4h, swing_win, swing_lb)
        else:
            sl_price = get_sl_for_short(bars_4h, swing_win, swing_lb)

        # --- 現在価格取得 --- #
        bid, ask = self.broker.get_current_price(instrument)
        entry_price = ask if sig == 'long' else bid

        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            self.log.warning(f"[{instrument}] SL距離=0 → スキップ")
            return

        # SL が entry と逆側にあるか確認 (中途半端なSL禁止)
        if sig == 'long'  and sl_price >= entry_price:
            self.log.warning(
                f"[{instrument}] SL({sl_price:.3f}) >= entry({entry_price:.3f}) "
                "スイングローが現在価格より上 → スキップ"
            )
            return
        if sig == 'short' and sl_price <= entry_price:
            self.log.warning(
                f"[{instrument}] SL({sl_price:.3f}) <= entry({entry_price:.3f}) "
                "スイングハイが現在価格より下 → スキップ"
            )
            return

        # --- TP計算 --- #
        rr_target = float(cfg.get('rr_target', 2.5))  # TP = entry + rr_target × SL_dist
        if sig == 'long':
            tp_price = entry_price + rr_target * sl_dist
        else:
            tp_price = entry_price - rr_target * sl_dist

        # --- RRフィルター (1:2 未満はスキップ) --- #
        actual_rr = abs(tp_price - entry_price) / sl_dist
        min_rr    = float(cfg.get('min_rr', 2.0))
        if actual_rr < min_rr:
            self.log.info(
                f"[{instrument}] RR={actual_rr:.2f} < {min_rr:.1f} → スキップ"
            )
            return

        # --- ポジションサイズ計算 --- #
        # lot = (総資金 × risk_pct) / (entry - SL) / lot_divisor
        risk_pct    = float(cfg.get('risk_pct',    0.02))
        lot_divisor = float(cfg.get('lot_divisor', 1.0))
        max_units   = cfg.get('max_units', None)

        units = calc_units(balance, risk_pct, entry_price, sl_price,
                           lot_divisor, max_units)
        if units <= 0:
            self.log.warning(f"[{instrument}] ユニット数=0 → スキップ")
            return

        order_units = units if sig == 'long' else -units

        self.log.info(
            f"[{instrument}] 発注準備:\n"
            f"  方向:  {sig.upper()}\n"
            f"  Units: {abs(order_units):,}\n"
            f"  Entry: {entry_price:.3f}\n"
            f"  SL:    {sl_price:.3f}  (距離: {sl_dist:.3f})\n"
            f"  TP:    {tp_price:.3f}  (RR: {actual_rr:.2f})\n"
            f"  Risk:  ${balance * risk_pct:,.0f} "
            f"({risk_pct:.0%} of ${balance:,.0f})"
        )

        # --- 発注 --- #
        try:
            resp = self.broker.place_market_order(
                instrument=instrument,
                units=order_units,
                sl_price=sl_price,
                tp_price=tp_price,
            )
            tx  = resp.get('orderFillTransaction') or resp.get('orderCreateTransaction', {})
            oid = tx.get('id', 'N/A')
            fpx = tx.get('price', 'N/A')
            self.log.info(
                f"[{instrument}] ✓ 発注完了 | OrderID={oid} | 約定={fpx}"
            )
        except Exception as e:
            self.log.error(f"[{instrument}] 発注エラー: {e}")


# ------------------------------------------------------------------ #
#  エントリーポイント                                                 #
# ------------------------------------------------------------------ #

def main():
    root = Path(__file__).parent.parent
    config_path = root / 'config.yaml'

    if not config_path.exists():
        print(f"[ERROR] config.yaml が見つかりません: {config_path}")
        print("        setup.bat を実行するか config.example.yaml をコピーしてください。")
        sys.exit(1)

    config = load_config(str(config_path))

    log_dir   = config.get('bot', {}).get('log_dir',   'logs')
    log_level = config.get('bot', {}).get('log_level', 'INFO')
    setup_logging(str(root / log_dir), log_level)

    # PIDファイル (stop_bot_v2.bat がプロセスを終了するために使用)
    pid_file = root / 'bot_v2.pid'
    pid_file.write_text(str(os.getpid()))

    try:
        bot = TradingBotV2(config)
        bot.run()
    finally:
        if pid_file.exists():
            pid_file.unlink()


if __name__ == '__main__':
    main()
