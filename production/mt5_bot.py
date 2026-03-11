"""
mt5_bot.py - YAGAMI改 本番MT5自動取引ボット（Exness対応）
==========================================================
【セットアップ】
  1. .env を作成して MT5_LOGIN / MT5_PASSWORD / MT5_SERVER を設定
  2. MetaTrader5 ターミナルを起動してログイン済みにしておく
  3. python mt5_bot.py を実行

【採用銘柄・ロジック（2026/3/9 確定）】
  XAUUSD  : Gold Logic（日足EMA20 + E2）  OOS PF=3.44
  GBPUSD  : Gold Logic                    OOS PF=2.29
  AUDUSD  : Gold Logic                    OOS PF=2.19
  NZDUSD  : Gold Logic                    OOS PF=1.78（様子見）
  SPX500  : Gold Logic（SP500）           OOS PF=2.03

【資金管理】
  - AdaptiveRiskManager: DD連動動的リスク逓減
  - 基本リスク: EQUITY_JPY × BASE_RISK_PCT
  - 半利確: 1R到達でポジション50%決済 → SLをBEへ
  - 最大同時ポジション: MAX_POSITIONS

【ループ間隔】60秒
"""
from __future__ import annotations

import os, sys, csv, json, time, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── MetaTrader5（Windows専用ライブラリ） ─────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 library not found. Running in DRY_RUN mode.")

# ── dotenv ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── シグナルエンジン ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from production.signal_engine import (
    build_indicators, check_signal, is_spike_bar, calc_entry_price, _calc_atr,
)
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ────────────────────────────────────────────────────────────────
# 設定（.envから読み込み）
# ────────────────────────────────────────────────────────────────
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN",    "0"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER",   "Exness-MT5Real8")

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
LOG_DIR         = Path(os.environ.get("LOG_DIR", Path(__file__).parent / "trade_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_POSITIONS   = int(os.environ.get("MAX_POSITIONS", "3"))
DRY_RUN         = os.environ.get("DRY_RUN", "false").lower() == "true"

LOOP_INTERVAL   = 60   # 秒

# ── ログ設定 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# 採用銘柄設定
# ────────────────────────────────────────────────────────────────
# mt5_sym: ExnessでのMT5銘柄名（環境によって末尾に'm'などが付く場合あり）
# risk_mult: 銘柄固有のリスク乗数（NZDUSD は様子見のため0.5×）
SYMBOLS = {
    "XAUUSD": {"mt5_sym": "XAUUSDm", "risk_mult": 1.0, "pip": 0.01,   "spread_pips": 5.2, "qt": "B"},
    "GBPUSD": {"mt5_sym": "GBPUSDm", "risk_mult": 1.0, "pip": 0.0001, "spread_pips": 0.1, "qt": "B"},
    "AUDUSD": {"mt5_sym": "AUDUSDm", "risk_mult": 1.0, "pip": 0.0001, "spread_pips": 0.0, "qt": "B"},
    "NZDUSD": {"mt5_sym": "NZDUSDm", "risk_mult": 0.5, "pip": 0.0001, "spread_pips": 0.5, "qt": "B"},
    "SPX500": {"mt5_sym": "SP500m",  "risk_mult": 1.0, "pip": 0.1,    "spread_pips": 0.1, "qt": "D"},
}

# ── リスク管理（シンプル3段階） ──────────────────────────────────
# DD < 5%  → 3.0%（好調・攻め）
# DD 5-10% → 2.5%（標準）
# DD ≥ 10% → 2.0%（守り）
_peak_equity: float = 0.0

def get_risk_pct(equity: float) -> float:
    global _peak_equity
    if equity > _peak_equity:
        _peak_equity = equity
    if _peak_equity <= 0:
        return 0.025
    dd = (_peak_equity - equity) / _peak_equity
    if dd >= 0.10:
        return 0.020
    if dd >= 0.05:
        return 0.025
    return 0.030

# ── ポジション・シグナル状態管理 ─────────────────────────────────
# pending_signals: { "XAUUSD": { signal_dict, "half_done": False }, ... }
pending_signals: dict[str, dict] = {}

# open_positions: { ticket_id: { "sym", "dir", "ep", "sl", "tp", "risk",
#                                "lots_total", "lots_remaining", "half_done",
#                                "peak_equity_at_entry" } }
open_positions: dict[int, dict] = {}


# ════════════════════════════════════════════════════════════════
# MT5 ユーティリティ
# ════════════════════════════════════════════════════════════════
def mt5_init() -> bool:
    """MT5に接続（ログイン）"""
    if not MT5_AVAILABLE:
        return False
    if MT5_LOGIN == 0:
        log.warning("MT5_LOGIN未設定 → DRY_RUNモードで動作")
        return False
    ok = mt5.initialize(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
        timeout=10000,
    )
    if not ok:
        log.error(f"MT5接続失敗: {mt5.last_error()}")
    return ok


def mt5_shutdown():
    if MT5_AVAILABLE:
        mt5.shutdown()


def get_account_equity() -> float:
    """口座残高（JPY）を取得"""
    if not MT5_AVAILABLE or DRY_RUN:
        return float(os.environ.get("EQUITY_JPY", "1000000"))
    info = mt5.account_info()
    if info is None:
        log.warning("口座情報取得失敗 → デフォルト100万円")
        return 1_000_000.0
    return info.equity


def get_usdjpy_rate() -> float:
    """USDJPYレートを取得"""
    if not MT5_AVAILABLE or DRY_RUN:
        return 150.0
    tick = mt5.symbol_info_tick("USDJPYm")
    if tick is None:
        tick = mt5.symbol_info_tick("USDJPY")
    if tick is None:
        return 150.0
    return (tick.bid + tick.ask) / 2


def fetch_ohlc(mt5_sym: str, tf_mt5: int, n: int) -> pd.DataFrame | None:
    """MT5からOHLCデータを取得してDataFrameに変換"""
    if not MT5_AVAILABLE or DRY_RUN:
        return None
    rates = mt5.copy_rates_from_pos(mt5_sym, tf_mt5, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("timestamp").rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "tick_volume": "volume"
    })
    return df[["open", "high", "low", "close", "volume"]]


def fetch_all_tf(sym: str) -> tuple[pd.DataFrame | None, ...]:
    """1m / 15m / 4h / 1d 足を取得"""
    mt5_sym = SYMBOLS[sym]["mt5_sym"]
    d1m  = fetch_ohlc(mt5_sym, mt5.TIMEFRAME_M1,  500)   # E2判定用
    d15m = fetch_ohlc(mt5_sym, mt5.TIMEFRAME_M15, 300)   # 1H生成用
    d4h  = fetch_ohlc(mt5_sym, mt5.TIMEFRAME_H4,  200)
    d1d  = fetch_ohlc(mt5_sym, mt5.TIMEFRAME_D1,  100)
    return d1m, d15m, d4h, d1d


def current_price(mt5_sym: str) -> float | None:
    """現在のask/midを取得"""
    if not MT5_AVAILABLE or DRY_RUN:
        return None
    tick = mt5.symbol_info_tick(mt5_sym)
    return (tick.bid + tick.ask) / 2 if tick else None


# ════════════════════════════════════════════════════════════════
# ロットサイズ計算（MT5ネイティブ仕様対応）
# ════════════════════════════════════════════════════════════════
def calc_lots(sym: str, sl_distance: float, ep: float,
              equity_jpy: float, usdjpy: float) -> float:
    """
    DD連動3段階リスク（2/2.5/3%）でロット計算し、MT5標準ロット数に変換。

    MT5ロット換算:
      FX / XAUUSD: units / 100,000（1標準ロット = 100,000通貨）
      SPX500: ExnessのSP500は1lot=$10/pt → units / 100
    """
    if sym not in SYMBOL_CONFIG:
        return 0.01

    risk_pct   = get_risk_pct(equity_jpy) * SYMBOLS[sym]["risk_mult"]
    rm         = RiskManager(sym, risk_pct=risk_pct)
    lot_units  = rm.calc_lot(
        equity      = equity_jpy,
        sl_distance = sl_distance,
        ref_price   = ep,
        usdjpy_rate = usdjpy,
    )

    # MT5標準ロットに変換
    cfg = SYMBOLS[sym]
    if cfg["qt"] in ("B", "A"):
        lot_mt5 = lot_units / 100_000
    elif cfg["qt"] == "C":
        lot_mt5 = lot_units / 100_000
    elif cfg["qt"] == "D":
        # SPX500: lot_units = risk_jpy / (sl_distance × usdjpy)
        # ExnessのSP500は1lot=$10/pt → 0.1lot=$1/pt
        # 上式で求まったunits ≈ contracts×100 なので /100
        lot_mt5 = lot_units / 100.0
    else:
        lot_mt5 = lot_units / 100_000

    # MT5最小・刻み・最大に丸める
    if MT5_AVAILABLE:
        info = mt5.symbol_info(SYMBOLS[sym]["mt5_sym"])
        if info:
            step = info.volume_step
            lot_mt5 = max(info.volume_min,
                          min(info.volume_max,
                              round(lot_mt5 / step) * step))

    return round(lot_mt5, 2)


# ════════════════════════════════════════════════════════════════
# 注文 / 決済
# ════════════════════════════════════════════════════════════════
def place_order(sym: str, direction: int, lots: float,
                ep: float, sl: float, tp: float) -> int | None:
    """
    成行注文を発注。
    Returns: ticket_id または None（失敗時）
    """
    mt5_sym = SYMBOLS[sym]["mt5_sym"]
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    request = {
        "action":        mt5.TRADE_ACTION_DEAL,
        "symbol":        mt5_sym,
        "volume":        lots,
        "type":          order_type,
        "sl":            round(sl, 5),
        "tp":            round(tp, 5),
        "type_filling":  mt5.ORDER_FILLING_IOC,
        "comment":       "YAGAMI_GOLD",
        "magic":         20260309,
    }

    if DRY_RUN:
        log.info(f"[DRY_RUN] 発注: {sym} dir={direction} lots={lots} "
                 f"ep={ep:.5f} sl={sl:.5f} tp={tp:.5f}")
        return 99999999   # ダミーチケット

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        log.error(f"注文失敗: {sym} retcode={code}")
        return None

    log.info(f"注文成功: {sym} ticket={result.order} "
             f"lots={lots} fill={result.price:.5f}")
    return result.order


def close_position_partial(ticket: int, lots: float, sym: str) -> bool:
    """ポジションの一部決済（半利確用）"""
    mt5_sym = SYMBOLS[sym]["mt5_sym"]
    pos_list = mt5.positions_get(ticket=ticket) if MT5_AVAILABLE else []
    if not pos_list:
        return False

    pos = pos_list[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(mt5_sym)
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       mt5_sym,
        "volume":       round(lots, 2),
        "type":         close_type,
        "position":     ticket,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment":      "YAGAMI_HALF_EXIT",
        "magic":        20260309,
    }

    if DRY_RUN:
        log.info(f"[DRY_RUN] 半利確: ticket={ticket} lots={lots:.2f}")
        return True

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"半利確失敗: ticket={ticket} retcode={getattr(result,'retcode','?')}")
        return False
    log.info(f"半利確成功: ticket={ticket} lots={lots:.2f}")
    return True


def modify_sl(ticket: int, new_sl: float, tp: float) -> bool:
    """SLをBEに移動"""
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       round(new_sl, 5),
        "tp":       round(tp,     5),
    }
    if DRY_RUN:
        log.info(f"[DRY_RUN] SL修正: ticket={ticket} new_sl={new_sl:.5f}")
        return True
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ════════════════════════════════════════════════════════════════
# 取引ログ
# ════════════════════════════════════════════════════════════════
TRADE_LOG = LOG_DIR / "paper_trades.csv"
TRADE_FIELDS = [
    "trade_id", "pair", "dir", "ep", "sl", "tp", "lots",
    "exit_price", "exit_type", "pnl_pips", "strategy",
    "entry_time", "exit_time", "entry_hour", "risk_pips",
]

def log_trade(row: dict):
    write_header = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════════
# Discord通知
# ════════════════════════════════════════════════════════════════
def notify(msg: str, color: int = 0x00ff00):
    if not DISCORD_WEBHOOK:
        return
    payload = {"embeds": [{"description": msg, "color": color}]}
    try:
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        log.warning(f"Discord通知失敗: {e}")


# ════════════════════════════════════════════════════════════════
# ポジション管理（半利確 + SL→BE）
# ════════════════════════════════════════════════════════════════
def manage_open_positions():
    """
    open_positions を巡回し、1R到達 → 半利確 + SL=BE を実施。
    MT5でクローズ済みのポジションは削除。
    """
    if not MT5_AVAILABLE or DRY_RUN:
        return

    # MT5の現在のオープンポジションと同期
    mt5_tickets = set()
    positions = mt5.positions_get()
    if positions:
        mt5_tickets = {p.ticket for p in positions}

    closed_tickets = [t for t in open_positions if t not in mt5_tickets and t != 99999999]
    for t in closed_tickets:
        pos = open_positions.pop(t)
        log.info(f"ポジションクローズ検出（MT5）: {pos['sym']} ticket={t}")

    # 半利確チェック
    for ticket, pos in list(open_positions.items()):
        if pos.get("half_done"):
            continue
        sym   = pos["sym"]
        price = current_price(SYMBOLS[sym]["mt5_sym"])
        if price is None:
            continue
        ep, risk, direction = pos["ep"], pos["risk"], pos["dir"]
        half_target = ep + direction * risk   # 1R

        hit = (direction == 1 and price >= half_target) or \
              (direction == -1 and price <= half_target)
        if not hit:
            continue

        # 半利確実施
        half_lots = round(pos["lots_total"] * 0.5, 2)
        if close_position_partial(ticket, half_lots, sym):
            pos["half_done"] = True
            pos["lots_remaining"] = pos["lots_total"] - half_lots
            # SL → BE
            modify_sl(ticket, ep, pos["tp"])
            pos["sl"] = ep
            notify(
                f"🔔 半利確\n{sym}  dir={direction:+d}\n"
                f"EP={ep:.5f}  1R={half_target:.5f}\n"
                f"SL→BE={ep:.5f}",
                color=0xffa500
            )


# ════════════════════════════════════════════════════════════════
# シグナルチェックとE2エントリー試行
# ════════════════════════════════════════════════════════════════
def process_symbol(sym: str, now: datetime, equity_jpy: float, usdjpy: float):
    """1銘柄のシグナルチェック→エントリー試行"""
    cfg = SYMBOLS[sym]
    mt5_sym = cfg["mt5_sym"]
    spread  = cfg["spread_pips"] * cfg["pip"]

    # ── データ取得 ──────────────────────────────────────────────
    d1m, d15m, d4h, d1d = fetch_all_tf(sym)
    if d4h is None or d15m is None:
        log.debug(f"{sym}: データ取得失敗")
        return

    df4h, df1d, df1h = build_indicators(d4h, d1d, d15m)
    if df1h is None or len(df1h) < 3:
        return

    # ── シグナルチェック ─────────────────────────────────────────
    sig = check_signal(df1h, df4h, df1d, spread, now)
    if sig is None:
        return

    log.info(f"{sym}: シグナル検出 dir={sig['dir']:+d} "
             f"raw_ep={sig['raw_ep']:.5f} sl={sig['sl']:.5f} tp={sig['tp']:.5f}")

    # ── E2エントリー試行（スパイクフィルター） ──────────────────
    if d1m is None or len(d1m) < 14:
        return

    # 1m足ATR
    atr_1m_series = _calc_atr(d1m)
    atr_1m = atr_1m_series.iloc[-1] if len(atr_1m_series) > 0 else None

    # 直近1m足（現在バー）でスパイク確認
    cur_bar = d1m.iloc[-1]
    if atr_1m and is_spike_bar(cur_bar["high"], cur_bar["low"], atr_1m):
        log.info(f"{sym}: スパイクバーのためE2スキップ (range={cur_bar['high']-cur_bar['low']:.5f})")
        return

    # エントリー価格計算
    ep = calc_entry_price(sig["raw_ep"], sig["dir"], spread)
    sl = sig["sl"]
    tp = sig["tp"]
    risk = abs(ep - sl)

    # ロット計算
    lots = calc_lots(sym, risk, ep, equity_jpy, usdjpy)
    if lots <= 0:
        log.warning(f"{sym}: ロット計算失敗 lots={lots}")
        return

    # ── 発注 ────────────────────────────────────────────────────
    ticket = place_order(sym, sig["dir"], lots, ep, sl, tp)
    if ticket is None:
        return

    entry_time = now.isoformat()
    entry_hour = now.hour

    open_positions[ticket] = {
        "sym":          sym,
        "dir":          sig["dir"],
        "ep":           ep,
        "sl":           sl,
        "tp":           tp,
        "risk":         risk,
        "lots_total":   lots,
        "lots_remaining": lots,
        "half_done":    False,
        "entry_time":   entry_time,
    }

    risk_pips  = round(risk / cfg["pip"], 1)
    risk_pct_used = get_risk_pct(equity_jpy) * cfg["risk_mult"]

    notify(
        f"📈 エントリー\n{sym}  {'BUY' if sig['dir']==1 else 'SELL'}\n"
        f"EP={ep:.5f}  SL={sl:.5f}  TP={tp:.5f}\n"
        f"lots={lots:.2f}  risk={risk_pips:.1f}pips  ({risk_pct_used*100:.1f}%)",
        color=0x00b0ff if sig["dir"] == 1 else 0xff5252
    )

    # ログ（決済時に上書きするためシグナル時点では entry のみ記録）
    log_trade({
        "trade_id":   ticket,
        "pair":       sym,
        "dir":        sig["dir"],
        "ep":         round(ep, 5),
        "sl":         round(sl, 5),
        "tp":         round(tp, 5),
        "lots":       lots,
        "exit_price": "",
        "exit_type":  "open",
        "pnl_pips":   "",
        "strategy":   "gold_logic",
        "entry_time": entry_time,
        "exit_time":  "",
        "entry_hour": entry_hour,
        "risk_pips":  risk_pips,
    })

    log.info(f"{sym}: 注文完了 ticket={ticket} lots={lots:.2f} risk={risk_pips:.1f}pips")


# ════════════════════════════════════════════════════════════════
# メインループ
# ════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("YAGAMI改 MT5ボット起動")
    log.info(f"  銘柄: {list(SYMBOLS.keys())}")
    log.info(f"  DRY_RUN: {DRY_RUN}")
    log.info(f"  MAX_POSITIONS: {MAX_POSITIONS}")
    log.info(f"  BASE_RISK_PCT: {BASE_RISK_PCT*100:.1f}%")
    log.info("=" * 60)

    # MT5接続
    connected = mt5_init()
    if not connected and not DRY_RUN:
        log.error("MT5接続失敗 → 終了")
        sys.exit(1)

    notify("🤖 YAGAMI改 ボット起動\n" + "  ".join(SYMBOLS.keys()), color=0x7289da)

    while True:
        try:
            loop_start = time.time()
            now = datetime.now(timezone.utc)

            equity_jpy = get_account_equity()
            usdjpy     = get_usdjpy_rate()

            log.debug(f"[ループ] equity={equity_jpy:,.0f}JPY  USDJPY={usdjpy:.2f}")

            # 1. オープンポジション管理（半利確・SL移動）
            manage_open_positions()

            # 2. 新規シグナルチェック（ポジション上限未満の場合）
            if len(open_positions) < MAX_POSITIONS:
                for sym in SYMBOLS:
                    # 同一銘柄で既にポジションあればスキップ
                    has_pos = any(p["sym"] == sym for p in open_positions.values())
                    if has_pos:
                        continue
                    try:
                        process_symbol(sym, now, equity_jpy, usdjpy)
                    except Exception as e:
                        log.exception(f"{sym} 処理エラー: {e}")

            # 60秒ごとに実行
            elapsed = time.time() - loop_start
            sleep_sec = max(0, LOOP_INTERVAL - elapsed)
            log.debug(f"スリープ {sleep_sec:.1f}秒")
            time.sleep(sleep_sec)

        except KeyboardInterrupt:
            log.info("手動停止")
            break
        except Exception as e:
            log.exception(f"メインループエラー: {e}")
            time.sleep(10)

    mt5_shutdown()
    notify("🛑 YAGAMI改 ボット停止", color=0xff0000)
    log.info("ボット停止")


if __name__ == "__main__":
    main()
