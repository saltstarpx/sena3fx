"""
mt5_live_trader.py
==================
YAGAMI改 — Exness × MT5 本番ライブトレーダー
Phase2: 1.0%リスク × 7銘柄

【動作環境】
  Windows VPS + MetaTrader5 ターミナル起動済み
  Python 3.10+ (Windows)
  pip install MetaTrader5 pandas numpy

【起動方法】
  python mt5_live_trader.py

【アーキテクチャ】
  60秒ごとにポーリング → 1H足更新検知 → シグナル判定 → MT5で発注
  ポジション監視: 1R到達でハーフクローズ → SLをBEへ移動

【採用7銘柄 / Phase2 (1.0%)】
  USDJPY   オーパーツYAGAMI (Logic-C)  → Sharpe=6.18
  GBPUSD   GOLDYAGAMI      (Logic-A)  → Sharpe=7.12
  EURUSD   オーパーツYAGAMI (Logic-C)  → Sharpe=6.18
  USDCAD   GOLDYAGAMI      (Logic-A)  → Sharpe=5.62
  NZDUSD   GOLDYAGAMI      (Logic-A)  → Sharpe=5.45
  XAUUSD   GOLDYAGAMI      (Logic-A)  → Sharpe=3.42
  AUDUSD   ADX+Streak      (Logic-B)  → Sharpe=3.66

【本番運用ルール（運用ルール書 準拠）】
  1トレードリスク : 1.0%（Phase2）
  最大同時保有    : 20ポジション
  USD同方向上限  : 3本まで（4本目以降見送り）
  日次停止        : 累計 -2R 到達
  週次停止        : 累計 -4R 到達
  月次DD停止      : -8% → Phase降格アラート
"""
import sys
import time
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 パッケージが見つかりません")
    print("  pip install MetaTrader5")
    sys.exit(1)

# ── 設定 ─────────────────────────────────────────────────────────
RISK_PCT        = 0.010   # Phase2: 1.0% / 銘柄
INIT_CASH       = 1_000_000
RR_RATIO        = 2.5
HALF_R          = 1.0
POLL_SEC        = 60      # ポーリング間隔（秒）
MAX_POSITIONS   = 20      # 最大同時保有数
USD_SAME_DIR_MAX = 3      # USD同方向エクスポージャ上限

# 日次・週次・月次ストップ
DAILY_STOP_R    = -2.0
WEEKLY_STOP_R   = -4.0
MONTHLY_DD_ALERT = -0.08  # -8% でアラート

# ログ
LOG_DIR = Path(__file__).parent.parent / "trade_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"mt5_live_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── 採用銘柄設定 ──────────────────────────────────────────────────
SYMBOLS = {
    "USDJPY": {"logic": "C", "pip": 0.01,   "spread_max": 0.5,  "priority": 1, "usd_pair": True,  "usd_side": "quote"},
    "GBPUSD": {"logic": "A", "pip": 0.0001, "spread_max": 0.3,  "priority": 2, "usd_pair": True,  "usd_side": "base_inv"},
    "EURUSD": {"logic": "C", "pip": 0.0001, "spread_max": 0.3,  "priority": 3, "usd_pair": True,  "usd_side": "base_inv"},
    "USDCAD": {"logic": "A", "pip": 0.0001, "spread_max": 0.5,  "priority": 4, "usd_pair": True,  "usd_side": "base"},
    "NZDUSD": {"logic": "A", "pip": 0.0001, "spread_max": 0.5,  "priority": 5, "usd_pair": True,  "usd_side": "base_inv"},
    "XAUUSD": {"logic": "A", "pip": 0.01,   "spread_max": 1.0,  "priority": 6, "usd_pair": False, "usd_side": None},
    "AUDUSD": {"logic": "B", "pip": 0.0001, "spread_max": 0.5,  "priority": 7, "usd_pair": True,  "usd_side": "base_inv"},
}

# ── シグナル生成定数 ──────────────────────────────────────────────
KLOW_THR        = 0.0015
EMA_DIST_MIN    = 1.0
TOL_ATR         = 0.30
ADX_MIN         = 20
STREAK_MIN      = 4
E2_SPIKE_ATR    = 2.0

# ── 状態ファイル ─────────────────────────────────────────────────
STATE_FILE = LOG_DIR / "live_state.json"


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week  = datetime.now(timezone.utc).strftime("%Y-W%W")
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    return {
        "daily_r":    {"date": today, "r": 0.0},
        "weekly_r":   {"week": week,  "r": 0.0},
        "monthly_dd": {"month": month, "peak_eq": INIT_CASH, "cur_eq": INIT_CASH},
        "be_done":    {},   # ticket -> True/False
        "half_done":  {},   # ticket -> True/False
    }

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)


# ── MT5 接続 ─────────────────────────────────────────────────────
def connect_mt5():
    if not mt5.initialize():
        logger.error(f"MT5 初期化失敗: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        logger.error("口座情報取得失敗。MT5 ターミナルを起動してログイン済みか確認してください")
        return False
    logger.info(f"MT5 接続成功: {info.login} / {info.server} / 残高={info.balance:.2f} {info.currency}")
    return True


# ── データ取得（MT5から） ──────────────────────────────────────────
def get_rates(sym, tf, n):
    """MT5からOHLCデータ取得"""
    rates = mt5.copy_rates_from_pos(sym, tf, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]

def get_1h(sym):  return get_rates(sym, mt5.TIMEFRAME_H1, 200)
def get_4h(sym):  return get_rates(sym, mt5.TIMEFRAME_H4, 200)
def get_1d(sym):  return get_rates(sym, mt5.TIMEFRAME_D1, 100)
def get_1m(sym):  return get_rates(sym, mt5.TIMEFRAME_M1, 30)


# ── インジケーター ────────────────────────────────────────────────
def atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def ema(df, n=20):
    return df["close"].ewm(span=n, adjust=False).mean()

def adx(df, n=14):
    h = df["high"]; l = df["low"]
    pdm = h.diff().clip(lower=0); mdm = (-l.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0.0; mdm[mdm < pdm] = 0.0
    a1  = atr(df, 1).ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / a1.replace(0, np.nan)
    dim = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / a1.replace(0, np.nan)
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)


# ── シグナル生成 ──────────────────────────────────────────────────
def check_signal(sym):
    """
    最新1H足で二番底/天井パターン + 4Hフィルターを確認
    戻り値: {"dir": 1/-1, "sl": ..., "tp": ..., "risk": ...} or None
    """
    logic = SYMBOLS[sym]["logic"]
    cfg   = SYMBOLS[sym]

    d1h = get_1h(sym)
    d4h = get_4h(sym)
    if d1h is None or d4h is None or len(d1h) < 20 or len(d4h) < 20:
        return None

    # ── 4H インジケーター ──────────────────────────────────────
    d4h["atr"]   = atr(d4h, 14)
    d4h["ema20"] = ema(d4h, 20)
    d4h["trend"] = np.where(d4h["close"] > d4h["ema20"], 1, -1)
    d4h["adx"]   = adx(d4h, 14)

    h4l   = d4h.iloc[-2]   # 直前確定4H足
    trend = int(h4l["trend"])
    h4atr = float(h4l["atr"])

    if np.isnan(h4atr) or h4atr <= 0:
        return None

    # ── Logic-A: 日足EMA20 ────────────────────────────────────
    if logic == "A":
        d1d = get_1d(sym)
        if d1d is None or len(d1d) < 25:
            return None
        d1d["ema20"]   = ema(d1d, 20)
        d1d["trend1d"] = np.where(d1d["close"] > d1d["ema20"], 1, -1)
        if int(d1d.iloc[-2]["trend1d"]) != trend:
            return None

    # ── Logic-B: ADX + Streak ─────────────────────────────────
    elif logic == "B":
        if float(h4l.get("adx", 0)) < ADX_MIN:
            return None
        streak = d4h["trend"].iloc[-STREAK_MIN-1:-1].values
        if not all(int(t) == trend for t in streak):
            return None

    # ── KMID / KLOW ───────────────────────────────────────────
    kmid_ok = (trend == 1 and h4l["close"] > h4l["open"]) or \
              (trend == -1 and h4l["close"] < h4l["open"])
    if not kmid_ok:
        return None

    lower_wick = min(h4l["open"], h4l["close"]) - h4l["low"]
    if h4l["open"] > 0 and lower_wick / h4l["open"] >= KLOW_THR:
        return None

    # ── EMA距離（Logic-C以外） ────────────────────────────────
    if logic != "C":
        ema_dist = abs(h4l["close"] - h4l["ema20"])
        if np.isnan(h4l["atr"]) or h4l["atr"] <= 0 or ema_dist < h4l["atr"] * EMA_DIST_MIN:
            return None

    # ── 1H 二番底/天井パターン ────────────────────────────────
    d1h["atr"] = atr(d1h, 14)
    h1_last    = d1h.iloc[-1]   # 現在の未確定足（パターン確認後足）
    h1_p1      = d1h.iloc[-2]   # 1つ前確定
    h1_p2      = d1h.iloc[-3]   # 2つ前確定

    atr1h = float(h1_last["atr"])
    if np.isnan(atr1h) or atr1h <= 0:
        return None

    v1 = float(h1_p2["low"])  if trend == 1 else float(h1_p2["high"])
    v2 = float(h1_p1["low"])  if trend == 1 else float(h1_p1["high"])
    if abs(v1 - v2) > atr1h * TOL_ATR:
        return None

    # ── Logic-C: 確認足チェック（前足が方向一致実体）────────────
    if logic == "C":
        if trend == 1 and h1_p1["close"] <= h1_p1["open"]:
            return None
        if trend == -1 and h1_p1["close"] >= h1_p1["open"]:
            return None

    # ── スプレッド確認 ────────────────────────────────────────
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return None
    spread_pips = abs(tick.ask - tick.bid) / cfg["pip"]
    if spread_pips > cfg["spread_max"] * 3:
        logger.warning(f"{sym} スプレッド異常: {spread_pips:.1f}pips → スキップ")
        return None

    entry = tick.ask if trend == 1 else tick.bid
    sl    = (min(v1, v2) - atr1h * 0.15) if trend == 1 else (max(v1, v2) + atr1h * 0.15)
    risk  = abs(entry - sl)

    if risk <= 0 or risk > h4atr * 2:
        return None

    tp = entry + trend * risk * RR_RATIO

    return {
        "sym":   sym,
        "dir":   trend,
        "entry": entry,
        "sl":    sl,
        "tp":    tp,
        "risk":  risk,
        "logic": logic,
    }


# ── ロット計算 ────────────────────────────────────────────────────
def calc_lot(sym, risk_price, entry_price, risk_pct):
    info     = mt5.account_info()
    balance  = info.balance
    risk_jpy = balance * risk_pct

    sym_info = mt5.symbol_info(sym)
    if sym_info is None:
        return 0.0
    contract = sym_info.trade_contract_size  # 通常 100000

    # 口座通貨（JPY）に換算
    # JPY建て (USDJPY等)
    tick = mt5.symbol_info_tick(sym)
    quote_price = tick.ask

    if "JPY" in sym:
        pips_risk = risk_price / 0.01
        lot = risk_jpy / (pips_risk * 0.01 * contract / quote_price)
    elif sym.startswith("XAU"):
        # XAUUSDはUSD建て → JPY変換
        usdjpy_tick = mt5.symbol_info_tick("USDJPY")
        usdjpy_rate = usdjpy_tick.bid if usdjpy_tick else 150.0
        lot = risk_jpy / (risk_price * contract * usdjpy_rate)
    else:
        # USD建て通貨ペア
        usdjpy_tick = mt5.symbol_info_tick("USDJPY")
        usdjpy_rate = usdjpy_tick.bid if usdjpy_tick else 150.0
        lot = risk_jpy / (risk_price * contract * usdjpy_rate)

    # ロットの正規化
    step = sym_info.volume_step
    lot  = max(sym_info.volume_min, round(lot / step) * step)
    lot  = min(lot, sym_info.volume_max)
    return round(lot, 2)


# ── リスクチェック ────────────────────────────────────────────────
def check_risk_rules(sym, direction, state):
    """発注前リスクルール確認"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week  = datetime.now(timezone.utc).strftime("%Y-W%W")

    # 日付リセット
    if state["daily_r"]["date"] != today:
        state["daily_r"] = {"date": today, "r": 0.0}
    if state["weekly_r"]["week"] != week:
        state["weekly_r"] = {"week": week, "r": 0.0}

    # 日次ストップ
    if state["daily_r"]["r"] <= DAILY_STOP_R:
        logger.info(f"  日次ストップ発動中 ({state['daily_r']['r']:.1f}R) → 見送り")
        return False

    # 週次ストップ
    if state["weekly_r"]["r"] <= WEEKLY_STOP_R:
        logger.info(f"  週次ストップ発動中 ({state['weekly_r']['r']:.1f}R) → 見送り")
        return False

    # 月次DD確認
    info = mt5.account_info()
    peak_eq = state["monthly_dd"].get("peak_eq", info.balance)
    cur_dd  = (peak_eq - info.balance) / peak_eq if peak_eq > 0 else 0
    if cur_dd >= abs(MONTHLY_DD_ALERT):
        logger.warning(f"  ⚠️ 月次DD警告: -{cur_dd*100:.1f}% → Phase降格を検討してください")

    # 最大同時保有確認
    positions = mt5.positions_get()
    n_pos = len(positions) if positions else 0
    if n_pos >= MAX_POSITIONS:
        logger.info(f"  最大同時保有({MAX_POSITIONS})到達 → 見送り")
        return False

    # USD同方向エクスポージャ確認
    cfg = SYMBOLS[sym]
    if cfg["usd_pair"] and cfg["usd_side"]:
        usd_same_dir = 0
        for pos in (positions or []):
            pos_sym = pos.symbol
            if pos_sym not in SYMBOLS:
                continue
            pos_cfg = SYMBOLS[pos_sym]
            if not pos_cfg["usd_pair"]:
                continue
            pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            # USD方向を統一（USD売り or USD買いで比較）
            usd_dir_pos = pos_dir if pos_cfg["usd_side"] == "base" else -pos_dir
            usd_dir_new = direction if cfg["usd_side"] == "base" else -direction
            if usd_dir_pos == usd_dir_new:
                usd_same_dir += 1
        if usd_same_dir >= USD_SAME_DIR_MAX:
            logger.info(f"  USD同方向エクスポージャ上限({USD_SAME_DIR_MAX}) → 見送り")
            return False

    return True


# ── 発注 ─────────────────────────────────────────────────────────
def place_order(signal, risk_pct, state):
    sym   = signal["sym"]
    d     = signal["dir"]
    ep    = signal["entry"]
    sl    = signal["sl"]
    tp    = signal["tp"]
    risk  = signal["risk"]

    lot = calc_lot(sym, risk, ep, risk_pct)
    if lot <= 0:
        logger.error(f"  {sym} ロット計算失敗")
        return None

    order_type = mt5.ORDER_TYPE_BUY if d == 1 else mt5.ORDER_TYPE_SELL
    tick       = mt5.symbol_info_tick(sym)
    price      = tick.ask if d == 1 else tick.bid

    request = {
        "action":        mt5.TRADE_ACTION_DEAL,
        "symbol":        sym,
        "volume":        lot,
        "type":          order_type,
        "price":         price,
        "sl":            round(sl, 5),
        "tp":            round(tp, 5),
        "deviation":     20,
        "magic":         20260311,
        "comment":       f"YAGAMI-{signal['logic']}-{signal['sym']}",
        "type_time":     mt5.ORDER_TIME_GTC,
        "type_filling":  mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = result.comment if result else mt5.last_error()
        logger.error(f"  {sym} 発注失敗: {err}")
        return None

    ticket = result.order
    logger.info(
        f"  ✅ 発注成功 | {sym} {'BUY' if d==1 else 'SELL'} | "
        f"Lot={lot} | EP={price:.5f} | SL={sl:.5f} | TP={tp:.5f} | "
        f"Ticket={ticket}"
    )
    state["be_done"][str(ticket)]   = False
    state["half_done"][str(ticket)] = False

    # ログファイルに記録
    _log_trade(sym, signal, lot, price, ticket)
    return ticket


def _log_trade(sym, signal, lot, price, ticket):
    """毎トレード記録（運用ルール書 §10 準拠）"""
    row = {
        "datetime":       datetime.now(timezone.utc).isoformat(),
        "symbol":         sym,
        "logic":          signal["logic"],
        "direction":      "Long" if signal["dir"] == 1 else "Short",
        "entry_price":    price,
        "sl":             signal["sl"],
        "risk_1r":        signal["risk"],
        "lot":            lot,
        "ticket":         ticket,
        "half_close":     "",
        "be_move":        "",
        "final_pnl_r":    "",
        "final_pnl_jpy":  "",
        "slippage":       round(abs(price - signal["entry"]) / SYMBOLS[sym]["pip"], 1),
        "spread_ok":      "yes",
        "news_time":      "no",
        "manual_action":  "",
    }
    log_csv = LOG_DIR / f"trades_{datetime.now().strftime('%Y%m')}.csv"
    header  = not log_csv.exists()
    pd.DataFrame([row]).to_csv(log_csv, mode="a", header=header, index=False)


# ── ポジション管理（ハーフクローズ & BE移動）─────────────────────
def manage_positions(state):
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        ticket = str(pos.ticket)
        sym    = pos.symbol
        if sym not in SYMBOLS:
            continue

        d       = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        ep      = pos.price_open
        sl      = pos.sl
        tp      = pos.tp
        lot     = pos.volume
        risk    = abs(ep - sl) if sl > 0 else 0
        if risk <= 0:
            continue

        tick    = mt5.symbol_info_tick(sym)
        cur_bid = tick.bid
        cur_ask = tick.ask
        cur     = cur_bid if d == 1 else cur_ask

        half_target = ep + d * risk * HALF_R
        half_done   = state["half_done"].get(ticket, False)
        be_done     = state["be_done"].get(ticket, False)

        # ── ハーフクローズ（+1R到達）─────────────────────────
        if not half_done:
            triggered = (d == 1 and cur >= half_target) or (d == -1 and cur <= half_target)
            if triggered:
                half_lot = round(lot / 2, 2)
                if half_lot >= mt5.symbol_info(sym).volume_min:
                    close_type = mt5.ORDER_TYPE_SELL if d == 1 else mt5.ORDER_TYPE_BUY
                    close_price = cur_bid if d == 1 else cur_ask
                    req = {
                        "action":       mt5.TRADE_ACTION_DEAL,
                        "symbol":       sym,
                        "volume":       half_lot,
                        "type":         close_type,
                        "position":     pos.ticket,
                        "price":        close_price,
                        "deviation":    20,
                        "magic":        20260311,
                        "comment":      "YAGAMI-HALF",
                        "type_time":    mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        state["half_done"][ticket] = True
                        logger.info(f"  ✂️  ハーフクローズ | {sym} | Lot={half_lot} | Price={close_price:.5f}")
                    else:
                        logger.error(f"  ハーフクローズ失敗 {sym}: {res.comment if res else mt5.last_error()}")

        # ── BE移動（ハーフクローズ後）─────────────────────────
        if half_done and not be_done:
            new_sl = round(ep, 5)
            if (d == 1 and sl < ep) or (d == -1 and sl > ep):
                req = {
                    "action":   mt5.TRADE_ACTION_SLTP,
                    "symbol":   sym,
                    "position": pos.ticket,
                    "sl":       new_sl,
                    "tp":       tp,
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    state["be_done"][ticket] = True
                    logger.info(f"  🔒 BE移動完了 | {sym} | SL → EP={new_sl:.5f}")
                else:
                    logger.error(f"  BE移動失敗 {sym}: {res.comment if res else mt5.last_error()}")


# ── 決済後R計算（日次・週次ストップ用）──────────────────────────
def update_r_after_close(state):
    """直近決済トレードのR損益を集計してstateに反映"""
    today = datetime.now(timezone.utc)
    from_dt = int((today - timedelta(hours=24)).timestamp())
    deals = mt5.history_deals_get(from_dt, int(today.timestamp()))
    if deals is None:
        return

    today_str = today.strftime("%Y-%m-%d")
    week_str  = today.strftime("%Y-W%W")
    if state["daily_r"]["date"] != today_str:
        state["daily_r"] = {"date": today_str, "r": 0.0}
    if state["weekly_r"]["week"] != week_str:
        state["weekly_r"] = {"week": week_str, "r": 0.0}

    daily_pnl = sum(d.profit for d in deals if d.entry == 1)  # entry=1は決済
    # 簡易: JPY損益 / (残高×リスク%) ≈ R
    info = mt5.account_info()
    if info and info.balance > 0:
        r_today = daily_pnl / (info.balance * RISK_PCT)
        state["daily_r"]["r"]  = r_today
        state["weekly_r"]["r"] += r_today  # 累積（週次リセットで使う）

    # 月次DD更新
    month_str = today.strftime("%Y-%m")
    if state["monthly_dd"].get("month") != month_str:
        state["monthly_dd"] = {"month": month_str, "peak_eq": info.balance, "cur_eq": info.balance}
    else:
        if info:
            if info.balance > state["monthly_dd"]["peak_eq"]:
                state["monthly_dd"]["peak_eq"] = info.balance
            state["monthly_dd"]["cur_eq"] = info.balance


# ── 1H足更新検知 ──────────────────────────────────────────────────
_last_1h_bar = {}

def is_new_1h_bar(sym):
    """1時間足の新バーが開いたかチェック"""
    d1h = get_1h(sym)
    if d1h is None or len(d1h) < 2:
        return False
    latest = d1h.index[-1]
    if _last_1h_bar.get(sym) != latest:
        _last_1h_bar[sym] = latest
        return True
    return False


# ── メインループ ─────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  YAGAMI改 MT5 ライブトレーダー 起動")
    logger.info(f"  Phase2: {RISK_PCT*100:.1f}%リスク × 7銘柄")
    logger.info("=" * 60)

    if not connect_mt5():
        sys.exit(1)

    state = load_state()

    # シンボル確認
    for sym in SYMBOLS:
        info = mt5.symbol_info(sym)
        if info is None:
            logger.warning(f"  {sym} がMT5で見つかりません。銘柄リストに追加してください")

    logger.info("  監視開始... Ctrl+C で停止")

    try:
        while True:
            now = datetime.now(timezone.utc)
            logger.debug(f"  Poll: {now.strftime('%H:%M:%S')} UTC")

            # ── 既存ポジション管理 ──────────────────────────
            manage_positions(state)

            # ── R損益更新 ───────────────────────────────────
            update_r_after_close(state)

            # ── 各銘柄シグナルチェック ──────────────────────
            for tgt_sym in sorted(SYMBOLS, key=lambda s: SYMBOLS[s]["priority"]):
                if not is_new_1h_bar(tgt_sym):
                    continue

                logger.info(f"  [{tgt_sym}] 1H足更新検知 → シグナルチェック")

                # リスクルール事前確認（仮direction=1でUSD上限チェック、後で実方向確認）
                signal = check_signal(tgt_sym)
                if signal is None:
                    logger.info(f"  [{tgt_sym}] シグナルなし")
                    continue

                logger.info(f"  [{tgt_sym}] シグナル検出: dir={'Long' if signal['dir']==1 else 'Short'} "
                            f"EP={signal['entry']:.5f} SL={signal['sl']:.5f} TP={signal['tp']:.5f}")

                if not check_risk_rules(tgt_sym, signal["dir"], state):
                    continue

                ticket = place_order(signal, RISK_PCT, state)
                if ticket:
                    save_state(state)
                    time.sleep(2)  # 連続発注防止

            save_state(state)
            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        logger.info("  停止シグナル受信。シャットダウン中...")
    finally:
        save_state(state)
        mt5.shutdown()
        logger.info("  MT5 切断完了")


if __name__ == "__main__":
    main()
