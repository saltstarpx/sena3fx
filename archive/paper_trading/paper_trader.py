"""
paper_trader.py
===============
v76ロジックを使ったOANDAデモ口座ペーパートレード自動実行bot

【実行サイクル】
1分ごとに:
  1. OANDA APIから最新ローソク足を取得（1m/15m/4h）
  2. v76.generate_signals() でシグナルを確認
  3. 新シグナルがあれば注文を発行（デモ口座）
  4. 既存ポジションのSL/TP到達を確認してログ記録
  5. JPY相関リスク: 同方向最大2本制限

【ログ】
- paper_trades.csv  : 1トレード1行（決済時に記録）
- paper_signals.csv : シグナル発生ごとに記録（スキップ含む）
- paper_system.log  : 1分ごとの実行ログ

【Discord通知】
- エントリー時
- 決済時（SL/TP）
- エラー時
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# ── パス設定 ─────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
STRAT_DIR   = BASE_DIR.parent / "strategies"
LOG_DIR     = BASE_DIR / "logs"
sys.path.insert(0, str(STRAT_DIR.parent))

from strategies.yagami_mtf_v76 import generate_signals

# ── OANDA設定 ─────────────────────────────────────────
API_TOKEN  = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
ACCOUNT_ID = "101-009-38652105-001"
BASE_URL   = "https://api-fxpractice.oanda.com"
HEADERS    = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# ── Discord設定 ───────────────────────────────────────
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1329418335259197490/59-rzRn2tvHmvMetqMlJPtoo4CApLk3yGoBZoTfUexmXQzrUrTBI1X8sL7RbFfvoQG5k"

# ── 取引設定 ─────────────────────────────────────────
PAIRS = {
    "USDJPY": {"oanda": "USD_JPY", "spread": 0.4, "units": 1000},
    "EURJPY": {"oanda": "EUR_JPY", "spread": 1.1, "units": 1000},
    "GBPJPY": {"oanda": "GBP_JPY", "spread": 1.5, "units": 1000},
}
MAX_SAME_DIR = 2   # 同方向最大保有数（JPY相関リスク対策）
RR_RATIO     = 2.5
CANDLE_COUNT = 200  # 取得するローソク足の本数

# ── ロガー設定 ─────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "paper_system.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ── CSVログ初期化 ─────────────────────────────────────
TRADE_LOG   = LOG_DIR / "paper_trades.csv"
SIGNAL_LOG  = LOG_DIR / "paper_signals.csv"

if not TRADE_LOG.exists():
    pd.DataFrame(columns=[
        "entry_time", "exit_time", "pair", "dir", "ep", "sl", "tp",
        "risk", "spread", "tf", "exit_price", "exit_type", "pnl",
        "oanda_order_id", "slippage"
    ]).to_csv(TRADE_LOG, index=False)

if not SIGNAL_LOG.exists():
    pd.DataFrame(columns=[
        "signal_time", "pair", "dir", "tf", "ep", "sl", "tp", "status"
    ]).to_csv(SIGNAL_LOG, index=False)

# ── ポジション管理（メモリ） ──────────────────────────
open_positions = {}  # key: oanda_trade_id, value: dict

# 定時レポート送信済みフラグ（時刻重複防止）
_last_report_hour = -1

# ── Discord通知 ───────────────────────────────────────

def send_discord(content: str, embeds: list = None):
    """Discordにメッセージを送信する"""
    payload = {"content": content}
    if embeds:
        payload["embeds"] = embeds
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        if r.status_code not in (200, 204):
            logger.warning(f"Discord通知失敗: {r.status_code} {r.text[:100]}")
    except Exception as e:
        logger.warning(f"Discord通知エラー: {e}")


def notify_entry(pair, dir_, ep, sl, tp, tf, risk, slippage):
    """エントリー通知"""
    dir_str  = "🟢 LONG" if dir_ > 0 else "🔴 SHORT"
    rr_pips  = risk * 100 * RR_RATIO
    sl_pips  = risk * 100
    now_jst  = datetime.now(timezone.utc).strftime("%m/%d %H:%M") + " UTC"
    color    = 0x00c853 if dir_ > 0 else 0xd50000  # 緑/赤

    embed = {
        "title": f"📈 エントリー: {pair} {dir_str}",
        "color": color,
        "fields": [
            {"name": "EP",       "value": f"`{ep:.3f}`",              "inline": True},
            {"name": "SL",       "value": f"`{sl:.3f}` (-{sl_pips:.1f}pips)", "inline": True},
            {"name": "TP",       "value": f"`{tp:.3f}` (+{rr_pips:.1f}pips)", "inline": True},
            {"name": "足種",     "value": f"`{tf}`",                  "inline": True},
            {"name": "スリッページ", "value": f"`{slippage:+.2f}pips`", "inline": True},
            {"name": "時刻",     "value": now_jst,                    "inline": True},
        ],
        "footer": {"text": "sena3fx paper trading v76"}
    }
    send_discord("", embeds=[embed])
    logger.info(f"Discord通知送信: エントリー {pair}")


def notify_close(pair, dir_, ep, exit_price, exit_type, pnl, tf):
    """決済通知"""
    if exit_type == "TP":
        emoji = "✅"
        color = 0x00c853
    elif exit_type == "SL":
        emoji = "❌"
        color = 0xd50000
    else:
        emoji = "⚪"
        color = 0x9e9e9e

    dir_str = "LONG" if dir_ > 0 else "SHORT"
    pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.1f}pips"
    now_jst = datetime.now(timezone.utc).strftime("%m/%d %H:%M") + " UTC"

    embed = {
        "title": f"{emoji} 決済: {pair} {dir_str} [{exit_type}]",
        "color": color,
        "fields": [
            {"name": "損益",     "value": f"**`{pnl_str}`**",        "inline": True},
            {"name": "EP",       "value": f"`{ep:.3f}`",              "inline": True},
            {"name": "決済価格", "value": f"`{exit_price:.3f}`",      "inline": True},
            {"name": "足種",     "value": f"`{tf}`",                  "inline": True},
            {"name": "時刻",     "value": now_jst,                    "inline": True},
        ],
        "footer": {"text": "sena3fx paper trading v76"}
    }
    send_discord("", embeds=[embed])
    logger.info(f"Discord通知送信: 決済 {pair} {pnl_str}")


def notify_error(msg: str):
    """エラー通知（重大なもののみ）"""
    send_discord(f"⚠️ **エラー**: {msg}")


def send_daily_report():
    """定時レポートをDiscordに送信する（朝9時・夜21時 JST）"""
    now_utc = datetime.now(timezone.utc)
    now_jst_str = (now_utc.strftime("%Y/%m/%d %H:%M")) + " UTC"

    # OANDA残高取得
    try:
        r = requests.get(
            f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/summary",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            d = r.json()["account"]
            balance    = float(d["balance"])
            unrealized = float(d.get("unrealizedPL", 0))
            nav        = float(d.get("NAV", balance))
            open_count = int(d["openPositionCount"])
        else:
            balance = unrealized = nav = 0.0
            open_count = 0
    except Exception as e:
        logger.warning(f"定時レポート: 残高取得失敗 {e}")
        balance = unrealized = nav = 0.0
        open_count = 0

    # トレードログ集計
    try:
        df = pd.read_csv(TRADE_LOG)
        total_trades = len(df)
        if total_trades > 0:
            wins       = len(df[df["pnl"] > 0])
            losses     = len(df[df["pnl"] <= 0])
            win_rate   = wins / total_trades * 100
            total_pnl  = df["pnl"].sum()
            avg_win    = df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
            avg_loss   = df[df["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
            pf         = abs(df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] <= 0]["pnl"].sum()) if losses > 0 else 999
        else:
            wins = losses = 0
            win_rate = total_pnl = avg_win = avg_loss = 0.0
            pf = 0.0
    except Exception:
        total_trades = wins = losses = 0
        win_rate = total_pnl = avg_win = avg_loss = pf = 0.0

    # オープンポジション一覧
    pos_lines = ""
    if open_positions:
        for tid, pos in open_positions.items():
            dir_str = "LONG" if pos["dir"] > 0 else "SHORT"
            pos_lines += f"  • {pos['pair']} {dir_str} ep={pos['ep']:.3f} sl={pos['sl']:.3f} tp={pos['tp']:.3f}\n"
    else:
        pos_lines = "  なし"

    color = 0x1565c0  # 青（定時レポート）
    embed = {
        "title": "📊 定時レポート",
        "color": color,
        "fields": [
            {"name": "💰 残高",         "value": f"`{balance:,.0f} JPY`",          "inline": True},
            {"name": "📈 含み損益",     "value": f"`{unrealized:+,.0f} JPY`",      "inline": True},
            {"name": "🏦 純資産(NAV)",  "value": f"`{nav:,.0f} JPY`",             "inline": True},
            {"name": "📋 総トレード数", "value": f"`{total_trades}回`",            "inline": True},
            {"name": "✅ 勝率",         "value": f"`{win_rate:.1f}% ({wins}勝{losses}敗)`", "inline": True},
            {"name": "⚖️ PF",           "value": f"`{pf:.2f}`",                   "inline": True},
            {"name": "💹 累計損益",     "value": f"`{total_pnl:+.1f}pips`",       "inline": True},
            {"name": "📉 平均利益/損失","value": f"`+{avg_win:.1f} / {avg_loss:.1f}pips`", "inline": True},
            {"name": "🔓 保有中",       "value": f"`{open_count}件`",             "inline": True},
            {"name": "📌 ポジション詳細", "value": f"```\n{pos_lines}```",         "inline": False},
        ],
        "footer": {"text": f"sena3fx paper trading v76 | {now_jst_str}"}
    }
    send_discord("", embeds=[embed])
    logger.info("定時レポート送信完了")


# ── OANDA API ヘルパー ────────────────────────────────

def get_candles(instrument, granularity, count=CANDLE_COUNT):
    """OANDAからローソク足を取得してDataFrameに変換"""
    r = requests.get(
        f"{BASE_URL}/v3/instruments/{instrument}/candles",
        headers=HEADERS,
        params={"granularity": granularity, "count": count, "price": "M"},
        timeout=10
    )
    if r.status_code != 200:
        logger.error(f"candles error {instrument}/{granularity}: {r.text[:100]}")
        return None
    candles = r.json()["candles"]
    rows = []
    for c in candles:
        if not c.get("complete", True) and granularity != "M1":
            continue  # 未確定足はスキップ（1分足は最新足も使う）
        m = c["mid"]
        rows.append({
            "timestamp": pd.Timestamp(c["time"]).tz_convert("UTC"),
            "open":   float(m["o"]),
            "high":   float(m["h"]),
            "low":    float(m["l"]),
            "close":  float(m["c"]),
            "volume": int(c.get("volume", 0))
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("timestamp")
    # 週末除去
    df = df[df.index.dayofweek < 5]
    return df


def get_open_trades():
    """現在のオープントレード一覧を取得"""
    r = requests.get(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/openTrades",
        headers=HEADERS, timeout=10
    )
    if r.status_code != 200:
        logger.error(f"openTrades error: {r.text[:100]}")
        return []
    return r.json().get("trades", [])


def place_order(instrument, units, sl_price, tp_price):
    """成行注文を発行（SL/TP付き）"""
    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "stopLossOnFill": {
                "price": f"{sl_price:.3f}",
                "timeInForce": "GTC"
            },
            "takeProfitOnFill": {
                "price": f"{tp_price:.3f}",
                "timeInForce": "GTC"
            }
        }
    }
    r = requests.post(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders",
        headers=HEADERS,
        json=body,
        timeout=10
    )
    if r.status_code in (200, 201):
        resp = r.json()
        if resp.get("orderFillTransaction"):
            fill = resp["orderFillTransaction"]
            return {
                "order_id": fill.get("id"),
                "trade_id": fill.get("tradeOpened", {}).get("tradeID"),
                "fill_price": float(fill.get("price", 0))
            }
    logger.error(f"order error {instrument}: {r.status_code} {r.text[:200]}")
    return None


def close_trade(trade_id):
    """トレードを手動決済"""
    r = requests.put(
        f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/close",
        headers=HEADERS, timeout=10
    )
    return r.status_code in (200, 201)


# ── ログ書き込み ──────────────────────────────────────

def log_signal(sig, pair, status):
    row = {
        "signal_time": sig["time"].isoformat(),
        "pair": pair,
        "dir": sig["dir"],
        "tf": sig["tf"],
        "ep": round(sig["ep"], 3),
        "sl": round(sig["sl"], 3),
        "tp": round(sig["tp"], 3),
        "status": status
    }
    pd.DataFrame([row]).to_csv(SIGNAL_LOG, mode="a", header=False, index=False)


def log_trade(pos, exit_price, exit_type, oanda_trade_id=""):
    ep  = pos["ep"]
    pnl = (exit_price - ep) * 100 * pos["dir"]
    slippage = (pos["fill_price"] - ep) * 100 * pos["dir"] if pos.get("fill_price") else 0.0
    row = {
        "entry_time":     pos["entry_time"],
        "exit_time":      datetime.now(timezone.utc).isoformat(),
        "pair":           pos["pair"],
        "dir":            pos["dir"],
        "ep":             round(ep, 3),
        "sl":             round(pos["sl"], 3),
        "tp":             round(pos["tp"], 3),
        "risk":           round(pos["risk"], 3),
        "spread":         round(pos["spread"], 3),
        "tf":             pos["tf"],
        "exit_price":     round(exit_price, 3),
        "exit_type":      exit_type,
        "pnl":            round(pnl, 2),
        "oanda_order_id": oanda_trade_id,
        "slippage":       round(slippage, 2)
    }
    pd.DataFrame([row]).to_csv(TRADE_LOG, mode="a", header=False, index=False)
    logger.info(f"TRADE CLOSED: {pos['pair']} {'+' if pnl >= 0 else ''}{pnl:.1f}pips [{exit_type}]")

    # Discord通知
    notify_close(pos["pair"], pos["dir"], ep, exit_price, exit_type, pnl, pos["tf"])


# ── メインサイクル ────────────────────────────────────

def run_cycle():
    now = datetime.now(timezone.utc)
    logger.info(f"cycle start")

    # ── 1. 既存ポジションの状態確認 ──────────────────
    if open_positions:
        oanda_trades = {t["id"]: t for t in get_open_trades()}
        closed_ids = []
        for tid, pos in open_positions.items():
            if tid not in oanda_trades:
                # OANDAで決済済み → ログに記録
                pair_cfg = PAIRS[pos["pair"]]
                instr = pair_cfg["oanda"]
                df1m = get_candles(instr, "M1", count=5)
                if df1m is not None and len(df1m) > 0:
                    last_price = float(df1m.iloc[-1]["close"])
                else:
                    last_price = pos["sl"]

                dist_sl = abs(last_price - pos["sl"])
                dist_tp = abs(last_price - pos["tp"])
                if dist_sl < dist_tp:
                    exit_type  = "SL"
                    exit_price = pos["sl"]
                else:
                    exit_type  = "TP"
                    exit_price = pos["tp"]

                log_trade(pos, exit_price, exit_type, tid)
                closed_ids.append(tid)

        for tid in closed_ids:
            del open_positions[tid]

    # ── 2. シグナル生成 ──────────────────────────────
    for pair, cfg in PAIRS.items():
        instr  = cfg["oanda"]
        spread = cfg["spread"]

        df1m  = get_candles(instr, "M1",  count=300)
        df15m = get_candles(instr, "M15", count=300)
        df4h  = get_candles(instr, "H4",  count=200)

        if df1m is None or df15m is None or df4h is None:
            logger.warning(f"{pair}: データ取得失敗、スキップ")
            continue

        try:
            signals = generate_signals(df1m, df15m, df4h,
                                       spread_pips=spread, rr_ratio=RR_RATIO)
        except Exception as e:
            logger.error(f"{pair}: generate_signals エラー: {e}")
            continue

        if not signals:
            continue

        # 最新シグナルのみ処理（現在時刻から5分以内）
        latest   = signals[-1]
        sig_time = latest["time"]
        if hasattr(sig_time, "tzinfo") and sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")
        age_min = (now - sig_time).total_seconds() / 60
        if age_min > 5:
            continue

        # ── 3. JPY相関リスクチェック ─────────────────
        same_dir_count = sum(
            1 for p in open_positions.values() if p["dir"] == latest["dir"]
        )
        if same_dir_count >= MAX_SAME_DIR:
            logger.info(f"{pair}: SKIPPED_CORR (同方向{same_dir_count}本保有中)")
            log_signal(latest, pair, "SKIPPED_CORR")
            continue

        # 同ペアで既にポジションがある場合はスキップ
        pair_already = any(p["pair"] == pair for p in open_positions.values())
        if pair_already:
            log_signal(latest, pair, "SKIPPED_POS")
            continue

        # ── 4. 注文発行 ──────────────────────────────
        units  = cfg["units"] * latest["dir"]
        result = place_order(instr, units, latest["sl"], latest["tp"])

        if result:
            trade_id = result["trade_id"]
            fill_price = result.get("fill_price", latest["ep"])
            slippage   = (fill_price - latest["ep"]) * 100 * latest["dir"]

            open_positions[trade_id] = {
                "pair":       pair,
                "dir":        latest["dir"],
                "ep":         latest["ep"],
                "sl":         latest["sl"],
                "tp":         latest["tp"],
                "risk":       latest["risk"],
                "spread":     latest["spread"],
                "tf":         latest["tf"],
                "entry_time": now.isoformat(),
                "fill_price": fill_price,
                "half_done":  False
            }
            logger.info(
                f"{pair}: ENTERED {'LONG' if latest['dir'] > 0 else 'SHORT'} "
                f"ep={latest['ep']:.3f} sl={latest['sl']:.3f} tp={latest['tp']:.3f} "
                f"tf={latest['tf']} slip={slippage:+.2f}pips (order_id={result['order_id']})"
            )
            log_signal(latest, pair, "ENTERED")

            # Discord通知
            notify_entry(
                pair, latest["dir"], latest["ep"],
                latest["sl"], latest["tp"], latest["tf"],
                latest["risk"], slippage
            )
        else:
            logger.error(f"{pair}: 注文発行失敗")
            log_signal(latest, pair, "ORDER_FAILED")

    # ── 5. 定時レポート判定（朝9時・夜21時 JST = UTC 0時・12時） ──
    global _last_report_hour
    now_jst_hour = (now.hour + 9) % 24  # UTC → JST
    if now_jst_hour in (9, 21) and _last_report_hour != now_jst_hour:
        send_daily_report()
        _last_report_hour = now_jst_hour
    elif now_jst_hour not in (9, 21):
        _last_report_hour = -1  # リセット（次の該当時刻で再送できるように）

    # ── 6. サマリー ──────────────────────────────────
    logger.info(f"open positions: {len(open_positions)}件 | cycle end")


def main():
    logger.info("=" * 60)
    logger.info("ペーパートレードbot 起動")
    logger.info(f"対象ペア: {list(PAIRS.keys())}")
    logger.info(f"同方向最大保有: {MAX_SAME_DIR}本")
    logger.info(f"RR比: {RR_RATIO}")
    logger.info("=" * 60)

    # 起動通知
    send_discord(
        "🚀 **ペーパートレードbot 起動**\n"
        f"対象ペア: USDJPY / EURJPY / GBPJPY\n"
        f"同方向最大保有: {MAX_SAME_DIR}本 | RR: {RR_RATIO} | 戦略: v76"
    )

    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            logger.info("手動停止")
            send_discord("🛑 ペーパートレードbot 停止")
            break
        except Exception as e:
            logger.error(f"予期しないエラー: {e}", exc_info=True)
            notify_error(str(e)[:200])

        now  = datetime.now(timezone.utc)
        wait = 60 - now.second
        logger.info(f"次のサイクルまで {wait}秒待機...")
        time.sleep(wait)


if __name__ == "__main__":
    main()
