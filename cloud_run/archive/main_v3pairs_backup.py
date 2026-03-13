"""
main.py - Cloud Run用ペーパートレードbot
=========================================
Cloud Schedulerから1分ごとにHTTP POSTを受け取り、
v77ロジックで1サイクルのシグナル判定・注文処理を実行する。

ポジション状態はGCS（Cloud Storage）に保存して永続化する。
ログ（trades/signals）もGCSに追記保存する。
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request
from google.cloud import storage

# ── 環境変数 ─────────────────────────────────────────
OANDA_TOKEN  = os.environ.get("OANDA_TOKEN", "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467")
ACCOUNT_ID   = os.environ.get("OANDA_ACCOUNT", "101-009-38652105-001")
BASE_URL     = "https://api-fxpractice.oanda.com"
OANDA_HEADERS = {
    "Authorization": f"Bearer {OANDA_TOKEN}",
    "Content-Type": "application/json"
}

DISCORD_WEBHOOK = os.environ.get(
    "DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1329418335259197490/59-rzRn2tvHmvMetqMlJPtoo4CApLk3yGoBZoTfUexmXQzrUrTBI1X8sL7RbFfvoQG5k"
)
GCS_BUCKET   = os.environ.get("GCS_BUCKET", "sena3fx-paper-trading")
PROJECT_ID   = os.environ.get("GCP_PROJECT", "aiyagami")

# ── 取引設定 ─────────────────────────────────────────
PAIRS = {
    "USDJPY": {"oanda": "USD_JPY", "spread": 0.4, "units": 1000},
    "EURJPY": {"oanda": "EUR_JPY", "spread": 1.1, "units": 1000},
    "GBPJPY": {"oanda": "GBP_JPY", "spread": 1.5, "units": 1000},
}
MAX_SAME_DIR = 2
RR_RATIO     = 2.5
CANDLE_COUNT = 200

# ── ロガー ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s UTC [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ── FastAPI ───────────────────────────────────────────
app = FastAPI()

# ── GCS ヘルパー ──────────────────────────────────────

def gcs_client():
    return storage.Client(project=PROJECT_ID)

def gcs_read_json(blob_name: str, default=None):
    """GCSからJSONを読み込む（存在しなければdefaultを返す）"""
    try:
        client = gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning(f"GCS read error {blob_name}: {e}")
    return default if default is not None else {}

def gcs_write_json(blob_name: str, data):
    """GCSにJSONを書き込む"""
    try:
        client = gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data, ensure_ascii=False, default=str),
                                content_type="application/json")
    except Exception as e:
        logger.error(f"GCS write error {blob_name}: {e}")

def gcs_append_csv(blob_name: str, row: dict):
    """GCSのCSVに1行追記する"""
    try:
        client = gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        if blob.exists():
            existing = blob.download_as_text()
            new_line = ",".join(str(row.get(k, "")) for k in row.keys())
            blob.upload_from_string(existing + new_line + "\n",
                                    content_type="text/csv")
        else:
            header  = ",".join(row.keys())
            values  = ",".join(str(v) for v in row.values())
            blob.upload_from_string(header + "\n" + values + "\n",
                                    content_type="text/csv")
    except Exception as e:
        logger.error(f"GCS append error {blob_name}: {e}")

# ── Discord通知 ───────────────────────────────────────

def send_discord(content: str, embeds: list = None):
    payload = {"content": content}
    if embeds:
        payload["embeds"] = embeds
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        if r.status_code not in (200, 204):
            logger.warning(f"Discord失敗: {r.status_code}")
    except Exception as e:
        logger.warning(f"Discord error: {e}")

def notify_entry(pair, dir_, ep, sl, tp, tf, risk, slippage):
    dir_str = "🟢 LONG" if dir_ > 0 else "🔴 SHORT"
    sl_pips = risk * 100
    tp_pips = risk * 100 * RR_RATIO
    now_str = datetime.now(timezone.utc).strftime("%m/%d %H:%M") + " UTC"
    color   = 0x00c853 if dir_ > 0 else 0xd50000
    embed = {
        "title": f"📈 エントリー: {pair} {dir_str}",
        "color": color,
        "fields": [
            {"name": "EP",           "value": f"`{ep:.3f}`",                       "inline": True},
            {"name": "SL",           "value": f"`{sl:.3f}` (-{sl_pips:.1f}pips)",  "inline": True},
            {"name": "TP",           "value": f"`{tp:.3f}` (+{tp_pips:.1f}pips)",  "inline": True},
            {"name": "足種",         "value": f"`{tf}`",                            "inline": True},
            {"name": "スリッページ", "value": f"`{slippage:+.2f}pips`",            "inline": True},
            {"name": "時刻",         "value": now_str,                              "inline": True},
        ],
        "footer": {"text": "sena3fx paper trading v77 | Cloud Run"}
    }
    send_discord("", embeds=[embed])

def notify_close(pair, dir_, ep, exit_price, exit_type, pnl, tf):
    emoji = "✅" if exit_type == "TP" else "❌" if exit_type == "SL" else "⚪"
    color = 0x00c853 if exit_type == "TP" else 0xd50000 if exit_type == "SL" else 0x9e9e9e
    dir_str = "LONG" if dir_ > 0 else "SHORT"
    pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.1f}pips"
    now_str = datetime.now(timezone.utc).strftime("%m/%d %H:%M") + " UTC"
    embed = {
        "title": f"{emoji} 決済: {pair} {dir_str} [{exit_type}]",
        "color": color,
        "fields": [
            {"name": "損益",     "value": f"**`{pnl_str}`**",   "inline": True},
            {"name": "EP",       "value": f"`{ep:.3f}`",         "inline": True},
            {"name": "決済価格", "value": f"`{exit_price:.3f}`", "inline": True},
            {"name": "足種",     "value": f"`{tf}`",             "inline": True},
            {"name": "時刻",     "value": now_str,               "inline": True},
        ],
        "footer": {"text": "sena3fx paper trading v77 | Cloud Run"}
    }
    send_discord("", embeds=[embed])

def send_daily_report(open_positions: dict):
    now_str = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M") + " UTC"
    try:
        r = requests.get(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/summary",
                         headers=OANDA_HEADERS, timeout=10)
        if r.status_code == 200:
            d = r.json()["account"]
            balance    = float(d["balance"])
            unrealized = float(d.get("unrealizedPL", 0))
            nav        = float(d.get("NAV", balance))
            open_count = int(d["openPositionCount"])
        else:
            balance = unrealized = nav = 0.0
            open_count = 0
    except Exception:
        balance = unrealized = nav = 0.0
        open_count = 0

    # GCSからトレードログ集計
    try:
        client = gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob("logs/paper_trades.csv")
        if blob.exists():
            from io import StringIO
            df = pd.read_csv(StringIO(blob.download_as_text()))
            total_trades = len(df)
            if total_trades > 0:
                wins     = len(df[df["pnl"] > 0])
                losses   = len(df[df["pnl"] <= 0])
                win_rate = wins / total_trades * 100
                total_pnl = df["pnl"].sum()
                avg_win  = df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
                avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
                pf       = abs(df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] <= 0]["pnl"].sum()) if losses > 0 else 999
            else:
                wins = losses = 0
                win_rate = total_pnl = avg_win = avg_loss = pf = 0.0
        else:
            total_trades = wins = losses = 0
            win_rate = total_pnl = avg_win = avg_loss = pf = 0.0
    except Exception:
        total_trades = wins = losses = 0
        win_rate = total_pnl = avg_win = avg_loss = pf = 0.0

    pos_lines = ""
    for tid, pos in open_positions.items():
        dir_str = "LONG" if pos["dir"] > 0 else "SHORT"
        pos_lines += f"  • {pos['pair']} {dir_str} ep={pos['ep']:.3f}\n"
    if not pos_lines:
        pos_lines = "  なし"

    embed = {
        "title": "📊 定時レポート",
        "color": 0x1565c0,
        "fields": [
            {"name": "💰 残高",           "value": f"`{balance:,.0f} JPY`",                    "inline": True},
            {"name": "📈 含み損益",       "value": f"`{unrealized:+,.0f} JPY`",                "inline": True},
            {"name": "🏦 純資産(NAV)",    "value": f"`{nav:,.0f} JPY`",                        "inline": True},
            {"name": "📋 総トレード数",   "value": f"`{total_trades}回`",                      "inline": True},
            {"name": "✅ 勝率",           "value": f"`{win_rate:.1f}% ({wins}勝{losses}敗)`",  "inline": True},
            {"name": "⚖️ PF",             "value": f"`{pf:.2f}`",                              "inline": True},
            {"name": "💹 累計損益",       "value": f"`{total_pnl:+.1f}pips`",                 "inline": True},
            {"name": "📉 平均利益/損失",  "value": f"`+{avg_win:.1f} / {avg_loss:.1f}pips`",  "inline": True},
            {"name": "🔓 保有中",         "value": f"`{open_count}件`",                        "inline": True},
            {"name": "📌 ポジション詳細", "value": f"```\n{pos_lines}```",                     "inline": False},
        ],
        "footer": {"text": f"sena3fx paper trading v77 | Cloud Run | {now_str}"}
    }
    send_discord("", embeds=[embed])
    logger.info("定時レポート送信完了")

# ── OANDA ヘルパー ────────────────────────────────────

def get_candles(instrument, granularity, count=CANDLE_COUNT):
    r = requests.get(
        f"{BASE_URL}/v3/instruments/{instrument}/candles",
        headers=OANDA_HEADERS,
        params={"granularity": granularity, "count": count, "price": "M"},
        timeout=15
    )
    if r.status_code != 200:
        logger.error(f"candles error {instrument}/{granularity}: {r.text[:100]}")
        return None
    candles = r.json()["candles"]
    rows = []
    for c in candles:
        if not c.get("complete", True) and granularity != "M1":
            continue
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
    df = df[df.index.dayofweek < 5]
    return df

def get_open_trades():
    r = requests.get(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/openTrades",
                     headers=OANDA_HEADERS, timeout=10)
    if r.status_code != 200:
        return []
    return r.json().get("trades", [])

def place_order(instrument, units, sl_price, tp_price):
    body = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "stopLossOnFill":   {"price": f"{sl_price:.3f}", "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": f"{tp_price:.3f}", "timeInForce": "GTC"}
        }
    }
    r = requests.post(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders",
                      headers=OANDA_HEADERS, json=body, timeout=10)
    if r.status_code in (200, 201):
        resp = r.json()
        if resp.get("orderFillTransaction"):
            fill = resp["orderFillTransaction"]
            return {
                "order_id":   fill.get("id"),
                "trade_id":   fill.get("tradeOpened", {}).get("tradeID"),
                "fill_price": float(fill.get("price", 0))
            }
    logger.error(f"order error {instrument}: {r.status_code} {r.text[:200]}")
    return None

# ── メインサイクル ────────────────────────────────────

def run_cycle():
    now = datetime.now(timezone.utc)
    logger.info(f"cycle start: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # GCSからポジション状態を読み込む
    open_positions = gcs_read_json("state/open_positions.json", default={})
    last_report    = gcs_read_json("state/last_report.json",    default={"hour": -1})

    # ── 1. 既存ポジションの決済確認 ──────────────────
    if open_positions:
        oanda_trades = {t["id"]: t for t in get_open_trades()}
        closed_ids   = []
        for tid, pos in open_positions.items():
            if tid not in oanda_trades:
                # 決済済み → SL/TP判定
                instr = PAIRS[pos["pair"]]["oanda"]
                df1m  = get_candles(instr, "M1", count=5)
                if df1m is not None and len(df1m) > 0:
                    last_price = float(df1m.iloc[-1]["close"])
                else:
                    last_price = pos["sl"]

                dist_sl = abs(last_price - pos["sl"])
                dist_tp = abs(last_price - pos["tp"])
                exit_type  = "SL" if dist_sl < dist_tp else "TP"
                exit_price = pos["sl"] if exit_type == "SL" else pos["tp"]

                ep  = pos["ep"]
                pnl = (exit_price - ep) * 100 * pos["dir"]
                row = {
                    "entry_time":     pos["entry_time"],
                    "exit_time":      now.isoformat(),
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
                    "oanda_order_id": tid,
                    "slippage":       round(pos.get("slippage", 0), 2)
                }
                gcs_append_csv("logs/paper_trades.csv", row)
                notify_close(pos["pair"], pos["dir"], ep, exit_price, exit_type, pnl, pos["tf"])
                logger.info(f"CLOSED: {pos['pair']} {pnl:+.1f}pips [{exit_type}]")
                closed_ids.append(tid)

        for tid in closed_ids:
            del open_positions[tid]

    # ── 2. シグナル生成 ──────────────────────────────
    # v77をインポート（Cloud Run環境ではsys.pathに追加）
    import sys
    sys.path.insert(0, "/app")
    from strategies.yagami_mtf_v77 import generate_signals

    for pair, cfg in PAIRS.items():
        instr  = cfg["oanda"]
        spread = cfg["spread"]

        df1m  = get_candles(instr, "M1",  count=300)
        df15m = get_candles(instr, "M15", count=300)
        df4h  = get_candles(instr, "H4",  count=200)

        if df1m is None or df15m is None or df4h is None:
            logger.warning(f"{pair}: データ取得失敗")
            continue

        try:
            signals = generate_signals(df1m, df15m, df4h,
                                       spread_pips=spread, rr_ratio=RR_RATIO)
        except Exception as e:
            logger.error(f"{pair}: generate_signals error: {e}")
            continue

        if not signals:
            continue

        latest   = signals[-1]
        sig_time = latest["time"]
        if hasattr(sig_time, "tzinfo") and sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")
        age_min = (now - sig_time).total_seconds() / 60
        if age_min > 5:
            continue

        # JPY相関リスクチェック
        same_dir_count = sum(1 for p in open_positions.values() if p["dir"] == latest["dir"])
        if same_dir_count >= MAX_SAME_DIR:
            logger.info(f"{pair}: SKIPPED_CORR (同方向{same_dir_count}本)")
            gcs_append_csv("logs/paper_signals.csv", {
                "signal_time": sig_time.isoformat(), "pair": pair,
                "dir": latest["dir"], "tf": latest["tf"],
                "ep": round(latest["ep"], 3), "sl": round(latest["sl"], 3),
                "tp": round(latest["tp"], 3), "status": "SKIPPED_CORR"
            })
            continue

        pair_already = any(p["pair"] == pair for p in open_positions.values())
        if pair_already:
            gcs_append_csv("logs/paper_signals.csv", {
                "signal_time": sig_time.isoformat(), "pair": pair,
                "dir": latest["dir"], "tf": latest["tf"],
                "ep": round(latest["ep"], 3), "sl": round(latest["sl"], 3),
                "tp": round(latest["tp"], 3), "status": "SKIPPED_POS"
            })
            continue

        # 注文発行
        units  = cfg["units"] * latest["dir"]
        result = place_order(instr, units, latest["sl"], latest["tp"])

        if result:
            trade_id   = result["trade_id"]
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
                "slippage":   slippage
            }
            gcs_append_csv("logs/paper_signals.csv", {
                "signal_time": sig_time.isoformat(), "pair": pair,
                "dir": latest["dir"], "tf": latest["tf"],
                "ep": round(latest["ep"], 3), "sl": round(latest["sl"], 3),
                "tp": round(latest["tp"], 3), "status": "ENTERED"
            })
            notify_entry(pair, latest["dir"], latest["ep"],
                         latest["sl"], latest["tp"], latest["tf"],
                         latest["risk"], slippage)
            logger.info(f"ENTERED: {pair} {'LONG' if latest['dir']>0 else 'SHORT'} "
                        f"ep={latest['ep']:.3f} tf={latest['tf']}")
        else:
            logger.error(f"{pair}: 注文失敗")

    # ── 3. 定時レポート判定（朝9時・夜21時 JST） ──────
    now_jst_hour = (now.hour + 9) % 24
    if now_jst_hour in (9, 21) and last_report.get("hour") != now_jst_hour:
        send_daily_report(open_positions)
        last_report = {"hour": now_jst_hour}
    elif now_jst_hour not in (9, 21):
        last_report = {"hour": -1}

    # ── 4. 状態をGCSに保存 ───────────────────────────
    gcs_write_json("state/open_positions.json", open_positions)
    gcs_write_json("state/last_report.json",    last_report)

    logger.info(f"cycle end: open={len(open_positions)}件")
    return {"status": "ok", "open_positions": len(open_positions)}


# ── エンドポイント ────────────────────────────────────

@app.post("/run")
async def run_endpoint(request: Request):
    """Cloud Schedulerから呼ばれるエンドポイント"""
    try:
        result = run_cycle()
        return result
    except Exception as e:
        logger.error(f"cycle error: {e}", exc_info=True)
        send_discord(f"⚠️ **Cloud Run エラー**: {str(e)[:200]}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    """ヘルスチェック"""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/report")
async def report_endpoint():
    """4時間ごとの定時レポート"""
    try:
        open_positions = gcs_read_json("state/open_positions.json", default={})
        send_daily_report(open_positions)
        return {"status": "ok", "message": "定時レポートを送信しました"}
    except Exception as e:
        logger.error(f"report error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/notify_test")
async def notify_test_endpoint():
    """Discord接続テスト通知"""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        send_discord(
            content=None,
            embeds=[{
                "title": "🔔 接続テスト",
                "description": "Cloud Run → Discord の接続確認です。\nペーパートレードbotは正常に稼働しています。",
                "color": 0x00bfff,
                "fields": [
                    {"name": "サービス", "value": "Cloud Run (asia-northeast1)", "inline": True},
                    {"name": "実行間隔", "value": "毎分（Cloud Scheduler）", "inline": True},
                    {"name": "ステータス", "value": "✅ 正常稼働中", "inline": False}
                ],
                "footer": {"text": f"sena3fx paper trader | {now_str} UTC"}
            }]
        )
        return {"status": "ok", "message": "Discord通知を送信しました"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/status")
async def status():
    """現在のポジション状態を返す"""
    open_positions = gcs_read_json("state/open_positions.json", default={})
    return {
        "open_positions": len(open_positions),
        "positions": open_positions,
        "time": datetime.now(timezone.utc).isoformat()
    }
