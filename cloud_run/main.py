"""
main.py - Cloud Run用ペーパートレードbot (v77 全68ペア対応版)
=================================================================
変更点:
- PAIRS: USDJPY/EURJPY/GBPJPY → OANDAデモ全68通貨ペアに拡張
- 毎週月曜0時(JST)に週次フィードバックレポートをGCSに蓄積
- 週次レポートはDiscordにも送信
- 同時オープン最大10件（同一ペアは1件まで）
- 週次フィードバックCSV: logs/weekly_feedback.csv に蓄積
"""
import os, json, logging, requests, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from google.cloud import storage

OANDA_TOKEN  = os.environ.get("OANDA_TOKEN", "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467")
ACCOUNT_ID   = os.environ.get("OANDA_ACCOUNT", "101-009-38652105-001")
BASE_URL     = "https://api-fxpractice.oanda.com"
OANDA_HEADERS = {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1329418335259197490/59-rzRn2tvHmvMetqMlJPtoo4CApLk3yGoBZoTfUexmXQzrUrTBI1X8sL7RbFfvoQG5k")
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "sena3fx-paper-trading")
PROJECT_ID  = os.environ.get("GCP_PROJECT", "aiyagami")

PAIRS = {
    "AUDCAD":{"oanda":"AUD_CAD","spread":0.5,"pip_size":0.0001,"units":1000},
    "AUDCHF":{"oanda":"AUD_CHF","spread":0.5,"pip_size":0.0001,"units":1000},
    "AUDHKD":{"oanda":"AUD_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "AUDJPY":{"oanda":"AUD_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "AUDNZD":{"oanda":"AUD_NZD","spread":0.7,"pip_size":0.0001,"units":1000},
    "AUDSGD":{"oanda":"AUD_SGD","spread":1.0,"pip_size":0.0001,"units":1000},
    "AUDUSD":{"oanda":"AUD_USD","spread":0.3,"pip_size":0.0001,"units":1000},
    "CADCHF":{"oanda":"CAD_CHF","spread":0.5,"pip_size":0.0001,"units":1000},
    "CADHKD":{"oanda":"CAD_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "CADJPY":{"oanda":"CAD_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "CADSGD":{"oanda":"CAD_SGD","spread":1.0,"pip_size":0.0001,"units":1000},
    "CHFHKD":{"oanda":"CHF_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "CHFJPY":{"oanda":"CHF_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "CHFZAR":{"oanda":"CHF_ZAR","spread":3.0,"pip_size":0.0001,"units":1000},
    "EURAUD":{"oanda":"EUR_AUD","spread":0.7,"pip_size":0.0001,"units":1000},
    "EURCAD":{"oanda":"EUR_CAD","spread":0.5,"pip_size":0.0001,"units":1000},
    "EURCHF":{"oanda":"EUR_CHF","spread":0.5,"pip_size":0.0001,"units":1000},
    "EURCZK":{"oanda":"EUR_CZK","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURDKK":{"oanda":"EUR_DKK","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURGBP":{"oanda":"EUR_GBP","spread":0.3,"pip_size":0.0001,"units":1000},
    "EURHKD":{"oanda":"EUR_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "EURHUF":{"oanda":"EUR_HUF","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURJPY":{"oanda":"EUR_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "EURNOK":{"oanda":"EUR_NOK","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURNZD":{"oanda":"EUR_NZD","spread":0.8,"pip_size":0.0001,"units":1000},
    "EURPLN":{"oanda":"EUR_PLN","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURSEK":{"oanda":"EUR_SEK","spread":2.0,"pip_size":0.0001,"units":1000},
    "EURSGD":{"oanda":"EUR_SGD","spread":1.0,"pip_size":0.0001,"units":1000},
    "EURTRY":{"oanda":"EUR_TRY","spread":5.0,"pip_size":0.0001,"units":1000},
    "EURUSD":{"oanda":"EUR_USD","spread":0.2,"pip_size":0.0001,"units":1000},
    "EURZAR":{"oanda":"EUR_ZAR","spread":3.0,"pip_size":0.0001,"units":1000},
    "GBPAUD":{"oanda":"GBP_AUD","spread":0.8,"pip_size":0.0001,"units":1000},
    "GBPCAD":{"oanda":"GBP_CAD","spread":0.8,"pip_size":0.0001,"units":1000},
    "GBPCHF":{"oanda":"GBP_CHF","spread":0.8,"pip_size":0.0001,"units":1000},
    "GBPHKD":{"oanda":"GBP_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "GBPJPY":{"oanda":"GBP_JPY","spread":0.8,"pip_size":0.01,  "units":1000},
    "GBPNZD":{"oanda":"GBP_NZD","spread":1.0,"pip_size":0.0001,"units":1000},
    "GBPPLN":{"oanda":"GBP_PLN","spread":2.0,"pip_size":0.0001,"units":1000},
    "GBPSGD":{"oanda":"GBP_SGD","spread":1.0,"pip_size":0.0001,"units":1000},
    "GBPUSD":{"oanda":"GBP_USD","spread":0.3,"pip_size":0.0001,"units":1000},
    "GBPZAR":{"oanda":"GBP_ZAR","spread":3.0,"pip_size":0.0001,"units":1000},
    "HKDJPY":{"oanda":"HKD_JPY","spread":1.0,"pip_size":0.0001,"units":1000},
    "NZDCAD":{"oanda":"NZD_CAD","spread":0.5,"pip_size":0.0001,"units":1000},
    "NZDCHF":{"oanda":"NZD_CHF","spread":0.5,"pip_size":0.0001,"units":1000},
    "NZDHKD":{"oanda":"NZD_HKD","spread":1.0,"pip_size":0.0001,"units":1000},
    "NZDJPY":{"oanda":"NZD_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "NZDSGD":{"oanda":"NZD_SGD","spread":1.0,"pip_size":0.0001,"units":1000},
    "NZDUSD":{"oanda":"NZD_USD","spread":0.3,"pip_size":0.0001,"units":1000},
    "SGDCHF":{"oanda":"SGD_CHF","spread":1.0,"pip_size":0.0001,"units":1000},
    "SGDJPY":{"oanda":"SGD_JPY","spread":0.5,"pip_size":0.01,  "units":1000},
    "TRYJPY":{"oanda":"TRY_JPY","spread":3.0,"pip_size":0.0001,"units":1000},
    "USDCAD":{"oanda":"USD_CAD","spread":0.3,"pip_size":0.0001,"units":1000},
    "USDCHF":{"oanda":"USD_CHF","spread":0.3,"pip_size":0.0001,"units":1000},
    "USDCNH":{"oanda":"USD_CNH","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDCZK":{"oanda":"USD_CZK","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDDKK":{"oanda":"USD_DKK","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDHKD":{"oanda":"USD_HKD","spread":0.5,"pip_size":0.0001,"units":1000},
    "USDHUF":{"oanda":"USD_HUF","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDJPY":{"oanda":"USD_JPY","spread":0.4,"pip_size":0.01,  "units":1000},
    "USDMXN":{"oanda":"USD_MXN","spread":3.0,"pip_size":0.0001,"units":1000},
    "USDNOK":{"oanda":"USD_NOK","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDPLN":{"oanda":"USD_PLN","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDSEK":{"oanda":"USD_SEK","spread":2.0,"pip_size":0.0001,"units":1000},
    "USDSGD":{"oanda":"USD_SGD","spread":0.5,"pip_size":0.0001,"units":1000},
    "USDTHB":{"oanda":"USD_THB","spread":3.0,"pip_size":0.0001,"units":1000},
    "USDTRY":{"oanda":"USD_TRY","spread":5.0,"pip_size":0.0001,"units":1000},
    "USDZAR":{"oanda":"USD_ZAR","spread":3.0,"pip_size":0.0001,"units":1000},
    "ZARJPY":{"oanda":"ZAR_JPY","spread":1.0,"pip_size":0.0001,"units":1000},
}

MAX_OPEN_POSITIONS = 10
MAX_SAME_DIR       = 5
RR_RATIO           = 2.5
CANDLE_COUNT       = 200

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
app = FastAPI()

# ── GCS ──────────────────────────────────────────────
def gcs_client(): return storage.Client(project=PROJECT_ID)

def gcs_read_json(blob_name, default=None):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        if b.exists(): return json.loads(b.download_as_text())
    except Exception as e: logger.error(f"GCS read {blob_name}: {e}")
    return default if default is not None else {}

def gcs_write_json(blob_name, data):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        b.upload_from_string(json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json")
    except Exception as e: logger.error(f"GCS write {blob_name}: {e}")

def gcs_append_csv(blob_name, row):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        line = ",".join(str(row.get(k,"")) for k in row.keys())
        if b.exists():
            b.upload_from_string(b.download_as_text().rstrip("\n") + "\n" + line + "\n", content_type="text/csv")
        else:
            b.upload_from_string(",".join(row.keys()) + "\n" + line + "\n", content_type="text/csv")
    except Exception as e: logger.error(f"GCS append {blob_name}: {e}")

def gcs_read_csv(blob_name):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        if b.exists():
            from io import StringIO
            return pd.read_csv(StringIO(b.download_as_text()))
    except Exception as e: logger.error(f"GCS read csv {blob_name}: {e}")
    return pd.DataFrame()

# ── Discord ───────────────────────────────────────────
def send_discord(content, embeds=None):
    try:
        r = requests.post(DISCORD_WEBHOOK, json={"content": content, **({"embeds": embeds} if embeds else {})}, timeout=10)
        if r.status_code not in (200, 204): logger.warning(f"Discord: {r.status_code}")
    except Exception as e: logger.warning(f"Discord error: {e}")

def notify_entry(pair, dir_, ep, sl, tp, tf, slippage):
    cfg = PAIRS.get(pair, {}); ps = cfg.get("pip_size", 0.0001)
    embed = {"title": f"📈 {'🟢 LONG' if dir_>0 else '🔴 SHORT'}: {pair}",
             "color": 0x00c853 if dir_>0 else 0xd50000,
             "fields": [
                 {"name":"EP","value":f"`{ep:.5f}`","inline":True},
                 {"name":"SL","value":f"`{sl:.5f}` (-{abs(ep-sl)/ps:.1f}p)","inline":True},
                 {"name":"TP","value":f"`{tp:.5f}` (+{abs(tp-ep)/ps:.1f}p)","inline":True},
                 {"name":"足種","value":f"`{tf}`","inline":True},
                 {"name":"Slip","value":f"`{slippage:+.2f}p`","inline":True},
                 {"name":"時刻","value":datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC"),"inline":True},
             ], "footer":{"text":"sena3fx v77 | 全68ペア"}}
    send_discord("", embeds=[embed])

def notify_close(pair, dir_, ep, exit_price, exit_type, pnl, tf):
    emoji = "✅" if exit_type=="TP" else "❌"
    embed = {"title": f"{emoji} 決済: {pair} {'LONG' if dir_>0 else 'SHORT'} [{exit_type}]",
             "color": 0x00c853 if exit_type=="TP" else 0xd50000,
             "fields": [
                 {"name":"損益","value":f"**`{pnl:+.1f}pips`**","inline":True},
                 {"name":"EP","value":f"`{ep:.5f}`","inline":True},
                 {"name":"決済","value":f"`{exit_price:.5f}`","inline":True},
                 {"name":"足種","value":f"`{tf}`","inline":True},
                 {"name":"時刻","value":datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC"),"inline":True},
             ], "footer":{"text":"sena3fx v77 | 全68ペア"}}
    send_discord("", embeds=[embed])

# ── 日次レポート ──────────────────────────────────────
def send_daily_report(open_positions):
    df = gcs_read_csv("logs/paper_trades.csv")
    n = len(df)
    wins = len(df[df["pnl"]>0]) if n>0 else 0
    wr   = wins/n*100 if n>0 else 0
    pf   = abs(df[df["pnl"]>0]["pnl"].sum()/df[df["pnl"]<=0]["pnl"].sum()) if n>0 and len(df[df["pnl"]<=0])>0 else 0
    pnl_sum = df["pnl"].sum() if n>0 else 0
    pos_str = "\n".join(f"  {p['pair']} {'L' if p['dir']>0 else 'S'} {p['ep']:.5f}" for p in open_positions.values()) or "  なし"
    embed = {"title":"📊 日次レポート","color":0x1565c0,
             "fields":[
                 {"name":"累計トレード","value":f"`{n}件`","inline":True},
                 {"name":"勝率","value":f"`{wr:.1f}%`","inline":True},
                 {"name":"PF","value":f"`{pf:.2f}`","inline":True},
                 {"name":"累計損益","value":f"`{pnl_sum:+.1f}pips`","inline":True},
                 {"name":"オープン","value":f"`{len(open_positions)}件`","inline":True},
                 {"name":"監視ペア","value":f"`{len(PAIRS)}ペア`","inline":True},
                 {"name":"ポジション詳細","value":f"```{pos_str}```","inline":False},
             ], "footer":{"text":f"sena3fx v77 | {datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M UTC')}"}}
    send_discord("", embeds=[embed])

# ── 週次フィードバック ────────────────────────────────
def send_weekly_feedback():
    now_utc = datetime.now(timezone.utc)
    week_start = now_utc - timedelta(days=7)
    df = gcs_read_csv("logs/paper_trades.csv")
    if len(df)==0:
        send_discord(f"📅 **週次FB ({week_start.strftime('%m/%d')}〜)**\nデータなし"); return

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    wdf = df[df["exit_time"] >= week_start]
    n = len(df); wn = len(wdf)
    wwr  = len(wdf[wdf["pnl"]>0])/wn*100 if wn>0 else 0
    wpnl = wdf["pnl"].sum() if wn>0 else 0
    wpf  = abs(wdf[wdf["pnl"]>0]["pnl"].sum()/wdf[wdf["pnl"]<=0]["pnl"].sum()) if wn>0 and len(wdf[wdf["pnl"]<=0])>0 else 0
    pf_all = abs(df[df["pnl"]>0]["pnl"].sum()/df[df["pnl"]<=0]["pnl"].sum()) if len(df[df["pnl"]<=0])>0 else 0

    # 本番実装判断
    if n>=500 and pf_all>=2.0:   stage="🟢 段階3達成（本番フルサイズGO）"
    elif n>=300 and pf_all>=1.8: stage="🟡 段階2達成（小額→標準ロット可）"
    elif n>=100 and pf_all>=1.5: stage="🟠 段階1達成（ペーパー→小額実取引可）"
    else:                         stage=f"🔴 蓄積中 ({n}/100本)"

    # ペア別TOP/WORST
    top3_str = bot3_str = "データなし"
    if "pair" in wdf.columns and wn>0:
        ps = wdf.groupby("pair").agg(trades=("pnl","count"),pnl=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()*100)).sort_values("pnl",ascending=False)
        top3_str = "\n".join(f"  {i}: {r['pnl']:+.1f}p ({r['trades']}件,{r['wr']:.0f}%)" for i,r in ps.head(3).iterrows())
        bot3_str = "\n".join(f"  {i}: {r['pnl']:+.1f}p ({r['trades']}件,{r['wr']:.0f}%)" for i,r in ps.tail(3).iterrows())
        # ペア別統計をGCSに保存
        gcs_write_json("logs/pair_stats.json", df.groupby("pair").agg(trades=("pnl","count"),pnl=("pnl","sum"),wr=("pnl",lambda x:(x>0).mean()*100)).round(2).to_dict(orient="index"))

    # 週次フィードバックCSVに蓄積
    gcs_append_csv("logs/weekly_feedback.csv", {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end":   now_utc.strftime("%Y-%m-%d"),
        "week_trades": wn, "week_wr": round(wwr,1),
        "week_pnl": round(wpnl,1), "week_pf": round(wpf,2),
        "total_trades": n, "total_pf": round(pf_all,2), "stage": stage,
    })

    embed = {"title":f"📅 週次フィードバック ({week_start.strftime('%m/%d')}〜{now_utc.strftime('%m/%d')})","color":0x7b1fa2,
             "fields":[
                 {"name":"今週トレード","value":f"`{wn}件`","inline":True},
                 {"name":"今週勝率","value":f"`{wwr:.1f}%`","inline":True},
                 {"name":"今週PF","value":f"`{wpf:.2f}`","inline":True},
                 {"name":"今週損益","value":f"`{wpnl:+.1f}pips`","inline":True},
                 {"name":"累計トレード","value":f"`{n}件`","inline":True},
                 {"name":"累計PF","value":f"`{pf_all:.2f}`","inline":True},
                 {"name":"本番実装判断","value":stage,"inline":False},
                 {"name":"今週TOP3","value":f"```{top3_str}```","inline":False},
                 {"name":"今週WORST3","value":f"```{bot3_str}```","inline":False},
             ], "footer":{"text":f"sena3fx v77 | 全68ペア | {now_utc.strftime('%Y/%m/%d UTC')}"}}
    send_discord("", embeds=[embed])
    logger.info(f"週次FB送信: {wn}件, PF={wpf:.2f}, 累計{n}件")

# ── OANDA ヘルパー ────────────────────────────────────
def get_candles(instrument, granularity, count=CANDLE_COUNT):
    try:
        r = requests.get(f"{BASE_URL}/v3/instruments/{instrument}/candles",
            headers=OANDA_HEADERS, params={"granularity":granularity,"count":count,"price":"M"}, timeout=15)
        if r.status_code!=200: return pd.DataFrame()
        rows = [{"timestamp":pd.Timestamp(c["time"]),"open":float(c["mid"]["o"]),"high":float(c["mid"]["h"]),
                 "low":float(c["mid"]["l"]),"close":float(c["mid"]["c"]),"volume":int(c.get("volume",0))}
                for c in r.json()["candles"] if c.get("complete",True)]
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).set_index("timestamp").sort_index()
    except Exception as e: logger.error(f"candles {instrument}: {e}"); return pd.DataFrame()

def get_current_price(instrument):
    try:
        r = requests.get(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing",
            headers=OANDA_HEADERS, params={"instruments":instrument}, timeout=10)
        if r.status_code==200:
            p = r.json().get("prices",[])
            if p: return (float(p[0]["asks"][0]["price"])+float(p[0]["bids"][0]["price"]))/2
    except Exception as e: logger.error(f"price {instrument}: {e}")
    return 0.0

def place_order(instrument, units, sl, tp):
    try:
        r = requests.post(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders",
            headers=OANDA_HEADERS, timeout=10,
            json={"order":{"type":"MARKET","instrument":instrument,"units":str(units),
                "stopLossOnFill":{"price":f"{sl:.5f}","timeInForce":"GTC"},
                "takeProfitOnFill":{"price":f"{tp:.5f}","timeInForce":"GTC"},
                "timeInForce":"FOK","positionFill":"DEFAULT"}})
        if r.status_code in (200,201):
            fill = r.json().get("orderFillTransaction",{})
            return {"trade_id":fill.get("tradeOpened",{}).get("tradeID",""),"fill_price":float(fill.get("price",0))}
        logger.error(f"order {instrument}: {r.status_code} {r.text[:100]}")
    except Exception as e: logger.error(f"order {instrument}: {e}")
    return {}

def close_trade(trade_id):
    try:
        r = requests.put(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/close",
            headers=OANDA_HEADERS, timeout=10)
        if r.status_code==200:
            return {"exit_price":float(r.json().get("orderFillTransaction",{}).get("price",0))}
    except Exception as e: logger.error(f"close {trade_id}: {e}")
    return {}

# ── メインサイクル ────────────────────────────────────
def run_cycle():
    now = datetime.now(timezone.utc)
    logger.info(f"cycle start: {now.isoformat()} | {len(PAIRS)}ペア")

    open_positions = gcs_read_json("state/open_positions.json", default={})
    last_report    = gcs_read_json("state/last_report.json",    default={"hour":-1})
    last_weekly    = gcs_read_json("state/last_weekly.json",    default={"week":-1})

    sys.path.insert(0, "/app/strategies")
    try:
        from yagami_mtf_v77 import generate_signals
    except ImportError:
        logger.error("yagami_mtf_v77 import failed"); return {"status":"error","message":"import failed"}

    # ── 1. 既存ポジション決済チェック ────────────────
    for tid in list(open_positions.keys()):
        pos = open_positions[tid]; pair = pos["pair"]
        cfg = PAIRS.get(pair, {}); ps = cfg.get("pip_size", 0.0001)
        instr = cfg.get("oanda", pair)
        price = get_current_price(instr)
        if price == 0.0: continue
        ep=pos["ep"]; sl=pos["sl"]; tp=pos["tp"]; d=pos["dir"]
        exit_type = None
        if d==1:
            if price<=sl: exit_type="SL"
            elif price>=tp: exit_type="TP"
        else:
            if price>=sl: exit_type="SL"
            elif price<=tp: exit_type="TP"
        if exit_type:
            res = close_trade(tid)
            xp = res.get("exit_price", price)
            pnl = (xp-ep)*d/ps
            notify_close(pair, d, ep, xp, exit_type, pnl, pos.get("tf","?"))
            gcs_append_csv("logs/paper_trades.csv", {
                "trade_id":tid,"pair":pair,"dir":d,"tf":pos.get("tf","?"),
                "ep":round(ep,5),"sl":round(sl,5),"tp":round(tp,5),
                "exit_price":round(xp,5),"exit_type":exit_type,
                "pnl":round(pnl,1),"entry_time":pos.get("entry_time",""),"exit_time":now.isoformat()})
            del open_positions[tid]
            logger.info(f"CLOSED: {pair} {exit_type} pnl={pnl:.1f}p")

    # ── 2. 新規シグナルチェック ───────────────────────
    if len(open_positions) < MAX_OPEN_POSITIONS:
        for pair, cfg in PAIRS.items():
            if len(open_positions) >= MAX_OPEN_POSITIONS: break
            if any(p["pair"]==pair for p in open_positions.values()): continue
            instr=cfg["oanda"]; ps=cfg["pip_size"]; spread_p=cfg["spread"]*ps
            d1m=get_candles(instr,"M1",300); d15m=get_candles(instr,"M15",300); d4h=get_candles(instr,"H4",200)
            if any(len(d)<50 for d in [d1m,d15m,d4h]): continue
            try:
                sigs = generate_signals(d1m, d15m, d4h, spread_pips=spread_p)
            except Exception as e: logger.error(f"{pair} signal: {e}"); continue
            if not sigs: continue
            latest = max(sigs, key=lambda s: s["time"])
            age = (now - latest["time"].replace(tzinfo=timezone.utc)).total_seconds()/60 if latest["time"].tzinfo is None else (now - latest["time"]).total_seconds()/60
            if age > 5: continue
            if sum(1 for p in open_positions.values() if p["dir"]==latest["dir"]) >= MAX_SAME_DIR:
                gcs_append_csv("logs/paper_signals.csv",{"signal_time":latest["time"].isoformat(),"pair":pair,"dir":latest["dir"],"tf":latest.get("tf","?"),"ep":round(latest["ep"],5),"sl":round(latest["sl"],5),"tp":round(latest["tp"],5),"status":"SKIPPED_CORR"})
                continue
            res = place_order(instr, cfg["units"]*latest["dir"], latest["sl"], latest["tp"])
            if res:
                tid=res["trade_id"]; fp=res.get("fill_price",latest["ep"])
                slip=(fp-latest["ep"])*latest["dir"]/ps
                open_positions[tid]={"pair":pair,"dir":latest["dir"],"ep":latest["ep"],"sl":latest["sl"],"tp":latest["tp"],"tf":latest.get("tf","?"),"entry_time":now.isoformat(),"fill_price":fp,"slippage":slip}
                gcs_append_csv("logs/paper_signals.csv",{"signal_time":latest["time"].isoformat(),"pair":pair,"dir":latest["dir"],"tf":latest.get("tf","?"),"ep":round(latest["ep"],5),"sl":round(latest["sl"],5),"tp":round(latest["tp"],5),"status":"ENTERED"})
                notify_entry(pair, latest["dir"], latest["ep"], latest["sl"], latest["tp"], latest.get("tf","?"), slip)
                logger.info(f"ENTERED: {pair} {'L' if latest['dir']>0 else 'S'} ep={latest['ep']:.5f}")

    # ── 3. 定時レポート（9時・21時 JST） ─────────────
    now_jst_h = (now.hour+9)%24
    if now_jst_h in (9,21) and last_report.get("hour")!=now_jst_h:
        send_daily_report(open_positions); last_report={"hour":now_jst_h}
    elif now_jst_h not in (9,21): last_report={"hour":-1}

    # ── 4. 週次フィードバック（月曜0時 JST） ─────────
    now_jst = now + timedelta(hours=9)
    week_no = now_jst.isocalendar()[1]
    if now_jst.weekday()==0 and now_jst.hour==0 and last_weekly.get("week")!=week_no:
        send_weekly_feedback(); last_weekly={"week":week_no}

    # ── 5. 状態保存 ──────────────────────────────────
    gcs_write_json("state/open_positions.json", open_positions)
    gcs_write_json("state/last_report.json",    last_report)
    gcs_write_json("state/last_weekly.json",    last_weekly)
    logger.info(f"cycle end: open={len(open_positions)}/{MAX_OPEN_POSITIONS}")
    return {"status":"ok","open_positions":len(open_positions),"pairs_monitored":len(PAIRS)}

# ── エンドポイント ────────────────────────────────────
@app.post("/run")
async def run_endpoint(request: Request):
    try: return run_cycle()
    except Exception as e:
        logger.error(f"cycle error: {e}", exc_info=True)
        send_discord(f"⚠️ **Cloud Run エラー**: {str(e)[:200]}")
        return {"status":"error","message":str(e)}

@app.get("/health")
async def health():
    return {"status":"ok","pairs_monitored":len(PAIRS),"time":datetime.now(timezone.utc).isoformat()}

@app.post("/report")
async def report_endpoint():
    try:
        send_daily_report(gcs_read_json("state/open_positions.json",default={}))
        return {"status":"ok"}
    except Exception as e: return {"status":"error","message":str(e)}

@app.post("/weekly_feedback")
async def weekly_feedback_endpoint():
    try: send_weekly_feedback(); return {"status":"ok","message":"週次フィードバック送信完了"}
    except Exception as e: return {"status":"error","message":str(e)}

@app.get("/feedback_history")
async def feedback_history_endpoint():
    df = gcs_read_csv("logs/weekly_feedback.csv")
    return {"status":"ok","weeks":len(df),"data":df.to_dict(orient="records") if len(df)>0 else []}

@app.get("/pair_stats")
async def pair_stats_endpoint():
    stats = gcs_read_json("logs/pair_stats.json",default={})
    return {"status":"ok","pairs":len(stats),"data":stats}

@app.post("/notify_test")
async def notify_test_endpoint():
    send_discord("",embeds=[{"title":"🔔 接続テスト（全68ペア版）","color":0x00bfff,
        "fields":[{"name":"監視ペア","value":f"{len(PAIRS)}ペア","inline":True},
                  {"name":"週次FB","value":"毎週月曜0時(JST)","inline":True},
                  {"name":"ステータス","value":"✅ 正常稼働中","inline":False}],
        "footer":{"text":f"sena3fx v77 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"}}])
    return {"status":"ok"}

@app.get("/status")
async def status():
    op = gcs_read_json("state/open_positions.json",default={})
    wf = gcs_read_csv("logs/weekly_feedback.csv")
    return {"open_positions":len(op),"positions":op,"pairs_monitored":len(PAIRS),
            "weekly_feedbacks_stored":len(wf),"time":datetime.now(timezone.utc).isoformat()}
