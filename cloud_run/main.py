"""
main.py - Cloud Run ペーパートレードbot (YAGAMI改 採用銘柄集中版)
=================================================================
【改修方針】
旧版: 全68ペアをv77で一律スキャン、同一2%リスク
新版: バックテスト検証済み採用銘柄のみ、Kelly基準のティア別リスク配分

【APPROVED_UNIVERSE (採用銘柄)】
  USDJPY: v77  / Tier1 / base 3%  (OOS PF=4.96, Kelly=0.608)
  XAUUSD: v79A / Tier2 / base 2%  (OOS PF=2.16, Kelly≈0.45)
  GBPUSD: v79BC+v81C / Tier3 / base 1%  (OOS PF=2.01, 試験的採用)

【動的リスク調整】
  直近30トレードの勝率に基づいて乗数を自動調整:
  WR ≥ 60%: ×1.5 (好調 → 増量)
  WR 35-60%: ×1.0 (標準)
  WR < 35%:  ×0.5 (不調 → 減量)
  最大 5% / 最小 0.5% の安全上限

【戦略バリアント】
  USDJPY: yagami_mtf_v77 (KMID+KLOWフィルター)
  XAUUSD: yagami_mtf_v79 (use_1d_trend=True)
  GBPUSD: yagami_mtf_v79 (adx_min=20, streak_min=4, ema_dist_min=1.0)
"""
import os, json, logging, requests, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── 採用銘柄ユニバース ─────────────────────────────────────────
# バックテスト検証済み銘柄のみ。OOS PF・Kelly基準に基づくティア配分。
APPROVED_UNIVERSE = {
    "USDJPY": {
        "oanda":         "USD_JPY",
        "pip_size":      0.01,
        "spread_pips":   0.4,
        "strategy":      "v77",
        "strategy_params": {},                    # v77デフォルト（KMID+KLOW）
        "tier":          1,
        "base_risk_pct": 0.03,                    # Kelly=0.608 → Half-Kelly≈3%
        "oos_pf":        4.96,
        "kelly":         0.608,
        "note":          "YAGAMI改 旗艦銘柄",
    },
    "XAUUSD": {
        "oanda":         "XAU_USD",
        "pip_size":      0.01,
        "spread_pips":   5.2,
        "strategy":      "v79",
        "strategy_params": {"use_1d_trend": True}, # v79A: 日足EMA20方向一致
        "tier":          2,
        "base_risk_pct": 0.02,                    # Kelly≈0.45 → 2%
        "oos_pf":        2.16,
        "kelly":         0.45,
        "note":          "METALS v79A",
    },
    "GBPUSD": {
        "oanda":         "GBP_USD",
        "pip_size":      0.0001,
        "spread_pips":   0.1,
        "strategy":      "v79",
        "strategy_params": {                       # v79BC+v81C: ADX+Streak+EMA距離
            "adx_min":       20,
            "streak_min":    4,
            "ema_dist_min":  1.0,
        },
        "tier":          3,
        "base_risk_pct": 0.01,                    # 試験的採用 → リスク最小
        "oos_pf":        2.01,
        "kelly":         0.30,
        "note":          "FX v79BC+v81C (試験)",
    },
}

# ペア数削減: 全68→採用3銘柄のみ
MAX_OPEN_POSITIONS = 3    # 採用銘柄数に合わせる（銘柄ごとに1ポジション上限）
RR_RATIO           = 2.5
CANDLE_COUNT       = 200

ACCOUNT_BALANCE_JPY = 3_000_000   # 証拠金 300万円
MIN_UNITS           = 100
MAX_UNITS           = 100_000


# ── リスク計算 ────────────────────────────────────────────────
def calc_units_from_risk(ep: float, sl: float, pip_size: float,
                         oanda_pair: str, risk_pct: float) -> int:
    """
    指定risk_pct（銘柄別・動的）でロット数を計算する。

    pip_value（1pip = 何円か）:
        JPYクロス (XXX_JPY): pip_size円/unit
        USD建て・その他:      pip_size × 150円/unit（USDJPY≈150想定）
    """
    sl_pips = abs(ep - sl) / pip_size
    if sl_pips <= 0:
        return MIN_UNITS

    pair = oanda_pair.replace("_", "")
    if pair.endswith("JPY"):
        pip_val_per_unit = pip_size
    else:
        pip_val_per_unit = pip_size * 150.0

    risk_amount_jpy = ACCOUNT_BALANCE_JPY * risk_pct
    units = int(risk_amount_jpy / (sl_pips * pip_val_per_unit))
    return max(MIN_UNITS, min(units, MAX_UNITS))


def calc_dynamic_risk_pct(pair: str, trades_df: pd.DataFrame,
                          base_risk_pct: float, min_sample: int = 30) -> float:
    """
    直近 min_sample トレードの勝率に基づいてリスク乗数を決定する。

    WR ≥ 60% → ×1.5 (好調: 増量)
    WR 35-60% → ×1.0 (標準)
    WR < 35%  → ×0.5 (不調: 減量)

    データ不足時はbase_risk_pctをそのまま返す。
    上限 5% / 下限 0.5% でキャップ。
    """
    if trades_df is None or len(trades_df) == 0:
        return base_risk_pct
    if "pair" not in trades_df.columns:
        return base_risk_pct

    pt = trades_df[trades_df["pair"] == pair].tail(min_sample)
    if len(pt) < min_sample:
        return base_risk_pct  # データ不足 → ベース維持

    pnl_vals = pd.to_numeric(pt["pnl"], errors="coerce").dropna()
    if len(pnl_vals) == 0:
        return base_risk_pct

    wr = (pnl_vals > 0).mean()
    if wr >= 0.60:
        mult = 1.5
    elif wr < 0.35:
        mult = 0.5
    else:
        mult = 1.0

    adjusted = base_risk_pct * mult
    return max(0.005, min(adjusted, 0.05))


logging.basicConfig(level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
app = FastAPI()


# ── GCS ──────────────────────────────────────────────────────
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
        b.upload_from_string(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type="application/json")
    except Exception as e: logger.error(f"GCS write {blob_name}: {e}")

def gcs_append_csv(blob_name, row):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        line = ",".join(str(row.get(k, "")) for k in row.keys())
        if b.exists():
            b.upload_from_string(
                b.download_as_text().rstrip("\n") + "\n" + line + "\n",
                content_type="text/csv")
        else:
            b.upload_from_string(
                ",".join(row.keys()) + "\n" + line + "\n",
                content_type="text/csv")
    except Exception as e: logger.error(f"GCS append {blob_name}: {e}")

def gcs_read_csv(blob_name):
    try:
        b = gcs_client().bucket(GCS_BUCKET).blob(blob_name)
        if b.exists():
            from io import StringIO
            return pd.read_csv(StringIO(b.download_as_text()))
    except Exception as e: logger.error(f"GCS read csv {blob_name}: {e}")
    return pd.DataFrame()


# ── Discord ───────────────────────────────────────────────────
def send_discord(content, embeds=None):
    try:
        r = requests.post(DISCORD_WEBHOOK,
            json={"content": content, **({"embeds": embeds} if embeds else {})},
            timeout=10)
        if r.status_code not in (200, 204):
            logger.warning(f"Discord: {r.status_code}")
    except Exception as e: logger.warning(f"Discord error: {e}")


def notify_entry(pair, dir_, ep, sl, tp, tf, slippage, units, risk_pct, tier):
    cfg = APPROVED_UNIVERSE.get(pair, {})
    ps  = cfg.get("pip_size", 0.0001)
    tier_label = f"Tier{tier}"
    embed = {
        "title": f"{'🟢 LONG' if dir_ > 0 else '🔴 SHORT'}: {pair} [{tier_label}]",
        "color": 0x00c853 if dir_ > 0 else 0xd50000,
        "fields": [
            {"name": "EP",    "value": f"`{ep:.5f}`",                         "inline": True},
            {"name": "SL",    "value": f"`{sl:.5f}` (-{abs(ep-sl)/ps:.1f}p)", "inline": True},
            {"name": "TP",    "value": f"`{tp:.5f}` (+{abs(tp-ep)/ps:.1f}p)", "inline": True},
            {"name": "足種",  "value": f"`{tf}`",                             "inline": True},
            {"name": "ロット","value": f"`{units:,} units`",                  "inline": True},
            {"name": "リスク","value": f"`{risk_pct*100:.1f}%`",              "inline": True},
            {"name": "Slip",  "value": f"`{slippage:+.2f}p`",                 "inline": True},
            {"name": "OOS PF","value": f"`{cfg.get('oos_pf','?')}`",          "inline": True},
            {"name": "時刻",  "value": datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC"), "inline": True},
        ],
        "footer": {"text": f"YAGAMI改 | {cfg.get('note','?')} | {cfg.get('strategy','?')}"}
    }
    send_discord("", embeds=[embed])


def notify_close(pair, dir_, ep, exit_price, exit_type, pnl, tf):
    emoji = "✅" if exit_type == "TP" else "❌"
    cfg = APPROVED_UNIVERSE.get(pair, {}); ps = cfg.get("pip_size", 0.0001)
    embed = {
        "title": f"{emoji} 決済: {pair} {'LONG' if dir_>0 else 'SHORT'} [{exit_type}]",
        "color": 0x00c853 if exit_type == "TP" else 0xd50000,
        "fields": [
            {"name": "損益", "value": f"**`{pnl:+.1f}pips`**",              "inline": True},
            {"name": "EP",   "value": f"`{ep:.5f}`",                         "inline": True},
            {"name": "決済", "value": f"`{exit_price:.5f}`",                  "inline": True},
            {"name": "足種", "value": f"`{tf}`",                             "inline": True},
            {"name": "時刻", "value": datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC"), "inline": True},
        ],
        "footer": {"text": f"YAGAMI改 | {cfg.get('note','?')}"}
    }
    send_discord("", embeds=[embed])


# ── レポート ──────────────────────────────────────────────────
def send_daily_report(open_positions, trades_df):
    n  = len(trades_df) if len(trades_df) > 0 else 0
    wr = (trades_df["pnl"] > 0).mean() * 100 if n > 0 else 0
    pf_pos = trades_df[trades_df["pnl"] > 0]["pnl"].sum() if n > 0 else 0
    pf_neg = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum()) if n > 0 else 1
    pf = pf_pos / pf_neg if pf_neg > 0 else 0

    # 銘柄別成績
    sym_stats = ""
    if n > 0 and "pair" in trades_df.columns:
        for sym, cfg in APPROVED_UNIVERSE.items():
            pt = trades_df[trades_df["pair"] == sym]
            if len(pt) == 0:
                continue
            sw = (pt["pnl"] > 0).sum(); sn = len(pt)
            sp = pt["pnl"].sum()
            recent_risk = calc_dynamic_risk_pct(sym, trades_df, cfg["base_risk_pct"])
            sym_stats += f"  {sym}: {sn}件 {sw/sn*100:.0f}% PnL:{sp:+.0f}p risk:{recent_risk*100:.1f}%\n"

    pos_str = "\n".join(
        f"  {p['pair']} {'L' if p['dir']>0 else 'S'} {p['ep']:.5f}"
        for p in open_positions.values()) or "  なし"

    embed = {
        "title": "📊 日次レポート (YAGAMI改)",
        "color": 0x1565c0,
        "fields": [
            {"name": "累計トレード", "value": f"`{n}件`",         "inline": True},
            {"name": "勝率",         "value": f"`{wr:.1f}%`",     "inline": True},
            {"name": "PF",           "value": f"`{pf:.2f}`",      "inline": True},
            {"name": "オープン",     "value": f"`{len(open_positions)}件`", "inline": True},
            {"name": "採用銘柄",     "value": f"`{len(APPROVED_UNIVERSE)}銘柄`", "inline": True},
            {"name": "銘柄別成績",   "value": f"```{sym_stats or 'データなし'}```", "inline": False},
            {"name": "ポジション",   "value": f"```{pos_str}```", "inline": False},
        ],
        "footer": {"text": f"YAGAMI改 | {datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M UTC')}"}
    }
    send_discord("", embeds=[embed])


def send_weekly_feedback(trades_df):
    now_utc    = datetime.now(timezone.utc)
    week_start = now_utc - timedelta(days=7)

    if len(trades_df) == 0:
        send_discord(f"📅 **週次FB ({week_start.strftime('%m/%d')}〜)**\nデータなし")
        return

    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")
    wdf = trades_df[trades_df["exit_time"] >= week_start]
    n   = len(trades_df); wn = len(wdf)
    wwr = (wdf["pnl"] > 0).mean() * 100 if wn > 0 else 0
    wpnl = wdf["pnl"].sum() if wn > 0 else 0

    # ペア別今週成績
    sym_fields = []
    if wn > 0 and "pair" in wdf.columns:
        for sym, cfg in APPROVED_UNIVERSE.items():
            pt = wdf[wdf["pair"] == sym]
            if len(pt) == 0: continue
            sw = (pt["pnl"] > 0).sum(); sn = len(pt)
            sp = pt["pnl"].sum()
            dynamic_risk = calc_dynamic_risk_pct(sym, trades_df, cfg["base_risk_pct"])
            sym_fields.append({
                "name": f"{sym} (Tier{cfg['tier']})",
                "value": (f"`{sn}件` | WR: `{sw/sn*100:.0f}%` | "
                          f"PnL: `{sp:+.0f}p` | 次回リスク: `{dynamic_risk*100:.1f}%`"),
                "inline": False,
            })

    pf_all = (trades_df[trades_df["pnl"] > 0]["pnl"].sum() /
              abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
              if len(trades_df[trades_df["pnl"] < 0]) > 0 else 0)

    if n >= 300 and pf_all >= 3.0:   stage = "🟢 採用基準到達（本番移行検討可）"
    elif n >= 100 and pf_all >= 2.0: stage = "🟡 蓄積中（基準PF接近）"
    else:                             stage = f"🔴 蓄積中 ({n}/100件)"

    embed = {
        "title": f"📅 週次FB ({week_start.strftime('%m/%d')}〜{now_utc.strftime('%m/%d')})",
        "color": 0x7b1fa2,
        "fields": [
            {"name": "今週トレード", "value": f"`{wn}件`",         "inline": True},
            {"name": "今週勝率",     "value": f"`{wwr:.1f}%`",     "inline": True},
            {"name": "今週損益",     "value": f"`{wpnl:+.1f}p`",   "inline": True},
            {"name": "累計トレード", "value": f"`{n}件`",           "inline": True},
            {"name": "累計PF",       "value": f"`{pf_all:.2f}`",   "inline": True},
            {"name": "本番判断",     "value": stage,                "inline": False},
            *sym_fields,
        ],
        "footer": {"text": f"YAGAMI改 | {now_utc.strftime('%Y/%m/%d UTC')}"}
    }
    send_discord("", embeds=[embed])

    gcs_append_csv("logs/weekly_feedback.csv", {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end":   now_utc.strftime("%Y-%m-%d"),
        "week_trades": wn, "week_wr": round(wwr, 1),
        "week_pnl": round(wpnl, 1), "total_trades": n, "total_pf": round(pf_all, 2),
        "stage": stage,
    })
    logger.info(f"週次FB送信: {wn}件, 累計{n}件, PF={pf_all:.2f}")


# ── OANDA ヘルパー ────────────────────────────────────────────
def get_candles(instrument, granularity, count=CANDLE_COUNT):
    try:
        r = requests.get(f"{BASE_URL}/v3/instruments/{instrument}/candles",
            headers=OANDA_HEADERS,
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=15)
        if r.status_code != 200: return pd.DataFrame()
        rows = [
            {"timestamp": pd.Timestamp(c["time"]),
             "open":  float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low":   float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
             "volume": int(c.get("volume", 0))}
            for c in r.json()["candles"] if c.get("complete", True)
        ]
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).set_index("timestamp").sort_index()
    except Exception as e:
        logger.error(f"candles {instrument}: {e}"); return pd.DataFrame()


def get_current_price(instrument):
    try:
        r = requests.get(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing",
            headers=OANDA_HEADERS, params={"instruments": instrument}, timeout=10)
        if r.status_code == 200:
            p = r.json().get("prices", [])
            if p: return (float(p[0]["asks"][0]["price"]) + float(p[0]["bids"][0]["price"])) / 2
    except Exception as e: logger.error(f"price {instrument}: {e}")
    return 0.0


def place_order(instrument, units, sl, tp):
    try:
        r = requests.post(f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders",
            headers=OANDA_HEADERS, timeout=10,
            json={"order": {
                "type": "MARKET", "instrument": instrument, "units": str(units),
                "stopLossOnFill":   {"price": f"{sl:.5f}", "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": f"{tp:.5f}", "timeInForce": "GTC"},
                "timeInForce": "FOK", "positionFill": "DEFAULT"}})
        if r.status_code in (200, 201):
            fill = r.json().get("orderFillTransaction", {})
            return {"trade_id":  fill.get("tradeOpened", {}).get("tradeID", ""),
                    "fill_price": float(fill.get("price", 0))}
        logger.error(f"order {instrument}: {r.status_code} {r.text[:100]}")
    except Exception as e: logger.error(f"order {instrument}: {e}")
    return {}


def close_trade(trade_id):
    try:
        r = requests.put(
            f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades/{trade_id}/close",
            headers=OANDA_HEADERS, timeout=10)
        if r.status_code == 200:
            return {"exit_price": float(r.json().get("orderFillTransaction", {}).get("price", 0))}
    except Exception as e: logger.error(f"close {trade_id}: {e}")
    return {}


# ── シグナル取得（採用銘柄専用） ──────────────────────────────
def fetch_symbol_signal(pair, sym_cfg, open_positions,
                        gen_v77, gen_v79, now):
    """
    1銘柄のシグナルを取得して返す（ThreadPoolExecutor用）。
    銘柄設定に従い v77 / v79 を自動選択。
    """
    # 既存ポジションがあればスキップ
    if any(p["pair"] == pair for p in open_positions.values()):
        return None

    instr = sym_cfg["oanda"]
    d1m   = get_candles(instr, "M1",  300)
    d15m  = get_candles(instr, "M15", 300)
    d4h   = get_candles(instr, "H4",  200)
    if any(len(d) < 50 for d in [d1m, d15m, d4h]):
        return None

    strategy_fn     = gen_v77 if sym_cfg["strategy"] == "v77" else gen_v79
    strategy_params = sym_cfg["strategy_params"]

    try:
        sigs = strategy_fn(
            d1m, d15m, d4h,
            spread_pips=sym_cfg["spread_pips"],
            pip_size=sym_cfg["pip_size"],
            **strategy_params
        )
    except Exception as e:
        logger.error(f"{pair} signal error: {e}"); return None

    if not sigs:
        return None

    latest = max(sigs, key=lambda s: s["time"])
    age = (now - latest["time"].replace(tzinfo=timezone.utc)).total_seconds() / 60 \
          if latest["time"].tzinfo is None else \
          (now - latest["time"]).total_seconds() / 60
    if age > 2:
        return None  # 2分以上古いシグナルは無効

    return (pair, sym_cfg, latest)


# ── メインサイクル ────────────────────────────────────────────
def run_cycle():
    now = datetime.now(timezone.utc)
    logger.info(f"cycle start: {now.isoformat()} | 採用{len(APPROVED_UNIVERSE)}銘柄")

    open_positions = gcs_read_json("state/open_positions.json", default={})
    last_report    = gcs_read_json("state/last_report.json",    default={"hour": -1})
    last_weekly    = gcs_read_json("state/last_weekly.json",    default={"week": -1})

    # 戦略インポート（v77 と v79 の両方）
    sys.path.insert(0, "/app/strategies")
    try:
        from yagami_mtf_v77 import generate_signals as gen_v77
        from yagami_mtf_v79 import generate_signals as gen_v79
    except ImportError as e:
        logger.error(f"strategy import failed: {e}")
        return {"status": "error", "message": str(e)}

    # 累積トレードログ読み込み（動的リスク調整に使用）
    trades_df = gcs_read_csv("logs/paper_trades.csv")
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce")

    # ── 1. 既存ポジション決済チェック（並列） ──────────────────
    def check_close(tid_pos):
        tid, pos = tid_pos
        pair = pos["pair"]
        cfg   = APPROVED_UNIVERSE.get(pair, {})
        instr = cfg.get("oanda", pair)
        price = get_current_price(instr)
        if price == 0.0: return None
        ep, sl, tp, d = pos["ep"], pos["sl"], pos["tp"], pos["dir"]
        if d == 1:
            if price <= sl: return (tid, pos, "SL", price)
            if price >= tp: return (tid, pos, "TP", price)
        else:
            if price >= sl: return (tid, pos, "SL", price)
            if price <= tp: return (tid, pos, "TP", price)
        return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        close_results = list(ex.map(check_close, list(open_positions.items())))

    for result in close_results:
        if result is None: continue
        tid, pos, exit_type, price = result
        pair = pos["pair"]
        cfg  = APPROVED_UNIVERSE.get(pair, {}); ps = cfg.get("pip_size", 0.0001)
        ep   = pos["ep"]; d = pos["dir"]
        res  = close_trade(tid)
        xp   = res.get("exit_price", price)
        pnl  = (xp - ep) * d / ps
        notify_close(pair, d, ep, xp, exit_type, pnl, pos.get("tf", "?"))
        gcs_append_csv("logs/paper_trades.csv", {
            "trade_id":  tid, "pair": pair, "dir": d, "tf": pos.get("tf", "?"),
            "ep": round(ep, 5), "sl": round(pos["sl"], 5), "tp": round(pos["tp"], 5),
            "exit_price": round(xp, 5), "exit_type": exit_type,
            "pnl": round(pnl, 1), "strategy": cfg.get("strategy", "?"),
            "entry_time": pos.get("entry_time", ""), "exit_time": now.isoformat()
        })
        if tid in open_positions: del open_positions[tid]
        logger.info(f"CLOSED: {pair} {exit_type} pnl={pnl:.1f}p")

    # ── 2. 新規シグナルチェック（採用銘柄のみ・並列） ───────────
    if len(open_positions) < MAX_OPEN_POSITIONS:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(fetch_symbol_signal, pair, cfg,
                          open_positions, gen_v77, gen_v79, now): pair
                for pair, cfg in APPROVED_UNIVERSE.items()
                if not any(p["pair"] == pair for p in open_positions.values())
            }
            candidates = []
            for future in as_completed(futures, timeout=30):
                result = future.result()
                if result:
                    candidates.append(result)

        for pair, sym_cfg, latest in candidates:
            if len(open_positions) >= MAX_OPEN_POSITIONS: break
            if any(p["pair"] == pair for p in open_positions.values()): continue

            ps    = sym_cfg["pip_size"]
            instr = sym_cfg["oanda"]

            # 動的リスク調整
            risk_pct = calc_dynamic_risk_pct(
                pair, trades_df, sym_cfg["base_risk_pct"])

            # 現在価格再取得（スリッページ確認）
            current_price = get_current_price(instr)
            if current_price == 0.0:
                logger.warning(f"{pair}: 現在価格取得失敗、スキップ")
                continue

            price_drift = abs(current_price - latest["ep"]) / ps
            if price_drift > 3.0:
                logger.info(f"{pair}: 価格乖離 {price_drift:.1f}p > 3p、スキップ")
                gcs_append_csv("logs/paper_signals.csv", {
                    "signal_time": latest["time"].isoformat(), "pair": pair,
                    "dir": latest["dir"], "tf": latest.get("tf", "?"),
                    "ep": round(latest["ep"], 5), "status": "SKIPPED_DRIFT",
                    "drift_pips": round(price_drift, 1)
                })
                continue

            # SL/TP を現在価格ベースで再計算（幅は維持）
            sl_dist = abs(latest["ep"] - latest["sl"])
            tp_dist = abs(latest["tp"] - latest["ep"])
            new_sl  = current_price - latest["dir"] * sl_dist
            new_tp  = current_price + latest["dir"] * tp_dist

            # ロット計算（銘柄別・動的リスク）
            order_units = calc_units_from_risk(
                current_price, new_sl, ps, instr, risk_pct)

            res = place_order(instr, order_units * latest["dir"], new_sl, new_tp)
            if res:
                tid = res["trade_id"]; fp = res.get("fill_price", current_price)
                slip = (fp - latest["ep"]) * latest["dir"] / ps
                open_positions[tid] = {
                    "pair": pair, "dir": latest["dir"],
                    "ep": fp, "sl": new_sl, "tp": new_tp,
                    "tf": latest.get("tf", "?"),
                    "strategy": sym_cfg["strategy"],
                    "risk_pct": risk_pct,
                    "entry_time": now.isoformat(),
                    "fill_price": fp, "slippage": slip,
                    "signal_ep": latest["ep"],
                }
                gcs_append_csv("logs/paper_signals.csv", {
                    "signal_time": latest["time"].isoformat(), "pair": pair,
                    "dir": latest["dir"], "tf": latest.get("tf", "?"),
                    "ep": round(fp, 5), "signal_ep": round(latest["ep"], 5),
                    "sl": round(new_sl, 5), "tp": round(new_tp, 5),
                    "status": "ENTERED", "strategy": sym_cfg["strategy"],
                    "risk_pct": round(risk_pct * 100, 1),
                    "slippage_pips": round(slip, 2)
                })
                notify_entry(pair, latest["dir"], fp, new_sl, new_tp,
                             latest.get("tf", "?"), slip,
                             order_units, risk_pct, sym_cfg["tier"])
                logger.info(
                    f"ENTERED: {pair} {'L' if latest['dir']>0 else 'S'} "
                    f"ep={fp:.5f} units={order_units} "
                    f"risk={risk_pct*100:.1f}% slip={slip:+.2f}p "
                    f"strategy={sym_cfg['strategy']}")

    # ── 3. 定時レポート（9時・21時 JST） ─────────────────────
    now_jst_h = (now.hour + 9) % 24
    if now_jst_h in (9, 21) and last_report.get("hour") != now_jst_h:
        send_daily_report(open_positions, trades_df)
        last_report = {"hour": now_jst_h}
    elif now_jst_h not in (9, 21):
        last_report = {"hour": -1}

    # ── 4. 週次フィードバック（月曜0時 JST） ─────────────────
    now_jst = now + timedelta(hours=9)
    week_no = now_jst.isocalendar()[1]
    if now_jst.weekday() == 0 and now_jst.hour == 0 and last_weekly.get("week") != week_no:
        send_weekly_feedback(trades_df)
        last_weekly = {"week": week_no}

    # ── 5. 状態保存 ────────────────────────────────────────
    gcs_write_json("state/open_positions.json", open_positions)
    gcs_write_json("state/last_report.json",    last_report)
    gcs_write_json("state/last_weekly.json",    last_weekly)

    logger.info(f"cycle end: open={len(open_positions)}/{MAX_OPEN_POSITIONS}")
    return {
        "status": "ok",
        "open_positions": len(open_positions),
        "symbols_monitored": list(APPROVED_UNIVERSE.keys()),
        "dynamic_risk": {
            pair: round(calc_dynamic_risk_pct(
                pair, trades_df, cfg["base_risk_pct"]) * 100, 1)
            for pair, cfg in APPROVED_UNIVERSE.items()
        }
    }


# ── エンドポイント ────────────────────────────────────────────
@app.post("/run")
async def run_endpoint(request: Request):
    try:
        return run_cycle()
    except Exception as e:
        logger.error(f"cycle error: {e}", exc_info=True)
        send_discord(f"⚠️ **Cloud Run エラー**: {str(e)[:200]}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "symbols_monitored": list(APPROVED_UNIVERSE.keys()),
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.post("/report")
async def report_endpoint():
    try:
        open_positions = gcs_read_json("state/open_positions.json", default={})
        trades_df = gcs_read_csv("logs/paper_trades.csv")
        send_daily_report(open_positions, trades_df)
        now_jst_h = (datetime.now(timezone.utc).hour + 9) % 24
        gcs_write_json("state/last_report.json", {"hour": now_jst_h})
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/weekly_feedback")
async def weekly_feedback_endpoint():
    try:
        trades_df = gcs_read_csv("logs/paper_trades.csv")
        send_weekly_feedback(trades_df)
        return {"status": "ok", "message": "週次フィードバック送信完了"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/feedback_history")
async def feedback_history_endpoint():
    df = gcs_read_csv("logs/weekly_feedback.csv")
    return {"status": "ok", "weeks": len(df),
            "data": df.to_dict(orient="records") if len(df) > 0 else []}


@app.get("/status")
async def status():
    op = gcs_read_json("state/open_positions.json", default={})
    trades_df = gcs_read_csv("logs/paper_trades.csv")
    dynamic_risk = {
        pair: round(calc_dynamic_risk_pct(
            pair, trades_df, cfg["base_risk_pct"]) * 100, 1)
        for pair, cfg in APPROVED_UNIVERSE.items()
    }
    return {
        "open_positions":    len(op),
        "positions":         op,
        "symbols_monitored": list(APPROVED_UNIVERSE.keys()),
        "dynamic_risk_pct":  dynamic_risk,
        "universe": {
            pair: {"tier": cfg["tier"], "oos_pf": cfg["oos_pf"],
                   "strategy": cfg["strategy"], "note": cfg["note"]}
            for pair, cfg in APPROVED_UNIVERSE.items()
        },
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.post("/notify_test")
async def notify_test_endpoint():
    lines = "\n".join(
        f"  {pair}: Tier{cfg['tier']} | {cfg['strategy']} | "
        f"OOS PF={cfg['oos_pf']} | base risk={cfg['base_risk_pct']*100:.0f}%"
        for pair, cfg in APPROVED_UNIVERSE.items()
    )
    send_discord("", embeds=[{
        "title": "🔔 YAGAMI改 採用銘柄集中版",
        "color": 0x00bfff,
        "fields": [
            {"name": "採用銘柄",    "value": f"```{lines}```", "inline": False},
            {"name": "同時上限",    "value": f"`{MAX_OPEN_POSITIONS}件`", "inline": True},
            {"name": "動的リスク",  "value": "直近30件WRで±50%調整", "inline": True},
            {"name": "ステータス",  "value": "✅ 正常稼働中", "inline": True},
        ],
        "footer": {"text": f"YAGAMI改 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"}
    }])
    return {"status": "ok", "universe": list(APPROVED_UNIVERSE.keys())}
