"""
main.py - Cloud Run 自動取引bot (YAGAMI改 採用銘柄集中版)
=================================================================
【ブローカー切り替え】
  BROKER=oanda   → OANDA v20 REST API（ペーパートレード）
  BROKER=exness  → MetaApi経由 Exness MT5（本番取引、Windows不要）

【APPROVED_UNIVERSE (採用銘柄)】
  USDJPY: v77       / Tier1 / base 3%  (OOS PF=4.96, Kelly=0.608)
  XAUUSD: v79A      / Tier2 / base 2%  (OOS PF=2.16, Goldロジック=日足EMA20)
  GBPUSD: v79A      / Tier3 / base 1%  (OOS PF=1.86, Goldロジック=日足EMA20, FXカテゴリ最高)

【動的リスク調整】
  直近30トレードの勝率に基づいて乗数を自動調整:
  WR ≥ 60%: ×1.5 (好調 → 増量)
  WR 35-60%: ×1.0 (標準)
  WR < 35%:  ×0.5 (不調 → 減量)
  最大 5% / 最小 0.5% の安全上限

【戦略バリアント】
  USDJPY: yagami_mtf_v77 (KMID+KLOWフィルター)
  XAUUSD: yagami_mtf_v79 (use_1d_trend=True, Goldロジック)
  GBPUSD: yagami_mtf_v79 (use_1d_trend=True, Goldロジック ← ADX+Streakより優位 PF1.86 vs 1.66)
"""
import os, json, logging, requests, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Request
from google.cloud import storage

# ── ブローカー選択 ────────────────────────────────────────────
BROKER_TYPE = os.environ.get("BROKER", "oanda").lower()

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK",
    "https://discord.com/api/webhooks/1481623536182362255/i-Bo7MBagWWA7-4L93F9aORYeRUeBDuqSlIkvVcygyIy4s9LKcIfI1ng1XdR5mfkBZnd")
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "sena3fx-paper-trading")
PROJECT_ID  = os.environ.get("GCP_PROJECT", "aiyagami")

def _create_broker():
    """環境変数に基づいてブローカーインスタンスを生成"""
    if BROKER_TYPE == "exness":
        from broker_metaapi import MetaApiBroker
        return MetaApiBroker(
            token=os.environ.get("METAAPI_TOKEN", ""),
            account_id=os.environ.get("METAAPI_ACCOUNT_ID", ""),
            equity_jpy=float(os.environ.get("EQUITY_JPY", "1000000")),
        )
    else:
        from broker_oanda import OandaBroker
        return OandaBroker(
            token=os.environ.get("OANDA_TOKEN",
                "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"),
            account_id=os.environ.get("OANDA_ACCOUNT", "101-009-38652105-001"),
        )

broker = _create_broker()

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
        "strategy_params": {"use_1d_trend": True},  # Goldロジック: 日足EMA20方向一致
        "tier":          3,
        "base_risk_pct": 0.01,                    # FXカテゴリ次点候補 → リスク最小
        "oos_pf":        1.86,
        "kelly":         0.30,
        "note":          "FX v79A Goldロジック (次点候補)",
    },
}

# ペア数削減: 全68→採用3銘柄のみ
MAX_OPEN_POSITIONS = 3    # 採用銘柄数に合わせる（銘柄ごとに1ポジション上限）
RR_RATIO           = 2.5
HALF_R             = 1.0   # 半利確トリガー: 1R到達で50%決済 → SLをBEへ
CANDLE_COUNT       = 200

ACCOUNT_BALANCE_JPY = float(os.environ.get("EQUITY_JPY", "1000000"))  # .envフォールバック
MIN_UNITS           = 100
MAX_UNITS           = 100_000

_cached_equity = {"value": 0.0, "ts": 0.0}   # エクイティキャッシュ（5分間有効）

def _get_equity_jpy() -> float:
    """ブローカーからエクイティを動的取得。5分キャッシュ、失敗時は.env値。"""
    import time
    now = time.time()
    if _cached_equity["value"] > 0 and (now - _cached_equity["ts"]) < 300:
        return _cached_equity["value"]
    try:
        eq = broker.get_account_equity()
        if eq > 0:
            _cached_equity["value"] = eq
            _cached_equity["ts"] = now
            return eq
    except Exception as e:
        logger.warning(f"equity fetch failed: {e}")
    return _cached_equity["value"] if _cached_equity["value"] > 0 else ACCOUNT_BALANCE_JPY


# ── リスク計算 ────────────────────────────────────────────────
def calc_units_from_risk(ep: float, sl: float, pip_size: float,
                         pair: str, risk_pct: float) -> int:
    """
    指定risk_pct（銘柄別・動的）でロット数を計算する。
    エクイティはブローカーから動的取得（JPY換算済み）。

    pip_value（1pip = 何円か）:
        JPYクロス (XXX_JPY): pip_size円/unit
        USD建て・その他:      pip_size × 150円/unit（USDJPY≈150想定）
    """
    sl_pips = abs(ep - sl) / pip_size
    if sl_pips <= 0:
        return MIN_UNITS

    pair = pair.replace("_", "")
    if pair.endswith("JPY"):
        pip_val_per_unit = pip_size
    else:
        pip_val_per_unit = pip_size * 150.0

    equity_jpy = _get_equity_jpy()
    risk_amount_jpy = equity_jpy * risk_pct
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


def notify_half_profit(pair, dir_, ep, half_price, pnl_pips, tf):
    cfg = APPROVED_UNIVERSE.get(pair, {}); ps = cfg.get("pip_size", 0.0001)
    embed = {
        "title": f"🔶 半利確: {pair} {'LONG' if dir_>0 else 'SHORT'}",
        "color": 0xff9800,
        "fields": [
            {"name": "EP",       "value": f"`{ep:.5f}`",              "inline": True},
            {"name": "半利確値", "value": f"`{half_price:.5f}`",      "inline": True},
            {"name": "1R損益",   "value": f"**`{pnl_pips:+.1f}pips`**", "inline": True},
            {"name": "処理",     "value": "50%決済 → SLをBEへ移動",    "inline": False},
            {"name": "足種",     "value": f"`{tf}`",                   "inline": True},
            {"name": "時刻",     "value": datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC"), "inline": True},
        ],
        "footer": {"text": f"YAGAMI改 | {cfg.get('note','?')} | 残50%はTP or BEで決済"}
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
    if n > 0 and "pnl" in trades_df.columns:
        trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce")
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

    # 現在のオープンポジション（含み損益付き）
    pos_lines = []
    for tid, p in open_positions.items():
        pair = p["pair"]
        cfg  = APPROVED_UNIVERSE.get(pair, {})
        cur = get_current_price(pair)
        ps  = cfg.get("pip_size", 0.0001)
        if cur > 0:
            unreal = (cur - p["ep"]) * p["dir"] / ps
            pos_lines.append(
                f"  {pair} {'L' if p['dir']>0 else 'S'} EP:{p['ep']:.5f} "
                f"現値:{cur:.5f} 含み:{unreal:+.1f}p"
            )
        else:
            pos_lines.append(f"  {pair} {'L' if p['dir']>0 else 'S'} EP:{p['ep']:.5f}")
    pos_str = "\n".join(pos_lines) or "  なし"

    embed = {
        "title": "📊 朝9時レポート (YAGAMI改)",
        "color": 0x1565c0,
        "fields": [
            {"name": "累計トレード", "value": f"`{n}件`",         "inline": True},
            {"name": "勝率",         "value": f"`{wr:.1f}%`",     "inline": True},
            {"name": "PF",           "value": f"`{pf:.2f}`",      "inline": True},
            {"name": "オープン",     "value": f"`{len(open_positions)}件`", "inline": True},
            {"name": "監視銘柄",     "value": f"`{', '.join(APPROVED_UNIVERSE.keys())}`", "inline": True},
            {"name": "銘柄別成績",   "value": f"```{sym_stats or 'データなし'}```", "inline": False},
            {"name": "ポジション（含み損益）", "value": f"```{pos_str}```", "inline": False},
        ],
        "footer": {"text": f"YAGAMI改 | {datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M UTC')}"}
    }
    send_discord("", embeds=[embed])


def log_weekly_feedback(trades_df):
    """週次成績をGCSにログ記録（Discord通知なし）。"""
    now_utc    = datetime.now(timezone.utc)
    week_start = now_utc - timedelta(days=7)
    n = len(trades_df)
    if n == 0:
        logger.info("週次FB: データなし")
        return

    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")
    wdf  = trades_df[trades_df["exit_time"] >= week_start]
    wn   = len(wdf)
    wwr  = (wdf["pnl"] > 0).mean() * 100 if wn > 0 else 0
    wpnl = wdf["pnl"].sum() if wn > 0 else 0
    pf_all = (trades_df[trades_df["pnl"] > 0]["pnl"].sum() /
              abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
              if len(trades_df[trades_df["pnl"] < 0]) > 0 else 0)

    gcs_append_csv("logs/weekly_feedback.csv", {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end":   now_utc.strftime("%Y-%m-%d"),
        "week_trades": wn, "week_wr": round(wwr, 1),
        "week_pnl": round(wpnl, 1), "total_trades": n, "total_pf": round(pf_all, 2),
    })
    logger.info(f"週次GCSログ: {wn}件, 累計{n}件, PF={pf_all:.2f}")


# ── ブローカー経由ヘルパー（OANDA / Exness 共通） ──────────────
def get_candles(symbol, granularity, count=CANDLE_COUNT):
    return broker.get_candles(symbol, granularity, count)

def get_current_price(symbol):
    return broker.get_current_price(symbol)

def place_order(symbol, units, sl, tp):
    return broker.place_order(symbol, units, sl, tp)

def close_trade(trade_id):
    return broker.close_trade(trade_id)


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

    d1m   = get_candles(pair, "M1",  300)
    d15m  = get_candles(pair, "M15", 300)
    d4h   = get_candles(pair, "H4",  200)
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

    open_positions  = gcs_read_json("state/open_positions.json",  default={})
    last_report     = gcs_read_json("state/last_report.json",     default={"key": ""})
    last_weekly     = gcs_read_json("state/last_weekly.json",     default={"week": -1})
    notified_sigs   = gcs_read_json("state/notified_signals.json", default={})
    # notified_sigs: {pair: "signal_iso_time"} — 同一シグナルへの重複通知防止

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

    # ── 1a. 半利確チェック（1R到達で50%決済 → SLをBEへ） ─────
    broker_positions = broker.get_open_positions()

    for tid, pos in list(open_positions.items()):
        if pos.get("half_done"):
            continue  # 既に半利確済み
        if tid not in broker_positions:
            continue  # ブローカーに存在しない（決済済み）→ 1bで処理

        pair = pos["pair"]
        cfg  = APPROVED_UNIVERSE.get(pair, {})
        ps   = cfg.get("pip_size", 0.0001)
        ep   = pos["ep"]; d = pos["dir"]
        risk = abs(ep - pos["sl"])
        half_target = ep + d * risk * HALF_R  # 1R到達価格

        price = get_current_price(pair)
        if price <= 0:
            continue

        # 1R到達判定
        reached = (d == 1 and price >= half_target) or \
                  (d == -1 and price <= half_target)
        if not reached:
            continue

        # ブローカーのポジションvolumeを取得して50%決済
        bp = broker_positions[tid]
        full_vol = bp.get("volume", 0)
        half_vol = round(full_vol / 2, 2)
        if half_vol < 0.01:
            half_vol = 0.01

        res = broker.partial_close(tid, half_vol)
        if res:
            # SLをBE（エントリー価格）に移動
            broker.modify_position(tid, sl=ep, tp=pos["tp"])
            pos["half_done"] = True
            pos["sl"] = ep  # 内部状態もBEに更新
            half_pnl = (half_target - ep) * d / ps
            notify_half_profit(pair, d, ep, half_target, half_pnl, pos.get("tf", "?"))
            logger.info(f"HALF PROFIT: {pair} {half_vol}lot closed at 1R, SL→BE")

    # ── 1b. 既存ポジション決済チェック ──────────────────────────
    # ブローカー側でSL/TPが約定するため、Cloud Run側では
    # ポジションの消滅を検知して記録する（二重決済を防止）。
    # broker_positions は 1a で取得済み
    closed_tids = []

    for tid, pos in list(open_positions.items()):
        pair = pos["pair"]
        cfg  = APPROVED_UNIVERSE.get(pair, {})
        ps   = cfg.get("pip_size", 0.0001)
        ep   = pos["ep"]; d = pos["dir"]

        if tid in broker_positions:
            continue  # まだブローカー側にポジションが存在 → 未決済

        # ブローカー側にポジションがない → SL/TPで約定済み
        # 現在価格で損益を推定（実際の約定価格はブローカー側で確定済み）
        price = get_current_price(pair)
        if price == 0.0:
            price = ep  # 価格取得失敗時はEPをフォールバック

        # exit_type を SL/TP で判定（距離ベース）
        sl_dist = abs(price - pos["sl"]) * d / ps
        tp_dist = abs(price - pos["tp"]) * d / ps
        exit_type = "TP" if tp_dist < sl_dist else "SL"
        pnl = (price - ep) * d / ps

        notify_close(pair, d, ep, price, exit_type, pnl, pos.get("tf", "?"))
        entry_time_str = pos.get("entry_time", "")
        try:
            entry_hour = datetime.fromisoformat(entry_time_str).hour
        except Exception:
            entry_hour = ""
        risk_pips = round(abs(ep - pos["sl"]) / ps, 1)
        gcs_append_csv("logs/paper_trades.csv", {
            "trade_id":  tid, "pair": pair, "dir": d, "tf": pos.get("tf", "?"),
            "ep": round(ep, 5), "sl": round(pos["sl"], 5), "tp": round(pos["tp"], 5),
            "exit_price": round(price, 5), "exit_type": exit_type,
            "pnl": round(pnl, 1), "strategy": cfg.get("strategy", "?"),
            "entry_time": entry_time_str, "exit_time": now.isoformat(),
            "entry_hour": entry_hour, "risk_pips": risk_pips
        })
        closed_tids.append(tid)
        logger.info(f"CLOSED: {pair} {exit_type} pnl={pnl:.1f}p (broker confirmed)")

    for tid in closed_tids:
        del open_positions[tid]

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

            # 重複シグナル防止: 同一シグナル時刻は1回のみエントリー通知
            sig_key = latest["time"].isoformat() if hasattr(latest["time"], "isoformat") else str(latest["time"])
            if notified_sigs.get(pair) == sig_key:
                logger.info(f"{pair}: シグナル重複スキップ ({sig_key})")
                continue

            ps    = sym_cfg["pip_size"]

            # 動的リスク調整
            risk_pct = calc_dynamic_risk_pct(
                pair, trades_df, sym_cfg["base_risk_pct"])

            # 現在価格再取得（スリッページ確認）
            current_price = get_current_price(pair)
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
                current_price, new_sl, ps, pair, risk_pct)

            res = place_order(pair, order_units * latest["dir"], new_sl, new_tp)
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
                    "half_done": False,
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
                notified_sigs[pair] = sig_key  # 重複通知防止ラベル更新
                notify_entry(pair, latest["dir"], fp, new_sl, new_tp,
                             latest.get("tf", "?"), slip,
                             order_units, risk_pct, sym_cfg["tier"])
                logger.info(
                    f"ENTERED: {pair} {'L' if latest['dir']>0 else 'S'} "
                    f"ep={fp:.5f} units={order_units} "
                    f"risk={risk_pct*100:.1f}% slip={slip:+.2f}p "
                    f"strategy={sym_cfg['strategy']}")

    # ── 3. 朝9時レポート（JST 9:00のみ・1日1回） ────────────
    now_jst    = now + timedelta(hours=9)
    now_jst_h  = now_jst.hour
    report_key = now_jst.strftime("%Y-%m-%d-09")   # "2026-03-10-09"
    if now_jst_h == 9 and last_report.get("key") != report_key:
        send_daily_report(open_positions, trades_df)
        last_report = {"key": report_key}
        gcs_write_json("state/last_report.json", last_report)   # 即時保存（重複防止）

    # ── 4. 週次GCSログ（月曜0時 JST・Discord通知なし） ───────
    week_no = now_jst.isocalendar()[1]
    if now_jst.weekday() == 0 and now_jst_h == 0 and last_weekly.get("week") != week_no:
        log_weekly_feedback(trades_df)
        last_weekly = {"week": week_no}

    # ── 5. 状態保存 ────────────────────────────────────────
    gcs_write_json("state/open_positions.json",  open_positions)
    gcs_write_json("state/last_report.json",     last_report)
    gcs_write_json("state/last_weekly.json",     last_weekly)
    gcs_write_json("state/notified_signals.json", notified_sigs)

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
    """手動レポート送信用（Schedulerからは呼ばない）。"""
    try:
        open_positions = gcs_read_json("state/open_positions.json", default={})
        trades_df = gcs_read_csv("logs/paper_trades.csv")
        send_daily_report(open_positions, trades_df)
        # /run 側の重複防止キーと同じ形式で書き込む
        now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
        report_key = now_jst.strftime("%Y-%m-%d-%H")
        gcs_write_json("state/last_report.json", {"key": report_key})
        return {"status": "ok", "note": "手動送信完了（/run側と重複防止キー統一済み）"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/weekly_feedback")
async def weekly_feedback_endpoint():
    try:
        trades_df = gcs_read_csv("logs/paper_trades.csv")
        log_weekly_feedback(trades_df)
        return {"status": "ok", "message": "週次GCSログ記録完了（Discord通知なし）"}
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


@app.post("/test_trade")
async def test_trade_endpoint():
    """テスト取引: 最小ロットでGBPUSD成行買い → 即決済。ブローカー接続・応答速度確認用。"""
    import time
    results = {"steps": []}

    # 1. 価格取得テスト
    t0 = time.time()
    price = broker.get_current_price("GBPUSD")
    t_price = round(time.time() - t0, 3)
    results["steps"].append({"action": "get_price", "price": price, "time_sec": t_price})
    if price <= 0:
        results["error"] = "価格取得失敗"
        return results

    # 2. 最小ロット成行買い（SL/TPは広めに設定）
    sl = round(price - 0.0100, 5)   # 100pips下
    tp = round(price + 0.0100, 5)   # 100pips上
    t0 = time.time()
    order = broker.place_order("GBPUSD", 100, sl, tp)  # 100 units = 0.01 lot (最小)
    t_order = round(time.time() - t0, 3)
    results["steps"].append({"action": "place_order", "result": order, "time_sec": t_order})
    if not order:
        results["error"] = "注文失敗"
        return results

    trade_id = order.get("trade_id", "")

    # 3. 即決済
    t0 = time.time()
    close = broker.close_trade(trade_id)
    t_close = round(time.time() - t0, 3)
    results["steps"].append({"action": "close_trade", "result": close, "time_sec": t_close})

    results["total_time_sec"] = round(t_price + t_order + t_close, 3)
    results["status"] = "ok" if close else "close_failed"

    # Discord通知
    send_discord("", embeds=[{
        "title": "🧪 テスト取引完了",
        "color": 0x00ff00 if close else 0xff0000,
        "fields": [
            {"name": "銘柄", "value": "GBPUSD (0.01 lot)", "inline": True},
            {"name": "価格取得", "value": f"{price:.5f} ({t_price}s)", "inline": True},
            {"name": "約定", "value": f"{order.get('fill_price', 'N/A')} ({t_order}s)", "inline": True},
            {"name": "決済", "value": f"{close.get('exit_price', 'N/A')} ({t_close}s)", "inline": True},
            {"name": "合計時間", "value": f"{results['total_time_sec']}秒", "inline": True},
        ],
    }])
    return results
